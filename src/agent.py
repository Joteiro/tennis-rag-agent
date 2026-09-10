"""Agente LangGraph con RAG jerárquico y memoria de conversación.

El agente dispone de DOS herramientas de recuperación y un system prompt que le
impone la jerarquía de consulta: primero el reglamento del torneo (Moncloa) y,
solo si allí no está la respuesta, el reglamento general de la ITF.
"""
from __future__ import annotations

import re
import time

from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

from . import config


def _resolver_modelo_groq(preferido: str) -> str:
    """Auto-recuperación: devuelve un modelo Groq que esté DISPONIBLE ahora mismo.

    Consulta la lista de modelos de la cuenta. Si el ``preferido`` sigue vivo, lo
    usa; si fue deprecado, cae al primer respaldo disponible de
    ``config.GROQ_MODEL_FALLBACKS``. Si no se puede consultar la lista (sin red),
    devuelve el preferido y deja que la llamada falle de forma normal.
    """
    import os

    try:
        from groq import Groq

        disponibles = {m.id for m in Groq(api_key=os.environ["GROQ_API_KEY"]).models.list().data}
    except Exception:  # noqa: BLE001 - sin conectividad/lista: intentamos con el preferido
        return preferido

    if preferido in disponibles:
        return preferido
    for alt in config.GROQ_MODEL_FALLBACKS:
        if alt in disponibles:
            print(f"⚠️ Modelo Groq '{preferido}' no disponible; usando respaldo '{alt}'.")
            return alt
    print(f"⚠️ Ni '{preferido}' ni los respaldos están disponibles en Groq.")
    return preferido


def _crear_llm():
    """Instancia el LLM de chat según ``config.LLM_PROVIDER`` (groq o gemini)."""
    if config.LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        modelo = _resolver_modelo_groq(config.GROQ_MODEL)
        return ChatGroq(model=modelo, temperature=0)
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=config.CHAT_MODEL, temperature=0)
from .rag import formatear_documentos, obtener_vectorstores

# --- System prompt ------------------------------------------------------------
# Decisiones de diseño (justificadas en el README):
#   1. Rol acotado -> evita que Gemini improvise fuera del dominio.
#   2. Jerarquía explícita torneo -> ITF -> refleja cómo se resuelven las dudas
#      reglamentarias en la práctica (la norma local prevalece salvo vacío).
#   3. Citar fuente y artículo -> respuestas verificables, no "alucinadas".
#   4. Regla de "no sé" -> el agente admite cuando la info no está en los docs.
SYSTEM_PROMPT = """Eres un asistente experto en la normativa del "Ranking Federado \
del Distrito de Moncloa-Aravaca 2025-2026", un torneo de tenis amateur.

Tu base de conocimiento tiene DOS reglamentos y DEBES respetar esta jerarquía:

1. FUENTE PRINCIPAL — el reglamento del torneo (herramienta `buscar_reglamento_torneo`).
   Úsala SIEMPRE primero para cualquier pregunta sobre inscripción, ranking, puntos,
   partidos, plazos, categorías, sanciones o funcionamiento del torneo.

2. FUENTE DE RESPALDO — las Reglas del Tenis de la ITF (herramienta `buscar_reglamento_itf`).
   Úsala SOLO cuando el reglamento del torneo NO cubra la duda: reglas generales del
   juego (puntuación de un partido, tie-break, saque, cambios de lado, etc.).

Cómo debes comportarte:
- Antes de responder algo técnico, consulta la(s) herramienta(s) correspondiente(s).
- Si la respuesta está en el reglamento del torneo, respóndela con esa fuente y NO
  consultes la ITF.
- Si el torneo no lo cubre, indícalo y consulta la ITF como norma general.
- Cita SIEMPRE de dónde sacaste la información (reglamento y, si es posible, artículo/página).
- Si ninguno de los dos reglamentos contiene la respuesta, dilo con honestidad; NO inventes.
- Responde en español, de forma clara y concisa. Mantén la coherencia con lo hablado antes."""


def construir_agente(persistir_chroma: bool = True):
    """Construye el agente LangGraph completo (tools + LLM + memoria).

    Devuelve el grafo compilado. Para conversar hay que pasar un
    ``config={"configurable": {"thread_id": "..."}}`` en cada ``invoke``.
    """
    vs_torneo, vs_itf = obtener_vectorstores(persistir=persistir_chroma)
    ret_torneo = vs_torneo.as_retriever(search_kwargs={"k": config.TOP_K})
    ret_itf = vs_itf.as_retriever(search_kwargs={"k": config.TOP_K})

    @tool
    def buscar_reglamento_torneo(consulta: str) -> str:
        """Busca en el reglamento del torneo Ranking Federado Moncloa-Aravaca.
        Úsala para inscripción, sistema de puntos, ranking, disputa de partidos,
        plazos, categorías, penalizaciones y todo lo específico del torneo."""
        return formatear_documentos(ret_torneo.invoke(consulta))

    @tool
    def buscar_reglamento_itf(consulta: str) -> str:
        """Busca en las Reglas del Tenis de la ITF 2026 (normativa general).
        Úsala solo cuando el reglamento del torneo no cubra la duda: reglas
        generales del juego como puntuación, tie-break, saque o cambios de lado."""
        return formatear_documentos(ret_itf.invoke(consulta))

    llm = _crear_llm()

    agente = create_react_agent(
        model=llm,
        tools=[buscar_reglamento_torneo, buscar_reglamento_itf],
        prompt=SYSTEM_PROMPT,
        checkpointer=InMemorySaver(),  # memoria de conversación por thread_id
    )
    return agente


def extraer_texto(content) -> str:
    """Normaliza el ``content`` de la respuesta a texto plano.

    Gemini 2.5 puede devolver el contenido como lista de bloques
    (``[{"type": "text", "text": ...}]``) en vez de una cadena; unificamos.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        partes = []
        for bloque in content:
            if isinstance(bloque, dict) and bloque.get("type") == "text":
                partes.append(bloque.get("text", ""))
            elif isinstance(bloque, str):
                partes.append(bloque)
        return "\n".join(p for p in partes if p).strip()
    return str(content)


# Errores transitorios de la API que conviene reintentar: cuota por minuto (429)
# y sobrecarga temporal del modelo (503). No reintentamos otros errores.
_TRANSITORIOS = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "high demand")


def preguntar(agente, pregunta: str, thread_id: str = "sesion-demo", reintentos: int = 4) -> str:
    """Envía una pregunta al agente y devuelve solo el texto de la respuesta.

    Reintenta ante errores transitorios (429 por cuota, 503 por sobrecarga del
    modelo), respetando el ``retryDelay`` que sugiere la API cuando existe.
    """
    for intento in range(reintentos + 1):
        try:
            resultado = agente.invoke(
                {"messages": [{"role": "user", "content": pregunta}]},
                config={"configurable": {"thread_id": thread_id}},
            )
            return extraer_texto(resultado["messages"][-1].content)
        except Exception as e:  # noqa: BLE001 - reintento solo ante errores transitorios
            msg = str(e)
            if any(t in msg for t in _TRANSITORIOS) and intento < reintentos:
                m = re.search(r"retry.{0,3}(\d+(?:\.\d+)?)s", msg, re.IGNORECASE)
                espera = float(m.group(1)) + 1 if m else 8 * (intento + 1)
                print(f"  · Error transitorio de la API; reintentando en {espera:.0f}s…")
                time.sleep(espera)
            else:
                raise
