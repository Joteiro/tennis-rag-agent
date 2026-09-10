# -*- coding: utf-8 -*-
"""Health check del asistente de reglamentos de tenis.

Verifica que el sistema completo siga operativo:
  1. Las API keys necesarias están presentes.
  2. El modelo de chat de Groq está disponible (o hay un respaldo vivo).
  3. Los embeddings de Gemini responden.
  4. Una consulta de punta a punta devuelve una respuesta con sentido.

Uso:
    python scripts/healthcheck.py

Salida: reporte con ✅/❌ por chequeo. Código de salida 0 si todo pasa, 1 si algo
falla (lo aprovecha GitHub Actions para avisar). Está pensado para correr en un
cron y detectar a tiempo modelos deprecados, cuotas agotadas o dependencias rotas.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Permite ejecutar el script desde cualquier carpeta.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

resultados: list[tuple[str, bool, str]] = []


def check(nombre: str, fn) -> None:
    try:
        detalle = fn()
        resultados.append((nombre, True, detalle or "OK"))
    except Exception as e:  # noqa: BLE001 - queremos capturar cualquier fallo del chequeo
        resultados.append((nombre, False, f"{type(e).__name__}: {e}"))


def _check_keys() -> str:
    from src import config

    config.load_api_key()  # lanza si falta GOOGLE_API_KEY (o GROQ si provider=groq)
    return f"proveedor de chat = {config.LLM_PROVIDER}"


def _check_modelo_chat() -> str:
    from src import config

    if config.LLM_PROVIDER != "groq":
        return f"usando Gemini ({config.CHAT_MODEL})"
    from src.agent import _resolver_modelo_groq

    modelo = _resolver_modelo_groq(config.GROQ_MODEL)
    if modelo != config.GROQ_MODEL:
        raise RuntimeError(
            f"el modelo preferido '{config.GROQ_MODEL}' NO está disponible; "
            f"se usaría el respaldo '{modelo}'. Actualizá GROQ_MODEL."
        )
    return f"modelo Groq '{modelo}' disponible"


def _check_embeddings() -> str:
    from src import config
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    emb = GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL)
    vec = emb.embed_query("prueba de salud de embeddings")
    if not vec:
        raise RuntimeError("el embedding devolvió un vector vacío")
    return f"embeddings Gemini OK (dim={len(vec)})"


def _check_end_to_end() -> str:
    from src.agent import construir_agente, preguntar

    agente = construir_agente(persistir_chroma=True)
    respuesta = preguntar(agente, "¿cuánto tiempo de cortesía hay antes del W.O.?",
                          thread_id="healthcheck")
    if not respuesta or len(respuesta.strip()) < 20:
        raise RuntimeError(f"respuesta vacía o demasiado corta: {respuesta!r}")
    return f"respuesta de {len(respuesta)} caracteres"


def main() -> int:
    print("=" * 60)
    print("HEALTH CHECK — Asistente de Reglamentos de Tenis")
    print("=" * 60)

    check("1. API keys", _check_keys)
    # Los chequeos siguientes dependen de las keys; si fallaron, no seguimos.
    if resultados[-1][1]:
        check("2. Modelo de chat (Groq)", _check_modelo_chat)
        check("3. Embeddings (Gemini)", _check_embeddings)
        check("4. Consulta end-to-end", _check_end_to_end)

    print()
    ok = True
    for nombre, paso, detalle in resultados:
        icono = "✅" if paso else "❌"
        print(f"{icono} {nombre}: {detalle}")
        ok = ok and paso

    print()
    if ok:
        print("RESULTADO: ✅ TODO OPERATIVO")
        return 0
    print("RESULTADO: ❌ HAY FALLOS — revisá el detalle arriba.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
