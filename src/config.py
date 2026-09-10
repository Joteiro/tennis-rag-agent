"""Configuración central del asistente de reglamentos de tenis.

Todas las constantes y la carga de la API key de Gemini viven aquí para que
el notebook y la app de Streamlit compartan la misma fuente de verdad.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Rutas del proyecto -------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = ROOT_DIR / "chroma_db"

# Documentos de la base de conocimiento. El orden importa: el reglamento del
# torneo es la fuente PRINCIPAL y el reglamento ITF es la fuente de RESPALDO.
DOC_TORNEO = DATA_DIR / "Normas Ranking Federado Distrito de Moncloa-Aravaca 2025-2026.pdf"
DOC_ITF = DATA_DIR / "itf-reglas-del-tenis-2026.pdf"

# Nombres de las colecciones en ChromaDB (una por reglamento).
COLLECTION_TORNEO = "reglamento_torneo_moncloa"
COLLECTION_ITF = "reglamento_itf"

# --- Modelos ------------------------------------------------------------------
# Proveedor del LLM de chat: "groq" (free tier amplio, miles de req/día) o "gemini"
# (~20 req/día en free tier). Configurable por variable de entorno LLM_PROVIDER.
# Los EMBEDDINGS siempre usan Gemini (su cuota es independiente de la de chat).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()

# Modelo de chat según proveedor. Groq deprecó la línea Llama 3.x; usamos el
# modelo abierto de OpenAI servido por Groq, que soporta tool-calling.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# Modelos de respaldo (con tool-calling) para AUTO-RECUPERACIÓN: si el modelo
# configurado deja de estar disponible en Groq, el agente cae al primero de esta
# lista que siga vivo, en vez de romperse. Ver src/agent.py::_resolver_modelo_groq.
GROQ_MODEL_FALLBACKS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
    "qwen/qwen3.6-27b",
]
CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash-lite")

# Embeddings de Gemini (cuota separada de la de chat).
EMBEDDING_MODEL = "models/gemini-embedding-001"

# --- Parámetros de troceado (chunking) y recuperación -------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4


def load_api_key() -> str:
    """Carga las API keys necesarias desde el entorno o un archivo .env.

    - Embeddings: siempre Gemini → se requiere ``GOOGLE_API_KEY`` (o ``GEMINI_API_KEY``).
      `langchain-google-genai` espera ``GOOGLE_API_KEY``, así que normalizamos.
    - Chat: si ``LLM_PROVIDER == "groq"`` se requiere además ``GROQ_API_KEY``.

    Nunca se escribe ninguna clave en el código (buena práctica de la consigna).
    """
    # Busca un .env en la raíz del proyecto y en la carpeta de clase.
    for candidate in (ROOT_DIR / ".env", ROOT_DIR / ".env" / ".env"):
        if candidate.is_file():
            load_dotenv(candidate)
            break
    else:
        load_dotenv()  # comportamiento por defecto (variables de entorno / .env local)

    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError(
            "No se encontró la API key de Gemini (necesaria para los embeddings). "
            "Definí GOOGLE_API_KEY o GEMINI_API_KEY en un archivo .env."
        )
    os.environ["GOOGLE_API_KEY"] = key  # normalizamos para las librerías de Google

    if LLM_PROVIDER == "groq" and not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "LLM_PROVIDER=groq pero falta GROQ_API_KEY. Añadila al .env o poné "
            "LLM_PROVIDER=gemini para usar Gemini como LLM de chat."
        )
    return key
