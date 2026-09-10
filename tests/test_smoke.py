# -*- coding: utf-8 -*-
"""Smoke tests que NO consumen APIs (corren en CI sin secrets).

Validan que el grafo de imports esté sano (que una actualización de dependencias
no rompa el proyecto en silencio) y que la lógica pura se comporte.
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src import config
from src.agent import SYSTEM_PROMPT, extraer_texto


def test_imports_ok():
    import src.agent  # noqa: F401
    import src.config  # noqa: F401
    import src.rag  # noqa: F401


def test_extraer_texto_string():
    assert extraer_texto("hola") == "hola"


def test_extraer_texto_bloques_gemini():
    assert extraer_texto([{"type": "text", "text": "hola"}]) == "hola"


def test_extraer_texto_vacio():
    assert extraer_texto([]) == ""


def test_system_prompt_define_jerarquia():
    p = SYSTEM_PROMPT.lower()
    assert "torneo" in p
    assert "itf" in p


def test_config_basica():
    assert config.LLM_PROVIDER in {"groq", "gemini"}
    assert config.EMBEDDING_MODEL
    assert config.GROQ_MODEL_FALLBACKS, "debe haber al menos un modelo de respaldo"
    assert all(isinstance(m, str) and m for m in config.GROQ_MODEL_FALLBACKS)
