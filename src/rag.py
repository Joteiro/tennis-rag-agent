"""Construcción de la base de conocimiento vectorial (RAG).

Se crean DOS colecciones independientes en ChromaDB, una por reglamento:

  * ``reglamento_torneo_moncloa`` -> fuente PRINCIPAL (torneo local)
  * ``reglamento_itf``            -> fuente de RESPALDO (reglas generales ITF)

Mantenerlas separadas permite que el agente enrute de forma explícita: primero
consulta el torneo y, si no encuentra la respuesta, recurre al reglamento ITF.
"""
from __future__ import annotations

import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from . import config

# El free tier de Gemini limita los embeddings a ~100 solicitudes/minuto. Llevamos
# una ventana deslizante (compartida entre ambas colecciones, que consumen la
# misma cuota) para no superarlo al indexar.
_EMB_TIMESTAMPS: list[float] = []


def _throttle_embeddings(n: int, rpm: int = 95) -> None:
    """Espera lo necesario para no exceder ``rpm`` embeddings en 60 s."""
    global _EMB_TIMESTAMPS
    now = time.time()
    _EMB_TIMESTAMPS = [t for t in _EMB_TIMESTAMPS if now - t < 60]
    while len(_EMB_TIMESTAMPS) + n > rpm and _EMB_TIMESTAMPS:
        espera = 60 - (now - _EMB_TIMESTAMPS[0]) + 0.5
        print(f"  · Límite de cuota alcanzado; esperando {espera:.0f}s…")
        time.sleep(max(espera, 1))
        now = time.time()
        _EMB_TIMESTAMPS = [t for t in _EMB_TIMESTAMPS if now - t < 60]
    _EMB_TIMESTAMPS.extend([time.time()] * n)


def _add_documentos_por_lotes(vs: Chroma, docs: list[Document], batch: int = 50) -> None:
    """Indexa los documentos en lotes, respetando el límite de cuota y reintentando."""
    for inicio in range(0, len(docs), batch):
        lote = docs[inicio : inicio + batch]
        _throttle_embeddings(len(lote))
        for intento in range(4):
            try:
                vs.add_documents(lote)
                break
            except Exception as e:  # noqa: BLE001 - reintento genérico ante 429
                if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                    espera = 15 * (intento + 1)
                    print(f"  · 429 recibido; reintento en {espera}s…")
                    time.sleep(espera)
                else:
                    raise
        else:
            raise RuntimeError("No se pudo indexar tras varios reintentos (cuota de Gemini).")


def _cargar_paginas(pdf_path: Path) -> list[Document]:
    """Lee un PDF con pypdf y devuelve un Document por página con texto."""
    reader = PdfReader(str(pdf_path))
    paginas = []
    for i, page in enumerate(reader.pages):
        texto = page.extract_text() or ""
        if texto.strip():
            paginas.append(Document(
                page_content=texto,
                metadata={"page": i, "source": pdf_path.name},
            ))
    return paginas


def _cargar_y_trocear(pdf_path: Path, fuente: str) -> list[Document]:
    """Carga un PDF, lo trocea y etiqueta cada fragmento con su fuente."""
    paginas = _cargar_paginas(pdf_path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(paginas)
    for ch in chunks:
        ch.metadata["fuente"] = fuente
        # page ya viene de PyPDFLoader (0-indexado); lo dejamos legible.
        ch.metadata["pagina"] = ch.metadata.get("page", 0) + 1
    return chunks


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=config.EMBEDDING_MODEL)


def construir_vectorstores(persistir: bool = True) -> tuple[Chroma, Chroma]:
    """Crea (o recrea) las dos colecciones de ChromaDB e indexa los documentos.

    Si ``persistir`` es True se guardan en disco (``chroma_db/``); si es False se
    construyen en memoria (útil en Streamlit Cloud, donde no hay disco persistente).
    """
    embeddings = _get_embeddings()
    persist_dir = str(config.CHROMA_DIR) if persistir else None

    docs_torneo = _cargar_y_trocear(config.DOC_TORNEO, "Reglamento Torneo Moncloa-Aravaca 2025-2026")
    docs_itf = _cargar_y_trocear(config.DOC_ITF, "Reglas del Tenis ITF 2026")

    vs_torneo = Chroma(
        collection_name=config.COLLECTION_TORNEO,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    vs_itf = Chroma(
        collection_name=config.COLLECTION_ITF,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    # Idempotencia: si la colección ya existía (re-ejecución de la celda), la
    # vaciamos antes de indexar para no acumular fragmentos duplicados.
    for vs in (vs_torneo, vs_itf):
        try:
            vs.reset_collection()
        except Exception:  # noqa: BLE001 - colección nueva; nada que resetear
            pass

    print(f"Indexando reglamento del torneo ({len(docs_torneo)} fragmentos)…")
    _add_documentos_por_lotes(vs_torneo, docs_torneo)
    print(f"Indexando reglamento ITF ({len(docs_itf)} fragmentos)…")
    _add_documentos_por_lotes(vs_itf, docs_itf)
    return vs_torneo, vs_itf


def cargar_vectorstores() -> tuple[Chroma, Chroma]:
    """Carga las colecciones ya persistidas en disco (sin reindexar)."""
    embeddings = _get_embeddings()
    persist_dir = str(config.CHROMA_DIR)
    vs_torneo = Chroma(
        collection_name=config.COLLECTION_TORNEO,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    vs_itf = Chroma(
        collection_name=config.COLLECTION_ITF,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    return vs_torneo, vs_itf


def obtener_vectorstores(persistir: bool = True) -> tuple[Chroma, Chroma]:
    """Devuelve las colecciones: las carga de disco si existen, o las construye."""
    if persistir and config.CHROMA_DIR.exists() and any(config.CHROMA_DIR.iterdir()):
        return cargar_vectorstores()
    return construir_vectorstores(persistir=persistir)


def formatear_documentos(docs: list[Document]) -> str:
    """Formatea los fragmentos recuperados citando fuente y página."""
    if not docs:
        return "No se encontró información relevante en este documento."
    bloques = []
    for i, doc in enumerate(docs, 1):
        fuente = doc.metadata.get("fuente", "desconocida")
        pagina = doc.metadata.get("pagina", "?")
        bloques.append(f"[Fragmento {i} · {fuente} · pág. {pagina}]\n{doc.page_content.strip()}")
    return "\n\n".join(bloques)
