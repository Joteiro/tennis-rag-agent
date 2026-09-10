# 🎾 Asistente Experto en Reglamentos de Tenis (RAG + Agente con Gemini)

[![CI](https://github.com/Joteiro/tennis-rag-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Joteiro/tennis-rag-agent/actions/workflows/ci.yml)
[![Health check](https://github.com/Joteiro/tennis-rag-agent/actions/workflows/healthcheck.yml/badge.svg)](https://github.com/Joteiro/tennis-rag-agent/actions/workflows/healthcheck.yml)
[![Abrir en Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://tennis-agent.streamlit.app//)

Proyecto final del módulo de **IA Generativa** (Máster en Data Science, Evolve).

Un agente conversacional que responde dudas sobre el **Ranking Federado del
Distrito de Moncloa-Aravaca 2025-2026** (un torneo de tenis amateur) aplicando
**RAG** sobre su reglamento. Cuando la duda es sobre reglas generales del juego
que el reglamento del torneo no cubre, el agente **recurre automáticamente** a
las *Reglas del Tenis de la ITF 2026* como fuente de respaldo.

### 🌐 App en vivo

**Probá el asistente desplegado:** **https://tennis-agent.streamlit.app//**

> Nota: usa el free tier de Gemini/Groq; si la app estuvo inactiva, la primera
> pregunta puede tardar unos segundos mientras indexa los reglamentos.

---

## 🧠 Dominio elegido

Normativa deportiva de tenis, organizada en **dos niveles jerárquicos**:

| Nivel | Documento | Rol | Páginas |
|-------|-----------|-----|---------|
| Local | Normas del Ranking Federado Moncloa-Aravaca 2025-2026 | **Fuente principal** | 6 |
| General | Reglas del Tenis ITF 2026 | **Fuente de respaldo** | 50 |

La gracia del proyecto es esa jerarquía: para una pregunta como *"¿cuántos
puntos gano si gano un partido?"* la respuesta está en el reglamento del torneo;
para *"¿cómo funciona un tie-break?"* no está en el torneo y el agente cae a la
ITF. Esa **decisión de enrutamiento** es lo que hace al sistema *agéntico*.

---

## 🏗️ Arquitectura

```
Pregunta ──▶ Agente LangGraph (create_react_agent, Gemini)
                 │  system prompt con jerarquía torneo ▶ ITF
                 ├─▶ tool: buscar_reglamento_torneo ──▶ Chroma (colección torneo)
                 └─▶ tool: buscar_reglamento_itf    ──▶ Chroma (colección ITF)
                 │
                 ▼
            Respuesta citando fuente + memoria por thread_id
```

- **LLM de chat:** configurable vía `LLM_PROVIDER` — **Groq** (`openai/gpt-oss-120b`,
  free tier amplio, por defecto) o **Gemini** (`gemini-2.5-flash-lite`). Ambos vía LangChain.
- **Embeddings:** Google Gemini (`gemini-embedding-001`) — su cuota es independiente de la de chat.
- **Base vectorial:** ChromaDB, **dos colecciones** (una por reglamento) para que
  el enrutamiento sea explícito.
- **Agente:** `create_react_agent` de **LangGraph**, con dos herramientas de
  recuperación y memoria de conversación (`InMemorySaver` + `thread_id`).
- **Chunking:** `RecursiveCharacterTextSplitter` (1000 / 150 de solape).

Código organizado como fuente única de verdad en `src/`, reutilizado tanto por el
notebook como por la app de Streamlit:

```
tennis-rag-agent/
├── data/                       # los dos PDFs
├── src/
│   ├── config.py               # rutas, modelos, carga de API key
│   ├── rag.py                  # carga PDFs, chunking, ChromaDB, retrievers
│   └── agent.py                # tools + system prompt + agente LangGraph
├── notebooks/
│   └── asistente_tenis.ipynb   # ENTREGABLE principal (demo + 5 ejemplos)
├── app.py                      # interfaz Streamlit (bonus)
├── requirements.txt
├── .env.example
└── README.md
```

---

## ✍️ System prompt: justificación de las decisiones

El system prompt completo está en [`src/agent.py`](src/agent.py). Decisiones clave:

1. **Rol acotado al torneo.** Se define al agente como experto en la normativa de
   Moncloa-Aravaca, no como un chatbot de tenis genérico. Esto evita que Gemini
   improvise con conocimiento externo no verificable.
2. **Jerarquía de fuentes explícita (torneo ▶ ITF).** Se instruye a usar *siempre
   primero* la herramienta del torneo y recurrir a la ITF *solo* si el torneo no
   cubre la duda. Refleja cómo se resuelven las dudas reglamentarias en la
   práctica: la norma local prevalece salvo vacío normativo.
3. **Obligación de citar la fuente.** Cada respuesta debe indicar de qué
   reglamento (y, si es posible, artículo/página) proviene → respuestas
   verificables, no alucinadas.
4. **Regla de honestidad ("no lo sé").** Si ninguno de los dos reglamentos
   contiene la respuesta, el agente debe admitirlo en lugar de inventar.
5. **Idioma y estilo.** Respuestas en español, claras y concisas, manteniendo la
   coherencia con los turnos previos de la conversación.

---

## 🚀 Instalación y ejecución

### 1. Requisitos
- Python **3.11** (recomendado; `chromadb` aún no publica wheels para 3.14).
- Una API key gratuita de Gemini: <https://aistudio.google.com/apikey>

### 2. Entorno
```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

### 3. API key
Copiá `.env.example` a `.env` y pegá tu clave:
```
GOOGLE_API_KEY=tu_api_key_aqui
```

### 4. Ejecutar el notebook
Abrí `notebooks/asistente_tenis.ipynb` y ejecutá las celdas de arriba a abajo.
Incluye la construcción de la base vectorial, el agente, la demostración de la
memoria y 5 preguntas de ejemplo documentadas.

### 5. (Bonus) Interfaz web
```bash
streamlit run app.py
```

### 6. (Bonus) Despliegue en Streamlit Cloud
1. Subí el repo a un GitHub público.
2. En <https://share.streamlit.io> conectá el repo y elegí `app.py`.
3. En **Settings ▸ Secrets** de la app, agregá:
   ```toml
   GOOGLE_API_KEY = "tu_api_key_aqui"
   ```

---

## 📦 Requisitos (dependencias)

Ver [`requirements.txt`](requirements.txt). Principales: `langchain`,
`langgraph`, `langchain-google-genai`, `langchain-chroma`, `chromadb`, `pypdf`,
`python-dotenv`, `streamlit`.

---

## 🔧 MLOps y robustez

El sistema depende de servicios externos (modelos de Groq y Gemini) que pueden
cambiar sin aviso —de hecho, Groq deprecó un modelo en pleno uso—. Para que eso no
rompa el proyecto en silencio hay una capa de robustez y monitoreo:

- **Auto-recuperación de modelo.** Al arrancar, el agente verifica que el modelo de
  Groq configurado siga disponible; si fue deprecado, **cae automáticamente** a un
  modelo de respaldo (`src/agent.py::_resolver_modelo_groq`). La app no se cae.
- **Manejo de errores transitorios.** `preguntar()` reintenta ante 429/503 (cuota o
  sobrecarga) con back-off; la app muestra un mensaje legible si algo falla.
- **Health check** (`scripts/healthcheck.py`): valida keys, disponibilidad de
  modelos, embeddings y una consulta de punta a punta. Corrélo con
  `python scripts/healthcheck.py` (exit 0 = todo OK).
- **Monitoreo automático (GitHub Actions).**
  - `ci.yml` — en cada push corre los *smoke tests* (`tests/`): valida que el grafo
    de imports y la lógica sigan sanos, sin consumir APIs.
  - `healthcheck.yml` — cron **semanal** (y manual) que corre el health check con
    APIs reales; si falla, **abre/actualiza un Issue** y GitHub avisa por email.
- **Dependencias con tope de *major*** (`requirements.txt`): una actualización con
  cambios incompatibles no puede romper el deploy en silencio.

### Para activar el monitoreo (una sola vez)

En **Settings ▸ Secrets and variables ▸ Actions** del repo, agregá dos secrets:
`GOOGLE_API_KEY` y `GROQ_API_KEY`. Sin ellos, el workflow de health check se omite
solo (no genera falsas alertas).

### Consideraciones conocidas

- **Arranque en frío en Streamlit:** la app reindexa los reglamentos en memoria al
  despertar, consumiendo cuota de *embeddings* de Gemini (finita por día). Para
  tráfico bajo no es problema, y el health check avisaría si la cuota se agota. Para
  un despliegue 100 % independiente de cuota, se pueden usar embeddings locales de
  HuggingFace (mayor consumo de memoria en el contenedor).
- **Versión de Python del deploy:** Streamlit elige la versión en *Advanced
  settings*. El CI valida en **3.12**; se recomienda fijar la misma allí para paridad.

## ⚠️ Notas

- La API key **nunca** se versiona: vive en `.env` (gitignored) o en los Secrets
  de Streamlit.
- La carpeta `chroma_db/` se reconstruye a partir de los PDFs; también está
  gitignored.

## 📊 Cuota del free tier de Gemini

La API key gratuita de Google AI Studio tiene límites estrictos que conviene conocer:

- **Chat (`gemini-2.5-flash-lite`):** ~20 solicitudes por día. Cada pregunta al agente
  consume 1–2 solicitudes (razonar qué herramienta usar + generar la respuesta), así
  que alcanzan para el demo y las 5 preguntas de ejemplo, pero **no para iterar mucho**.
  La cuota **se reinicia cada día**.
- **Embeddings (`gemini-embedding-001`):** ~100 por minuto. Por eso la indexación se
  hace **por lotes con control de ritmo** (`src/rag.py`) y las colecciones se
  **persisten en disco** para no reindexar en cada ejecución.

Consejos: ejecutá el notebook con `RECONSTRUIR = False` (usa la base ya indexada) y, si
querés desarrollar sin límites, activá facturación en Google AI Studio (el consumo real
de este proyecto es de céntimos) o repartí las pruebas a lo largo del día.
