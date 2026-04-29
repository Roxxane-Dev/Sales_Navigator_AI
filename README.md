# MILA - Ferreycorp Sales Navigator AI

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Supabase](https://img.shields.io/badge/Supabase-Postgres-3ECF8E?logo=supabase&logoColor=white)](https://supabase.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-LLM-412991?logo=openai&logoColor=white)](https://platform.openai.com/)

Asistente comercial con IA para priorizar clientes por propension de compra, explorar comportamiento (EDA) y consultar insights en lenguaje natural.

## Demo Scope

- Dashboard ejecutivo con KPIs y segmentacion.
- EDA dinamico en Streamlit usando datos reales de `compras_data` en Supabase.
- Chat con MILA (tool-calling) sobre `predictions`.
- Recomendaciones accionables para equipos comerciales.

## Arquitectura

```text
compras_data (Supabase) ---------> EDA (Streamlit)
                                     |
                                     v
predictions (Supabase) --> tools.py --> agent.py --> Chat MILA (Streamlit)
                                     ^
                                     |
                          train_model.py + supabase_loader.py
```

## Requisitos

- Python 3.11+
- Proyecto en Supabase
- API key de OpenAI

## Variables de entorno

Crear `.env` con:

```env
OPENAI_API_KEY=...
SUPABASE_URL=...
SUPABASE_KEY=...
```

## Como correr local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Publicacion en Streamlit Community Cloud

1. Conecta tu repo de GitHub en [share.streamlit.io](https://share.streamlit.io/).
2. Configura:
   - Branch: `main`
   - Main file: `app.py`
3. En `Secrets` agrega `OPENAI_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY`.
4. Deploy.

## Estructura relevante

```text
src/
  agent.py             # orquestacion LLM + tool-calls
  tools.py             # consultas a Supabase (predictions, semantic search, KPIs)
  predict.py           # helper de inferencia por cliente (usa predictions en Supabase)
  feature_engineering.py
  train_model.py
  supabase_loader.py
app.py                 # Streamlit app (dashboard + EDA + chat)
notebooks/01_EDA.ipynb # analisis exploratorio base
```

## Fuente de datos en runtime

- `compras_data` (Supabase): alimenta EDA en tiempo real.
- `predictions` (Supabase): alimenta dashboard, chat y consultas por cliente.

Con esto, la app no depende de tener los CSV locales para funcionar en cloud.

## Roadmap corto

- Feedback loop de respuestas del chat (tabla `feedback` con RLS correcto).
- Filtros avanzados por segmento/canal en dashboard.
- Monitoreo de drift de scores en `predictions`.
