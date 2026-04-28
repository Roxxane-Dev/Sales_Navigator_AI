# AI Sales Intelligence Assistant (Ferreycorp)

Este proyecto implementa un pipeline completo para la predicción de propensión de compra de clientes, segmentación y un agente conversacional inteligente impulsado por OpenAI.

## Arquitectura

1. **Feature Engineering (`src/feature_engineering.py`)**: Construye un perfil de cliente a partir de datos de visitas. Incluye métricas de RFM, comportamiento de compra, sensibilidad a precios y promociones.
2. **Entrenamiento (`src/train_model.py`)**: Ejecuta un clustering con KMeans para segmentar a los usuarios y entrena un clasificador XGBoost con optimización de hiperparámetros vía Optuna. Añade explicabilidad mediante SHAP.
3. **Predicción (`src/predict.py`)**: Contiene la lógica para predecir e inferir acciones para visitas nuevas (listo para ser importado por una API).
4. **Carga a Supabase (`src/supabase_loader.py`)**: Crea los embeddings de texto del perfil de cliente (usando Voyage AI) y carga todos los datos a una tabla de Supabase optimizada para búsquedas vectoriales.
5. **Agente NINA (`src/agent.py` y `src/tools.py`)**: Un asistente basado en OpenAI que cuenta con herramientas para buscar predicciones vía SQL, buscar por similitud semántica y recuperar KPIs del negocio.

## Requisitos

- Python 3.9+
- Supabase (Proyecto creado con pgvector habilitado)
- API Keys de OpenAI y Voyage AI

## Instalación y Configuración

1. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configurar variables de entorno:**
   ```
   SUPABASE_URL=tu_url_aqui
   SUPABASE_KEY=tu_anon_key_aqui
   VOYAGE_API_KEY=tu_voyage_api_key
   OPENAI_API_KEY=tu_openai_api_key
   ```

## Orden de Ejecución

Para correr el proyecto desde cero, ejecuta los siguientes comandos en orden:

### 1. Preparar las características (Feature Engineering)
Toma el dataset crudo `data/compras_data.csv` y genera `data/client_features.csv`.
```bash
python src/feature_engineering.py
```

### 2. Entrenar el Modelo
Generará la segmentación, el modelo de propensión, gráficas en `outputs/` y la base de predicciones final `data/predictions.csv`.
```bash
python src/train_model.py
```

### 3. Carga en la Base de Datos (Supabase)
Ejecuta el script. El script imprimirá en consola un código SQL que **debes copiar y ejecutar en el SQL Editor de tu Dashboard en Supabase** para crear las tablas, el índice vectorial y las funciones RPC requeridas.
Una vez hecho, confirma en la terminal para que inicie la generación de embeddings y la inserción.
```bash
python src/supabase_loader.py
```

### 4. Lanzar el Agente (NINA)
Una vez que los datos están en Supabase, ya puedes usar al agente comercial:
```bash
python src/agent.py
```
Puedes preguntarle cosas como:
- *¿Cuáles son los KPIs generales de mis clientes?*
- *Muéstrame a los clientes con una probabilidad de compra mayor al 80%*
- *Busca perfiles similares a un comprador leal con ingresos altos*
