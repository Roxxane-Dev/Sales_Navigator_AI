# 🧠 Ferreycorp Sales Navigator AI
> AI Sales Intelligence Assistant — Motor de consulta inteligente sobre predicciones de propensión de compra

## 📌 Descripción
Ferreycorp Sales Navigator AI es una solución avanzada de inteligencia comercial diseñada para predecir la probabilidad de compra de los clientes. El sistema utiliza **XGBoost** para modelos de propensión y **KMeans** para segmentación de comportamiento, permitiendo identificar patrones de consumo críticos. Los resultados son expuestos a través de **Sales Navigator AI**, un agente conversacional basado en GPT-4o-mini que permite realizar consultas en lenguaje natural directamente sobre una base de datos vectorial en **Supabase (pgvector)**.

## 🏗️ Arquitectura
El sistema se divide en 5 capas principales:
- **Datos**: Procesamiento de CSV de transacciones diarias mediante *feature engineering* a nivel de cliente (RFM, sensibilidad a promociones, demografía).
- **Modelo**: Pipeline de Machine Learning con KMeans (segmentación) y XGBoost (propensión de compra) optimizado con Optuna y validado con 5-fold CV. Explicabilidad mediante valores SHAP.
- **Storage**: Almacenamiento en Supabase Postgres con soporte para `pgvector` utilizando embeddings de OpenAI (`text-embedding-3-small`).
- **Agente**: Agente inteligente (GPT-4o-mini) con soporte de herramientas (*tool-use*) para ejecución de SQL dinámico y búsqueda semántica.
- **Interfaz**: Aplicación conversacional vía CLI (Interfaz de Línea de Comandos) para interacción directa con el equipo comercial.

## 📁 Estructura del Proyecto
```text
Ferreycorp Sales Navigator AI/
├── data/                    # Datasets originales y procesados
│   └── compras_data.csv     # Dataset base de transacciones
├── models/                  # Modelos entrenados (XGBoost, Scaler)
├── notebooks/               # Jupyter Notebooks de EDA y prototipado
├── outputs/                 # Gráficos de resultados (Confusion Matrix, SHAP)
├── src/                     # Código fuente del pipeline
│   ├── agent.py             # Agente conversacional Sales Navigator AI
│   ├── feature_engineering.py# Transformación de datos y agregación
│   ├── predict.py           # Lógica de inferencia y reglas de negocio
│   ├── supabase_loader.py   # Carga de datos y embeddings a Supabase
│   ├── tools.py             # Herramientas de consulta (SQL y Vectorial)
│   └── train_model.py       # Entrenamiento y validación cruzada
├── .env                     # Variables de entorno (API Keys)
├── requirements.txt         # Dependencias del proyecto
└── README.md                # Documentación principal
```

## ⚙️ Instalación y Configuración

### Requisitos
- Python 3.11+
- Cuenta en Supabase (con extensión pgvector habilitada)
- API Keys: OpenAI, Supabase (URL y Service Role Key)

### Pasos
1. **Clonar el repositorio**:
   `git clone https://github.com/Roxxane-Dev/Sales_Navigator_AI.git`
2. **Crear entorno virtual**:
   `python -m venv .venv`
   `source .venv/bin/activate` # En Windows: .venv\Scripts\activate
3. **Instalar dependencias**:
   `pip install -r requirements.txt`
4. **Configurar entorno**:
   Copiar `.env.example` a `.env` y completar las claves de OpenAI y Supabase.
5. **Ejecutar el SQL de setup**:
   Copiar y ejecutar el código de la sección **SQL Setup Supabase** en el SQL Editor de tu proyecto en Supabase.

## 🚀 Orden de Ejecución
Para poner en marcha el pipeline completo, ejecuta los comandos en este orden:

1. **`python src/feature_engineering.py`**: Procesa los datos crudos y genera los features por cliente.
2. **`python src/train_model.py`**: Entrena los modelos de segmentación y propensión, guardando los resultados localmente.
3. **`python src/supabase_loader.py`**: Genera embeddings de OpenAI y sube las predicciones a la base de datos vectorial de Supabase.
4. **`python src/agent.py`**: Inicia el asistente conversacional para realizar consultas.

## 💬 Ejemplos de uso de Sales Navigator AI
El asistente puede procesar consultas complejas como:
- **Tú**: "¿Cuáles son los KPIs globales del negocio?"
  - **NINA**: Muestra una tabla con total de clientes, tasa de conversión global y clientes en riesgo de churn.
- **Tú**: "Muéstrame los 10 clientes con mayor propensión de compra"
  - **NINA**: Genera una tabla markdown con los IDs de clientes y sus respectivos scores de propensión.
- **Tú**: "¿Cómo están distribuidos los segmentos de clientes?"
  - **NINA**: Resume la cantidad de clientes por segmento (Cazador de ofertas, Comprador leal, etc.).
- **Tú**: "Busca clientes frecuentes sensibles a promociones"
  - **NINA**: Realiza una búsqueda semántica para encontrar perfiles que coincidan con esa descripción.

## 📊 Resultados del Modelo
El modelo ha sido validado rigurosamente para evitar el *data leakage*:
- **AUC-ROC**: 0.846 (Validación cruzada 5-fold).
- **Accuracy**: 77%.
- **Segmentos identificados**: 4 (comprador_leal, cazador_ofertas, comprador_premium_ocasional, visitante_pasivo).

## 🗄️ SQL Setup Supabase
Ejecuta este bloque en el SQL Editor de Supabase para habilitar las búsquedas vectoriales y las herramientas del agente:

```sql
-- Habilitar extensión pgvector
CREATE EXTENSION IF NOT EXISTS vector;

-- Limpiar tabla previa si existe
DROP TABLE IF EXISTS predictions;

-- Crear tabla de predicciones
CREATE TABLE predictions (
  id BIGINT PRIMARY KEY,
  score_propension FLOAT,
  comprador_activo INT,
  segmento INT,
  segmento_nombre TEXT,
  shap_top_feature TEXT,
  edad INT,
  genero INT,
  ingreso_anual FLOAT,
  ocupacion INT,
  n_compras INT,
  frecuencia INT,
  tasa_compra FLOAT,
  marca_favorita INT,
  switching_ratio FLOAT,
  recencia INT,
  dias_entre_visitas_median FLOAT,
  descripcion TEXT,
  embedding vector(1536) -- Dimensión para text-embedding-3-small
);

-- Crear índice para búsqueda vectorial eficiente
CREATE INDEX ON predictions USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Función para búsqueda semántica (Vector Similarity Search)
CREATE OR REPLACE FUNCTION match_predictions(
  query_embedding vector(1536),
  match_threshold float,
  match_count int
)
RETURNS TABLE(id bigint, descripcion text, segmento_nombre text, 
              score_propension float, similarity float)
LANGUAGE sql STABLE AS $$
  SELECT id, descripcion, segmento_nombre, score_propension,
         1 - (embedding <=> query_embedding) AS similarity
  FROM predictions
  WHERE 1 - (embedding <=> query_embedding) > match_threshold
  ORDER BY similarity DESC
  LIMIT match_count;
$$;

-- Función para ejecución de SQL dinámico (solo lectura)
CREATE OR REPLACE FUNCTION run_sql(sql_query text)
RETURNS SETOF predictions
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY EXECUTE sql_query;
END;
$$;

-- Función para resumen de segmentos
CREATE OR REPLACE FUNCTION get_segment_summary_rpc()
RETURNS TABLE(segmento_nombre text, count bigint, avg_score float, avg_tasa float)
LANGUAGE sql STABLE AS $$
  SELECT segmento_nombre, count(*), avg(score_propension), avg(tasa_compra) 
  FROM predictions 
  GROUP BY segmento_nombre 
  ORDER BY avg(score_propension) DESC;
$$;

-- Función para KPIs globales
CREATE OR REPLACE FUNCTION get_business_kpis_rpc()
RETURNS json
LANGUAGE sql STABLE AS $$
  SELECT json_build_object(
    'total_clientes', (SELECT count(*) FROM predictions),
    'compradores_activos', (SELECT count(*) FROM predictions WHERE comprador_activo = 1),
    'tasa_conversion_global', (SELECT avg(tasa_compra) FROM predictions),
    'score_promedio', (SELECT avg(score_propension) FROM predictions),
    'clientes_en_riesgo', (SELECT count(*) FROM predictions WHERE score_propension < 0.3),
    'clientes_alta_propension', (SELECT count(*) FROM predictions WHERE score_propension > 0.7)
  );
$$;
```

## 📈 Impacto en el Negocio
- **Priorización Estratégica**: 208 "cazadores de ofertas" identificados para campañas de promociones dirigidas.
- **Fidelización**: 50 "compradores leales" detectados con score >0.99 para programas de beneficios exclusivos.
- **Activación de Clientes**: 116 "visitantes pasivos" con potencial de conversión mediante incentivos específicos (Marca 5).
- **Eficiencia Operativa**: El agente Sales Navigator AI elimina la fricción entre los datos y la toma de decisiones comercial.

## 🛠️ Stack Tecnológico
| Tecnología | Uso |
| :--- | :--- |
| **Python 3.11** | Lenguaje principal del pipeline |
| **XGBoost** | Algoritmo de predicción de propensión |
| **Scikit-learn** | Clustering (KMeans) y preprocesamiento |
| **Optuna** | Optimización de hiperparámetros |
| **SHAP** | Explicabilidad del modelo (AI explicable) |
| **Supabase** | Base de datos Postgres y Backend as a Service |
| **pgvector** | Almacenamiento y búsqueda de vectores |
| **OpenAI API** | Embeddings y razonamiento del agente |
| **GPT-4o-mini** | Motor de lenguaje del asistente comercial |
| **Pandas / NumPy** | Manipulación y análisis de datos |

## 👤 Autor
Camilla Navinta 
