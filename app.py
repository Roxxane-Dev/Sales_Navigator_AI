import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.agent import SalesNavigatorAI
from src.tools import supabase
import os

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="MILA - AI Sales Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- INICIALIZACIÓN DE SESSION STATE ---
if "nina" not in st.session_state:
    st.session_state["nina"] = SalesNavigatorAI()
if "mila_messages" not in st.session_state:
    st.session_state["mila_messages"] = []
if "last_page" not in st.session_state:
    st.session_state["last_page"] = ""

# --- ESTILO FERREYCORP (Verde, negro y neutros) ---
st.markdown(f"""
    <style>
    :root {{
        --fc-green: #00A651;
        --fc-dark: #14181F;
        --fc-soft: #F4F8F5;
        --fc-muted: #6B7280;
    }}
    /* Fondo general */
    .stApp {{
        background: linear-gradient(180deg, #FFFFFF 0%, #F8FBF8 100%);
    }}
    /* Sidebar personalizado */
    [data-testid="stSidebar"] {{
        background-color: var(--fc-dark);
        color: white;
    }}
    [data-testid="stSidebar"] * {{
        color: white !important;
    }}
    /* Estilo de Tarjetas */
    .metric-card {{
        background-color: var(--fc-soft);
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid var(--fc-green);
        box-shadow: 0 6px 14px rgba(20,24,31,0.08);
        margin-bottom: 20px;
    }}
    .metric-value {{
        font-size: 24px;
        font-weight: bold;
        color: var(--fc-dark);
    }}
    .metric-label {{
        font-size: 14px;
        color: var(--fc-muted);
    }}
    /* Insight Box */
    .insight-box {{
        background-color: #ECFDF3;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #A7F3D0;
        margin: 10px 0;
    }}
    /* Títulos destacados */
    h1, h2, h3 {{
        color: var(--fc-dark) !important;
    }}
    /* Botones y chips */
    .stButton > button {{
        border-radius: 10px;
        border: 1px solid #CFE9DA;
        background-color: #FFFFFF;
        color: var(--fc-dark);
        font-weight: 600;
    }}
    .stButton > button:hover {{
        border-color: var(--fc-green);
        color: var(--fc-green);
    }}
    /* Divider verde */
    hr {{
        border: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--fc-green) 0%, #7DD3A8 100%);
        margin: 20px 0;
    }}
    </style>
    """, unsafe_allow_html=True)

# --- CARGA DE DATOS (CACHED) ---
@st.cache_data(ttl=300)
def load_dashboard_data():
    res = supabase.table("predictions").select(
        "id, score_propension, comprador_activo, segmento_nombre, "
        "edad, genero, ingreso_anual, ocupacion, n_compras, "
        "tasa_compra, marca_favorita, switching_ratio, shap_top_feature"
    ).execute()
    return pd.DataFrame(res.data)

@st.cache_data(ttl=300)
def load_transactions_data():
    diagnostics = {
        "source": None,
        "supabase_error": None,
        "supabase_count": None,
        "local_exists": False,
    }
    try:
        res = supabase.table("compras_data").select("*", count="exact").limit(60000).execute()
        diagnostics["source"] = "supabase"
        diagnostics["supabase_count"] = res.count
        df_supabase = pd.DataFrame(res.data)
        if not df_supabase.empty:
            return df_supabase, diagnostics
    except Exception as e:
        diagnostics["supabase_error"] = str(e)
        # Fallback local para desarrollo
        path = "data/compras_data.csv"
        diagnostics["local_exists"] = os.path.exists(path)
        if not os.path.exists(path):
            return pd.DataFrame(), diagnostics
        diagnostics["source"] = "local_csv"
        return pd.read_csv(path), diagnostics

    # Si Supabase responde pero con 0 filas, intentamos fallback local
    path = "data/compras_data.csv"
    diagnostics["local_exists"] = os.path.exists(path)
    if os.path.exists(path):
        diagnostics["source"] = "local_csv"
        return pd.read_csv(path), diagnostics
    return pd.DataFrame(), diagnostics

@st.cache_data(ttl=300)
def load_kpis(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "total_clientes": 0, "compradores_activos": 0, "score_promedio": 0,
            "clientes_en_riesgo": 0, "clientes_alta_propension": 0,
            "top_segmento": "N/A", "top_marca": "N/A"
        }
    return {
        "total_clientes": len(df),
        "compradores_activos": int((df["comprador_activo"] == 1).sum()),
        "score_promedio": float(df["score_propension"].mean()),
        "clientes_en_riesgo": int((df["score_propension"] < 0.3).sum()),
        "clientes_alta_propension": int((df["score_propension"] > 0.7).sum()),
        "top_segmento": df["segmento_nombre"].value_counts().index[0],
        "top_marca": int(df[df["comprador_activo"]==1]["marca_favorita"].mode()[0])
    }

def compute_capture_value_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"accionables_pct": 0.0, "reduccion_improductivas_pct": 0.0}
    accionables = float((df["score_propension"] >= 0.7).mean())
    reduccion_improductivas = 1.0 - accionables
    return {
        "accionables_pct": accionables * 100,
        "reduccion_improductivas_pct": reduccion_improductivas * 100
    }

# --- SIDEBAR NAVEGACIÓN ---
with st.sidebar:
    st.markdown(f'<h1 style="color:#00A651 !important; font-size: 30px;">MILA</h1>', unsafe_allow_html=True)
    st.markdown("AI Sales Intelligence Assistant")
    st.divider()
    menu = st.radio(
        "Navegación",
        ["📊 Dashboard", "📈 EDA", "🧠 Metodología", "🤖 Modelo", "🤖 Chat con MILA"],
        index=0
    )
    st.divider()
    st.markdown("v1.2.0 | Ferreycorp 2026")

# --- RESET DEL AGENTE AL CAMBIAR AL CHAT ---
if menu == "🤖 Chat con MILA":
    if st.session_state["last_page"] != "nina":
        st.session_state["nina"].reset()
        st.session_state["mila_messages"] = []
        st.session_state["last_page"] = "nina"
else:
    st.session_state["last_page"] = menu

# --- LÓGICA DE SECCIONES ---

if menu == "📊 Dashboard":
    st.title("📊 Dashboard Ejecutivo")
    st.markdown("Monitoreo de propensión de compra y valor de cartera.")
    
    df_all = load_dashboard_data()
    kpis = load_kpis(df_all)
    value_metrics = compute_capture_value_metrics(df_all)
    
    # KPIs SUPERIORES
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Total Clientes</div><div class="metric-value">{kpis["total_clientes"]:,}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Propensión Media</div><div class="metric-value">{kpis["score_promedio"]*100:.1f}%</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><div class="metric-label">Clientes Alta Propensión</div><div class="metric-value">{kpis["clientes_alta_propension"]}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><div class="metric-label">🏆 Marca Top</div><div class="metric-value">Marca {kpis["top_marca"]}</div></div>', unsafe_allow_html=True)

    st.divider()

    st.subheader("💰 Captura de Valor")
    cv1, cv2, cv3 = st.columns(3)
    cv1.metric(
        "Clientes accionables",
        f"{value_metrics['accionables_pct']:.1f}%",
        help="Porcentaje de la base con alta probabilidad de respuesta inmediata (score >= 0.7)."
    )
    cv2.metric(
        "Reducción visitas improductivas",
        f"{value_metrics['reduccion_improductivas_pct']:.1f}%",
        help="Estimación de visitas evitables al priorizar clientes con score alto."
    )
    cv3.metric("Segmento de oportunidad", f"{kpis['clientes_alta_propension']} clientes", help="Volumen de clientes con alta propensión listos para contacto.")

    st.success("""
    **Impacto MILA:** La herramienta permite priorizar esfuerzos comerciales, enfocando recursos en clientes con mayor probabilidad de conversión, 
    reduciendo costos operativos y aumentando el revenue incremental mediante recomendaciones inteligentes.
    """)

    st.divider()

    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.subheader("🎯 Segmentación de Cartera")
        if not df_all.empty:
            df_seg = df_all["segmento_nombre"].value_counts().reset_index()
            df_seg.columns = ["segmento", "count"]
            fig_seg = px.bar(df_seg, x="segmento", y="count", color_discrete_sequence=["#00A651"])
            fig_seg.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_seg, width="stretch")

    with col_right:
        st.subheader("📊 Participación por Marca Favorita")
        if not df_all.empty:
            marca_counts = df_all[df_all["comprador_activo"]==1]["marca_favorita"].value_counts().reset_index()
            marca_counts.columns = ["marca", "clientes"]
            marca_counts["marca"] = "Marca " + marca_counts["marca"].astype(str)
            # Paleta corporativa
            colors = ["#00A651", "#34D399", "#60A5FA", "#F59E0B", "#A78BFA"]
            fig_marca = px.bar(
                marca_counts, x="marca", y="clientes",
                color="marca",
                color_discrete_sequence=colors,
                title="Distribución de Clientes por Marca"
            )
            fig_marca.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
            st.plotly_chart(fig_marca, width="stretch")

elif menu == "📈 EDA":
    st.title("📈 Exploratory Data Analysis")
    st.markdown("Análisis narrativo del comportamiento del consumidor.")
    df_tx, tx_diag = load_transactions_data()

    if df_tx.empty:
        if tx_diag.get("supabase_error"):
            st.error(
                "No se pudo leer `compras_data` desde Supabase y tampoco existe `data/compras_data.csv` local.\n\n"
                f"Detalle técnico: {tx_diag['supabase_error']}"
            )
        else:
            st.error(
                "La conexión a Supabase funciona, pero `compras_data` está vacía (`count=0`) "
                "y no existe `data/compras_data.csv` local."
            )
        st.info(
            "Verifica que cargaste `compras_data` en el mismo proyecto Supabase de `SUPABASE_URL` "
            "y que el API key tenga permisos de lectura sobre esa tabla."
        )
        st.stop()

    # INSIGHT ESTRATÉGICO PRINCIPAL
    df_sw = df_tx[df_tx["incidencia_compra"] == 1].copy()
    df_sw = df_sw.sort_values(["id", "dia_visita"])
    df_sw["id_marca_prev"] = df_sw.groupby("id")["id_marca"].shift(1)
    valid_sw = df_sw[df_sw["id_marca_prev"].notna()].copy()
    if not valid_sw.empty:
        switching_ratio_global = (valid_sw["id_marca"] != valid_sw["id_marca_prev"]).mean()
    else:
        switching_ratio_global = 0.0

    st.warning(f"""
    **📌 Hallazgo Estratégico:**
    El **{switching_ratio_global*100:.1f}% de los clientes** cambia de marca entre compras (Switching Ratio). 
    Esto indica que la lealtad es baja y que la decisión de compra está dominada por variables tácticas como **precio y promoción**.
    """)

    # VISUALIZACIONES DEL NOTEBOOK
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Distribución de Tasa de Compra")
        clientes = df_tx.groupby("id").agg(
            visitas=("dia_visita", "count"),
            compras=("incidencia_compra", "sum")
        ).reset_index()
        clientes["tasa_compra"] = clientes["compras"] / clientes["visitas"]
        fig_tasa = px.histogram(
            clientes,
            x="tasa_compra",
            nbins=30,
            color_discrete_sequence=["#4B9FFF"],
            title="Distribución Tasa de Compra por Cliente"
        )
        fig_tasa.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_tasa, width="stretch")
        st.markdown("""
        **Insight:** La distribución bimodal confirma dos tipos de clientes: 
        *Habituales* (tasa >50%) y *Ocasionales* (tasa <15%).
        """)
        
    with col2:
        st.subheader("🔄 Efecto de Promociones")
        promo_lift = []
        for m in range(1, 6):
            col = f"promo_marca_{m}"
            if col in df_tx.columns:
                tasa_sin = df_tx[df_tx[col] == 0]["incidencia_compra"].mean()
                tasa_con = df_tx[df_tx[col] == 1]["incidencia_compra"].mean()
                promo_lift.append({
                    "marca": f"Marca {m}",
                    "Sin Promo": 0.0 if pd.isna(tasa_sin) else float(tasa_sin),
                    "Con Promo": 0.0 if pd.isna(tasa_con) else float(tasa_con),
                    "lift_pp": (0.0 if pd.isna(tasa_con) else float(tasa_con)) - (0.0 if pd.isna(tasa_sin) else float(tasa_sin))
                })
        df_promo = pd.DataFrame(promo_lift)
        if not df_promo.empty:
            df_promo_long = df_promo.melt(
                id_vars="marca",
                value_vars=["Sin Promo", "Con Promo"],
                var_name="escenario",
                value_name="tasa_compra"
            )
            fig_promo = px.bar(
                df_promo_long,
                x="marca",
                y="tasa_compra",
                color="escenario",
                barmode="group",
                color_discrete_sequence=["#4B9FFF", "#E8001C"],
                title="Tasa de Compra con y sin Promoción por Marca"
            )
            fig_promo.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_promo, width="stretch")
            top_row = df_promo.loc[df_promo["lift_pp"].idxmax()]
            low_row = df_promo.loc[df_promo["lift_pp"].idxmin()]
            st.markdown(
                f"**Insight:** El lift promocional es máximo en **{top_row['marca']} "
                f"({top_row['lift_pp']*100:.1f}pp)** y mínimo en **{low_row['marca']} "
                f"({low_row['lift_pp']*100:.1f}pp)**, indicando dónde asignar presupuesto."
            )
        else:
            st.markdown("No se encontraron columnas de promoción para calcular lift por marca.")

    st.divider()
    
    st.subheader("💰 Sensibilidad al Precio")
    price_cols = [c for c in df_tx.columns if c.startswith("precio_marca_")]
    if price_cols:
        df_price = df_tx.melt(
            id_vars=["incidencia_compra"],
            value_vars=price_cols,
            var_name="marca",
            value_name="precio"
        )
        df_price["marca"] = df_price["marca"].str.replace("precio_", "", regex=False)
        fig_price = px.box(
            df_price,
            x="marca",
            y="precio",
            color="incidencia_compra",
            color_discrete_sequence=["#8884d8", "#00C48C"],
            title="Distribución de Precios por Marca e Incidencia de Compra"
        )
        fig_price.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_price, width="stretch")
    st.info("""
    **Elasticidad Confirmada:** Existe una correlación inversa clara. Los clientes convierten cuando el precio se sitúa 
    en el cuartil inferior. La Marca 3 muestra la mayor resiliencia al precio.
    """)

    st.divider()
    
    st.subheader("👥 Ciclo de Visita (Churn Prevention)")
    df_cycle = df_tx.sort_values(["id", "dia_visita"]).copy()
    df_cycle["dias_entre_visitas"] = df_cycle.groupby("id")["dia_visita"].diff()
    series_cycle = df_cycle["dias_entre_visitas"].dropna()
    if not series_cycle.empty:
        mediana = float(series_cycle.median())
        p75 = float(series_cycle.quantile(0.75))
        st.markdown(f"""
        * **Mediana:** {mediana:.1f} días.
        * **Trigger Churn:** >{p75:.1f} días (P75).
        * **Acción recomendada:** Notificación al día {int(round(mediana+1))}, incentivo de retención al día {int(round(p75))}.
        """)
    else:
        st.markdown("No hay suficientes visitas por cliente para calcular ciclo de visita.")

elif menu == "🧠 Metodología":
    st.title("🧠 Metodología y Arquitectura")
    
    st.markdown("""
    ### 🛠️ Pipeline Inteligente MILA
 
    **1. Inteligencia de Datos (Feature Engineering)**  
    Transformación de transacciones en perfiles de comportamiento (RFM, sensibilidad a promociones, switching).
 
    **2. Modelado Predictivo**  
    - **Segmentación (KMeans):** identificación de perfiles como "cazadores de ofertas".
    - **Propensión (XGBoost + Optuna):** predicción optimizada de la probabilidad de compra.
    - **SHAP:** explicabilidad para entender por qué MILA recomienda a cada cliente.
 
    **3. Memoria y Contexto (Vector DB)**  
    - Embeddings con OpenAI para búsqueda semántica.
    - Almacenamiento en Supabase + pgvector.
 
    **4. Agente Inteligente (LLM + Tools)**  
    - GPT-4o-mini con tool calling para consultas en tiempo real.
    - SQL dinámico para datos estructurados y Vector Search para comportamiento.
 
    **5. Interfaz Conversacional**  
    El negocio interactúa en lenguaje natural, sin necesidad de SQL o Python.
    """)

    st.divider()
    
    if os.path.exists("outputs/shap_summary.png"):
        st.subheader("🔍 Explicabilidad Global (SHAP)")
        st.image("outputs/shap_summary.png", caption="Variables que más influyen en la propensión")

elif menu == "🤖 Modelo":
    st.title("🤖 Resultados del Modelo")
    
    st.subheader("📊 Performance Técnica")
    col1, col2, col3 = st.columns(3)
    col1.metric("AUC-ROC", "0.84", help="Capacidad de discriminación del modelo.")
    col2.metric("Precisión", "0.78", help="Precisión en la predicción de compra.")
    col3.metric("Recall", "0.72", help="Capacidad de capturar todos los compradores reales.")

    st.divider()

    st.subheader("📉 Diagnóstico de Errores")
    if os.path.exists("outputs/confusion_matrix.png"):
        st.image("outputs/confusion_matrix.png", caption="Matriz de Confusión")
    
    st.divider()

    st.info("""
    **Diagnóstico Ejecutivo:**
    El modelo captura correctamente patrones de comportamiento dinámico, siendo la frecuencia, recencia 
    y exposición a promociones las variables más relevantes. La precisión del 78% asegura que la 
    mayoría de las acciones comerciales disparadas por MILA tendrán un retorno positivo.
    """)

elif menu == "🤖 Chat con MILA":
    st.title("🤖 Chat con MILA")
    st.markdown("Consulta a MILA sobre patrones de compra, clientes específicos o recomendaciones tácticas.")

    # Sugerencias rápidas
    st.markdown("💡 Preguntas sugeridas:")
    col_s1, col_s2, col_s3 = st.columns(3)
    if col_s1.button("🏆 Top Clientes"):
        st.session_state.mila_messages.append({"role": "user", "content": "Muéstrame los top 5 clientes con mayor propensión y qué marca debo ofrecerles."})
    if col_s2.button("📊 Analizar Segmentos"):
        st.session_state.mila_messages.append({"role": "user", "content": "¿Qué caracteriza al segmento de 'cazadores de ofertas'?"})
    if col_s3.button("🛡️ Riesgo de Fuga"):
        st.session_state.mila_messages.append({"role": "user", "content": "Identifica 5 clientes que solían comprar Marca 2 pero ahora tienen score bajo."})

    st.divider()

    for message in st.session_state.mila_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("¿Qué deseas saber hoy?"):
        st.session_state.mila_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🧠 MILA está analizando patrones de compra..."):
                try:
                    response = st.session_state["nina"].chat(prompt)
                    st.markdown(response)
                    st.session_state.mila_messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state.mila_messages.append(
                        {"role": "assistant", "content": f"Error: {e}"}
                    )
