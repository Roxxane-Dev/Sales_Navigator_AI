import os
import re
from openai import OpenAI
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def sanitize_sql(sql_str: str) -> str:
    """Basic SQL sanitization."""
    forbidden = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', ';']
    upper_sql = sql_str.upper()
    for f in forbidden:
        if f in upper_sql:
            raise ValueError(f"Forbidden SQL keyword found: {f}")
    return sql_str

SAFE_COLUMNS = """
    id, score_propension, segmento_nombre,
    edad, genero, ingreso_anual, ocupacion,
    n_compras, tasa_compra, marca_favorita,
    switching_ratio, sensibilidad_promo, es_sensible_promo, es_leal_marca, descripcion
"""

SAFE_COLUMNS_BASE = """
    id, score_propension, segmento_nombre,
    edad, genero, ingreso_anual, ocupacion,
    n_compras, tasa_compra, marca_favorita,
    switching_ratio, descripcion
"""

ALLOWED_FILTER_COLUMNS = {
    "id",
    "score_propension",
    "segmento_nombre",
    "edad",
    "genero",
    "ingreso_anual",
    "ocupacion",
    "n_compras",
    "tasa_compra",
    "marca_favorita",
    "switching_ratio",
    "sensibilidad_promo",
    "es_sensible_promo",
    "es_leal_marca",
    "descripcion",
}

OP_MAP = {
    "=": "eq",
    "!=": "neq",
    ">": "gt",
    "<": "lt",
    ">=": "gte",
    "<=": "lte",
}

def _parse_value(raw_value: str):
    value = raw_value.strip()
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value

def _apply_simple_where(query, sql_where: str):
    where = (sql_where or "").strip()
    if not where or where == "1=1":
        return query

    parts = re.split(r"\s+AND\s+", where, flags=re.IGNORECASE)
    pattern = re.compile(r"^([a-zA-Z_][a-zA-Z0-9_]*)\s*(=|!=|>=|<=|>|<)\s*(.+)$")

    for cond in parts:
        cond = cond.strip()
        if cond == "1=1":
            continue
        match = pattern.match(cond)
        if not match:
            raise ValueError(f"Condición WHERE no soportada: {cond}")

        col, op, raw_value = match.groups()
        if col not in ALLOWED_FILTER_COLUMNS:
            raise ValueError(f"Columna no permitida en filtros: {col}")

        query = query.filter(col, OP_MAP[op], _parse_value(raw_value))

    return query

def _parse_order_by(order_by: str):
    order_raw = (order_by or "score_propension DESC").strip()
    first_clause = order_raw.split(",")[0].strip()
    match = re.match(r"^([a-zA-Z_][a-zA-Z0-9_]*)(?:\s+(ASC|DESC))?$", first_clause, flags=re.IGNORECASE)
    if not match:
        return "score_propension", True
    col, direction = match.groups()
    if col not in ALLOWED_FILTER_COLUMNS:
        return "score_propension", True
    return col, (direction or "DESC").upper() == "DESC"

def query_predictions(sql_where: str = "1=1", 
                      order_by: str = "score_propension DESC", 
                      limit: int = 15) -> dict:
    limit = min(limit, 15)  # nunca más de 15 filas
    sql_where = sanitize_sql(sql_where)
    try:
        order_col, order_desc = _parse_order_by(order_by)
        try:
            query = supabase.table("predictions").select(SAFE_COLUMNS)
            query = _apply_simple_where(query, sql_where)
            query = query.order(order_col, desc=order_desc).limit(limit)
            res = query.execute()
        except Exception:
            # Fallback para tablas aún no migradas con nuevas columnas
            query = supabase.table("predictions").select(SAFE_COLUMNS_BASE)
            query = _apply_simple_where(query, sql_where)
            query = query.order(order_col, desc=order_desc).limit(limit)
            res = query.execute()
        return {"data": res.data, "count": len(res.data)}
    except Exception as e:
        return {"error": str(e)}

def semantic_search(query: str, top_k: int = 8) -> dict:
    top_k = min(top_k, 8)
    try:
        response = openai_client.embeddings.create(
            input=[query],
            model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding
        res = supabase.rpc('match_predictions', {
            'query_embedding': query_embedding,
            'match_threshold': 0.3,  # era 0.7, muy restrictivo
            'match_count': top_k
        }).execute()
        return {"data": res.data, "count": len(res.data)}
    except Exception as e:
        return {"error": str(e)}

def get_segment_summary() -> dict:
    try:
        res = supabase.rpc('get_segment_summary_rpc').execute()
        return {"data": res.data}
    except Exception as e:
        return {"error": str(e)}

def get_business_kpis() -> dict:
    try:
        res = supabase.rpc('get_business_kpis_rpc').execute()
        return {"data": res.data}
    except Exception as e:
        return {"error": str(e)}
