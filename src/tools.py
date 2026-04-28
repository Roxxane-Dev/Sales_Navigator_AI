import os
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
    
    forbidden = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'TRUNCATE', ';']
    upper_sql = sql_str.upper()
    for f in forbidden:
        if f in upper_sql:
            raise ValueError(f"Forbidden SQL keyword found: {f}")
    return sql_str

def query_predictions(sql_where: str = "1=1", order_by: str = "score_propension DESC", limit: int = 20) -> dict:
    sql_where = sanitize_sql(sql_where)
    
    try:
        result = supabase.table("predictions")\
            .select("id, score_propension, comprador_activo, segmento_nombre, shap_top_feature, edad, genero, ingreso_anual, ocupacion, n_compras, tasa_compra, marca_favorita, switching_ratio")\
            .filter("score_propension", "gte", 0)\
            .order(order_by.replace(" DESC","").replace(" ASC",""), desc="DESC" in order_by)\
            .limit(limit)\
            .execute()
        
        return {"data": result.data, "count": len(result.data)}
    except Exception as e:
        return {"error": str(e)}

def semantic_search(query: str, top_k: int = 10) -> dict:
    top_k = min(top_k, 10)  
    try:
        response = openai_client.embeddings.create(
            input=[query],
            model="text-embedding-3-small"
        )
        query_embedding = response.data[0].embedding
        
        res = supabase.rpc('match_predictions', {
            'query_embedding': query_embedding,
            'match_threshold': 0.3,
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
