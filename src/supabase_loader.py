import os
import logging
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

def create_predictions_table():
    
    print("\n--- ATTENTION ---")
    print("Please execute the following SQL in your Supabase SQL Editor to create the table:")
    sql = 
    print(sql)
    print("-----------------\n")

def build_client_description(row) -> str:
    genero_str = 'Masculino' if row['genero'] == 0 else 'Femenino'
    desc = (f"Cliente {row['id']}: propensión {row['score_propension']:.0%}, "
            f"segmento {row['segmento_nombre']}. "
            f"Compra {row['tasa_compra']:.0%} de sus visitas, frecuencia {row['frecuencia']} visitas. "
            f"Marca favorita {row['marca_favorita']}, switching {row['switching_ratio']:.0%}. "
            f"Perfil: {genero_str}, {row['edad']} años, ingreso anual {row['ingreso_anual']:,.0f}. "
            f"Principal driver: {row['shap_top_feature']}.")
    return desc

def generate_embeddings(texts: list[str]) -> list[list[float]]:
    
    response = openai_client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [item.embedding for item in response.data]

def load_predictions(predictions_path: str = "data/predictions.csv"):
    df = pd.read_csv(predictions_path)
    logging.info("Building descriptions...")
    df["descripcion"] = df.apply(build_client_description, axis=1)

    logging.info("Generating embeddings via OpenAI...")
    all_embeddings = []
    batch_size = 100  

    for i in tqdm(range(0, len(df), batch_size), desc="Embeddings"):
        batch = df["descripcion"].iloc[i:i+batch_size].tolist()
        embeddings = generate_embeddings(batch)
        all_embeddings.extend(embeddings)

    df["embedding"] = all_embeddings

    logging.info("Uploading to Supabase...")
    upload_batch_size = 50
    for i in tqdm(range(0, len(df), upload_batch_size), desc="Uploading"):
        batch_df = df.iloc[i:i+upload_batch_size]
        records = []
        for _, row in batch_df.iterrows():
            row = row.fillna(0)
            records.append({
                "id": int(row["id"]),
                "score_propension": float(row["score_propension"]),
                "comprador_activo": int(row["comprador_activo"]),
                "segmento": int(row["segmento"]),
                "segmento_nombre": str(row["segmento_nombre"]),
                "shap_top_feature": str(row["shap_top_feature"]),
                "edad": int(row["edad"]),
                "genero": int(row["genero"]),
                "ingreso_anual": float(row["ingreso_anual"]),
                "ocupacion": int(row["ocupacion"]),
                "n_compras": int(row["n_compras"]),
                "frecuencia": int(row["frecuencia"]),
                "tasa_compra": float(row["tasa_compra"]),
                "marca_favorita": int(row["marca_favorita"]),
                "switching_ratio": float(row["switching_ratio"]),
                "recencia": int(row["recencia"]),
                "dias_entre_visitas_median": float(row["dias_entre_visitas_median"]),
                "descripcion": str(row["descripcion"]),
                "embedding": row["embedding"]
            })
        supabase.table("predictions").upsert(records).execute()

    logging.info(f"Done. {len(df)} records loaded successfully.")

def main():
    create_predictions_table()
    ans = input("Have you executed the SQL in Supabase? (y/n): ")
    if ans.lower() == 'y':
        load_predictions()
    else:
        print("Please execute the SQL and run this script again.")

if __name__ == '__main__':
    main()
