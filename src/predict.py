import pandas as pd
import json

def predict_new_visit(client_id: int, visit_context: dict) -> dict:
    
    try:
        df = pd.read_csv('data/predictions.csv')
    except Exception as e:
        return {"error": "Predictions file not found. Run train_model.py first."}
        
    client_data = df[df['id'] == client_id]
    if client_data.empty:
        return {"error": f"Client {client_id} not found"}
        
    client_row = client_data.iloc[0]
    
    score = float(client_row['score_propension'])
    dias_median = float(client_row['dias_entre_visitas_median'])
    recencia = float(client_row['recencia'])
    marca_fav = int(client_row['marca_favorita'])
    
    recomendacion = "Mantener estrategia actual."
    if score > 0.7:
        recomendacion = f"Cliente en ventana óptima de compra (propensión {score:.0%}). Activar promo marca {marca_fav}."
    elif recencia > (dias_median * 1.5) and dias_median > 0:
        recomendacion = f"Cliente fuera de ciclo habitual (lleva {recencia} días, ciclo = {dias_median}). Riesgo de churn."
    elif score < 0.3:
        recomendacion = "Cliente con baja probabilidad de compra. Ofrecer descuento agresivo para incentivar conversión."
        
    return {
        "client_id": client_id,
        "score_propension": score,
        "comprador_activo": int(client_row['comprador_activo']),
        "segmento_nombre": str(client_row['segmento_nombre']),
        "shap_top_feature": str(client_row['shap_top_feature']),
        "recomendacion": recomendacion
    }

if __name__ == '__main__':
    res = predict_new_visit(20002000, {"dia": 1, "promo_marca_1": 1})
    print(json.dumps(res, indent=2, ensure_ascii=False))
