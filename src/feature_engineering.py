import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_client_features(df: pd.DataFrame) -> pd.DataFrame:
    
    logging.info("Starting feature engineering...")
    
    df = df.sort_values(['id', 'dia_visita'])
    
    rfm = df.groupby('id').agg(
        recencia=('dia_visita', 'max'),
        frecuencia=('dia_visita', 'count'),
        n_compras=('incidencia_compra', 'sum')
    ).reset_index()
    
    rfm['tasa_compra'] = rfm['n_compras'] / rfm['frecuencia']
    
    def get_price_bought(row):
        marca = int(row['id_marca'])
        if marca == 0:
            return 0.0
        return row.get(f'precio_marca_{marca}', 0.0)
        
    df['precio_pagado'] = df.apply(get_price_bought, axis=1)
    df['gasto_visita'] = df['cantidad'] * df['precio_pagado']
    
    valor_monetario = df.groupby('id')['gasto_visita'].sum().reset_index(name='valor_monetario')
    rfm = pd.merge(rfm, valor_monetario, on='id')
    
    df['dias_entre_visitas'] = df.groupby('id')['dia_visita'].diff()
    behavioral = df.groupby('id').agg(
        dias_entre_visitas_median=('dias_entre_visitas', 'median'),
        dias_entre_visitas_std=('dias_entre_visitas', 'std')
    ).reset_index()
    
    behavioral['dias_entre_visitas_std'] = behavioral['dias_entre_visitas_std'].fillna(0)
    behavioral['dias_entre_visitas_median'] = behavioral['dias_entre_visitas_median'].fillna(0)
    
    df_compras = df[df['incidencia_compra'] == 1].copy()
    df_compras['is_switch'] = (df_compras['id_marca'] != df_compras['ultima_marca_comprada']).astype(int)
    
    switching = df_compras.groupby('id').agg(
        switching_ratio=('is_switch', 'mean'),
        marcas_distintas=('id_marca', 'nunique')
    ).reset_index()
    
    def get_favorite_brand(x):
        modes = x.mode()
        return modes.iloc[0] if not modes.empty else 0
        
    marca_fav = df_compras.groupby('id')['id_marca'].agg(get_favorite_brand).reset_index(name='marca_favorita')
    
    behavioral = pd.merge(behavioral, switching, on='id', how='left')
    behavioral = pd.merge(behavioral, marca_fav, on='id', how='left')
    
    behavioral['switching_ratio'] = behavioral['switching_ratio'].fillna(0)
    behavioral['marcas_distintas'] = behavioral['marcas_distintas'].fillna(0)
    behavioral['marca_favorita'] = behavioral['marca_favorita'].fillna(0)
    
    agg_funcs = {}
    for i in range(1, 6):
        if f'precio_marca_{i}' in df.columns:
            agg_funcs[f'precio_prom_marca_{i}'] = (f'precio_marca_{i}', 'mean')
        if f'promo_marca_{i}' in df.columns:
            agg_funcs[f'exposicion_promo_{i}'] = (f'promo_marca_{i}', 'mean')
            
    price_promo = df.groupby('id').agg(**agg_funcs).reset_index()
    
    compra_promo_dfs = []
    for i in range(1, 6):
        col = f'promo_marca_{i}'
        if col in df.columns:
            cp = df_compras[df_compras['id_marca'] == i].groupby('id')[col].mean().reset_index(name=f'compra_con_promo_{i}')
            compra_promo_dfs.append(cp)
        
    for cp in compra_promo_dfs:
        price_promo = pd.merge(price_promo, cp, on='id', how='left')
        
    for i in range(1, 6):
        if f'compra_con_promo_{i}' in price_promo.columns:
            price_promo[f'compra_con_promo_{i}'] = price_promo[f'compra_con_promo_{i}'].fillna(0)
        else:
            price_promo[f'compra_con_promo_{i}'] = 0.0
            
    global_prices = {}
    for i in range(1, 6):
        if f'precio_marca_{i}' in df.columns:
            global_prices[i] = df[f'precio_marca_{i}'].mean()
        else:
            global_prices[i] = 1.0
            
    client_features = pd.merge(rfm, behavioral, on='id')
    client_features = pd.merge(client_features, price_promo, on='id')
    
    def get_relative_price(row):
        fav = int(row['marca_favorita'])
        if fav == 0 or fav not in global_prices:
            return 1.0
        precio_prom_fav_cliente = row.get(f'precio_prom_marca_{fav}', global_prices[fav])
        global_prom = global_prices[fav]
        if global_prom == 0:
            return 1.0
        return precio_prom_fav_cliente / global_prom
        
    client_features['precio_relativo_favorita'] = client_features.apply(get_relative_price, axis=1)
    
    demo_cols = ['id', 'genero', 'estado_civil', 'edad', 'nivel_educacion', 'ingreso_anual', 'ocupacion']
    existing_demo_cols = [c for c in demo_cols if c in df.columns]
    demographics = df[existing_demo_cols].drop_duplicates('id')
    client_features = pd.merge(client_features, demographics, on='id')
    
    client_features['comprador_activo'] = (client_features['tasa_compra'] >= 0.20).astype(int)
    
    logging.info(f"Feature engineering completed. Resulting shape: {client_features.shape}")
    return client_features

if __name__ == '__main__':
    logging.info("Reading data/compras_data.csv...")
    try:
        df = pd.read_csv('data/compras_data.csv')
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        exit(1)
        
    if 'tamanio_ciudad' in df.columns:
        df = df.drop(columns=['tamanio_ciudad'])
        
    features_df = build_client_features(df)
    features_df.to_csv('data/client_features.csv', index=False)
    logging.info("Saved features to data/client_features.csv")
