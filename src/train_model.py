import pandas as pd
import numpy as np
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
import xgboost as xgb
import optuna
import shap
import pickle
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_pipeline(features_path='data/client_features.csv'):
    
    logging.info(f"Loading features from {features_path}...")
    df = pd.read_csv(features_path)
    
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    logging.info("Step A: Clustering")
    cluster_features = ['tasa_compra', 'frecuencia', 'ingreso_anual', 'edad', 'switching_ratio', 'precio_relativo_favorita']
    
    scaler = StandardScaler()
    X_cluster = scaler.fit_transform(df[cluster_features])
    
    inertias = []
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_cluster)
        inertias.append(km.inertia_)
        
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, 9), inertias, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia')
    plt.savefig('outputs/elbow.png')
    plt.close()
    
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['segmento'] = kmeans.fit_predict(X_cluster)
    
    cluster_means = df.groupby('segmento')[cluster_features].mean()
    
    idx_leal = (cluster_means['tasa_compra'] - cluster_means['switching_ratio']).idxmax()
    idx_cazador = (cluster_means['frecuencia'] + cluster_means['switching_ratio']).drop(idx_leal, errors='ignore').idxmax()
    idx_premium = (cluster_means['ingreso_anual'] - cluster_means['frecuencia']).drop([idx_leal, idx_cazador], errors='ignore').idxmax()
    
    remaining = [i for i in range(4) if i not in [idx_leal, idx_cazador, idx_premium]]
    idx_pasivo = remaining[0] if remaining else 0
    
    mapping = {
        idx_leal: "comprador_leal",
        idx_cazador: "cazador_ofertas",
        idx_premium: "comprador_premium_ocasional",
        idx_pasivo: "visitante_pasivo"
    }
    
    df['segmento_nombre'] = df['segmento'].map(mapping)
    logging.info(f"Segment distribution:\n{df['segmento_nombre'].value_counts()}")
    
    logging.info("Step B: Propensity Model Training")
    
    exclude_cols = [
        'id', 'tasa_compra', 'n_compras', 'comprador_activo', 'segmento_nombre',
        'valor_monetario', 'marca_favorita', 'marcas_distintas', 'switching_ratio'
    ]
    for i in range(1, 6):
        exclude_cols.append(f'compra_con_promo_{i}')
        
    features_model = [c for c in df.columns if c not in exclude_cols]
    
    X = df[features_model]
    y = df['comprador_activo']
    
    ratio = float(np.sum(y == 0)) / np.sum(y == 1)
    
    from sklearn.model_selection import StratifiedKFold
    
    def objective(trial):
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        params = {
            'n_estimators': 500,
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'scale_pos_weight': ratio,
            'random_state': 42,
            'eval_metric': 'auc',
            'early_stopping_rounds': 30
        }
        for train_idx, val_idx in cv.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            model_trial = xgb.XGBClassifier(**params)
            model_trial.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            preds = model_trial.predict_proba(X_val)[:, 1]
            scores.append(roc_auc_score(y_val, preds))
        return np.mean(scores)
        
    logging.info("Running Optuna tuning...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    
    best_params = study.best_params
    best_params['n_estimators'] = 500
    best_params['scale_pos_weight'] = ratio
    best_params['random_state'] = 42
    best_params['eval_metric'] = 'auc'
    best_params['early_stopping_rounds'] = 30
    
    logging.info(f"Best Optuna params: {best_params}")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds_proba = np.zeros(len(X))
    oof_preds_class = np.zeros(len(X))
    
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model_cv = xgb.XGBClassifier(**best_params)
        model_cv.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        oof_preds_proba[val_idx] = model_cv.predict_proba(X_val)[:, 1]
        oof_preds_class[val_idx] = model_cv.predict(X_val)
    
    roc_test = roc_auc_score(y, oof_preds_proba)
    logging.info(f"CV AUC-ROC: {roc_test:.4f}")
    logging.info(f"CV Classification Report:\n{classification_report(y, oof_preds_class)}")
    
    cm = confusion_matrix(y, oof_preds_class)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (CV)')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('outputs/confusion_matrix.png')
    plt.close()
    
    best_params_no_early = best_params.copy()
    if 'early_stopping_rounds' in best_params_no_early:
        del best_params_no_early['early_stopping_rounds']
    final_model = xgb.XGBClassifier(**best_params_no_early)
    final_model.fit(X, y, verbose=False)
    
    logging.info("Step C: SHAP Explainability")
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X)
    
    vals = shap_values[1] if isinstance(shap_values, list) else shap_values
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(vals, X, show=False)
    plt.savefig('outputs/shap_summary.png', bbox_inches='tight')
    plt.close()
    
    top_features = []
    feature_names = X.columns.tolist()
    for i in range(len(X)):
        idx = np.argmax(vals[i])
        top_features.append(feature_names[idx])
        
    df['shap_top_feature'] = top_features
    df['score_propension'] = final_model.predict_proba(X)[:, 1]
    
    logging.info("Step D: Saving outputs")
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    final_model.save_model('models/xgb_model.json')
    
    output_cols = [
        'id', 'score_propension', 'comprador_activo', 'segmento', 'segmento_nombre', 
        'shap_top_feature', 'edad', 'genero', 'ingreso_anual', 'ocupacion', 
        'n_compras', 'frecuencia', 'tasa_compra', 'marca_favorita', 
        'switching_ratio', 'recencia', 'dias_entre_visitas_median'
    ]
    existing_output_cols = [c for c in output_cols if c in df.columns]
    final_predictions = df[existing_output_cols]
    final_predictions.to_csv('data/predictions.csv', index=False)
    logging.info("Saved data/predictions.csv and models successfully.")

if __name__ == '__main__':
    train_pipeline()
