"""
explain.py
Computes unified feature importance for the 3-model ensemble
(XGBoost + LightGBM + CatBoost) using SHAP TreeExplainer.

It writes a small CSV that app.py/dashboard
reads at display time -- no SHAP computation happens in the app itself.
"""

import numpy as np
import pandas as pd
import joblib
import shap
from catboost import CatBoostRegressor
from pathlib import Path
from features import build_features,FEATURES

CURRENT_DIR = Path(__file__).resolve().parent
WEIGHTS = {
    "xgb": 1 / 3,
    "lgb": 1 / 3,
    "cat": 1 / 3,
}

MODEL_PATHS = {
    "xgb": CURRENT_DIR / 'models'/'xgb.pkl',
    "lgb": CURRENT_DIR / 'models'/'lgbm.pkl',
    "cat": CURRENT_DIR / 'models'/ 'catb.cbm',
}

FEATURES_PATH = CURRENT_DIR / 'data' /'player_data_2026.csv'
OUTPUT_PATH = CURRENT_DIR / 'data' /'feature_importance.csv'

def load_models():

    xgb_model = joblib.load(MODEL_PATHS["xgb"])

    lgb_model = joblib.load(MODEL_PATHS["lgb"])

    cat_model = CatBoostRegressor()
    cat_model.load_model(MODEL_PATHS["cat"])


    return xgb_model, lgb_model, cat_model


def compute_shap_values(xgb_model, lgb_model, cat_model, X: pd.DataFrame):
    explainer_xgb = shap.TreeExplainer(xgb_model)
    explainer_lgb = shap.TreeExplainer(lgb_model)
    explainer_cat = shap.TreeExplainer(cat_model)

    shap_xgb = explainer_xgb.shap_values(X)
    shap_lgb = explainer_lgb.shap_values(X)
    shap_cat = explainer_cat.shap_values(X)

    return shap_xgb, shap_lgb, shap_cat


def combine_shap(shap_xgb, shap_lgb, shap_cat, weights: dict):
    return (
        weights["xgb"] * shap_xgb
        + weights["lgb"] * shap_lgb
        + weights["cat"] * shap_cat
    )


def main():
    data = pd.read_csv(FEATURES_PATH)
    X = build_features(data, objective="predict")
    X = X[FEATURES]

    xgb_model, lgb_model, cat_model = load_models()
    shap_xgb, shap_lgb, shap_cat = compute_shap_values(xgb_model, lgb_model, cat_model, X)
    ensemble_shap = combine_shap(shap_xgb, shap_lgb, shap_cat, WEIGHTS)

    # Global importance: mean absolute SHAP value per feature, across all players
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": np.abs(ensemble_shap).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    importance_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved ensemble feature importance to {OUTPUT_PATH}")
    print(importance_df.head(10))

if __name__ == "__main__":
    main()