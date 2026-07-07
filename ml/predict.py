# predict.py — called weekly by Airflow
import pandas as pd
import numpy as np
import catboost, joblib
from pathlib import Path
import sys

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.append(str(CURRENT_DIR))

from features import build_features, FEATURES, STATS

catb = catboost.CatBoostRegressor()
catb.load_model(CURRENT_DIR / 'models'/ 'catb.cbm')
xgb  = joblib.load(CURRENT_DIR / 'models'/'xgb.pkl')
lgbm = joblib.load(CURRENT_DIR / 'models'/'lgbm.pkl')

data_path = CURRENT_DIR / 'data' /'player_data_2026.csv'
data = pd.read_csv(data_path)

def predict_mvp_race(current_season_raw: pd.DataFrame) -> pd.DataFrame:
    df = build_features(current_season_raw, objective="predict")

    preds = np.mean([
        np.clip(catb.predict(df[FEATURES]), 0, None),
        np.clip(xgb.predict(df[FEATURES]),  0, None),
        np.clip(lgbm.predict(df[FEATURES]), 0, None),
    ], axis=0)

    df['Predicted_Share'] = preds
    return (df[['Player','Team', 'Pos','Season', 'Predicted_Share'] + STATS]
              .sort_values('Predicted_Share', ascending=False)
              .reset_index(drop=True))

predictions = predict_mvp_race(data)
predictions.to_csv(CURRENT_DIR / 'data' /'predictions.csv')

