import pandas as pd
import xgboost, catboost, lightgbm, joblib
from features import build_features, FEATURES, TARGET
import os
from dotenv import load_dotenv
from pathlib import Path
   
load_dotenv()

CURRENT_DIR = Path(__file__).resolve().parent
MODELS_DIR = CURRENT_DIR / 'models'

# create the 'models' folder if it does not exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)

data_path = CURRENT_DIR / 'data' /'wnba_players_data.csv'
data = pd.read_csv(data_path)
df = build_features(data)

X = df[FEATURES]
y = df[TARGET]

xgb = xgboost.XGBRegressor(**{'max_depth': 3, 'eta': 0.0114, 'n_estimators': 363, 'subsample': 0.67, 'random_state':0})
catb = catboost.CatBoostRegressor(**{'depth': 5, 'learning_rate': 0.025, 'iterations': 670, 'l2_leaf_reg': 7.21, 'random_state':0, 'verbose':0})
lgbm = lightgbm.LGBMRegressor(**{'num_leaves': 112, 'learning_rate': 0.049, 'feature_fraction': 0.67, 'colsample_bytree': 0.62, 'min_child_samples': 16, 'random_state':0, 'verbose':-1})

xgb.fit(X, y)
catb.fit(X, y)
lgbm.fit(X, y)

catb.save_model(str(MODELS_DIR / 'catb.cbm'))
joblib.dump(xgb, MODELS_DIR / 'xgb.pkl')
joblib.dump(lgbm, MODELS_DIR/ 'lgbm.pkl')

print('Training completed. Models saved.')