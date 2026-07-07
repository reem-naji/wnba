# 🏀 2026 WNBA MVP Race Predictor

An end-to-end machine learning project that predicts the WNBA MVP race using a tuned gradient boosting ensemble, trained on 29 seasons of historical data and deployed as a live, interactive dashboard.

**Live Dashboard:** [wnba-project.streamlit.app](https://wnba-project.streamlit.app/)
**Dataset (Kaggle):** [wnba-player-statistics-with-mvp-votes-1997-2025](https://www.kaggle.com/datasets/reemmalih/wnba-player-statistics-with-mvp-votes-1997-2025)
**Modeling Notebook (Kaggle):** [wnba-mvp-prediction-gradient-boosting-ensemble](https://www.kaggle.com/code/reemmalih/wnba-mvp-prediction-gradient-boosting-ensemble)

---

## Project Motivation

As a WNBA fan, I noticed there wasn't a rigorous, data-driven way to track the MVP race in real time — most "predictions" during a season are narrative-driven takes rather than models grounded in historical voting patterns. This project builds that: a model trained on every MVP vote since 1997, applied to weekly updated 2026 season data, with the reasoning behind each prediction made transparent through feature importance analysis.

---

## Key Results

| Metric                              | Value                       |
| ----------------------------------- | --------------------------- |
| R² (walk-forward CV, 2013–2025)     | ≈ 0.78                      |
| Hit@3 (true MVP in top-2 predicted) | 100%                        |
| Kaggle dataset usability score      | 10.0                        |
| Training window                     | 1997–2025 (29 seasons)      |
| Models ensembled                    | XGBoost, LightGBM, CatBoost |

---

## Data

Player statistics and MVP voting shares for **1997–2025** were scraped from [Basketball-Reference](https://www.basketball-reference.com/), but the plus minus of each player is scraped from the nba_api open source module in python (https://github.com/swar/nba_api), then was cleaned, merged, and published as a public dataset on Kaggle. The 2026 season is scraped separately each time predictions are refreshed, using the same feature pipeline.

## Methodology

1. **EDA** — explored scoring trends, usage patterns, and historical MVP voting behavior across eras (`ml/notebooks/eda.ipynb`)
2. **Feature engineering** — built the statistical feature set used for training (`ml/features.py`)
3. **Hyperparameter tuning** — XGBoost, LightGBM, and CatBoost each tuned independently with **Optuna**
4. **Walk-forward cross-validation** — models trained and validated season-by-season in chronological order, avoiding lookahead bias (a common mistake with standard k-fold on time-series sports data)
5. **Ensemble** — final predicted award share is the average of the three tuned models' outputs
6. **Explainability** — since native feature importances aren't comparable across XGBoost/LightGBM/CatBoost (different internal metrics), unified importance is computed by combining **SHAP values** from all three models using the same weights as the prediction ensemble (`ml/explain.py`)

Full modeling workflow, including tuning and validation results, is documented in the [Kaggle notebook](https://www.kaggle.com/code/reemmalih/wnba-mvp-prediction-gradient-boosting-ensemble).

## 2026 Prediction Pipeline

- `ingestion/extractors/bbref_scraper.py` — scrapes current-season player stats
- `ingestion/loaders/load_data.py` — cleans and loads scraped data into the feature pipeline
- `ml/predict.py` — generates predicted award share for every active player using the saved ensemble
- `ml/explain.py` — computes ensemble-level SHAP feature importance for the dashboard's explainability chart

## Dashboard

Built with Streamlit and deployed on Streamlit Community Cloud. The layout walks the viewer through: **who** the model predicts as MVP , **why** (feature importance) , **where** the frontrunner stands statistically among the league (interactive scatter plot) , **how large the gap is** between MVP-caliber players and the rest of the league (distribution/outlier analysis).

- MVP frontrunner card with predicted award share
- Top 5 contenders point plot
- SHAP-based feature importance chart
- Interactive scatter plot (player stats, hover for details, adjustable axes)
- Distribution analysis (histogram + boxplot) comparing MVP-caliber players vs. the rest of the league

---

## Project Structure

```
wnba-predictor/
├── .github/
│   └── workdlows/
│   │   └── weekly_updater.yml                      # WIP: GitHub Actions automated weekly pipeline execution
├── airflow/
│   └── dags/
│   │   └── weekly_update_dag.py            # DAG to orchestrate weekly scraping and prediction updates
│   │   └── docker-compose.yml
│   │   └── Dockerfile
├── ingestion/
│   ├── extractors/
│   │   └── bbref_scraper.py     # scrapes Basketball-Reference
│   └── loaders/
│       └── load_data.py         # cleans and loads scraped data
├── ml/
│   ├── data/                    # feature/prediction CSVs
│   ├── notebooks/
│   │   ├── eda.ipynb
│   │   ├── scrape.ipynb
│   │   └── merge.ipynb
│   ├── features.py
│   ├── train.py
│   └── predict.py
│   └── explain.py                   # ensemble SHAP feature importance
├── api/
│   └── main.py                  # scaffolded for future API deployment
├── dashboard/
│   ├── .streamlit/config.toml
│   └── app.py
│   └── Dockerfile
├── docker-compose.yml           # scaffolded for future containerized deployment
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
```

## Tech Stack

**Modeling:** Python, XGBoost, LightGBM, CatBoost, Optuna, SHAP, scikit-learn
**Data:** BeautifulSoup, Pandas
**Dashboard:** Streamlit
**Planned:** FastAPI

---

## Running Locally

```bash
git clone <repo-url>
cd wnba-predictor
pip install -r requirements.txt
streamlit run dashboard/app.py
```

Run the Airflow Scheduler (via Docker):

```bash
cd airflow
docker-compose up -d
```

You can access the Airflow UI at http://localhost:8080

---

## Roadmap

This project currently runs its scraping, prediction, and explainability steps manually. Planned next steps:

- [x] Automate weekly data refresh (Airflow DAG implemented and active; GitHub Actions workflow in progress)
- [x] Implement the scaffolded Airflow DAGs for production orchestration
- [ ] Deploy `api/main.py` as a public REST endpoint serving live predictions
- [ ] Add unit tests and CI (GitHub Actions)
- [ ] Track training experiments with MLflow

```

```
