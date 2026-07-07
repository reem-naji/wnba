from datetime import datetime, timedelta
from airflow.sdk import dag, task

@dag(
    dag_id="weekly_data_prediction_updater",
    start_date=datetime(2026, 7, 1),
    schedule='0 10 * * 1',
    catchup=False,
)
def weekly_data_prediction_updater():

    @task.bash
    def run_ingestion():
        return 'cd /opt/airflow && python ingestion/loaders/load_data.py'

    @task.bash
    def run_prediction():
        return 'cd /opt/airflow && python ml/predict.py'

    run_ingestion() >> run_prediction()

weekly_data_prediction_updater()