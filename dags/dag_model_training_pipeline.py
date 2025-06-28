from airflow import DAG
from airflow.operators.bash import BashOperator

from datetime import datetime, timedelta

from configs.data import gold_data_dirs
from configs.models import DEFAULT_TRAINING_VALUES

# Define the project root to add to the Python path.
PYTHON_DIR = "/opt/airflow" 

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'dag_model_training_pipeline',
    default_args=default_args,
    description='training pipeline runs after manual trigger',
    schedule=None,  # No schedule, manual trigger only
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:
 
    # --- check availability of data in feature store and label store ---

    check_label_store_availability = BashOperator(
        task_id='check_label_store_availability',
        bash_command=(
            f'PYTHONPATH={PYTHON_DIR} python /opt/airflow/scripts/ml_processing/data_availability_check.py '
            f'--store_dir {gold_data_dirs["label_store"]} '
            # FIX: Removed trailing space from inside the quotes
            f'--start_date "{{{{ dag_run.conf.get("start_date", "{DEFAULT_TRAINING_VALUES["start_date"]}") }}}}" '
            f'--end_date "{{{{ dag_run.conf.get("end_date", "{DEFAULT_TRAINING_VALUES["end_date"]}") }}}}"'
        )
    )

    check_feature_store_availability = BashOperator(
        task_id='check_feature_store_availability',
        bash_command=(
            f'PYTHONPATH={PYTHON_DIR} python /opt/airflow/scripts/ml_processing/data_availability_check.py '
            f'--store_dir {gold_data_dirs["feature_store"]} '
            # FIX: Removed trailing space from inside the quotes
            f'--start_date "{{{{ dag_run.conf.get("start_date", "{DEFAULT_TRAINING_VALUES["start_date"]}") }}}}" '
            f'--end_date "{{{{ dag_run.conf.get("end_date", "{DEFAULT_TRAINING_VALUES["end_date"]}") }}}}"'
        )
    )
    # --- prepare data for model training ---

    prepare_data = BashOperator(
        task_id='prepare_data',
        bash_command=(
            f'PYTHONPATH={PYTHON_DIR} python /opt/airflow/scripts/ml_processing/training_data_prep.py '
            # FIX: Removed trailing spaces from inside the quotes
            f'--start "{{{{ dag_run.conf.get("start_date", "{DEFAULT_TRAINING_VALUES["start_date"]}") }}}}" '
            f'--end "{{{{ dag_run.conf.get("end_date", "{DEFAULT_TRAINING_VALUES["end_date"]}") }}}}" '
            f'--oot "{{{{ dag_run.conf.get("oot", "{DEFAULT_TRAINING_VALUES["oot"]}") }}}}"'
        )
    )

    # --- model training ---

    model_gbt_train = BashOperator(
        task_id='model_gbt_train',
        bash_command=(
            f'PYTHONPATH={PYTHON_DIR} python /opt/airflow/scripts/ml_processing/model_trainer.py --model_type gbt'
        ),
    )

    model_logreg_train = BashOperator(
        task_id='model_logreg_train',
        bash_command=(
            f'PYTHONPATH={PYTHON_DIR} python /opt/airflow/scripts/ml_processing/model_trainer.py --model_type logreg'
        ),
    )

    model_promote_best = BashOperator(
        task_id='model_promote_best',
        bash_command=(
            f'PYTHONPATH={PYTHON_DIR} python /opt/airflow/scripts/ml_processing/model_selector.py'
        ),
    )

    # --- Task Dependencies ---

    check_label_store_availability >> prepare_data
    check_feature_store_availability >> prepare_data

    prepare_data >> model_gbt_train >> model_promote_best
    prepare_data >> model_logreg_train >> model_promote_best
