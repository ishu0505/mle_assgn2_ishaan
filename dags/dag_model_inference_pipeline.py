from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.sensors.external_task import ExternalTaskSensor

from datetime import datetime, timedelta


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'dag_model_inference_pipeline',
    default_args=default_args,
    description='inference pipeline run once a month after data pipeline',
    schedule='0 15 1 * *',  # At 00:15 on day-of-month 1 --> 1 hour after midnight since data pipeline has to run first
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 2),
    catchup=True,
) as dag:
 
    # --- wait for data pipeline to complete ---

    label_store_completed = ExternalTaskSensor(
        task_id='wait_for_label_store_completed',
        external_dag_id='dag_data_pipeline',
        external_task_id='label_store_completed',
        allowed_states=['success'],
        failed_states=['failed', 'skipped'],
        mode='reschedule',
        timeout=60 * 10,  # Wait for 10 minutes
    )

    feature_store_completed = ExternalTaskSensor(
        task_id='wait_for_feature_store_completed',
        external_dag_id='dag_data_pipeline',
        external_task_id='feature_store_completed',
        allowed_states=['success'],
        failed_states=['failed', 'skipped'],
        mode='reschedule',
        timeout=60 * 10,  # Wait for 10 minutes
    )

    # --- check if trained models are available ---

    dep_check_trained_models = PythonOperator(
        task_id='dep_check_trained_models',
        python_callable=lambda: print("Checking if trained models are available...")
        # This is a placeholder; replace with actual check logic if needed
    )

    # --- model inference ---

    model_inference = EmptyOperator(task_id="model_inference")

    # --- model monitoring ---

    model_monitoring = EmptyOperator(task_id="model_1_monitor")

    # --- Task Dependencies ---

    label_store_completed >> model_inference # type: ignore
    feature_store_completed >> model_inference # type: ignore
    dep_check_trained_models >> model_inference # type: ignore
    model_inference >> model_monitoring # type: ignore

    # --- model auto training ---

    model_automl_start = EmptyOperator(task_id="model_automl_start")
    
    model_1_automl = EmptyOperator(task_id="model_1_automl")

    model_2_automl = EmptyOperator(task_id="model_2_automl")

    model_automl_completed = EmptyOperator(task_id="model_automl_completed")
    
    # Define task dependencies to run scripts sequentially
    feature_store_completed >> model_automl_start
    label_store_completed >> model_automl_start
    model_automl_start >> model_1_automl >> model_automl_completed
    model_automl_start >> model_2_automl >> model_automl_completed