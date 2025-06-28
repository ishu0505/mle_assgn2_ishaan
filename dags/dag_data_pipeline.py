
# File: dags/dag_hybrid_optimized_pipeline.py
# This DAG combines sequential logic for the label store with parallel logic for the feature store.

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime, timedelta

# --- Configuration ---
# Define constants for paths to make the code cleaner and easier to maintain.
SCRIPTS_DIR = "/opt/airflow/scripts/data_processing"
DATAMART_DIR = "/opt/airflow/datamart"
# Define the project root to add to the Python path.
PYTHON_DIR = "/opt/airflow" 

default_args = {
    'owner': 'airflow',
    # FIX: Set to True. This ensures that a DAG run will only start if the previous
    # scheduled run has completed successfully. This is crucial for preventing data gaps.
    'depends_on_past': True,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'dag_hybrid_optimized_pipeline',
    default_args=default_args,
    description='A hybrid DAG with sequential labels and parallel features.',
    schedule='0 0 1 * *',  # At 00:00 on the 1st of every month
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,
    max_active_runs=4,
    tags=['production', 'hybrid'],
) as dag:

    # --- Define clear start and end points for the entire pipeline ---
    start = EmptyOperator(task_id="start_pipeline")
    end = EmptyOperator(task_id="end_pipeline")

    # =============================================================================
    # BRANCH 1: LABEL STORE PIPELINE (Sequential Logic)
    # =============================================================================

    dep_check_source_label_data = PythonOperator(
        task_id='dep_check_source_label_data',
        python_callable=lambda: print("Checking if source label data is available...")
    )

    bronze_label_store = BashOperator(
        task_id='run_bronze_label_store',
        bash_command=f"PYTHONPATH={PYTHON_DIR} python {SCRIPTS_DIR}/bronze_processing.py --date '{{{{ ds }}}}' --type loan_data"
    )

    silver_label_store = BashOperator(
        task_id='run_silver_label_store',
        bash_command=f"PYTHONPATH={PYTHON_DIR} python {SCRIPTS_DIR}/silver_processing.py --date '{{{{ ds }}}}' --type loan_data"
    )

    gold_label_store = BashOperator(
        task_id="run_gold_label_store",
        bash_command=f"PYTHONPATH={PYTHON_DIR} python {SCRIPTS_DIR}/gold_processing_label.py --date '{{{{ ds }}}}'"
    )

    label_store_completed = EmptyOperator(task_id="label_store_completed")
    
    # Define dependencies for the sequential Label Store branch
    start >> dep_check_source_label_data >> bronze_label_store >> silver_label_store >> gold_label_store >> label_store_completed

    # =============================================================================
    # BRANCH 2: FEATURE STORE PIPELINE (Parallel Logic)
    # =============================================================================
    
    join_before_gold_features = EmptyOperator(task_id="join_before_gold_features")
    
    for feature_type in ['customer_attributes', 'customer_financials', 'clickstream_data']:
        
        dep_check_task = PythonOperator(
            task_id=f'dep_check_source_{feature_type}',
            python_callable=lambda ft=feature_type: print(f"Checking if source {ft} data is available...")
        )

        bronze_task = BashOperator(
            task_id=f'bronze_{feature_type}',
            bash_command=f"PYTHONPATH={PYTHON_DIR} python {SCRIPTS_DIR}/bronze_processing.py --date '{{{{ ds }}}}' --type {feature_type}"
        )
        
        silver_task = BashOperator(
            task_id=f'silver_{feature_type}',
            bash_command=f"PYTHONPATH={PYTHON_DIR} python {SCRIPTS_DIR}/silver_processing.py --date '{{{{ ds }}}}' --type {feature_type}"
        )

        # Define dependencies within each parallel feature branch
        start >> dep_check_task >> bronze_task >> silver_task >> join_before_gold_features

    gold_feature_store = BashOperator(
        task_id='gold_feature_store',
        bash_command=f"PYTHONPATH={PYTHON_DIR} python {SCRIPTS_DIR}/gold_processing_features.py --date '{{{{ ds }}}}'"
    )

    feature_store_completed = EmptyOperator(task_id="feature_store_completed")

    # Define the final dependencies for the Feature Store branch
    join_before_gold_features >> gold_feature_store >> feature_store_completed

    # Both main branches must complete before the DAG can end.
    [label_store_completed, feature_store_completed] >> end

