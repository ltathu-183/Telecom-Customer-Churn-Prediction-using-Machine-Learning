"""
Airflow DAG for end-to-end churn prediction pipeline.
Demonstrates MLOps capabilities required by Viettel.
"""

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.providers.slack.operators.slack import SlackAPIPostOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import os

# Default arguments
default_args = {
    'owner': 'data_science_team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email': ['ai-team@viettel.com.vn'],
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'start_date': days_ago(1),
}

# DAG definition
dag = DAG(
    'churn_prediction_pipeline',
    default_args=default_args,
    description='End-to-end churn prediction pipeline for Viettel',
    schedule_interval='0 2 * * *',  # Run daily at 2 AM ICT
    catchup=False,
    tags=['churn', 'ml', 'production', 'viettel'],
)


def check_data_quality(**context):
    """Check data quality before processing"""
    import pandas as pd
    
    data_path = context['ti'].xcom_pull(task_ids='extract_data')
    df = pd.read_csv(data_path)
    
    # Check missing values threshold
    missing_pct = df.isnull().mean() * 100
    if (missing_pct > 10).any():
        context['ti'].xcom_push(key='quality_check', value='failed')
        raise ValueError(f"Data quality check failed: columns with >10% missing: {missing_pct[missing_pct > 10]}")
    
    # Check row count
    if len(df) < 1000:
        context['ti'].xcom_push(key='quality_check', value='failed')
        raise ValueError(f"Insufficient data: {len(df)} rows (minimum 1000 required)")
    
    context['ti'].xcom_push(key='quality_check', value='passed')
    return 'preprocess_data'


def monitor_model_drift(**context):
    """Monitor model drift and trigger retraining if needed"""
    import pandas as pd
    from src.monitoring.drift_detection import DriftMonitor
    
    # Load reference data
    reference_data = pd.read_csv('/data/reference_data.csv')
    
    # Get current predictions
    prediction_path = context['ti'].xcom_pull(task_ids='predict')
    current_data = pd.read_csv(prediction_path)
    
    # Detect drift
    monitor = DriftMonitor(reference_data)
    drift_result = monitor.detect_data_drift(current_data)
    
    # Push drift result
    context['ti'].xcom_push(key='drift_result', value=drift_result)
    
    # Trigger retraining if drift detected
    if drift_result['dataset_drift_detected']:
        context['ti'].xcom_push(key='trigger_retraining', value=True)
        return 'trigger_retraining_alert'
    else:
        return 'load_to_database'


def send_slack_alert(**context):
    """Send Slack alert for critical issues"""
    drift_result = context['ti'].xcom_pull(task_ids='monitor_drift', key='drift_result')
    
    message = f"""
🚨 *MODEL DRIFT DETECTED* 🚨
Pipeline: churn_prediction_pipeline
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Drift Share: {drift_result['drift_share']:.2%}
Drifted Features: {drift_result['number_of_drifted_columns']}
Action: Retraining pipeline triggered automatically
"""
    
    return message


# Task definitions
extract_data = BashOperator(
    task_id='extract_data',
    bash_command='python /opt/airflow/scripts/extract_data.py --output /tmp/data_{{ ds }}.csv',
    dag=dag,
)

check_quality = BranchPythonOperator(
    task_id='check_data_quality',
    python_callable=check_data_quality,
    dag=dag,
)

quality_failed = BashOperator(
    task_id='quality_check_failed',
    bash_command='echo "Data quality check failed - aborting pipeline"',
    dag=dag,
)

preprocess_data = DockerOperator(
    task_id='preprocess_data',
    image='viettel/churn-preprocessing:latest',
    command='python preprocess.py --input /tmp/data_{{ ds }}.csv --output /tmp/processed_{{ ds }}.csv',
    docker_url='unix://var/run/docker.sock',
    network_mode='bridge',
    dag=dag,
)

generate_features = DockerOperator(
    task_id='generate_features',
    image='viettel/churn-feature-engineering:latest',
    command='python feature_builder.py --input /tmp/processed_{{ ds }}.csv --output /tmp/features_{{ ds }}.csv',
    docker_url='unix://var/run/docker.sock',
    network_mode='bridge',
    dag=dag,
)

predict = DockerOperator(
    task_id='predict',
    image='viettel/churn-model:latest',
    command='python predict.py --input /tmp/features_{{ ds }}.csv --output /tmp/predictions_{{ ds }}.csv',
    docker_url='unix://var/run/docker.sock',
    network_mode='bridge',
    dag=dag,
)

monitor_drift = BranchPythonOperator(
    task_id='monitor_drift',
    python_callable=monitor_model_drift,
    dag=dag,
)

trigger_retraining_alert = SlackAPIPostOperator(
    task_id='trigger_retraining_alert',
    slack_conn_id='slack_conn',
    text=send_slack_alert(),
    dag=dag,
)

trigger_retraining = BashOperator(
    task_id='trigger_retraining',
    bash_command='airflow dags trigger churn_retraining_pipeline --conf \'{"triggered_by": "drift_detection"}\'',
    dag=dag,
)

load_to_database = BashOperator(
    task_id='load_to_database',
    bash_command='python /opt/airflow/scripts/load_to_database.py --input /tmp/predictions_{{ ds }}.csv',
    dag=dag,
)

send_daily_report = SlackAPIPostOperator(
    task_id='send_daily_report',
    slack_conn_id='slack_conn',
    text='✅ *Churn Prediction Pipeline Completed*\n\n'
         f'Date: {{ ds }}\n'
         'Status: SUCCESS\n'
         'Predictions Generated: {{ ti.xcom_pull(task_ids="predict") | length }}\n'
         'High Risk Customers: {{ ti.xcom_pull(task_ids="predict") | selectattr("risk_level", "equalto", "High Risk") | list | length }}',
    dag=dag,
)

# Task dependencies
extract_data >> check_quality
check_quality >> [quality_failed, preprocess_data]
preprocess_data >> generate_features >> predict >> monitor_drift
monitor_drift >> [trigger_retraining_alert, load_to_database]
trigger_retraining_alert >> trigger_retraining
load_to_database >> send_daily_report