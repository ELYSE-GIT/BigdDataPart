from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 31),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
}

dag = DAG(
    'credit_model_training', 
    default_args=default_args, 
    description='Trains a LightGBM model for credit scoring', 
    schedule_interval=timedelta(minutes=10),
    catchup=False
)

def train_model():
    # code pour entraîner le modèle, identique au code dans src/train.py
    # 
    return ''
train_task = PythonOperator(
    task_id='train_model', 
    python_callable=train_model, 
    dag=dag
)

