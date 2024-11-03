
from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
from airflow import configuration as conf

import os
import sys
# Adjust this path to match the absolute or relative path to the parent directory containing `Data`
# sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append('/opt/airflow/src')

from src.preprocessing import process_images
# from airflow.operators.email import EmailOperator
# from airflow.operators.trigger_dagrun import TriggerDagRunOperator
# from airflow.utils.trigger_rule import TriggerRule

# Enable pickle support for XCom
conf.set('core', 'enable_xcom_pickling', 'True')

# Set up custom file logger
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, 'airflow.log')

DAG_NAME = 'Data_Preprocessing_Steps'


original_data_folder = '/content/drive/My Drive/MLOPs Project/sampled_data'
preprocessed_data_folder = '/content/drive/My Drive/MLOPs Project/preprocessed_data'
csv_path = '/content/drive/My Drive/MLOPs Project/sampled_train_data_entry.csv'
label_json_path = '/content/drive/My Drive/MLOPs Project/label_indices.json'

default_args = {
    'owner': 'MLOPs_Team10',
    'depends_on_past': False,
    'start_date': datetime(2024, 10, 11),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1
}

dag = DAG(
    DAG_NAME,
    default_args=default_args,
    description='Data preprocessing Steps of Images',
    schedule_interval=None,  # Set the schedule interval or use None for manual triggering
    catchup=False,
    tags=['Group10']
    )

start_task = DummyOperator(
    task_id='Start',
    dag=dag
)

end_task = DummyOperator(
    task_id='End',
    dag=dag
)

process_images_task = PythonOperator(
    task_id='Process_Images',
    python_callable=process_images,
    op_kwargs={
        'original_data_folder': original_data_folder,
        'preprocessed_data_folder': preprocessed_data_folder,
        'csv_path': csv_path,
        'label_json_path': label_json_path
    },
    dag=dag
)

start_task >> process_images_task >> end_task