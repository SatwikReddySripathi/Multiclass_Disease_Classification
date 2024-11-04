from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
from airflow import configuration as conf

import os
import sys
import logging

# Set the logging directory to a known directory inside the Airflow container
LOG_DIR = '/opt/airflow/logs'
LOG_FILE_PATH = os.path.join(LOG_DIR, 'data_preprocessing.log')

# Ensure the logs directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging to write to the specified file
logging.basicConfig(filename=LOG_FILE_PATH, level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add the source directory to the path for imports
sys.path.append('/opt/airflow/dags/src')

from src.preprocessing import process_images  # Ensure this path is valid in your container

# Enable pickle support for XCom (if required by your tasks)
conf.set('core', 'enable_xcom_pickling', 'True')

# DAG definition
DAG_NAME = 'Data_Preprocessing_Step'
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

with DAG(DAG_NAME, default_args=default_args, schedule_interval=None, catchup=False) as dag:
    
    start = DummyOperator(task_id='start')

    preprocess_task = PythonOperator(
        task_id='process_images',
        python_callable=process_images,
    )

    end = DummyOperator(task_id='end')

    # Set task dependencies
    start >> preprocess_task >> end

logging.info("DAG loaded successfully")