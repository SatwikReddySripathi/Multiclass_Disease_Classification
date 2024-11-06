from airflow import DAG
from airflow.operators.dummy import DummyOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
from airflow import configuration as conf

import os
import sys
import logging
# Adjust this path to match the absolute or relative path to the parent directory containing `Data`
# sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
# sys.path.append('/opt/airflow/src')

from src.preprocessing import process_images_airflow
from src.cloud_data_management import extracting_data_from_gcp
from src.schema_generation import validate_data_schema
from src.anomaly_detection import anomalies_detect
# from airflow.operators.email import EmailOperator
# from airflow.operators.trigger_dagrun import TriggerDagRunOperator
# from airflow.utils.trigger_rule import TriggerRule

# Set the logging directory to a known directory inside the Airflow container
LOG_DIR = '/opt/airflow/logs'
LOG_FILE_PATH = os.path.join(LOG_DIR, 'data_preprocessing.log')

# Ensure the logs directory exists
os.makedirs(LOG_DIR, exist_ok=True)

# Configure logging to write to the specified file
logging.basicConfig(filename=LOG_FILE_PATH, level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

bucket_name = 'nih-dataset-mlops'

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
    
    start = DummyOperator(
        task_id='start'
    )

    extracting_data = PythonOperator(
        task_id='Data_Extraction',
        python_callable=extracting_data_from_gcp,
        op_kwargs={
            'bucket_name': bucket_name
        }
    )

    data_schema = PythonOperator(
        task_id='Data_Schema',
        python_callable=validate_data_schema,
        op_kwargs={
            'bucket_name': bucket_name
        }
    )

    anomaly_detection = PythonOperator(
        task_id='Anomaly_Detection',
        python_callable=anomalies_detect,
        op_kwargs={
            'bucket_name': bucket_name
        }
    )

    preprocess_task = PythonOperator(
        task_id='Preprocssing_images',
        python_callable=process_images_airflow,
    )

    end = DummyOperator(
        task_id='end'
    )

    # Set task dependencies
    # start >> extracting_data >> anomaly_detection >> end
    start >> extracting_data >> data_schema >> anomaly_detection >> preprocess_task >> end

logging.info("DAG loaded successfully")