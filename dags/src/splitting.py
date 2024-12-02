import os
import shutil
import pickle
# import pandas as pd
# from google.colab import files
from sklearn.model_selection import train_test_split

import logging
from airflow.utils.log.logging_mixin import LoggingMixin

# Set up Airflow logger
airflow_logger = LoggingMixin().log
# Set the project directory
# PROJECT_DIR = '/opt/airflow'
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_DIR2 = os.path.join(PROJECT_DIR, 'dags', 'src','data')
INPUT_FILEPATH = os.path.join(PROJECT_DIR, 'Processed_Data', 'raw_compressed_data.pkl')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'Processed_Data')

LOG_DIR = os.path.join(PROJECT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, 'splitting_data.log')


logger = logging.getLogger('file_logger')
logger.setLevel(logging.DEBUG)  # Setting to DEBUG to capture all log messages or else it might not log info and error messages

file_handler = logging.FileHandler(LOG_FILE_PATH)
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("Logging configuration is set. Logs will be saved to: {}".format(LOG_FILE_PATH))

def custom_log(message, level=logging.INFO):
    """Log to both Airflow and custom file logger"""
    if level == logging.INFO:
        airflow_logger.info(message)
        logger.info(message)
    elif level == logging.ERROR:
        airflow_logger.error(message)
        logger.error(message)
    elif level == logging.WARNING:
        airflow_logger.warning(message)
        logger.warning(message)


########
## Splitting Pickle files


def split_pickle_data(pickle_file, output_folder, test_size=0.2, random_state=42):
    
    i= 0

    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)

    #   print(type(data))
    #   print(data)
    custom_log(type(data))
    custom_log(data)

    if isinstance(data, dict):
        keys = list(data.keys())
        train_keys, test_keys = train_test_split(keys, test_size=test_size, random_state=random_state)

        train_data = {key: data[key] for key in train_keys}
        for key in train_keys:
            if i==1:
                i+=1
                # print(key)
                # print(train_data[key])
                custom_log(key)
                custom_log(train_data[key])

        test_data = {key: data[key] for key in test_keys}

    elif isinstance(data, (list, tuple)):
        train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)

    else:
        raise ValueError(f"Unsupported data type: {type(data)}. Only dictionaries and lists are supported.")

    #train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    # os.makedirs(output_folder, exist_ok=True)

    train_pickle_file = f"{output_folder}/train_data.pkl"
    with open(train_pickle_file, 'wb') as f:
        pickle.dump(train_data, f)
    # print(f"Train data saved to {train_pickle_file}")
    custom_log(f"Train data saved to {train_pickle_file}")
    #files.download(train_pickle_file)

    test_pickle_file = f"{output_folder}/test_data.pkl"
    with open(test_pickle_file, 'wb') as f:
        pickle.dump(test_data, f)
    # print(f"Test data saved to {test_pickle_file}")
    custom_log(f"Test data saved to {test_pickle_file}")
    #files.download(test_pickle_file)


def splitting_airflow():

    input_pickle_file = INPUT_FILEPATH
    output_folder = OUTPUT_DIR

    with open(input_pickle_file, 'rb') as f:
        data = pickle.load(f)

    custom_log(type(data))
    custom_log(f"Number of keys: {len(data)}")
    custom_log(f"Sample keys: {list(data.keys())[:5]}")
    custom_log(f"Sample value of the first key: {data[list(data.keys())[0]]}")
    split_pickle_data(input_pickle_file, output_folder, test_size=0.2, random_state=42)

if __name__ == "__main__":
    splitting_airflow()