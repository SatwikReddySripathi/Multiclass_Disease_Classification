import os
import pandas as pd
import tensorflow_data_validation as tfdv
import pickle
from IPython.display import display
import logging
from airflow.utils.log.logging_mixin import LoggingMixin

# Set up Airflow logger
airflow_logger = LoggingMixin().log
# Set the project directory
PROJECT_DIR = '/opt/airflow'

# PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__name__))))
INPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'Processed_Data', 'raw_compressed_data.pkl')
SCHEMA_DIR = os.path.join(PROJECT_DIR, 'Processed_Data')
LOG_DIR = os.path.join(PROJECT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, 'schema_generation.log')

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


def load_data_from_pickle(file_path):
    """Load data from a pickle file."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            print(f"Loaded data from pickle file with {len(data)} records.")
            custom_log(f"Loaded data from pickle file with {len(data)} records.")
            return pd.DataFrame.from_dict(data, orient='index')
    except Exception as e:
        print("Error:",e)
        custom_log(f"Error: {e}",level=logging.ERROR)
        raise

def prepare_train_data(df):
    """Prepare training, evaluation, and serving data from the full dataset."""
    total_len = len(df)
    train_len = int(total_len * 0.6)
    eval_len = int(total_len * 0.2)

    train_df = df.iloc[:train_len].reset_index(drop=True)
    eval_df = df.iloc[train_len:train_len + eval_len].reset_index(drop=True)
    serving_df = df.iloc[train_len + eval_len:].drop(columns='image_label').reset_index(drop=True)

    return train_df, eval_df, serving_df

def generate_train_stats(train_df):
    """Generate statistics from the training dataset."""
    train_stats = tfdv.generate_statistics_from_dataframe(train_df)
    print("Generated training data statistics.")
    custom_log("Generated training data statistics.")
    return train_stats

def generate_serving_stats(serving_df):
    """Generate statistics from the serving dataset."""
    serving_stats = tfdv.generate_statistics_from_dataframe(serving_df)
    print("Generated serving data statistics.")
    custom_log("Generated serving data statistics.")
    return serving_stats

def infer_schema(train_stats):
    """Infer schema from the computed statistics."""
    schema = tfdv.infer_schema(statistics=train_stats)
    print("Inferred schema from training data statistics.")
    custom_log("Inferred schema from training data statistics.")
    return schema

def save_schema(schema, output_dir, suffix=''):
    """Save the schema to a file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    schema_file = os.path.join(output_dir, f'schema{suffix}.pbtxt')
    tfdv.write_schema_text(schema, schema_file)
    print(f"Schema saved to {schema_file}")
    custom_log(f"Schema saved to {schema_file}")
    return schema_file

def visualize_statistics(lhs_stats, rhs_stats, lhs_name="TRAIN_DATASET", rhs_name="EVAL_DATASET"):
    """Visualize statistics for comparison between two datasets."""
    tfdv.visualize_statistics(lhs_statistics=lhs_stats, rhs_statistics=rhs_stats, lhs_name=lhs_name, rhs_name=rhs_name)
    return "Statistics visualization complete."

def save_to_pickle(obj, file_path):
    """Save an object to a pickle file."""
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Saved object to pickle file at {file_path}")
    custom_log(f"Saved object to pickle file at {file_path}")

def save_to_csv(df, file_path):
    """Save DataFrame to a CSV file."""
    df.to_csv(file_path, index=False)
    print(f"DataFrame saved to CSV file at {file_path}")
    custom_log(f"DataFrame saved to CSV file at {file_path}")

def validate_data_schema():
    df = load_data_from_pickle(INPUT_PICKLE_PATH)
    train_df, eval_df, serving_df = prepare_train_data(df)
    train_stats = generate_train_stats(train_df)

    schema = infer_schema(train_stats)
    
    schema_file = save_schema(schema,SCHEMA_DIR)
    
    serving_stats = generate_serving_stats(serving_df)
    
    eval_stats = generate_train_stats(eval_df)

    
    visualization_result = visualize_statistics(lhs_stats=train_stats, rhs_stats=eval_stats)
    

# if __name__ == "__main__":
#     validate_data_schema()
