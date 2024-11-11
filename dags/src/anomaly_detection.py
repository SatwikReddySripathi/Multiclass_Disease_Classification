import os
import pandas as pd
import pickle
import tensorflow_data_validation as tfdv
from tensorflow_data_validation.utils import schema_util
from PIL import Image
import numpy as np
import io
from collections import Counter
import ast
import logging
from airflow.utils.log.logging_mixin import LoggingMixin

# Set up Airflow logger
airflow_logger = LoggingMixin().log
# Set the project directory
PROJECT_DIR = '/opt/airflow'
# PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__name__))))
INPUT_PICKLE_PATH = os.path.join(PROJECT_DIR, 'Processed_Data', 'raw_compressed_data.pkl')
SCHEMA_PATH = os.path.join(PROJECT_DIR, 'Processed_Data', "schema.pbtxt")
LOG_DIR = os.path.join(PROJECT_DIR, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE_PATH = os.path.join(LOG_DIR, 'anomaly_detection.log')

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


def load_dataframe(file_path):
    """Load data from a pickle file."""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            print(f"Loaded data from pickle file with {len(data)} records.")
            custom_log(f"Loaded data from pickle file with {len(data)} records.")
            return pd.DataFrame.from_dict(data, orient='index')
    except Exception as e:
        print("Error:", e)
        custom_log(f"Error: {e}",level=logging.ERROR)
        raise

def load_schema(schema_path):
    """Load schema from a .pbtxt file."""
    try:
        schema = schema_util.load_schema_text(schema_path)
        print(f"Schema loaded from {schema_path}")
        custom_log(f"Schema loaded from {schema_path}")
        return schema
    
    except Exception as e:
        print(f"Failed to load schema: {e}")
        custom_log(f"Failed to load schema: {e}",level=logging.ERROR)
        raise

def prepare_data_splits(df):
    """Prepare training, evaluation, and serving data splits."""
    total_len = len(df)
    train_len = int(total_len * 0.6)
    eval_len = int(total_len * 0.2)

    train_df = df.iloc[:train_len].reset_index(drop=True)
    eval_df = df.iloc[train_len:train_len + eval_len].reset_index(drop=True)
    serving_df = df.iloc[train_len + eval_len:].drop(columns='image_label').reset_index(drop=True)

    print(f"Prepared data splits: train shape {train_df.shape}, eval shape {eval_df.shape}, serving shape {serving_df.shape}")
    custom_log(f"Prepared data splits: train shape {train_df.shape}, eval shape {eval_df.shape}, serving shape {serving_df.shape}")
    return train_df, eval_df, serving_df

def generate_statistics(train_df, eval_df, serving_df, schema):
    """Generate statistics for train and eval datasets and detect anomalies."""
    train_stats = tfdv.generate_statistics_from_dataframe(train_df)
    eval_stats = tfdv.generate_statistics_from_dataframe(eval_df)
    serving_stats = tfdv.generate_statistics_from_dataframe(
        serving_df, stats_options=tfdv.StatsOptions(schema=schema, infer_type_from_schema=True)
    )

    print("Generated statistics for training and evaluation data.")
    custom_log("Generated statistics for training and evaluation data.")
    return train_stats, eval_stats, serving_stats

def detect_anomalies(statistics, schema):
    """Detect anomalies between dataset statistics and schema."""
    anomalies = tfdv.validate_statistics(statistics=statistics, schema=schema)
    tfdv.display_anomalies(anomalies=anomalies)

def get_image_dimensions(image_data):
    """Get dimensions of the image from its byte data."""
    image = Image.open(io.BytesIO(image_data))
    return image.size  # Returns (width, height)

def is_grayscale(image_data):
    """Check if the image is grayscale."""
    try:
        image = Image.open(io.BytesIO(image_data))
        return image.mode == 'L'  # Return True if grayscale
    except Exception:
        return False  # Return False if there's an error processing the image

def check_image_data(df):
    """Ensure images are grayscale and compatible with JPEG."""
    incompatible_indices = df.index[~df['image_data'].apply(is_grayscale)].tolist()

    if incompatible_indices:
        custom_log(f"Incompatible images found at indices: {incompatible_indices}",level=logging.ERROR)
        raise ValueError(f"Incompatible images found at indices: {incompatible_indices}")
    
    print("All images are in grayscale (JPEG-compatible) format.")


def check_missing_or_invalid_labels(df):
    """Check for missing or invalid labels. Each label should be an integer or a list of integers."""
    # Check for missing labels
    missing_labels = df['image_label'].isna().sum()
    invalid_labels = []

    for index, label in df['image_label'].items():
        try:
            # Convert string representation of list to actual list
            label = ast.literal_eval(label)
            
            # Check if the label is either an integer or a list of integers
            if not (isinstance(label, int) or (isinstance(label, list) and all(isinstance(i, int) for i in label))):
                invalid_labels.append(index)
        except (ValueError, SyntaxError):
            # Handle cases where the string cannot be converted
            invalid_labels.append(index)

    if missing_labels > 0:
        print(f"Found {missing_labels} missing labels.")
        custom_log(f"Found {missing_labels} missing labels.")
    if invalid_labels:
        print(f"Invalid labels found at indices: {invalid_labels}")
        custom_log(f"Invalid labels found at indices: {invalid_labels}")
    else:
        print("All labels are valid and correctly formatted as integers or lists of integers.")
        custom_log("All labels are valid and correctly formatted as integers or lists of integers.")



def check_class_distribution(df):
    """Check for class distribution imbalances."""
    class_counts = pd.Series([label for labels in df['image_label'] for label in labels]).value_counts()
    if class_counts.min() < 0.1 * class_counts.sum():
        print("Warning: Potential class imbalance detected.")
        custom_log("Warning: Potential class imbalance detected.",level = logging.WARNING)

def check_image_dimensions(df):
    """Check the dimensions of images and alert for high variation."""
    # Apply the function to the 'image_data' column
    dimensions = df['image_data'].apply(get_image_dimensions)

    # Count dimensions
    dimension_counts = Counter(dimensions)
    most_common_dim, most_common_count = dimension_counts.most_common(1)[0]
    
    # Calculate variance in dimensions
    unique_dims = len(dimension_counts)
    if unique_dims > 1:
        width_variation = np.std([dim[0] for dim in dimensions])
        height_variation = np.std([dim[1] for dim in dimensions])
        
        print(f"Most common dimension: {most_common_dim} with {most_common_count} images")
        print(f"Unique dimensions count: {unique_dims}")
        print(f"Width variation: {width_variation}, Height variation: {height_variation}")
        custom_log(f"Most common dimension: {most_common_dim} with {most_common_count} images")
        custom_log(f"Unique dimensions count: {unique_dims}")
        custom_log(f"Width variation: {width_variation}, Height variation: {height_variation}")
        
        # Alert if there is high variation
        if width_variation > 50 or height_variation > 50:  # Adjust thresholds as needed
            print("Alert: There is a high variation in image dimensions.")
            custom_log("Alert: There is a high variation in image dimensions.")
    else:
        print("All images have the same dimension:",most_common_dim)
        custom_log(f"All images have the same dimension: {most_common_dim}")


def anomalies_detect():
    """Main function to execute the next data processing task."""
    print("Starting next data processing task...")
    custom_log("Starting next data processing task...")

    # Load data and schema
    df = load_dataframe(INPUT_PICKLE_PATH)

    schema = load_schema(SCHEMA_PATH)

    # Prepare data splits
    train_df, eval_df, serving_df = prepare_data_splits(df)

    # Generate statistics and check anomalies
    train_stats, eval_stats, serving_stats = generate_statistics(train_df, eval_df, serving_df, schema)
    
    check_image_data(df)
    check_image_dimensions(df)
    check_missing_or_invalid_labels(df)
    check_class_distribution(df)
    
    print("Anomalies in Evaluation:")
    custom_log("Anomalies in Evaluation:")
    detect_anomalies(eval_stats, schema)
    print("Anomalies in serving set:")
    custom_log("Anomalies in serving set:")
    detect_anomalies(serving_stats, schema)

    

# if __name__ == "__main__":
#     anomalies_detect()
