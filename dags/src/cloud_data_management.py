from google.cloud import storage
from PIL import Image
import io
import matplotlib.pyplot as plt
import os
from google.cloud import storage
import logging
import json
# from airflow.utils.log.logging_mixin import LoggingMixin
import sys
sys.path.append('/opt/airflow/src')
from src.get_data_from_gcp import find_md5_hashes, get_file_contents_as_dict, create_final_json, download_and_compress_images

# sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

# airflow_logger = LoggingMixin().log

# import os

def print_directory_contents(path):
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_DIR2 = os.path.join(PROJECT_DIR, 'dags', 'src','data')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'Processed_Data', 'raw_compressed_data.pkl')
KEY_PATH = os.path.join(PROJECT_DIR, "config", "black-resource-440218-c3-5c0b7f31ce1f.json")

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = KEY_PATH

def view_blob_content(bucket_name,blob_name):
    storage_client = storage.Client(project = 'black-resource-440218-c3')
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    image_data = blob.download_as_bytes()
    image = Image.open(io.BytesIO(image_data))
    plt.imshow(image)
    plt.axis('off')


def list_blobs(bucket_name):
    storage_client = storage.Client()
    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name)
    # Note: The call returns a response only when the iterator is consumed.
    i = 0
    for blob in blobs:
        print(blob.name)
        i= i+1
    print(i)


def extracting_data_from_gcp(bucket_name):

    md5_hashes = find_md5_hashes(PROJECT_DIR2)

    # Print all extracted MD5 hashes
    print("Extracted MD5 hashes:", md5_hashes)    
    # find_md5_hashes()

    storage_client = storage.Client(project = 'black-resource-440218-c3')
    bucket_name = 'nih-dataset-mlops'
    bucket = storage_client.get_bucket(bucket_name)
    json_content_dict, csv_content_dict = get_file_contents_as_dict(bucket, md5_hashes)
    final_json = create_final_json(json_content_dict, csv_content_dict)


    # Display the final JSON structure
    print("Final JSON structure:")
    print(json.dumps(final_json, indent=2))

    # Load the JSON data containing MD5 and image index info
    # Assuming this is your JSON data loaded from the previous step
    output_pickle_file = OUTPUT_DIR

    download_and_compress_images(bucket, final_json, output_pickle_file)

def upload_data_to_gcp():

    # Initialize a client
    storage_client = storage.Client(project = 'black-resource-440218-c3')
    bucket_name = 'nih-dataset-mlops'
    file_path = 'Processed_Data'
    blob_folder_name = 'Data_Preprocessing_files'
    blob_name1 = 'train_preprocessed_data.pkl'
    blob_name2 = 'test_data.pkl'

    # Get the bucket
    bucket = storage_client.get_bucket(bucket_name)
    # Create a blob object
    blob1 = bucket.blob(blob_folder_name +"/"+blob_name1)
    # Upload the file
    blob1.upload_from_filename(os.path.join(PROJECT_DIR, file_path, blob_name1))

    print(f"File {blob_name1} uploaded to {file_path}")

    blob2 = bucket.blob(blob_folder_name +"/"+blob_name2)
    # Upload the file
    blob2.upload_from_filename(os.path.join(PROJECT_DIR, file_path, blob_name2))

    print(f"File {blob_name2} uploaded to {file_path}")
