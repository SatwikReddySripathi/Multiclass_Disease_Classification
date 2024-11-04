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
from get_data_from_gcp import find_md5_hashes, get_file_contents_as_dict, create_final_json, download_and_compress_images

# sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

# airflow_logger = LoggingMixin().log

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'Processed_Data', 'raw_compressed_data.pkl')
# PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__name__))))
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


# bucket_name = 'nih-dataset-mlops'
# list_blobs(bucket_name)

# # bucket_name = 'nih-dataset-mlops'
# # blob_name = 'files/md5/ff/f9357b1df5cc88a7b3c3a32082a041'
# view_blob_content(bucket_name,blob_name)

def extracting_data_from_gcp(bucket_name):

    # Run the function to get MD5 hashes
    md5_hashes = find_md5_hashes(PROJECT_DIR)

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