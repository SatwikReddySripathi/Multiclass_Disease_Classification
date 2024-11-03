from google.cloud import storage
from PIL import Image
import io
import matplotlib.pyplot as plt
import os
from google.cloud import storage
import logging
from airflow.utils.log.logging_mixin import LoggingMixin
import sys

# sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

airflow_logger = LoggingMixin().log

# PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__name__))))
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
    for blob in blobs:
        print(blob.name)

bucket_name = 'nih-dataset-mlops'
list_blobs(bucket_name)

bucket_name = 'nih-dataset-mlops'
blob_name = 'files/md5/ff/f9357b1df5cc88a7b3c3a32082a041'
view_blob_content(bucket_name,blob_name)