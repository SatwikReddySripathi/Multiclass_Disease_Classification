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

# PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__name__)))
#DVC_DIR = os.path.join(PROJECT_DIR)
def extract_md5_from_dvc(file_path):
    try:
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
            md5_hash = data.get("outs", [{}])[0].get("md5", "N/A")
            return md5_hash
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
def main():
    # Check if the DVC directory exists
    if not os.path.isdir(DVC_DIR):
        print(f"Directory '{DVC_DIR}' not found.")
        return
    
    # List all .dvc files in the DVC directory
    dvc_files = [f for f in os.listdir(PROJECT_DIR) if f.endswith(".dvc")]

    # Process each .dvc file
    for dvc_file in dvc_files:
        file_path = os.path.join(DVC_DIR, dvc_file)
        md5_hash = extract_md5_from_dvc(file_path)
        if md5_hash:
            print(f"MD5 hash for {dvc_file}: {md5_hash}")
        else:
            print(f"Failed to extract MD5 hash for {dvc_file}")

if __name__ == "__main__":
    main()


