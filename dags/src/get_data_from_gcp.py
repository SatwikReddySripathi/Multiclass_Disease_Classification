import os
import yaml
import glob
import io
from io import StringIO
import csv
import json
from google.cloud import storage
import pickle
from PIL import Image


def extract_md5_from_dvc(file_path):
    """Extract the MD5 hash from a .dvc file."""
    try:
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
            # Extract MD5 hash from 'outs' if available
            return data.get("outs", [{}])[0].get("md5")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        #custom_log(f"Error reading {file_path}: {e}",level=logging.ERROR)
        return None

def find_md5_hashes(project_dir):
    """Find and extract MD5 hashes from all .dvc files in the project directory."""
    # Find all .dvc files in the project directory
    dvc_files = glob.glob(os.path.join(project_dir, "*.dvc"))
    
    md5_keys = []
    for dvc_file in dvc_files:
        md5_hash = extract_md5_from_dvc(dvc_file)
        if md5_hash:
            md5_keys.append(md5_hash)
            print(f"MD5 hash for {os.path.basename(dvc_file)}: {md5_hash}")
            #custom_log(f"MD5 hash for {os.path.basename(dvc_file)}: {md5_hash}")
        else:
            print(f"No MD5 hash found in {os.path.basename(dvc_file)} or failed to extract.")
            #custom_log(f"No MD5 hash found in {os.path.basename(dvc_file)} or failed to extract.",level=logging.ERROR)

    return md5_keys


def get_file_contents_as_dict(bucket, md5_keys):
    json_content_dict = {}
    csv_data = {}
    
    for blob in bucket.list_blobs():
        blob_name = blob.name.split("/")[-1]
        for md5_key in md5_keys:
            if md5_key[2:] == blob_name:
                print(f'Reading content from {blob.name}...')
                #custom_log(f'Reading content from {blob.name}...')
                content = blob.download_as_text()  # Read the blob content as text
                
                # Check the file type based on the extension
                if blob.name.endswith(".dir"):
                    # Parse JSON content
                    try:
                        json_content_dict[md5_key] = json.loads(content)
                        print(f"JSON content loaded for {blob.name}.")
                        #custom_log(f"JSON content loaded for {blob.name}.")
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON content in {blob.name}.")
                        #custom_log(f"Error decoding JSON content in {blob.name}.",level=logging.ERROR)
                else:
                    # Parse CSV-like content
                    print(f"Parsing CSV content for {blob.name}...")
                    #custom_log(f"Parsing CSV content for {blob.name}...")
                    csv_reader = csv.DictReader(StringIO(content))
                    csv_data = {row['Image Index']: [row['Labels'],row['Age'],row['Gender']] for row in csv_reader}
                    print(f"CSV content loaded for {blob.name}.")
                    #custom_log(f"CSV content loaded for {blob.name}.")
    
    return json_content_dict, csv_data

def create_final_json(json_content_dict, csv_content_dict):
    final_data = []
    
    for md5_key, json_data in json_content_dict.items():
        for item in json_data:
            relpath = item['relpath']
            
            # Check if there's a matching label in the CSV content
            if relpath in csv_content_dict.keys():
                image_label = csv_content_dict[relpath][0]
                Age = csv_content_dict[relpath][1]
                Gender = csv_content_dict[relpath][2]
                
                # Create the final JSON structure for each matched entry
                final_data.append({
                    'md5': item['md5'],
                    'image_index': relpath,
                    'image_label': image_label,
                    'Age': Age,
                    'Gender': Gender
                })
                
    return final_data


def download_and_compress_images(bucket, md5_image_data, output_pickle_file):
    compressed_images = {}

    for item in md5_image_data:
        md5 = item["md5"]
        image_index = item["image_index"]
        image_label = item["image_label"]
        image_Age = item["Age"]
        image_Gender = item["Gender"]

        # Attempt to download the image from GCP bucket
        blob = bucket.blob(f'files/md5/{md5[:2]}/{md5[2:]}')
        
        try:
            # Download image as bytes
            image_bytes = blob.download_as_bytes()
            
            # Open image using PIL
            image = Image.open(io.BytesIO(image_bytes))

            # Converting RGBA to RGB if necessary
            if image.mode in ['RGB','RGBA']:
                image = image.convert('L')
                print(f"Converted {image_index}  to L for JPEG compatibility.")
                #custom_log(f"Converted {image_index}  to L for JPEG compatibility.")
            
            # Compress the image
            compressed_image = io.BytesIO()
            image.save(compressed_image, format="JPEG", quality=50)  # Adjust quality as needed
            compressed_image.seek(0)  # Rewind the buffer
            
            # Store compressed image in dictionary
            compressed_images[image_index] = {'image_data': compressed_image.getvalue(), 'image_label': image_label,'Age': image_Age,'Gender': image_Gender}
            print(f"Compressed and stored image: {image_index}")
            #custom_log(f"Compressed and stored image: {image_index}")

        except Exception as e:
            print(f"Failed to download or compress image {image_index} with MD5 {md5}: {e}")
            #custom_log(f"Failed to download or compress image {image_index} with MD5 {md5}: {e}",level=logging.ERROR)

    # Save all compressed images in a pickle file
    with open(output_pickle_file, 'wb') as f:
        pickle.dump(compressed_images, f)
        print(f"All compressed images saved to {output_pickle_file}")
        #custom_log(f"All compressed images saved to {output_pickle_file}")






