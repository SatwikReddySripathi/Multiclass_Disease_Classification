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
        else:
            print(f"No MD5 hash found in {os.path.basename(dvc_file)} or failed to extract.")

    return md5_keys

# Set the project directory (replace this with your actual project directory path)
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__name__))))
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'Processed_Data', 'raw_compressed_data.pkl')

# Run the function to get MD5 hashes
md5_hashes = find_md5_hashes(PROJECT_DIR)

# Print all extracted MD5 hashes
print("Extracted MD5 hashes:", md5_hashes)


def get_file_contents_as_dict(bucket, md5_keys):
    json_content_dict = {}
    csv_data = {}
    
    for blob in bucket.list_blobs():
        blob_name = blob.name.split("/")[-1]
        for md5_key in md5_keys:
            if md5_key[2:] == blob_name:
                print(f'Reading content from {blob.name}...')
                content = blob.download_as_text()  # Read the blob content as text
                
                # Check the file type based on the extension
                if blob.name.endswith(".dir"):
                    # Parse JSON content
                    try:
                        json_content_dict[md5_key] = json.loads(content)
                        print(f"JSON content loaded for {blob.name}.")
                    except json.JSONDecodeError:
                        print(f"Error decoding JSON content in {blob.name}.")
                else:
                    # Parse CSV-like content
                    print(f"Parsing CSV content for {blob.name}...")
                    csv_reader = csv.DictReader(StringIO(content))
                    csv_data = {row['Image Index']: row['Labels'] for row in csv_reader}
                    print(f"CSV content loaded for {blob.name}.")
    
    return json_content_dict, csv_data

def create_final_json(json_content_dict, csv_content_dict):
    final_data = []
    
    for md5_key, json_data in json_content_dict.items():
        for item in json_data:
            relpath = item['relpath']
            
            # Check if there's a matching label in the CSV content
            if relpath in csv_content_dict.keys():
                image_label = csv_content_dict[relpath]
                
                # Create the final JSON structure for each matched entry
                final_data.append({
                    'md5': item['md5'],
                    'image_index': relpath,
                    'image_label': image_label
                })
                
    return final_data

# Run the function and store the content in variables
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "black-resource-440218-c3-5c0b7f31ce1f.json"
storage_client = storage.Client(project = 'black-resource-440218-c3')
bucket_name = 'nih-dataset-mlops'
bucket = storage_client.get_bucket(bucket_name)
json_content_dict, csv_content_dict = get_file_contents_as_dict(bucket, md5_hashes)
final_json = create_final_json(json_content_dict, csv_content_dict)


# Display the final JSON structure
print("Final JSON structure:")
print(json.dumps(final_json, indent=2))


def download_and_compress_images(md5_image_data, output_pickle_file):
    compressed_images = {}

    for item in md5_image_data:
        md5 = item["md5"]
        image_index = item["image_index"]

        # Attempt to download the image from GCP bucket
        blob = bucket.blob(f'files/md5/{md5[:2]}/{md5[2:]}')
        
        try:
            # Download image as bytes
            image_bytes = blob.download_as_bytes()
            
            # Open image using PIL
            image = Image.open(io.BytesIO(image_bytes))

            # Convert RGBA to RGB if necessary
            if image.mode == 'RGBA':
                image = image.convert('RGB')
                print(f"Converted {image_index} from RGBA to RGB for JPEG compatibility.")
            
            # Compress the image
            compressed_image = io.BytesIO()
            image.save(compressed_image, format="JPEG", quality=50)  # Adjust quality as needed
            compressed_image.seek(0)  # Rewind the buffer
            
            # Store compressed image in dictionary
            compressed_images[image_index] = compressed_image.getvalue()
            print(f"Compressed and stored image: {image_index}")

        except Exception as e:
            print(f"Failed to download or compress image {image_index} with MD5 {md5}: {e}")

    # Save all compressed images in a pickle file
    with open(output_pickle_file, 'wb') as f:
        pickle.dump(compressed_images, f)
        print(f"All compressed images saved to {output_pickle_file}")

# Load the JSON data containing MD5 and image index info
# Assuming this is your JSON data loaded from the previous step
output_pickle_file = OUTPUT_DIR

download_and_compress_images(final_json, output_pickle_file)

