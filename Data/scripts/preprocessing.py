import os
import io
import cv2
import csv
import time
import json
import random
import pickle
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt

from PIL import Image
from PIL import ImageOps
from IPython.display import display

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator


log_directory = '/content/drive/My Drive/MLOPs Project'  
log_filename = 'logs.log'
log_file_path = os.path.join(log_directory, log_filename)


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Setting to DEBUG to capture all log messages or else it might not log info and error messages(got this error already)

file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

logger.info("Logging configuration is set. Logs will be saved to: {}".format(log_file_path))


def load_label_indices(json_path):
  try:
    logging.info(f"Attempting to load label indices from {json_path}")
    with open(json_path, 'r') as f:
      label_indices = json.load(f)
    logging.info(f"Successfully loaded label indices from {json_path}")
    return label_indices

  except FileNotFoundError:
    logging.error(f"File not found: {json_path}")
    return None

  except json.JSONDecodeError:
    logging.error(f"JSON decoding failed for file: {json_path}. Ensure it is in a valid JSON format.")
    return None

  except Exception as e:
    logging.error(f"Unexpected error occurred while loading label indices from {json_path}: {e}")
    return None
  

def preprocess_image(image):
  """
  Load an image, resize it, normalize pixel values, and apply CLAHE.
  Contrast-limited adaptive histogram equalization (CLAHE) is an image processing technique that improves contrast and reduces noise. This is a domain specific
  technique I guess? resizing and normalizing can be applied to everything. But for medical images it would be benificial to add CLAHE.
  """
  try:
    # Converting the PIL image to a NumPy array if needed
    if isinstance(image, Image.Image):
        image = np.array(image)

    # Ensuring image is in grayscale(to avoid problems with the gcp thingy)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    image = cv2.resize(image, (224, 224))

    if image.max() <= 1.0:
        image = (image * 255).astype('uint8')

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    image = image / 255.0

    return image

  except Exception as e:
    print(f"Error during image preprocessing: {e}")
    return None


def augment_generator():
  augmentation_generator = ImageDataGenerator(
      rotation_range=10,
      width_shift_range=0.1,
      height_shift_range=0.1,
      brightness_range=[0.8, 1.2],
      zoom_range=0.1
  )
  return augmentation_generator


def get_augmented_indices(class_data, total_images, target_percentage=0.05, undersampled_percent=0.05, random_percent=0.05, total_percent=0.10):
  """
  Identify indices for 5% of undersampled class images (hernia) and 5% random other images.
  """

  try:
    logging.info("Starting to calculate augmented indices.")
    class_counts = {item[0]: len(item[1]) for item in class_data.items()}
    undersampled_classes = {cls: count for cls, count in class_counts.items() if count < target_percentage * total_images}
    logging.info(f"Identified undersampled classes: {undersampled_classes}")

    if len(undersampled_classes) == 0:
      logging.info("No undersampled classes found; adjusting percentage values.")
      undersampled_percent = 0
      random_percent = total_percent

    undersampled_indices = [index for cls in undersampled_classes.keys() for index in class_data[cls]]
    logging.info(f"Collected {len(undersampled_indices)} undersampled indices.")

    non_undersampled_classes = set(class_data.keys()) - set(undersampled_classes.keys())
    random_indices = [index for cls in non_undersampled_classes for index in class_data[cls]]
    logging.info(f"Collected {len(random_indices)} random indices.")

    if len(random_indices) < int(random_percent * total_images):
      raise ValueError("Not enough non-undersampled images to sample from.")

    random_indices = random.sample(random_indices, k=int(random_percent * total_images))
    logging.info(f"Randomly sampled {len(random_indices)} indices.")

    result_indices= set(undersampled_indices[:int(undersampled_percent * total_images)] + random_indices)
    logging.info(f"Total augmented indices returned: {len(result_indices)}")

    return result_indices

  except KeyError as e:
    logging.error(f"KeyError: {e}. Please check your class_data for correct keys.")
    return set()
  except ValueError as e:
    logging.error(f"ValueError: {e}. Check the sample fraction or indices. {e}")
    return set()
  except Exception as e:
    logging.error(f"An error occurred: {e}")
    return set()
  

def apply_augmentation(image, augmentation_generator):
  """
  Apply augmentation and return augmented images with basic error handling.
  """
  augmented_images = []

  try:
    image = image.astype(np.float32)
    image = image.reshape((1, image.shape[0], image.shape[1], 1))  # Reshaping for the generator.  Here the height and width remains the same but it makes it suitable for the generator.
    aug_iter = augmentation_generator.flow(image, batch_size=1)

    for aug_num in range(2):  # Saving two augmentations per selected image
      try:
        aug_image = next(aug_iter)[0].squeeze() #It applies augmentation randomly and we could access it doing next(). No two augments are same.
        augmented_images.append(aug_image)
        logging.info(f"Augmented image {aug_num + 1} generated successfully.")

      except Exception as e:
        logging.error(f"Error during augmentation iteration {aug_num + 1}: {e}")
        continue  # Skipping to the next iteration if one fails, no cap

  except Exception as e:
    logging.error(f"An error occurred during the augmentation process: {e}")

  if not augmented_images:
    logging.warning("No augmented images were generated.")

  return augmented_images


def get_demographic_info(original_data_pickle, stats_file='demographics_stats_for_dummy.pkl'):
    """
    Reads demographic data from a pickle file, calculates mean, std for age,
    and creates a one-hot encoder for gender. Saves these stats to a file
    if they change, and returns the scaler and encoder.
    """

    """if os.path.exists(stats_file):
      with open(stats_file, 'rb') as f:
        saved_stats = pickle.load(f)
        print("Loaded demographics stats from file.")
        return saved_stats['age_scaler'], saved_stats['gender_encoder']"""


    ages= []
    genders= []

    with open(original_data_pickle, 'rb') as f:
        data = pickle.load(f)

    for item in data.values():
        ages.append(item['age'])
        genders.append(item['gender'])

    ages = np.array(ages)
    genders = np.array(genders).reshape(-1, 1)  # Gender should be a 2D array for OneHotEncoder

    age_scaler = StandardScaler().fit(ages.reshape(-1, 1))
    gender_encoder = OneHotEncoder(sparse_output=False).fit(genders)

    logging.info("Calculated and saved new demographics stats.")

    with open(stats_file, 'wb') as f:
        pickle.dump({'age_scaler': age_scaler, 'gender_encoder': gender_encoder}, f)

    """Since I'm dumping it to a pickle file, the states of the scaler and enconder are saved as such.
       I can load the pickle file and say age_scaler.transform(new_age) and it'll transform.
    """

    logging.info("Saved new demographics stats.")
    files.download(stats_file)

    return age_scaler, gender_encoder


def process_images(original_data_pickle, preprocessed_data_pickle, label_json_path, is_training=False):
  """
  Processes images by preprocessing and optionally augmenting them,
  then saves all processed images into a new pickle file.
  """
  try:
    logging.info("Starting image processing.")

    global augmentation_generator
    augmentation_generator = augment_generator()

    logging.info("Augmentation generator created.")

    with open(original_data_pickle, 'rb') as f:
      image_data = pickle.load(f)
    logging.info(f"Loaded data from {original_data_pickle} with {len(image_data)} entries.")
    age_scaler, gender_encoder = get_demographic_info(original_data_pickle)

    if is_training:
      logging.info("Loading augmented indices for training mode.")
      with open(label_json_path, 'r') as json_file:
        label_to_indices = json.load(json_file)
      augmented_indices = get_augmented_indices(label_to_indices, len(image_data))
      logging.info(f"Loaded {len(augmented_indices)} augmented indices.")

    processed_images = {}

    for image_index, image_info in tqdm(image_data.items(), desc="Processing images"):
      image_label = image_info['image_label']
      image_bytes = image_info['image_data']
      gender_raw= image_info['gender']
      age_raw= image_info['age']

      age = age_scaler.transform([[age_raw]])[0][0]
      gender= gender_encoder.transform([[gender_raw]])[0] #one-hot encoding of gender

      image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

      if is_training and int(image_index) in augmented_indices:
        augmented_images = apply_augmentation(image, augmentation_generator)
        for aug_num, aug_image in enumerate(augmented_images):
          aug_preprocessed_image = preprocess_image(aug_image)

          if aug_preprocessed_image.max() <= 1:
            aug_preprocessed_image = (aug_preprocessed_image * 255).astype(np.uint8)

          if isinstance(aug_preprocessed_image, np.ndarray):
            aug_preprocessed_image = Image.fromarray(aug_preprocessed_image).convert('L')

          aug_image_bytes = io.BytesIO()
          aug_preprocessed_image.save(aug_image_bytes, format='JPEG')
          processed_images[f'aug_{aug_num}_{image_index}'] = {
              'image_data': aug_image_bytes.getvalue(),
              'image_label': image_label,
              'gender': gender,
              'age': age
          }

      preprocessed_image = preprocess_image(image)

      if preprocessed_image.max() <= 1:
        preprocessed_image = (preprocessed_image * 255).astype(np.uint8)

      if isinstance(preprocessed_image, np.ndarray):
        preprocessed_image = Image.fromarray(preprocessed_image).convert('L')

      preprocessed_image_bytes = io.BytesIO()
      preprocessed_image.save(preprocessed_image_bytes, format='JPEG')
      processed_images[image_index] = {
          'image_data': preprocessed_image_bytes.getvalue(),
          'image_label': image_label,
          'gender': gender,
          'age': age
      }

    logging.info(f"Saving all processed images to {preprocessed_data_pickle}")

    with open(preprocessed_data_pickle, 'wb') as f:
      pickle.dump(processed_images, f)
      logging.info(f"All processed images saved to {preprocessed_data_pickle}")

    files.download(preprocessed_data_pickle)

  except Exception as e:
    logging.error(f"An error occurred during image processing: {e}")

  finally:
    logging.info("Image processing completed.")


"""
original_data_pickle= "raw_compressed_data.pkl"
preprocessed_data_pickle= "preprocessed_compressed_data.pkl"
csv_path = '/content/drive/My Drive/MLOPs Project/sampled_train_data_entry.csv'
label_json_path = '/content/drive/My Drive/MLOPs Project/label_to_indices.json'

process_images(original_data_pickle, preprocessed_data_pickle, label_json_path, is_training=False)

"""