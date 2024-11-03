import os
import cv2
import json
import random
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
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
  

def preprocess_image(image_path):
  """
  Load an image, resize it, normalize pixel values, and apply CLAHE.
  Contrast-limited adaptive histogram equalization (CLAHE) is an image processing technique that improves contrast and reduces noise. This is a domain specific
  technique I guess? resizing and normalizing can be applied to everything. But for medical images it would be benificial to add CLAHE.
  """
  try:
    logging.info(f"Starting to preprocess image from path: {image_path}")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
      raise ValueError(f"Image at path {image_path} could not be loaded. Ensure the path is correct and the file exists.")
    
    logging.info(f"Image loaded from path: {image_path}")

    image = cv2.resize(image, (224, 224))
    image = image / 255.0  # Normalizing pixel values to [0, 1]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply((image * 255).astype(np.uint8)) / 255.0
    return image

  except cv2.error as e:
    logging.error(f"OpenCV error while processing image {image_path}: {e}")
    return None

  except ValueError as e:
    logging.error(f"Value error while processing image {image_path}: {e}")
    return None

  except Exception as e:
    logging.error(f"Unexpected error processing image {image_path}: {e}")
    return None
  

def save_image(image, save_path, image_id):
  try:
    logging.info(f"Attempting to save image {image_id} to {save_path}")
    os.makedirs(save_path, exist_ok=True)
    cv2.imwrite(os.path.join(save_path, image_id), (image * 255).astype(np.uint8))
    logging.info(f"Successfully saved image {image_id} to {save_path}")

  except Exception as e:
    logging.error(f"Error saving image {image_id} to path {save_path}: {e}")


def display_image(image):
  try:
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()
  except Exception as e:
    print(f"Error displaying image: {e}")


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

def process_images(original_data_folder, preprocessed_data_folder, csv_path, label_json_path, is_training=False):
  """
  Processes images by preprocessing and optionally augmenting them. Saves preprocessed and augmented images.
  """
  try:
    logging.info("Starting image processing.")
    global augmentation_generator
    augmentation_generator = augment_generator()

    df = pd.read_csv(csv_path)
    logging.info(f"Loaded CSV data from {csv_path} with {len(df)} entries.")

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preprocessing images"):
      image_id = row['Image Index']
      image_path = os.path.join(original_data_folder, image_id)

      image = preprocess_image(image_path)
      if image is not None:
        save_image(image, preprocessed_data_folder, image_id)
        logging.info(f"Successfully preprocessed and saved image: {image_id}")
      else:
        logging.warning(f"Skipping image {image_id} due to preprocessing issues.")

    if is_training:
      logging.info("Starting augmentation process.")
      label_to_indices = load_label_indices(label_json_path)

      if not label_to_indices:
        logging.error(f"Label indices could not be loaded from {label_json_path}. Aborting augmentation.")
        return

      augmented_indices = get_augmented_indices(label_to_indices, len(df))
      logging.info(f"Generated {len(augmented_indices)} augmented indices.")

      for idx in tqdm(augmented_indices, desc="Augmenting images"):
        image_id = df.iloc[idx]['Image Index']
        image_path = os.path.join(original_data_folder, image_id)
        raw_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if raw_image is not None:
          augmented_images = apply_augmentation(raw_image, augmentation_generator)
          for aug_num, aug_image in enumerate(augmented_images):
            aug_preprocessed_image = preprocess_image(aug_image)
            save_image(aug_preprocessed_image, preprocessed_data_folder, f'aug_{aug_num}_{image_id}')

        else:
          logging.warning(f"Skipping raw image {image_id} due to loading issues.")

    logging.info("Data preprocessing and augmentation complete.")

  except Exception as e:
    logging.error(f"An unexpected error occurred during image processing: {e}")
  
  finally:
    logging.info("Image processing completed.")
  
  print("Data preprocessing and augmentation complete.")


"""
original_data_folder = '/content/drive/My Drive/MLOPs Project/sampled_data'
preprocessed_data_folder = '/content/drive/My Drive/MLOPs Project/preprocessed_data'
csv_path = '/content/drive/My Drive/MLOPs Project/sampled_train_data_entry.csv'
label_json_path = '/content/drive/My Drive/MLOPs Project/label_indices.json'

process_images(original_data_folder, preprocessed_data_folder, csv_path, label_json_path)

"""