# -*- coding: utf-8 -*-
"""model_selection.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/14OrEjodoJ46RoDh5wOPyequfOFw80a-1
"""

!pip install torchmetrics

import os
print(os.getcwd())  # Prints the current working directory
!ls /content

import io
import torch
import datetime
import logging
import importlib


import testing
importlib.reload(testing)
import retraining
importlib.reload(retraining)

from testing import main as test_main
from retraining import main as retrain_main
from retraining import CustomResNet18

print(testing.__file__)  # Prints the file path of the testing module

log_file_path = 'logs_model.log'
best_params_file = 'best_params.txt'
train_data_pickle = 'train_preprocessed_data.pkl'
test_data_pickle = 'test_preprocessed_data.pkl'
inference_data_pickle = 'test_preprocessed_data.pkl'
original_model_path= 'new_best_model.pt'
#retrained_model_path= 'retrained_best_model.pt'
combined_pickle= "combined_preprocessed_data.pkl"
final_model= ""

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

config = {
    "inference_data_pickle": inference_data_pickle,
    "test_data_pickle": test_data_pickle,
    "train_data_pickle": train_data_pickle,
    "combined_pickle": combined_pickle,
    "best_params_file": best_params_file,
    "test_batch_size": 64,
    "num_classes": 15,
    "num_demographics": 3,
    "train_percent": 0.8,
    "val_percent": 0.2,
}


if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"Using device: {device}")
  try:
    print("Testing the original model...")
    logging.info("Starting testing for the original model {original_model_path}.")
    original_accuracy = test_main(
        model_path=original_model_path,
        test_data_pickle=config["test_data_pickle"],
        batch_size=config["test_batch_size"],
        num_classes=config["num_classes"]
    )
    logging.info(f"Original model tested. Accuracy: {original_accuracy:.4f}")

    print("Retraining the model...")
    logging.info("Starting retraining process.")
    retrained_model_path = retrain_main(
        train_data_pickle=config["train_data_pickle"],
        inference_data_pickle=config["inference_data_pickle"],
        combined_pickle=config["combined_pickle"],
        best_params_file=config["best_params_file"],
        train_percent=config["train_percent"]
    )

    logging.info("Retraining completed successfully.")
    """torch.save(retrained_model, retrained_model_path)
    logging.info(f"Retrained model saved as {retrained_model_path}.")"""

    logging.info("Starting testing for the retrained model.")




    print("Testing the retrained model...")
    retrained_accuracy = test_main(
        model_path=retrained_model_path,
        test_data_pickle=config["test_data_pickle"],
        batch_size=config["test_batch_size"],
        num_classes=config["num_classes"]
    )

    logging.info(f"Retrained model tested. Accuracy: {retrained_accuracy:.4f}")

    if retrained_accuracy > original_accuracy:
      logging.info(
                f"Retrained model ({retrained_accuracy:.4f}) is better than the original model ({original_accuracy:.4f}). "
                f"Saving the retrained model as the original model."
            )
      print(f"Retrained model is better ({retrained_accuracy:.2f}% > {original_accuracy:.2f}%).")
      final_model = retrained_model_path
      #os.rename(retrained_model_path, original_model_path)

      timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

      logging.info(f"Retrained model saved as {retrained_model_path}.")

    else:
      logging.info(
          f"Original model ({original_accuracy:.4f}) is better or equal to the retrained model ({retrained_accuracy:.4f}). "
          f"Keeping the original model."
          )
      final_model = original_model_path
      print(f"Original model is better or equal ({original_accuracy:.2f}% >= {retrained_accuracy:.2f}%). Keeping it.")

  except Exception as e:
    print(f"An error occurred: {e}")