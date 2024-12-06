# -*- coding: utf-8 -*-
"""retraining.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1Q4xBecrsgIYGIwbJecF_yEB2etcbWO6a
"""

import os
import io
import ast
import json
import torch
import pickle
import sklearn
import logging
import datetime
import itertools
import numpy as np
from PIL import Image
from google.colab import files

import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import random_split, DataLoader, TensorDataset

best_params_file= "best_params.txt"
train_preprocessed_data= "train_preprocessed_data.pkl"
test_preprocessed_data= "test_preprocessed_data.pkl"
combined_preprocessed_data= "combined_preprocessed_data.pkl"
best_model= "new_best_model.pt"

def combine_pickles(pickle_files):
    combined_data = {}

    for file in pickle_files:
        with open(file, "rb") as f:
            data = pickle.load(f)

        for key, value in data.items():
            combined_data[key] = value

    return combined_data

def load_data(original_data_pickle, batch_size, train_percent, target_size=(224, 224), seed= 42):

  torch.manual_seed(seed)
  np.random.seed(seed)

  images = []
  demographics = []
  labels= []

  resize_transform = transforms.Compose([
      transforms.Resize(target_size),
      transforms.ToTensor()
  ])

  with open(original_data_pickle, 'rb') as f:
      data = pickle.load(f)

  for item in data.values():

    """
    The image data we get would be in bytes. We need to open it and convert it to grey scale and then resize. Recheck it. What are we doing with resizing before then?
    """
    image_data = item['image_data']
    image = Image.open(io.BytesIO(image_data)).convert('L')
    image = resize_transform(image)  # Resizing and converting to tensor with shape (1, H, W) --> got an error without it

    label= item['image_label']
    label = ast.literal_eval(label)
    label = np.array(label, dtype=int)

    age = torch.tensor([item['age']], dtype=torch.float32)
    gender = torch.tensor(item['gender'], dtype=torch.float32)

    images.append(image)
    demographics.append(torch.cat([age, gender]))
    labels.append(label)

  """
  Stacking images and demographics.
  images Shape: (num_samples, channels, height, width)
  demographics Shape: (num_samples, num_features)
  """
  images = torch.stack(images)
  demographics = torch.stack(demographics)
  labels = torch.stack([torch.tensor(label, dtype=torch.long) for label in labels])
  #labels = torch.tensor(labels, dtype= torch.long)

  dataset = TensorDataset(images, demographics, labels)

  train_size = int(train_percent * len(dataset))
  #val_size = int(val_percent * len(dataset))
  val_size = len(dataset) - train_size  # this coz it would then add the remaining images, to the val dataset else we get an error

  print(f"Train size: {train_size}, Validation size: {val_size}, length of dataset: {len(dataset)}")

  train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

  print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

  return train_loader, val_loader

class CustomResNet18(nn.Module):
    def __init__(self, demographic_fc_size, num_demographics, num_classes=15):
        super(CustomResNet18, self).__init__()

        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Modifying the first convolutional layer to accept grayscale images (1 channel) --> generally ResNet expects 3 channels
        #for RGB
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Removing the final fully connected layer in ResNet
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # this fc processes the demographics (age + gender)
        self.demographics_fc = nn.Sequential(
            nn.Linear(num_demographics, demographic_fc_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc = nn.Linear(512 + demographic_fc_size, num_classes)  # 512 from ResNet(it's how resnet is), 32 from demographics_fc, can make it 64?

    def forward(self, images, demographics):
        x = self.resnet(images)  # Passing images through the modified ResNet (without its last layer)
        x = x.view(x.size(0), -1)  # Flattening the ResNet output

        demographics_features = self.demographics_fc(demographics)
        x = torch.cat((x, demographics_features), dim=1)

        #print("Shape after concatenating demographics:", x.shape)

        x = self.fc(x)
        #print("Output shape before returning:", x.shape)

        return x

def freeze_unfreeze_layers(model, freeze=True, layers_to_train=["layer4", "demographics_fc", "fc"]):
    for name, param in model.named_parameters():
        if any(layer in name for layer in layers_to_train):
            param.requires_grad = not freeze
        else:
            param.requires_grad = freeze


def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs= 10):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.train()

  best_val_accuracy = 0
  for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, demographics, labels in train_loader:
      inputs, demographics, labels = inputs.to(device), demographics.to(device), labels.to(device)

      # Repeating grayscale images to make them 3 channels - this is not needed now since I changed the ResNet to accept grayscale,
      #in-general ResNet expects RGB images ig

      #inputs = inputs.repeat(1, 3, 1, 1)

      optimizer.zero_grad()

      outputs = model(inputs, demographics)
      loss = criterion(outputs, labels.float())

      loss.backward()
      optimizer.step()

      running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)


    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
      for inputs, demographics, labels in val_loader:
        inputs, demographics, labels = inputs.to(device), demographics.to(device), labels.to(device)
        outputs = model(inputs, demographics)

        val_loss += criterion(outputs, labels.float()).item()
        probabilities = torch.sigmoid(outputs)
        predicted = (probabilities >= 0.5).int()

        correct += (predicted == labels).sum().item()
        total += labels.numel()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total

    if val_accuracy > best_val_accuracy:
      best_val_accuracy = val_accuracy

    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}, Validation Accuracy: {val_accuracy}%")
  return best_val_accuracy

def retrain_model(train_loader, val_loader, best_params):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = torch.load(best_model, map_location=device).to(device)
  optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])
  criterion = nn.BCEWithLogitsLoss()

  best_val_accuracy = train_model(train_loader, val_loader, model, criterion, optimizer, best_params['num_epochs'])
  print(f"Retrained validation accuracy: {best_val_accuracy}")
  return model

def main(train_data_pickle: str, inference_data_pickle: str, combined_pickle: str, best_params_file: str, train_percent:float):

  #combined_pickle = combine_pickles([config["original_pickle"], config["inference_pickle"]])
  combined_data= combine_pickles([train_data_pickle, inference_data_pickle])

  with open(combined_pickle, "wb") as f:
    pickle.dump(combined_data, f)

  print("Combined data saved to 'combined_preprocessed_data.pkl'")

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  with open(best_params_file, "r") as f:
      best_accuracy_line, params_line = f.readlines()
      best_params = params_line.replace("Parameters: ", "").strip()
      best_params = ast.literal_eval(best_params)
      num_epochs, batch_size, learning_rate, demographics_fc_size = best_params
      best_params = {
          "num_epochs": num_epochs,
          "batch_size": batch_size,
          "learning_rate": learning_rate,
          "demographics_fc_size": demographics_fc_size
      }

  print(f"Best Params: Epochs={num_epochs}, Batch Size={batch_size}, Learning Rate={learning_rate}, Demographics FC Size={demographics_fc_size}")

  train_loader, val_loader = load_data(
        original_data_pickle= combined_pickle,
        batch_size=best_params["batch_size"],
        train_percent=train_percent,
        target_size=(224, 224)
    )

  retrained_model = retrain_model(
        train_loader=train_loader,
        val_loader=val_loader,
        best_params=best_params
    )

  timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  model_filename = f"retrained_best_model_{timestamp}.pt"

  torch.save(retrained_model, model_filename)
  print(f"Retrained model saved as {model_filename}")

  return model_filename

