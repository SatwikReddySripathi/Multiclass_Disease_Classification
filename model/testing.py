# -*- coding: utf-8 -*-
"""testing.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1RbGATcZb5UHRrLXo-9J-U66AldfG89fV
"""

!pip install torchmetrics

import io
import ast
import torch
import pickle
import numpy as np
from PIL import Image

import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision import transforms
from torch.utils.data import random_split, DataLoader, TensorDataset

from torchmetrics.classification import MultilabelPrecision, MultilabelRecall, MultilabelF1Score

test_data_pickle = "preprocessed_dummy_data.pkl"
model_path = "final_model.pth"
batch_size = 32

def load_test_data(original_data_pickle, batch_size, target_size=(224, 224)):

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

    #considering test preprocessing would come from the actual preprocessing pipeline, I'm not doing the age and gender transformation here
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

  test_dataset = TensorDataset(images, demographics, labels)
  test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

  print(f" samples: {len(test_dataset)}")

  return test_loader

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

def evaluate_model(test_loader, model, criterion, precision_metric, recall_metric, f1_metric, confidence= 0.3):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0

    precision_metric.reset()
    recall_metric.reset()
    f1_metric.reset()

    with torch.no_grad():
      for inputs, demographics, labels in test_loader:
        inputs, demographics, labels = inputs.to(device), demographics.to(device), labels.to(device)
        outputs = model(inputs, demographics)

        test_loss += criterion(outputs, labels.float()).item()

        probabilities = torch.sigmoid(outputs)
        predicted = (probabilities >= confidence).int()

        correct += (predicted == labels).sum().item()
        total += labels.numel()

        #print("predicted:", predicted)
        #print("labels: ", labels)

        precision_metric.update(predicted, labels)
        recall_metric.update(predicted, labels)
        f1_metric.update(predicted, labels)

    test_accuracy = 100 * correct / total
    precision = precision_metric.compute().item()
    recall = recall_metric.compute().item()
    f1_score = f1_metric.compute().item()
    avg_test_loss = test_loss / len(test_loader)


    print(f'Test Loss: {avg_test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    print(f'Test Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}')

    return test_accuracy, precision, recall, f1_score

if __name__ == "__main__":

  config = {
    "file_path": "preprocessed_data_new.pkl",
    "num_demographics": 3,
    "num_classes": 15,
    "train_percent": 0.7,
    "val_percent": 0.1
  }

  demographics_fc_size = 64

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  print("Loading the best model for evaluation...")

  model = CustomResNet18(demographics_fc_size,
                           num_demographics=config["num_demographics"],
                           num_classes=config["num_classes"])
  model.load_state_dict(torch.load(model_path))
  model.to(device)
  model.eval()

  test_loader = load_test_data(test_data_pickle, batch_size)

  precision_metric = MultilabelPrecision(num_labels= config["num_classes"], average='macro').to(device)
  recall_metric = MultilabelRecall(num_labels= config["num_classes"], average='macro').to(device)
  f1_metric = MultilabelF1Score(num_labels= config["num_classes"], average='macro').to(device)

  criterion = nn.BCEWithLogitsLoss()
  test_accuracy, precision, recall, f1_score= evaluate_model(test_loader, model, criterion, precision_metric, recall_metric, f1_metric)

  print(f"Test Accuracy of the best model: {test_accuracy:.4f}")