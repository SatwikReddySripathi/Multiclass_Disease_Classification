# -*- coding: utf-8 -*-
"""inference.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/16QZYuqdFMMCzU1vKE3rI-JL6rIqQA_XH
"""

import io
import pickle
from PIL import Image
import torch
from torchvision import transforms
from google.colab import files
import numpy as np
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

pickle_file_path = 'demographics_stats_for_dummy.pkl'
final_model_path = 'new_best_model.pt'

one_hot_encoding = {
    'No Finding': 0,
    'Atelectasis': 1,
    'Cardiomegaly': 2,
    'Effusion': 3,
    'Infiltration': 4,
    'Mass': 5,
    'Nodule': 6,
    'Pneumonia': 7,
    'Pneumothorax': 8,
    'Consolidation': 9,
    'Edema': 10,
    'Emphysema': 11,
    'Fibrosis': 12,
    'Pleural_Thickening': 13,
    'Hernia': 14
}

config = {
    "file_path": "preprocessed_data_new.pkl",
    "num_demographics": 3,
    "num_classes": 15,
    "train_percent": 0.7,
    "val_percent": 0.1
  }

demographics_fc_size= 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def load_model_and_preprocessors(pickle_path, model_path, device):

  with open(pickle_path, 'rb') as f:
      data = pickle.load(f)

  age_scaler = data['age_scaler']
  gender_encoder = data['gender_encoder']

  model = CustomResNet18(demographics_fc_size,
                          num_demographics=config["num_demographics"],
                          num_classes=config["num_classes"])
  model = torch.load(model_path, map_location=device)
  model.to(device)
  model.eval()

  return age_scaler, gender_encoder, model

def preprocess_inputs(images_data, ages, genders, age_scaler, gender_encoder, target_size=(224, 224)):

    resize_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])

    image_tensors, demographic_features = [], []
    for image_data, age, gender in zip(images_data, ages, genders):
        image = Image.open(io.BytesIO(image_data)).convert('L')
        image = resize_transform(image)
        image_tensors.append(image)

        age_scaled = age_scaler.transform([[age]])[0][0]
        gender_encoded = gender_encoder.transform([[gender]])[0]
        demographic_features.append(np.concatenate([[age_scaled], gender_encoded]))

    return torch.stack(image_tensors), torch.tensor(demographic_features, dtype=torch.float32)

def create_test_loader(image_tensors, demographic_tensors, batch_size=4):
    test_dataset = TensorDataset(image_tensors, demographic_tensors)
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def evaluate_model(test_loader, model, device, confidence=0.3):
    model.eval()
    predictions, probabilities_list = [], []

    with torch.no_grad():
        for inputs, demographics in test_loader:
            inputs, demographics = inputs.to(device), demographics.to(device)
            outputs = model(inputs, demographics)

            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities >= confidence).int()

            probabilities_list.append(probabilities.cpu())
            predictions.append(predicted.cpu())

    predictions = torch.cat(predictions, dim=0)
    probabilities_list = torch.cat(probabilities_list, dim=0)
    confidence_scores = probabilities_list.numpy().tolist()

    return predictions, probabilities_list, confidence_scores

def decode_predictions(predictions, probabilities, one_hot_encoding, threshold= 0.3):
    index_to_label = {v: k for k, v in one_hot_encoding.items()}
    predictions_list = predictions.cpu().numpy().tolist()
    probabilities_list = probabilities.cpu().numpy().tolist()

    predicted_labels_with_confidences = []
    for sample_pred, sample_prob in zip(predictions_list, probabilities_list):
        predicted_indices = [i for i, value in enumerate(sample_pred) if value == 1]
        labels_with_confidences = [
            (index_to_label[idx], sample_prob[idx]) for idx in predicted_indices if sample_prob[idx] >= threshold
        ]
        predicted_labels_with_confidences.append(labels_with_confidences)

    return predicted_labels_with_confidences

def run_pipeline(pickle_file_path, model_path, uploaded_files):

    age_scaler, gender_encoder, model = load_model_and_preprocessors(pickle_file_path, model_path, device)

    ages = [float(input(f"Enter the Age for {file}: ")) for file in uploaded_files.keys()]
    genders = [input(f"Enter the Gender (Male/Female) for {file}: ") for file in uploaded_files.keys()]

    images_data = [uploaded_files[file] for file in uploaded_files.keys()]
    image_tensors, demographic_tensors = preprocess_inputs(images_data, ages, genders, age_scaler, gender_encoder)

    test_loader = create_test_loader(image_tensors, demographic_tensors)

    predictions, probabilities, confidence_scores = evaluate_model(test_loader, model, device)

    predicted_labels_with_confidences = decode_predictions(predictions, probabilities, one_hot_encoding)

    for i, labels_with_confidences in enumerate(predicted_labels_with_confidences):
        if labels_with_confidences:
            print(f"Sample {i + 1}: Predicted Diseases:")
            for label, confidence in labels_with_confidences:
                print(f"  - {label} (Confidence: {confidence:.2f})")
        else:
            print(f"Sample {i + 1}: No Findings")

uploaded_files = files.upload()
run_pipeline(pickle_file_path, final_model_path, uploaded_files)