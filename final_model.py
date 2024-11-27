import os
import io
import ast
import pickle
import logging
import itertools
import numpy as np
from PIL import Image

import mlflow
import mlflow.pytorch

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import random_split, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MultilabelPrecision, MultilabelRecall, MultilabelF1Score

from google.cloud import storage

# Logging setup
log_file_path = 'logs_model.log'
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# TensorBoard writer setup
writer = SummaryWriter("runs/CustomResNet18_experiment")

# Load data from GCP bucket
def load_data_from_gcp(bucket_name, file_path, batch_size, train_percent, val_percent, target_size=(224, 224)):
    images = []
    demographics = []
    labels = []

    resize_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor()
    ])

    # Initialize GCP client and bucket
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file_path)

    # Read .pkl file from GCP
    with blob.open("rb") as f:
        data = pickle.load(f)

    for item in data.values():
        image_data = item['image_data']
        image = Image.open(io.BytesIO(image_data)).convert('L')
        image = resize_transform(image)

        label = item['image_label']
        label = ast.literal_eval(label)
        label = np.array(label, dtype=int)

        age = torch.tensor([item['age']], dtype=torch.float32)
        gender = torch.tensor(item['gender'], dtype=torch.float32)

        images.append(image)
        demographics.append(torch.cat([age, gender]))
        labels.append(label)

    images = torch.stack(images)
    demographics = torch.stack(demographics)
    labels = torch.stack([torch.tensor(label, dtype=torch.long) for label in labels])

    dataset = TensorDataset(images, demographics, labels)

    train_size = int(train_percent * len(dataset))
    val_size = int(val_percent * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

    return train_loader, val_loader, test_loader

# Custom ResNet18 Model
class CustomResNet18(nn.Module):
    def __init__(self, demographic_fc_size, num_demographics, num_classes=15):
        super(CustomResNet18, self).__init__()

        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        self.demographics_fc = nn.Sequential(
            nn.Linear(num_demographics, demographic_fc_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc = nn.Linear(512 + demographic_fc_size, num_classes)

    def forward(self, images, demographics):
        x = self.resnet(images)
        x = x.view(x.size(0), -1)
        demographics_features = self.demographics_fc(demographics)
        x = torch.cat((x, demographics_features), dim=1)
        x = self.fc(x)
        return x

# Training function
def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=10):
    model.train()
    best_val_accuracy = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, demographics, labels in train_loader:
            inputs, demographics, labels = inputs.to(device), demographics.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, demographics)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)

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
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Validation", val_accuracy, epoch)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}, Validation Accuracy: {val_accuracy}%")

    return best_val_accuracy

# Grid search function
def grid_search():
    param_grid = {
        "num_epochs": [5],
        "batch_size": [32, 64],
        "learning_rate": [1e-5, 1e-4],
        "demographics_fc_size": [64]
    }

    best_val_accuracy = 0
    best_params = None
    best_model = None
    all_combinations = list(itertools.product(*param_grid.values()))
    mlflow.set_experiment("Grid Search Experiment")

    for params in all_combinations:
        num_epochs, batch_size, learning_rate, demographic_fc_size = params

        with mlflow.start_run():
            mlflow.log_param("num_epochs", num_epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("demographic_fc_size", demographic_fc_size)

            train_loader, val_loader, test_loader = load_data_from_gcp(
                bucket_name="nih-dataset-mlops",
                file_path="Data_preproecssing_files/preprocessed_data.pkl",
                batch_size=batch_size,
                train_percent=config["train_percent"],
                val_percent=config["val_percent"]
            )

            model = CustomResNet18(demographic_fc_size, num_demographics=config["num_demographics"], num_classes=config["num_classes"]).to(device)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.BCEWithLogitsLoss()

            val_accuracy = train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs)
            mlflow.log_metric("val_accuracy", val_accuracy)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_params = params
                best_model = model

    if best_model is not None:
        output_dir = "/app/model_output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(best_model.state_dict(), os.path.join(output_dir, "best_model.pth"))
        print(f"Model saved at {output_dir}/best_model.pth with accuracy: {best_val_accuracy}%")

# Main script
if __name__ == "__main__":
    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = {
        "num_demographics": 3,
        "num_classes": 15,
        "train_percent": 0.7,
        "val_percent": 0.1
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Starting grid search...")
    grid_search()
