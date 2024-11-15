import io
import ast
import torch
import pickle
import numpy as np
from PIL import Image

import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import random_split, DataLoader, TensorDataset

from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MultilabelPrecision, MultilabelRecall, MultilabelF1Score
def load_data(original_data_pickle, batch_size, train_percent, val_percent, target_size=(224, 224)):
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
    val_size = int(val_percent * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

    return train_loader, val_loader, test_loader
class CustomResNet18(nn.Module):
    def __init__(self, num_demographics, num_classes=15):
        super(CustomResNet18, self).__init__()

        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Modifying the first convolutional layer to accept grayscale images (1 channel) --> generally ResNet expects 3 channels
        #for RGB
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        # Removing the final fully connected layer in ResNet
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # this fc processes the demographics (age + gender)
        self.demographics_fc = nn.Sequential(
            nn.Linear(num_demographics, 32),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc = nn.Linear(512 + 32, num_classes)  # 512 from ResNet, 32 from demographics_fc, can make it 64?

    def forward(self, images, demographics):
        x = self.resnet(images)  # Passing images through the modified ResNet (without its last layer)
        x = x.view(x.size(0), -1)  # Flattening the ResNet output

        demographics_features = self.demographics_fc(demographics)
        x = torch.cat((x, demographics_features), dim=1)

        #print("Shape after concatenating demographics:", x.shape)

        x = self.fc(x)
        #print("Output shape before returning:", x.shape)

        return x
writer = SummaryWriter("runs/CustomResNet18_experiment")

def freeze_unfreeze_layers(model, freeze=True, layers_to_train=["layer4", "fc"]):
    for name, param in model.named_parameters():
        if any(layer in name for layer in layers_to_train):
            param.requires_grad = not freeze
        else:
            param.requires_grad = freeze


def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs= 10):

  model.train()
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

    accuracy = 100 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}, Validation Accuracy: {val_accuracy}%")
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

    writer.add_scalar("Loss/Test", avg_test_loss)
    writer.add_scalar("Accuracy/Test", test_accuracy)
    writer.add_scalar("Precision/Test", precision)
    writer.add_scalar("Recall/Test", recall)
    writer.add_scalar("F1-Score/Test", f1_score)

    print(f'Test Loss: {avg_test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.2f}%')
    print(f'Test Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1_score:.4f}')
if __name__ == "__main__":
  config = {
      "file_path": "preprocessed_dummy_data.pkl",
      "batch_size": 32,
      "num_epochs": 10,
      "learning_rate": 1e-5,
      "num_demographics": 3,
      "num_classes": 15,
      "train_percent": 0.7,
      "val_percent": 0.1
  }

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  train_loader, val_loader, test_loader = load_data(config["file_path"], config["batch_size"], config["train_percent"], config["val_percent"])

  model = CustomResNet18(num_demographics=config["num_demographics"], num_classes=config["num_classes"]).to(device)
  freeze_unfreeze_layers(model, freeze=True, layers_to_train=["layer4", "fc"])

  criterion = nn.BCEWithLogitsLoss()
  optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["learning_rate"])

  """precision_metric = torchmetrics.Precision(average='micro').to(device)
  recall_metric = torchmetrics.Recall(average='micro').to(device)
  f1_metric = torchmetrics.F1Score(average='micro').to(device)"""

  precision_metric = MultilabelPrecision(num_labels= config["num_classes"], average='macro').to(device)
  recall_metric = MultilabelRecall(num_labels= config["num_classes"], average='macro').to(device)
  f1_metric = MultilabelF1Score(num_labels= config["num_classes"], average='macro').to(device)

  train_model(train_loader, val_loader, model, criterion, optimizer, config["num_epochs"])
  evaluate_model(test_loader, model, criterion, precision_metric, recall_metric, f1_metric)

  writer.close()
