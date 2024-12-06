from PIL import Image
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
from torchvision import models
import streamlit as st



class CustomResNet18(nn.Module):
    def __init__(self, demographic_fc_size, num_demographics, num_classes=15):
        super(CustomResNet18, self).__init__()

        # Load a pre-trained ResNet-18 model
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        # Fully connected layer for demographic inputs
        self.demographics_fc = nn.Sequential(
            nn.Linear(num_demographics, demographic_fc_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Final fully connected layer combining ResNet features and demographics
        self.fc = nn.Linear(512 + demographic_fc_size, num_classes)

    def forward(self, images, demographics):
        x = self.resnet(images)
        x = x.view(x.size(0), -1)
        demographics_features = self.demographics_fc(demographics)
        x = torch.cat((x, demographics_features), dim=1)
        x = self.fc(x)
        return x





# Load the PyTorch model
@st.cache_resource
def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((224, 224)),  # Resize the image to the model's expected input size
        transforms.ToTensor(),  # Convert the image to a tensor
        transforms.Normalize([0.5], [0.5])  # Normalize for grayscale input
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

def load_custom_model(model_path, demographic_fc_size=64, num_demographics=2, num_classes=15):
    model = CustomResNet18(
        demographic_fc_size=demographic_fc_size,
        num_demographics=num_demographics,
        num_classes=num_classes
    )
    model=torch.load(model_path, map_location=torch.device("cpu"))
    model.eval()
    return model
