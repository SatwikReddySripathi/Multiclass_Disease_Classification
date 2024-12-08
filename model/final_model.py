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
import os
import subprocess
gcp_credentials_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = gcp_credentials_path
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
def load_data_from_gcp(bucket_name, file_path, batch_size, train_percent, target_size=(224, 224), seed=42):

    torch.manual_seed(seed)
    np.random.seed(seed)
    
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
    val_size = len(dataset) - train_size
    #test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    #test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    return train_loader, val_loader

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

def freeze_unfreeze_layers(model, freeze=True, layers_to_train=["layer4", "demographics_fc","fc"]):
    for name, param in model.named_parameters():
        if any(layer in name for layer in layers_to_train):
            param.requires_grad = not freeze
        else:
            param.requires_grad = freeze

# Training function
def train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs=10):
    logging.info("#################### Entered TRAIN MODEL ################")
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
        logging.info(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss}, Validation Loss: {avg_val_loss}, Validation Accuracy: {val_accuracy}%")

    return best_val_accuracy

def save_model_as_torchscript(model_path, output_path):
    # Load the trained model
    model = torch.load(model_path)
    model.eval()

    # Create an example input for tracing (assuming the image size and demographic input)
    image_tensor = torch.rand((1, 1, 224, 224))  # Example input for grayscale image
    demographics_tensor = torch.rand((1, 3))  # Example demographic tensor ([gender_m, gender_f, age])

    # Trace the model
    traced_model = torch.jit.trace(model, (image_tensor, demographics_tensor))

    # Save the traced model
    traced_model.save(output_path)

def create_torch_model_archive(model_name, version, serialized_file, model_file, handler, export):
    """
    Function to create a Torch model archive using the torch-model-archiver command.

    Args:
        model_name (str): The name of the model to be archived.
        version (str): The version of the model.
        serialized_file (str): The path to the serialized PyTorch model file (.jit).
        handler (str): The path to the handler.py file.
    """
    try:
        command = [
            "torch-model-archiver",
            "--model-name", model_name,
            "--version", version,
            "--serialized-file", serialized_file,
            "--model-file", model_file,
            "--handler", handler,
            "--export-path", export
        ]
        
        # Execute the command
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        print("Model archive created successfully:")
        print(result.stdout.decode("utf-8"))
    except subprocess.CalledProcessError as e:
        print("Failed to create model archive:")
        print(e.stderr.decode("utf-8"))

# Grid search function
def grid_search():
    '''
    param_grid = {
        "num_epochs": [5],
        "batch_size": [32, 64],
        "learning_rate": [1e-5, 1e-4],
        "demographics_fc_size": [64]
    }
    '''

    param_grid = {
        "num_epochs": [5, 10, 15],
        "batch_size": [32, 64],
        "learning_rate": [1e-5, 1e-4, 1e-3],
        "demographics_fc_size": [64, 128]
    }

    best_val_accuracy = 0
    best_params = None
    best_model = None
    all_combinations = list(itertools.product(*param_grid.values()))
    mlflow.set_experiment("Grid Search Experiment")
    logging.info("Experiment is set in mlflow")

    for params in all_combinations:
        num_epochs, batch_size, learning_rate, demographic_fc_size = params

        with mlflow.start_run():
            mlflow.log_param("num_epochs", num_epochs)
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("learning_rate", learning_rate)
            mlflow.log_param("demographic_fc_size", demographic_fc_size)
            
            logging.info("##################################    DOWNLOADING MODEL      ####################################")
            model = CustomResNet18(demographic_fc_size, num_demographics=config["num_demographics"], num_classes=config["num_classes"]).to(device)
            logging.info("##################################    FREEZING MODEL      ####################################")
            freeze_unfreeze_layers(model, freeze=True)
            logging.info("############################# crossed one  ##################################################")
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            logging.info("############################## crossed two ###################################################")
            criterion = nn.BCEWithLogitsLoss()      
            logging.info("crossed three")

            logging.info("Splitting data")

            train_loader, val_loader  = load_data_from_gcp(
                bucket_name="nih-dataset-mlops",
                file_path="Data_Preprocessing_files/train_preprocessed_data.pkl",
                batch_size=batch_size,
                train_percent=config["train_percent"]
            )

            logging.info("Training of the model has started")

            val_accuracy = train_model(train_loader, val_loader, model, criterion, optimizer, num_epochs)
            mlflow.log_metric("val_accuracy", val_accuracy)

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_params = params
                best_model = model

    if best_model is not None:
        output_dir = "/app/model_output"
        if not os.path.exists(output_dir):
            logging.info("## Creating Directory")
            os.makedirs(output_dir)

        torch.save(best_model, os.path.join(output_dir, "best_model.pt"))
        print(f"Model saved at {output_dir}/best_model.pt with accuracy: {best_val_accuracy}%")
        save_model_as_torchscript(os.path.join(output_dir, "best_model.pt"), os.path.join(output_dir, "best_model.jit"))
        print(f"Model saved at {output_dir}/best_model.jit")
        
    best_param_path = os.path.join(os.getcwd(),"model","best_params.txt")
    with open(best_param_path,"w") as f:
        f.write(f"Best validation accuracy: {best_val_accuracy}\n")
        f.write(f"Parameters: {best_params}\n")
        
    
    handler_path = os.path.join(os.getcwd(),"model","model_handler.py")
    serialized_path = os.path.join(output_dir,"best_model.jit")
    model_path = os.path.join(os.getcwd(),"model","model.py")
    export_path = os.path.join(os.getcwd(),output_dir)
    create_torch_model_archive(
    model_name="model",
    version="1.0",
    serialized_file=serialized_path,
    model_file=model_path, 
    handler=handler_path,
    export= export_path)
    #dummy_items = os.listdir(export_path)
    logging.info("#########################  BEST PARAMETERS ##################")
    print(best_params)
    logging.info(best_params)
    return best_params

    
    

# Main script
if __name__ == "__main__":
#    PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config = {
        "num_demographics": 3,
        "num_classes": 15,
        "train_percent": 0.7,
        "val_percent": 0.1
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Starting grid search...")
    grid_search() 
