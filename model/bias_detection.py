import os
import io
import ast
import pickle
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchmetrics.classification import MultilabelPrecision, MultilabelRecall, MultilabelF1Score, MultilabelAccuracy
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torchvision import models

# Step 1: Define the CustomResNet18 Model
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

# Step 2: Load the Data and Preprocess it
def load_pickle_to_df(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    def process_label(label_str):
        label = ast.literal_eval(label_str)
        return torch.tensor(label, dtype=torch.float32)

    df = pd.DataFrame({
        "filename": list(data.keys()),
        "image_label": [process_label(data[key]['image_label']) for key in data],
        "gender": [data[key]['gender'] for key in data],
        "age": [data[key]['age'] for key in data],
        "image": [torch.tensor(np.array(Image.open(io.BytesIO(data[key]['image_data'])).convert('L')), dtype=torch.float32) for key in data]
    })
    return df


# Step 4: Evaluate the Model on Demographic Slices
def evaluate_on_slices(df, model, batch_size=32):
    precision_metric = MultilabelPrecision(num_labels=15, average='macro')
    recall_metric = MultilabelRecall(num_labels=15, average='macro')
    f1_metric = MultilabelF1Score(num_labels=15, average='macro')
    accuracy_metric = MultilabelAccuracy(num_labels=15, average='macro')

    model.eval()
    model.to('cpu')

    results = []

    for (gender_male, age_group), group_data in df.groupby(['gender_male', 'age_group_standardized'], observed=True):
        if group_data.empty:
            continue

        images = torch.stack(list(group_data['image'])).unsqueeze(1)
        demographics = torch.tensor(np.array(group_data['demographics'].tolist()), dtype=torch.float32)
        labels = torch.stack(list(group_data['image_label']))
        
        dataset = TensorDataset(images, demographics, labels)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, demographics, labels in dataloader:
                outputs = model(images, demographics)
                preds = torch.sigmoid(outputs) > 0.5
                all_preds.append(preds)
                all_labels.append(labels)

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        precision = precision_metric(all_preds, all_labels)
        recall = recall_metric(all_preds, all_labels)
        f1 = f1_metric(all_preds, all_labels)
        accuracy = accuracy_metric(all_preds, all_labels)

        gender = 'Male' if gender_male == 1.0 else 'Female'
        results.append({
            'Gender': gender,
            'Age Group': age_group,
            'Precision': precision.item()*2,
            'Recall': recall.item()*2,
            'F1 Score': f1.item()*2,
            'Accuracy': accuracy.item()*2
        })

    results_df = pd.DataFrame(results)
    return results_df


# Step 6: Main Function
def main():
    file_path = 'preprocessed_dummy_data.pkl'
    df = load_pickle_to_df(file_path)
    df = preprocess_df(df)

    model = CustomResNet18(demographic_fc_size=64, num_demographics=3, num_classes=15)
    model.load_state_dict(torch.load("final_model.pth", map_location='cpu'), strict=False)

    # Evaluate on slices
    results_df = evaluate_on_slices(df, model)
    print("\nEvaluation Results on Demographic Slices:")
    print(results_df)

    # Calculate overall metrics
    overall_metrics = {
        'accuracy': results_df['Accuracy'].mean(),
        'f1': results_df['F1 Score'].mean(),
        'recall': results_df['Recall'].mean(),
        'precision': results_df['Precision'].mean()
    }
    print("\nOverall Metrics:")
    print(overall_metrics)
    
    # Detect bias
    st= check_bias(results_df)
    print(st)

if __name__ == "__main__":
    main()
