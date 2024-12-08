from PIL import Image, ImageStat
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
from torchvision import models
import streamlit as st
import base64
import requests
import subprocess  
from io import BytesIO
import io
from google.auth import default
from google.auth.transport.requests import Request
import streamlit as st
import numpy as np
from torchvision import transforms
from torch.nn.functional import sigmoid
from google.cloud import storage
import json

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



# Disease Mapping and Information
DISEASE_INFO = {
    'No Finding':{
        "name": "No Finding",
        "description":"Our Model predicts that there might be any disese afftecting you, other than your own imagination",
        "symptoms":"Just some personal dissatisfaction",
        "":"",
        "":"",
        "":"",
        "":"",
        "":"",
    },
    'Atelectasis': {
        "name": "Atelectasis",
    "description": "Atelectasis (at-uh-LEK-tuh-sis) is the collapse of a lung or part of a lung, also known as a lobe.\n"
                "It happens when tiny air sacs within the lung, called alveoli, lose air.\n\n"
                "Atelectasis is one of the most common breathing complications after surgery.\n"
                "It's also a possible complication of other respiratory problems, including cystic fibrosis, lung tumors, chest injuries, fluid in the lung, and respiratory weakness.\n"
                "You may develop atelectasis if you breathe in a foreign object.\n\n"
                "This condition can make breathing hard, particularly if you already have lung disease.\n"
                "Treatment depends on what's causing the collapse and how severe it is.\n\n"
                "The definition of atelectasis is broader than pneumothorax (noo-moe-THOR-aks).\n"
                "Pneumothorax is when air leaks into the space between your lungs and chest wall, causing part or all of a lung to collapse.\n"
                "Pneumothorax is one of several causes of atelectasis.",
    "symptoms": ["Shortness of breath", "Coughing", "Chest pain"],
    "causes": [
        "A blocked airway, known as obstructive atelectasis.",
        "Pressure from outside the lung, known as nonobstructive atelectasis.",
        "General anesthesia, which affects the exchange of lung gases and often occurs after major surgery, including heart bypass surgery."
    ],
    "causes_details": {
        "Obstructive Causes": [
            "Mucus plug: A buildup of sputum or phlegm, common during and after surgery or in conditions like cystic fibrosis or severe asthma.",
            "Foreign body: Common in children who inhale objects such as peanuts or small toys.",
            "Tumor inside the airway: A growth that can narrow or block the airway."
        ],
        "Nonobstructive Causes": [
            "Injury: Chest trauma, such as from a fall or car accident.",
            "Pleural effusion: Fluid buildup in the space between the lung lining and chest wall.",
            "Pneumonia: A lung infection.",
            "Pneumothorax: Air leakage into the space between the lungs and chest wall.",
            "Scarring of lung tissue: From injury, lung disease, or surgery.",
            "Tumor: A large tumor pressing against the lung."
        ]
    },
    "risk_factors": [
        "Any condition that makes it hard to swallow.",
        "Prolonged bed rest without changing position.",
        "Lung diseases such as asthma, bronchiectasis, or cystic fibrosis.",
        "Recent surgery in the stomach or chest area.",
        "Recent general anesthesia.",
        "Weak breathing muscles due to neuromuscular conditions.",
        "Medicines that weaken breathing.",
        "Pain or injury, such as stomach pain or a broken rib.",
        "Smoking."
    ],
    "complications": [
        "Low blood oxygen (hypoxemia).",
        "Pneumonia: Mucus in a collapsed lung may lead to infection.",
        "Respiratory failure: Loss of a lobe or lung, especially in infants or people with lung disease, can be life-threatening."
    ],
    "prevention": [
        "In children, keep small objects out of their reach to avoid airway blockages.",
        "In adults, discuss breathing exercises and muscle training with your doctor before major surgery to reduce the risk of atelectasis."
    ],
    "image": "https://my.clevelandclinic.org/-/scassets/images/org/health/articles/17699-atelectasis",
    },
    'Cardiomegaly': {
        "name": "Cardiomegaly (Enlarged Heart)",
    "description": "Cardiomegaly, or an enlarged heart, occurs when the heart is abnormally thick or overly stretched, becoming larger than usual and struggling to pump blood effectively. The condition can be temporary or permanent. However, cardiomegaly is manageable, and most people can continue their normal activities.",
    "symptoms": [
        "Fatigue and dizziness",
        "Palpitations",
        "Shortness of breath",
        "Edema in the abdomen, legs, and feet"
    ],
    "causes": [
        "Ischemic heart disease",
        "Hypertension",
        "Valvular heart disease",
        "Cardiomyopathy",
        "Congenital heart disease",
        "Drug and alcohol use",
        "Arrhythmia",
        "Viral myocarditis"
    ],
    "causes_details": {
        "Detailed Causes": [
            "Ischemic heart disease: Reduced blood flow to the heart muscle.",
            "Hypertension: High blood pressure overworks the heart, causing it to enlarge.",
            "Valvular heart disease: Faulty heart valves lead to inefficient blood flow.",
            "Cardiomyopathy: Diseases of the heart muscle affect its function.",
            "Congenital heart disease: Structural abnormalities in the heart present at birth.",
            "Drug and alcohol use: Excessive consumption weakens the heart.",
            "Arrhythmia: Irregular heart rhythms stress the heart.",
            "Viral myocarditis: Viral infections can inflame and enlarge the heart."
        ]
    },
    "risk_factors": [
        "Inactive lifestyle",
        "Family history of heart attacks or enlarged heart",
        "Hypertension",
        "Smoking or excessive alcohol consumption"
    ],
    "complications": [
        "Heart failure due to left ventricular hypertrophy",
        "Heart attack or stroke caused by blood clots",
        "Heart murmur caused by valve problems",
        "Sudden cardiac death due to arrhythmias"
    ],
    "prevention": [
        "Eat healthily",
        "Exercise regularly for at least 30 minutes daily",
        "Get adequate sleep (at least 8 hours)",
        "Keep cholesterol levels and blood pressure under control",
        "Drink alcohol in moderation",
        "Quit smoking and avoid illicit substances"
    ],
    "image": "https://my.clevelandclinic.org/-/scassets/images/org/health/articles/21490-enlarged-heart",
    },
    'Effusion': {
        "name": "Pleural Effusion",
    "description": "Effusion refers to the abnormal accumulation of fluid in body cavities, commonly in the pleural cavity of the lungs. It can result from various underlying medical conditions and may cause breathing difficulties or discomfort. The condition requires prompt diagnosis to address the underlying cause.",
    "symptoms": [
        "Difficulty breathing",
        "Coughing",
        "Chest discomfort",
        "Shortness of breath"
    ],
    "causes": [
        "Heart failure",
        "Kidney disease",
        "Liver cirrhosis",
        "Pneumonia or lung infections",
        "Cancer or malignancy",
        "Pulmonary embolism",
        "Inflammatory diseases such as lupus or rheumatoid arthritis"
    ],
    "causes_details": {
        "Detailed Causes": [
            "Heart failure: Increased pressure in blood vessels can lead to fluid leakage.",
            "Kidney disease: Fluid retention and decreased excretion contribute to effusion.",
            "Liver cirrhosis: Reduced protein production causes fluid accumulation.",
            "Pneumonia or lung infections: Infections can inflame the pleura and cause effusion.",
            "Cancer: Tumors can block lymphatic drainage or produce fluid directly.",
            "Pulmonary embolism: Blockages in lung arteries can lead to fluid buildup.",
            "Inflammatory diseases: Autoimmune conditions can cause pleural inflammation and effusion."
        ]
    },
    "risk_factors": [
        "Chronic heart, liver, or kidney disease",
        "History of lung infections or pneumonia",
        "Smoking or exposure to toxins",
        "Autoimmune diseases such as lupus or rheumatoid arthritis",
        "Cancer or malignancy"
    ],
    "complications": [
        "Lung collapse due to excessive fluid accumulation (pleural effusion)",
        "Infection in the pleural space (empyema)",
        "Breathing difficulties due to lung compression",
        "Reduced oxygen levels in the body (hypoxemia)"
    ],
    "prevention": [
        "Maintain a healthy lifestyle to prevent chronic diseases such as heart and kidney conditions.",
        "Avoid smoking and exposure to environmental toxins.",
        "Seek prompt treatment for respiratory infections.",
        "Manage chronic illnesses effectively under medical supervision.",
        "Stay up-to-date on vaccinations to prevent lung infections."
    ],
    "image": "https://my.clevelandclinic.org/-/scassets/images/org/health/articles/pleural-effusion",
    },
    'Infiltration': {
        "name": "Infiltration",
    "description": "Infiltration refers to the abnormal accumulation of substances, such as fluids, cells, or foreign material, in lung tissues. This condition is often associated with infections, inflammation, or certain medical treatments. It can impair lung function and cause respiratory distress.",
    "symptoms": [
        "Cough",
        "Fever",
        "Shortness of breath",
        "Chest pain",
        "Fatigue"
    ],
    "causes": [
        "Bacterial, viral, or fungal infections",
        "Pulmonary edema due to heart failure",
        "Aspiration of foreign substances",
        "Autoimmune conditions such as sarcoidosis",
        "Interstitial lung disease",
        "Allergic reactions or hypersensitivity pneumonitis"
    ],
    "causes_details": {
        "Detailed Causes": [
            "Infections: Bacterial, viral, or fungal infections can inflame the lung tissues, leading to infiltration.",
            "Pulmonary edema: Fluid accumulation in the lungs due to heart failure or kidney problems.",
            "Aspiration: Inhaling foreign substances, such as food particles or stomach contents, into the lungs.",
            "Autoimmune conditions: Diseases like sarcoidosis or lupus can cause inflammatory lung infiltration.",
            "Interstitial lung disease: A group of disorders that cause scarring and infiltration in the lungs.",
            "Allergic reactions: Hypersensitivity pneumonitis from exposure to allergens or toxins."
        ]
    },
    "risk_factors": [
        "Chronic respiratory conditions such as asthma or COPD",
        "Weakened immune system due to diseases or medications",
        "Exposure to environmental toxins or allergens",
        "Heart or kidney disease",
        "History of lung infections or pneumonia",
        "Smoking or exposure to secondhand smoke"
    ],
    "complications": [
        "Impaired lung function leading to difficulty breathing",
        "Hypoxemia (low oxygen levels in the blood)",
        "Progression to chronic respiratory failure",
        "Development of pulmonary fibrosis (scarring of lung tissue)",
        "Increased risk of secondary infections"
    ],
    "prevention": [
        "Avoid smoking and exposure to environmental pollutants.",
        "Seek timely treatment for respiratory infections and chronic diseases.",
        "Maintain a strong immune system with a healthy diet and regular exercise.",
        "Minimize exposure to known allergens and irritants.",
        "Follow safety measures in workplaces with exposure to toxins."
    ],
    "image": "https://images.app.goo.gl/677X3mCgkZhnxGXA8",
    },
    'Mass': {
        "name": "Mass",
    "description": "A mass refers to an abnormal tissue growth in the lungs, which can be benign or malignant. It is often discovered through imaging and may require further diagnostic tests to determine its nature.",
    "symptoms": [
        "Persistent cough",
        "Chest pain",
        "Shortness of breath",
        "Unexplained weight loss",
        "Fatigue"
    ],
    "causes": [
        "Lung cancer",
        "Benign tumors",
        "Infections such as tuberculosis",
        "Inflammatory conditions",
        "Metastatic cancer from other organs"
    ],
    "causes_details": {
        "Detailed Causes": [
            "Lung cancer: Malignant growths originating in the lungs.",
            "Benign tumors: Non-cancerous growths such as hamartomas.",
            "Infections: Chronic infections like tuberculosis or fungal infections.",
            "Inflammatory conditions: Granulomas caused by sarcoidosis or other autoimmune diseases.",
            "Metastasis: Cancer spreading to the lungs from other parts of the body."
        ]
    },
    "risk_factors": [
        "Smoking or exposure to secondhand smoke",
        "Family history of lung cancer",
        "Chronic exposure to pollutants or toxins",
        "History of lung infections",
        "Weakened immune system"
    ],
    "complications": [
        "Obstruction of airways leading to difficulty breathing",
        "Spread of malignant tumors to other organs",
        "Infections in areas surrounding the mass",
        "Collapse of lung tissue"
    ],
    "prevention": [
        "Avoid smoking and exposure to secondhand smoke.",
        "Limit exposure to environmental toxins.",
        "Maintain a healthy immune system.",
        "Seek timely treatment for respiratory infections.",
        "Regular screenings for high-risk individuals."
    ],
    "image": "https://www.verywellhealth.com/thmb/6Wk1rc8GiRV3A8Rw_iDMJ0NKVdw=/750x0/filters:no_upscale():max_bytes(150000):strip_icc():format(webp)/lung-mass-possible-causes-and-what-to-expect-2249388-5bc3f847c9e77c00512dc818.png",
    },
    'Nodule': {
        "name": "Nodule",
    "description": "A nodule is a small, round growth in the lungs. Most nodules are benign and do not cause symptoms, but some may require monitoring or further investigation to rule out malignancy.",
    "symptoms": [
        "Usually asymptomatic",
        "Occasionally, mild cough or shortness of breath"
    ],
    "causes": [
        "Benign tumors",
        "Infections such as tuberculosis",
        "Inflammatory diseases like sarcoidosis",
        "Lung cancer",
        "Fungal infections"
    ],
    "causes_details": {
        "Detailed Causes": [
            "Benign tumors: Non-cancerous growths.",
            "Infections: Tuberculosis or fungal infections causing granulomas.",
            "Inflammatory diseases: Conditions like sarcoidosis or rheumatoid arthritis.",
            "Lung cancer: Small malignant growths in early stages.",
            "Fungal infections: Histoplasmosis or coccidioidomycosis."
        ]
    },
    "risk_factors": [
        "Smoking",
        "Family history of lung conditions",
        "Exposure to environmental toxins",
        "History of lung infections",
        "Chronic inflammatory diseases"
    ],
    "complications": [
        "Possible progression to malignancy",
        "Scarring of lung tissue",
        "Respiratory discomfort",
        "Risk of infection around the nodule"
    ],
    "prevention": [
        "Avoid smoking and exposure to environmental pollutants.",
        "Seek timely treatment for infections and inflammatory conditions.",
        "Monitor any detected nodules regularly through imaging.",
        "Maintain a healthy immune system."
    ],
    "image": "https://my.clevelandclinic.org/-/scassets/images/org/health/articles/pulmonary-nodules",
    },
    'Pneumonia': {
        "name": "Pneumonia",
    "description": "Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus, causing difficulty in breathing and other symptoms.",
    "symptoms": [
        "Cough with phlegm",
        "Fever",
        "Chills",
        "Shortness of breath",
        "Chest pain"
    ],
    "causes": [
        "Bacterial infections such as Streptococcus pneumoniae",
        "Viral infections including influenza",
        "Fungal infections",
        "Aspiration of food or liquids",
        "Weakened immune system"
    ],
    "causes_details": {
        "Detailed Causes": [
            "Bacterial infections: Streptococcus pneumoniae is the most common cause.",
            "Viral infections: Influenza viruses or coronaviruses.",
            "Fungal infections: Common in individuals with weakened immune systems.",
            "Aspiration: Inhaling food, liquid, or vomit into the lungs.",
            "Weakened immunity: Conditions like HIV/AIDS or chemotherapy."
        ]
    },
    "risk_factors": [
        "Age (very young or elderly)",
        "Chronic diseases like asthma or COPD",
        "Weakened immune system",
        "Smoking",
        "Hospitalization"
    ],
    "complications": [
        "Severe breathing difficulty",
        "Pleural effusion (fluid around the lungs)",
        "Lung abscess",
        "Sepsis"
    ],
    "prevention": [
        "Get vaccinated for pneumococcal pneumonia and influenza.",
        "Practice good hygiene to prevent infections.",
        "Avoid smoking.",
        "Strengthen your immune system with a healthy lifestyle."
    ],
    "image": "https://my.clevelandclinic.org/-/scassets/images/org/health/articles/4471-pneumonia-01",
    },
    'Pneumothorax': {
        "name": "Pneumothorax",
    "description": "Pneumothorax, or a collapsed lung, occurs when air leaks into the space between the lungs and the chest wall. This causes the lung to collapse partially or entirely, leading to breathing difficulties.",
    "symptoms": [
        "Sudden chest pain",
        "Shortness of breath",
        "Rapid heart rate",
        "Fatigue"
    ],
    "causes": [
        "Chest injury or trauma",
        "Lung diseases like COPD or asthma",
        "Ruptured air blisters (blebs)",
        "Medical procedures involving the chest",
        "Spontaneous occurrence without a known cause"
    ],
    "causes_details": {
        "Detailed Causes": [
            "Chest injury: Blunt trauma or penetration can cause air leakage.",
            "Lung diseases: Conditions like COPD or cystic fibrosis weaken the lungs.",
            "Ruptured air blisters: Small air-filled sacs can burst, causing leakage.",
            "Medical procedures: Accidental punctures during surgery or biopsies.",
            "Spontaneous occurrence: Sometimes happens in healthy individuals, often tall and thin young men."
        ]
    },
    "risk_factors": [
        "Smoking",
        "Family history of pneumothorax",
        "Chronic lung diseases like COPD or asthma",
        "Previous pneumothorax episodes",
        "Mechanical ventilation"
    ],
    "complications": [
        "Recurrence of pneumothorax",
        "Severe breathing difficulties",
        "Cardiac arrest in extreme cases",
        "Shock due to a large collapse"
    ],
    "prevention": [
        "Avoid smoking.",
        "Protect the chest from injury during activities.",
        "Seek timely treatment for chronic lung conditions.",
        "Regular follow-ups if you’ve had a pneumothorax before."
    ],
    "image": "https://my.clevelandclinic.org/-/scassets/images/org/health/articles/15304-pneumothorax-illustration",
    },
    'Consolidation': {
        "name": "Consolidation",
    "description": "Consolidation occurs when lung tissue that is normally filled with air becomes solid due to accumulation of fluids, blood, or other substances, often as a result of infections or inflammations.",
    "symptoms": [
        "Coughing",
        "Difficulty breathing",
        "Fever",
        "Chest pain",
        "Fatigue"
    ],
    "causes": [
        "Bacterial pneumonia",
        "Pulmonary hemorrhage",
        "Lung abscess",
        "Aspiration of foreign materials",
        "Lung cancer"
    ],
    "causes_details": {
        "Detailed Causes": [
            "Bacterial pneumonia: Infection causes fluid accumulation in alveoli.",
            "Pulmonary hemorrhage: Bleeding into the lungs leads to solidification.",
            "Lung abscess: Localized infection causes pus accumulation.",
            "Aspiration: Inhaling foreign substances can inflame lung tissue.",
            "Lung cancer: Tumors can block airways and lead to consolidation."
        ]
    },
    "risk_factors": [
        "Weakened immune system",
        "Chronic lung diseases",
        "Smoking",
        "Recent respiratory infections",
        "Hospitalization"
    ],
    "complications": [
        "Hypoxemia due to poor oxygen exchange",
        "Respiratory failure",
        "Spread of infection to other parts of the body",
        "Pleural effusion"
    ],
    "prevention": [
        "Get vaccinated against pneumonia.",
        "Avoid smoking and exposure to lung irritants.",
        "Practice good hygiene to prevent infections.",
        "Seek early treatment for respiratory symptoms."
    ],
    "image": "https://upload.wikimedia.org/wikipedia/commons/a/a6/Pneumonia_x-ray.jpgn",
    },
    'Edema': {
        "name": "Pulmonary Edema",
    "description": "Pulmonary edema occurs when excess fluid accumulates in the lungs, making it difficult to breathe. It is often caused by heart problems or other medical conditions.",
    "symptoms": [
        "Shortness of breath",
        "Wheezing",
        "Coughing up frothy sputum",
        "Rapid or irregular heartbeat",
        "Fatigue"
    ],
    "causes": [
        "Heart failure",
        "Kidney failure",
        "High-altitude sickness",
        "Lung infections",
        "Inhalation of toxins or smoke"
    ],
    "causes_details": {
        "Detailed Causes": [
            "Heart failure: Increased pressure in the pulmonary veins causes fluid leakage.",
            "Kidney failure: Impaired fluid balance can lead to pulmonary edema.",
            "High-altitude sickness: Rapid ascent can cause fluid buildup in the lungs.",
            "Lung infections: Severe infections like pneumonia can cause edema.",
            "Inhalation of toxins: Breathing harmful substances damages lung tissue."
        ]
    },
    "risk_factors": [
        "Heart disease or heart failure",
        "Kidney disease",
        "High-altitude travel",
        "Chronic respiratory diseases",
        "Exposure to toxins or irritants"
    ],
    "complications": [
        "Severe breathing difficulties",
        "Respiratory failure",
        "Cardiac arrest",
        "Organ failure due to low oxygen levels"
    ],
    "prevention": [
        "Manage heart and kidney conditions effectively.",
        "Avoid exposure to lung irritants and toxins.",
        "Follow gradual ascent guidelines when traveling to high altitudes.",
        "Maintain a healthy lifestyle to reduce risk factors."
    ],
    "image": "https://assets.mayoclinic.org/content/dam/media/en/images/2023/02/10/high-altitude-pulmonary-edema.jpg",
    },
    'Emphysema': {
        "name": "Emphysema",
    "description": "Emphysema is a chronic lung disease that damages the alveoli (air sacs) in the lungs. This leads to reduced oxygen exchange, shortness of breath, and other respiratory issues.",
    "symptoms": [
        "Shortness of breath, especially during physical activities",
        "Chronic cough",
        "Wheezing",
        "Fatigue",
        "Unintentional weight loss"
    ],
    "causes": [
        "Smoking (primary cause)",
        "Long-term exposure to airborne irritants such as chemical fumes and dust",
        "Genetic deficiency of alpha-1 antitrypsin",
        "Chronic bronchitis"
    ],
    "causes_details": {
        "Detailed Causes": [
            "Smoking: Long-term tobacco use damages lung tissue.",
            "Airborne irritants: Chemicals, dust, and pollutants can lead to chronic inflammation.",
            "Genetic factors: Alpha-1 antitrypsin deficiency increases susceptibility.",
            "Chronic bronchitis: Inflammation can contribute to alveolar damage."
        ]
    },
    "risk_factors": [
        "Smoking or exposure to secondhand smoke",
        "Long-term exposure to air pollution or workplace irritants",
        "Family history of emphysema",
        "Age, as damage accumulates over time"
    ],
    "complications": [
        "Collapsed lung (pneumothorax)",
        "Heart problems due to increased pressure in lung arteries",
        "Respiratory infections",
        "Severe breathing difficulties"
    ],
    "prevention": [
        "Quit smoking and avoid secondhand smoke.",
        "Limit exposure to air pollutants and workplace irritants.",
        "Get vaccinated against respiratory infections such as influenza and pneumonia.",
        "Regularly monitor lung function if at risk."
    ],
    "image": "https://my.clevelandclinic.org/-/scassets/images/org/health/articles/9370-emphysema",
    },
    'Fibrosis': {
        "name": "Fibrosis",
    "description": "Pulmonary fibrosis refers to the scarring and thickening of lung tissue, which leads to stiffness and reduced lung function. It can result from a variety of causes, including chronic inflammation and environmental exposure.",
    "symptoms": [
        "Chronic dry cough",
        "Shortness of breath",
        "Fatigue",
        "Chest discomfort",
        "Unexplained weight loss"
    ],
    "causes": [
        "Environmental exposure to toxins such as asbestos or silica",
        "Chronic infections",
        "Autoimmune diseases like rheumatoid arthritis or scleroderma",
        "Certain medications like chemotherapy drugs",
        "Idiopathic pulmonary fibrosis (unknown cause)"
    ],
    "causes_details": {
        "Detailed Causes": [
            "Environmental exposure: Long-term inhalation of harmful substances like asbestos.",
            "Chronic infections: Repeated infections can cause lung scarring.",
            "Autoimmune diseases: Conditions like rheumatoid arthritis or lupus.",
            "Medications: Some chemotherapy drugs or radiation therapy.",
            "Idiopathic: Unknown causes but linked to genetic and environmental factors."
        ]
    },
    "risk_factors": [
        "Age, as it typically occurs in middle-aged or older adults",
        "Smoking",
        "Long-term exposure to environmental toxins",
        "Family history of pulmonary fibrosis",
        "Autoimmune diseases"
    ],
    "complications": [
        "Respiratory failure",
        "Pulmonary hypertension",
        "Heart failure",
        "Increased susceptibility to lung infections"
    ],
    "prevention": [
        "Avoid exposure to environmental toxins like asbestos or silica.",
        "Quit smoking and avoid secondhand smoke.",
        "Seek early treatment for autoimmune diseases.",
        "Maintain regular health check-ups if at risk."
    ],
    "image": "https://upload.wikimedia.org/wikipedia/commons/c/ce/Ipf_NIH.jpg",
    },
    'Pleural_Thickening': {
        "name": "Pleural Thickening",
    "description": "Pleural thickening refers to the thickening of the pleura, the thin membrane surrounding the lungs. It is often caused by chronic inflammation, infections, or asbestos exposure.",
    "symptoms": [
        "Chest pain",
        "Shortness of breath",
        "Reduced lung capacity",
        "Fatigue"
    ],
    "causes": [
        "Chronic infections like tuberculosis or pneumonia",
        "Asbestos exposure",
        "Inflammatory conditions such as pleurisy",
        "Trauma to the chest area",
        "Radiation therapy"
    ],
    "causes_details": {
        "Detailed Causes": [
            "Infections: Chronic infections like tuberculosis or fungal diseases.",
            "Asbestos exposure: Long-term exposure to asbestos fibers.",
            "Inflammatory conditions: Conditions like pleurisy or lupus.",
            "Trauma: Injuries or surgery in the chest area.",
            "Radiation therapy: Damage to the pleura during cancer treatments."
        ]
    },
    "risk_factors": [
        "Occupational exposure to asbestos or other toxins",
        "History of pleural infections or inflammation",
        "Smoking",
        "Previous chest trauma or surgeries"
    ],
    "complications": [
        "Restricted lung expansion",
        "Difficulty breathing",
        "Reduced oxygen levels in the body",
        "Progression to more severe respiratory conditions"
    ],
    "prevention": [
        "Avoid exposure to asbestos and other lung irritants.",
        "Seek timely treatment for respiratory infections.",
        "Protect the chest area from injuries.",
        "Quit smoking and maintain a healthy lifestyle."
    ],
    "image": "https://i0.wp.com/www.clydesideactiononasbestos.org.uk/wp-content/uploads/2020/03/lungs.png?fit=711%2C305&ssl=1",
    },
    'Hernia': {
        "name": "Hernia",
    "description": "A hernia occurs when an organ or tissue protrudes through an opening or weak spot in the surrounding muscle or connective tissue, often affecting the diaphragm and chest cavity.",
    "symptoms": [
        "Chest discomfort",
        "Abdominal pain",
        "Difficulty breathing",
        "Acid reflux",
        "Nausea"
    ],
    "causes": [
        "Congenital weakness in the diaphragm",
        "Trauma or injury to the abdominal or chest area",
        "Obesity or rapid weight gain",
        "Pregnancy",
        "Chronic coughing or straining"
    ],
    "causes_details": {
        "Detailed Causes": [
            "Congenital defects: Structural abnormalities present at birth.",
            "Injury: Trauma can weaken or tear the diaphragm.",
            "Obesity: Excess weight increases abdominal pressure.",
            "Pregnancy: Increased intra-abdominal pressure.",
            "Chronic straining: Coughing or lifting heavy objects."
        ]
    },
    "risk_factors": [
        "Obesity",
        "Pregnancy",
        "Chronic coughing or constipation",
        "History of abdominal surgeries",
        "Aging, which weakens muscles"
    ],
    "complications": [
        "Organ strangulation due to restricted blood flow",
        "Severe abdominal pain",
        "Difficulty swallowing",
        "Infection in the affected area"
    ],
    "prevention": [
        "Maintain a healthy weight.",
        "Avoid heavy lifting or straining.",
        "Seek treatment for chronic coughing or constipation.",
        "Practice proper posture and exercise to strengthen abdominal muscles."
    ],
    "image": "https://www.mayoclinic.org/-/media/kcms/gbs/patient-consumer/images/2013/08/26/10/13/ds00099_im01191_ww5rl88thu_jpg.jpg",
    },
}

# Initialize Google Cloud Storage
def initialize_storage(bucket_name):
    """Initialize GCS client and get bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    return bucket

def upload_image_to_gcs(bucket, folder_name, image, image_name):
    """Upload an image to GCS."""
    blob = bucket.blob(f"{folder_name}/{image_name}")
    blob.upload_from_file(image, content_type="image/png")
    return blob.public_url

def append_to_jsonl(bucket, folder_name, instance_data, jsonl_filename="predicted_metadata.jsonl"):
    """Append metadata to a JSONL file in GCS."""
    blob = bucket.blob(f"{folder_name}/{jsonl_filename}")
    try:
        # Download existing JSONL file
        existing_data = blob.download_as_text()
        records = [json.loads(line) for line in existing_data.splitlines()]
    except Exception:  # If file doesn't exist
        records = []

    # Append the new record
    records.append(instance_data)

    # Upload updated JSONL file
    updated_data = "\n".join([json.dumps(record) for record in records])
    blob.upload_from_string(updated_data, content_type="application/jsonl")



def get_access_token():
    credentials, _ = default()
    credentials.refresh(Request())
    return credentials.token


def encode_image_to_base64(image):
    """Encodes an image file to a base64 string."""
    buffered = BytesIO()  
    image.save(buffered, format="JPEG")  
    img_bytes = buffered.getvalue()  
    image_base64 = base64.b64encode(img_bytes).decode('utf-8')  
    return image_base64

# Parameters
#endpoint_url = "https://us-east1-aiplatform.googleapis.com/v1/projects/812555529114/locations/us-east1/endpoints/5963526768684957696:predict"
endpoint_url2 = "https://us-east1-aiplatform.googleapis.com/v1/projects/812555529114/locations/us-east1/endpoints/5472634409301573632:predict"

# Brightness thresholds
BRIGHTNESS_MIN = 50  # Minimum acceptable brightness
BRIGHTNESS_MAX = 200  # Maximum acceptable brightness




def calculate_brightness(image):
    """Calculate the average brightness of an image."""
    grayscale_image = image.convert("L")  # Convert to grayscale
    stat = ImageStat.Stat(grayscale_image)
    return stat.mean[0]  # Return the mean pixel value


def validate_image(image):
    """
    Validate the uploaded image:
    - Check brightness.
    - Ensure it looks like an X-ray (basic grayscale check).
    """
    # Calculate brightness
    brightness = calculate_brightness(image)

    # Check if the brightness is within acceptable range
    if brightness < BRIGHTNESS_MIN:
        return False, f"The image is too dark (brightness: {brightness:.2f}). Please upload a clearer X-ray image."
    elif brightness > BRIGHTNESS_MAX:
        return False, f"The image is too bright (brightness: {brightness:.2f}). Please upload a properly exposed X-ray image."

    # Additional validation checks can be added here (e.g., X-ray classification).
    return True, "Image is valid for prediction."




# Initialize session state variables
if "step" not in st.session_state:
    st.session_state.step = "input"
if "feedback" not in st.session_state:
    st.session_state.feedback = None
if "selected_disease" not in st.session_state:
    st.session_state.selected_disease = None
if "restart" not in st.session_state:
    st.session_state.restart = False
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "file_uploader_visible" not in st.session_state:
    st.session_state.file_uploader_visible = True

if st.session_state.restart:
    # Reset all session states
    st.session_state.step = "input"
    st.session_state.feedback = None
    st.session_state.selected_disease = None
    st.session_state.restart = False
    st.session_state.uploaded_image = None
    st.session_state.file_uploader_visible = True
    st.rerun()

####################################################
# Bucket Details
bucket_name = "nih-dataset-mlops"
bucket = initialize_storage(bucket_name)
####################################################



# Title and instructions

st.title("ThorAIx - Disease Prediction")
st.write("Upload an X-ray image and provide demographic details.")

gender = st.selectbox("Select Gender", ["Male", "Female"])
if gender == "Male":
    gender = "M"
else:
    gender = "F"
st.session_state.gender = gender
age = st.number_input("Enter Age", min_value=0, max_value=120, step=1, value=50)
st.session_state.age = age

if st.session_state.file_uploader_visible:
    uploaded_image = st.file_uploader("Upload an X-ray Image", type=["png", "jpg", "jpeg"])
    if uploaded_image:
        st.session_state.uploaded_image = uploaded_image
        st.session_state.file_uploader_visible = False
# Step 1: Input and Prediction
if st.session_state.step == "input":
    if st.session_state.uploaded_image:
        # Show the uploaded image
        st.image(st.session_state.uploaded_image, caption="Uploaded X-ray Image", use_container_width =10+0)
        image = Image.open(st.session_state.uploaded_image)

        #################################################
        ############# For GCP ##########
        image_name = st.session_state.uploaded_image.name

        is_valid, validation_message = validate_image(image)
        if not is_valid:
            st.error(validation_message)
            if st.button("Upload a New Image"):
                st.session_state.file_uploader_visible = True
                st.session_state.uploaded_image = None
                st.rerun()
        else:
            st.success(validation_message)

            # Predict button
            if st.button("Predict"):
                image = Image.open(st.session_state.uploaded_image)
                processed_image = encode_image_to_base64(image)  
                #st.write("This is the first debug statement")
                access_token = get_access_token()
                #st.write("This is the second debug statement")

                payload = {
                    "instances": [
                        {
                            "data": processed_image,
                            "gender": gender,
                            "age": age
                        }
                    ]
                }

                
                print("\n\n\n",access_token)
                headers = {
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json"
                }


                response = requests.post(endpoint_url2, json=payload, headers=headers)
                #st.write("This is the third debug statement")
                # Debug statements
                #print("Response:")
                #print(response.json())  
                
                response_data = response.json()  
                predictions = response_data.get('predictions', [])[0]  

                prediction_probs = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
                print(prediction_probs)
                top_predictions = sorted(prediction_probs, key=lambda x: x[1], reverse=True)[:3]
                st.session_state.predictions = top_predictions
                st.session_state.step = "results"

# Step 2: Show Predictions and Collect Feedback
if st.session_state.step == "results":
    st.write("### Top Predicted Diseases:")
    for idx, (disease_id, prob) in enumerate(st.session_state.predictions, 1):
        disease_info = DISEASE_INFO.get(disease_id, {})
        disease = DISEASE_INFO.get(disease_id, {})
        st.subheader(f"{idx}. {disease_id} ({prob * 100:.2f}%)")
        if disease_id != 'No Finding':
            st.image(disease.get("image"), caption=disease.get("name"), width=450)
            st.subheader("Description")
            st.write(disease.get("description"))
            st.subheader("Symptoms")
            st.write("- " + "\n- ".join(disease.get("symptoms")))

            st.subheader("Causes")
            st.write("- " + "\n- ".join(disease.get("causes")))

            st.subheader("Detailed Causes")
            for cause_type, details in disease.get("causes_details").items():
                st.write(f"**{cause_type}:**")
                st.write("- " + "\n- ".join(details))

            st.subheader("Risk Factors")
            st.write("- " + "\n- ".join(disease.get("risk_factors")))

            st.subheader("Complications")
            st.write("- " + "\n- ".join(disease.get("complications")))

            st.subheader("Prevention")
            st.write("- " + "\n- ".join(disease.get("prevention")))


            st.markdown(
                        """
                        <div style="position: fixed; bottom: 0; right: 0; font-size: 12px; text-align: right; margin: 10px; color: gray;">
                            Information provided by <a href="https://www.clevelandclinic.org" target="_blank">Cleveland Clinic</a>, <a href="https://www.mayoclinic.org/" target="_blank">Mayo Clinic</a>  and other reputable sources. Images are used for educational purposes only.
                        </div>
                        """,
                        unsafe_allow_html=True
                        )                   
                                

    st.subheader("Please provide feedback on the predictions")                  
    st.write(" Please select on of the 2 options and proceed further for the feedback")
    # Feedback buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Correctly Predicted"):

            folder_name = "feedback/correctly_predicted"
            #################################################
            ############# For GCP ##########
            image_name = st.session_state.uploaded_image.name
            with BytesIO() as img_file:
                image = st.session_state.uploaded_image
                image.save(img_file, format="JPEG")
                img_file.seek(0)
                upload_image_to_gcs(bucket, folder_name, img_file, image_name)
                preds = st.session_state.predictions
            
            # Append metadata to JSONL file
            image_metadata = encode_image_to_base64(image)
            instance_data = {
                "image_name": image_name,
                "image_metadata": image_metadata,
                "age": age,
                "gender": gender,
                "predicted_labels": preds
            }
            append_to_jsonl(bucket, folder_name, instance_data)
            st.success(f"Image and metadata stored in '{folder_name}' folder successfully!")


            st.session_state.feedback = "Thank you for your feedback!"
            st.session_state.step = "thank_you"
    with col2:
        if st.button("Incorrect Predictions"):
            st.session_state.step = "incorrect"

# Step 3: Handle Incorrect Predictions
if st.session_state.step == "incorrect":
    st.write("Please select the correct disease prediction:")
    disease_options = list(DISEASE_INFO.keys())
    st.session_state.selected_disease = st.selectbox(
        "Select the correct disease",
        disease_options,
        index=0 if st.session_state.selected_disease is None else disease_options.index(st.session_state.selected_disease),
    )
    if st.button("Confirm Selection"):
        folder_name = "feedback/incorrectly_predicted"
        true_label = st.session_state.selected_disease

        #################################################
        ############# For GCP ##########
        image_name = st.session_state.uploaded_image.name
        image = st.session_state.uploaded_image
        with BytesIO() as img_file:
                image.save(img_file, format="JPEG")
                img_file.seek(0)
                upload_image_to_gcs(bucket, folder_name, img_file, image_name)
            
        # Append metadata to JSONL file
        image_metadata = encode_image_to_base64(image)
        instance_data = {
            "image_name": image_name,
            "image_metadata": image_metadata,
            "age": age,
            "gender": gender,
            "predicted_labels": true_label
        }
        append_to_jsonl(bucket, folder_name, instance_data)
        st.success(f"Image and metadata stored in '{folder_name}' folder successfully!")

        st.session_state.feedback = f"Thank you! As provided in the feedback "
        st.session_state.step = "thank_you"

# Step 4: Thank You Message
if st.session_state.step == "thank_you":
    st.success(st.session_state.feedback)
    #st.write("Your feedback has been recorded. Thank you for helping us improve our model!")
    if st.button("Restart"):
        st.session_state.restart = True
        st.session_state.uploaded_image = None
        st.rerun()
        