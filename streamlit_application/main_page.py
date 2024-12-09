import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.nn.functional import sigmoid
from custom_resnet import load_model
from custom_resnet import preprocess_image

# Disease Mapping and Information
DISEASE_INFO = {
    0: {
        "name": "No Finding",
        "description": "No abnormalities or findings detected in the X-ray.",
        "symptoms": ["None"],
        "image": "https://via.placeholder.com/224x224.png?text=No+Findings",  # Placeholder image
    },
    1: {
        "name": "Atelectasis",
        "description": "Partial or complete collapse of the lung.",
        "symptoms": ["Shortness of breath", "Coughing", "Chest pain"],
        "image": "https://upload.wikimedia.org/wikipedia/commons/a/a4/Atelectasis.jpg",
    },
    2: {
        "name": "Cardiomegaly",
        "description": "An enlarged heart, often caused by high blood pressure or heart disease.",
        "symptoms": ["Fatigue", "Shortness of breath", "Swelling in the legs"],
        "image": "https://upload.wikimedia.org/wikipedia/commons/b/b0/Cardiomegaly.jpg",
    },
    3: {
        "name": "Effusion",
        "description": "Accumulation of fluid in body cavities, commonly in the pleural cavity of the lungs.",
        "symptoms": ["Difficulty breathing", "Coughing", "Chest discomfort"],
        "image": "https://upload.wikimedia.org/wikipedia/commons/4/4b/Effusion.jpg",
    },
    4: {
        "name": "Infiltration",
        "description": "Substance accumulation in lung tissues, often associated with infections or inflammation.",
        "symptoms": ["Cough", "Fever", "Shortness of breath"],
        "image": "https://via.placeholder.com/224x224.png?text=Infiltration",  # Placeholder image
    },
    5: {
        "name": "Mass",
        "description": "Abnormal tissue growth in the lungs, which can be benign or malignant.",
        "symptoms": ["Coughing", "Chest pain", "Weight loss"],
        "image": "https://via.placeholder.com/224x224.png?text=Mass",  # Placeholder image
    },
    6: {
        "name": "Nodule",
        "description": "Small growths or lumps in the lungs, often benign but may require monitoring.",
        "symptoms": ["Usually asymptomatic", "Occasionally cough or shortness of breath"],
        "image": "https://via.placeholder.com/224x224.png?text=Nodule",  # Placeholder image
    },
    7: {
        "name": "Pneumonia",
        "description": "Infection that inflames the air sacs in one or both lungs.",
        "symptoms": ["Cough with phlegm", "Fever", "Chills", "Difficulty breathing"],
        "image": "https://upload.wikimedia.org/wikipedia/commons/f/fd/Pneumonia.jpg",
    },
    8: {
        "name": "Pneumothorax",
        "description": "Collapsed lung due to air leakage into the pleural space.",
        "symptoms": ["Sudden chest pain", "Shortness of breath"],
        "image": "https://upload.wikimedia.org/wikipedia/commons/3/39/Pneumothorax.jpg",
    },
    9: {
        "name": "Consolidation",
        "description": "Lung tissue that has filled with liquid instead of air, typically due to pneumonia.",
        "symptoms": ["Cough", "Difficulty breathing", "Chest discomfort"],
        "image": "https://via.placeholder.com/224x224.png?text=Consolidation",  # Placeholder image
    },
    10: {
        "name": "Edema",
        "description": "Excess fluid trapped in the lungs, commonly seen in heart failure.",
        "symptoms": ["Shortness of breath", "Rapid breathing", "Wheezing"],
        "image": "https://upload.wikimedia.org/wikipedia/commons/5/50/Pulmonary_Edema.jpg",
    },
    11: {
        "name": "Emphysema",
        "description": "A chronic lung disease that causes damage to the alveoli (air sacs).",
        "symptoms": ["Shortness of breath", "Chronic cough", "Wheezing"],
        "image": "https://upload.wikimedia.org/wikipedia/commons/9/94/Emphysema.jpg",
    },
    12: {
        "name": "Fibrosis",
        "description": "Scarring or thickening of lung tissue, leading to stiffness and difficulty breathing.",
        "symptoms": ["Chronic dry cough", "Fatigue", "Shortness of breath"],
        "image": "https://upload.wikimedia.org/wikipedia/commons/4/4e/Fibrosis.jpg",
    },
    13: {
        "name": "Pleural Thickening",
        "description": "Thickening of the pleural lining around the lungs, often due to chronic inflammation.",
        "symptoms": ["Chest pain", "Shortness of breath"],
        "image": "https://via.placeholder.com/224x224.png?text=Pleural+Thickening",  # Placeholder image
    },
    14: {
        "name": "Hernia",
        "description": "Protrusion of the abdominal contents into the chest cavity through the diaphragm.",
        "symptoms": ["Chest discomfort", "Abdominal pain", "Difficulty breathing"],
        "image": "https://via.placeholder.com/224x224.png?text=Hernia",  # Placeholder image
    },
}

# Top Doctors in Boston
DOCTORS_INFO = [
    {"name": "Dr. John Smith", "specialty": "Pulmonology", "phone": "+1 617-555-1234"},
    {"name": "Dr. Emily Johnson", "specialty": "Cardiology", "phone": "+1 617-555-5678"},
    {"name": "Dr. Michael Brown", "specialty": "Radiology", "phone": "+1 617-555-9876"},
]
"""
def main_page():

    st.title("ThorAIx - Disease Prediction")
    st.write("Upload an X-ray image and provide demographic details.")

    # Input for gender
    gender = st.selectbox("Select Gender", ["Male", "Female"])
    gender_encoded = [1.0, 0.0] if gender == "Male" else [0.0, 1.0]

    # Input for age
    age = st.number_input("Enter Age", min_value=0, max_value=120, step=1, value=25)

    # Image upload
    uploaded_file = st.file_uploader("Upload an X-ray Image", type=["png", "jpg", "jpeg"])

    if st.button("Predict"):
        if uploaded_file:
            try:
                # Load the image
                image = Image.open(uploaded_file)
                processed_image = preprocess_image(image)
                demographics = torch.tensor([[age] + gender_encoded], dtype=torch.float32)

                # Load the model
                model = load_model("models_best_model-latest.pt")

                # Perform prediction
                with torch.no_grad():
                    prediction = model(processed_image, demographics)
                    prediction_probs = sigmoid(prediction).squeeze().tolist()

                # Get top 3 predictions
                top_predictions = sorted(enumerate(prediction_probs, start=1), key=lambda x: x[1], reverse=True)[:3]

                # Display top 3 predictions
                st.write("### Top 3 Predicted Diseases:")
                for idx, (disease_id, probability) in enumerate(top_predictions, start=1):
                    disease_name = DISEASE_INFO.get(disease_id, {}).get("name", "Unknown")
                    st.write(f"**{idx}. {disease_name}** - Probability: {probability:.2f}")
                    if st.button(f"Learn More About {disease_name}", key=f"learn_more_{disease_id}"):
                        # Update session state to navigate to details page
                        st.session_state.page = "details"
                        st.session_state.selected_disease = disease_id
                        st.experimental_rerun()
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please upload an image.")
            """
def main_page(toggle_expansion):
    st.title("ThorAIx - Disease Prediction")
    st.write("Upload an X-ray image and provide demographic details.")

    # Input for gender
    gender = st.selectbox("Select Gender", ["Male", "Female"])
    gender_encoded = [1.0, 0.0] if gender == "Male" else [0.0, 1.0]

    # Input for age
    age = st.number_input("Enter Age", min_value=0, max_value=120, step=1, value=25)

    # Upload an image
    uploaded_file = st.file_uploader("Upload an X-ray Image", type=["png", "jpg", "jpeg"])

    if st.button("Predict"):
        if uploaded_file:
            try:
                # Process the uploaded image
                image = Image.open(uploaded_file)
                processed_image = preprocess_image(image)
                demographics = torch.tensor([[age] + gender_encoded], dtype=torch.float32)

                # Load the model
                model = load_model("models_best_model-latest.pt")

                # Perform prediction
                with torch.no_grad():
                    prediction = model(processed_image, demographics)
                    prediction_probs = sigmoid(prediction).squeeze().tolist()

                # Get top 3 predictions
                top_predictions = sorted(enumerate(prediction_probs, start=0), key=lambda x: x[1], reverse=True)[:3]

                st.write("### Top 3 Predicted Diseases:")
                for idx, (disease_id, probability) in enumerate(top_predictions, start=1):
                    disease_name = DISEASE_INFO.get(disease_id, {}).get("name", "Unknown")
                    st.write(f"**{idx}. {disease_name}** - Probability: {probability:.2f}")

                    # Add a "Learn More" or "Show Less" button
                    if st.button(
                        "Learn More" if st.session_state.expanded_disease != disease_id else "Show Less",
                        key=f"learn_more_{disease_id}"
                    ):
                        toggle_expansion(disease_id)

                    # Show detailed information if expanded
                    if st.session_state.expanded_disease == disease_id:
                        disease = DISEASE_INFO.get(disease_id, {})
                        st.image(disease.get("image"), caption=disease.get("name"), use_column_width=True)
                        st.subheader("Description")
                        st.write(disease.get("description"))
                        st.subheader("Symptoms")
                        st.write(", ".join(disease.get("symptoms")))
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please upload an image.")
