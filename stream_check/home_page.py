import streamlit as st

# Title and description
st.title("ThorAIx - Disease Prediction")
st.write("""
### Overview:
Welcome to **ThorAIx**, a cutting-edge application for X-ray-based disease prediction. 
This application leverages advanced machine learning and computer vision techniques to assist healthcare professionals 
in diagnosing thoracic diseases accurately and efficiently.

### Features:
- Upload an X-ray image and get predictions for potential thoracic diseases.
- Validates the uploaded image to ensure it's suitable for prediction (e.g., brightness check).
- Provides detailed explanations of the predicted diseases.
- Allows feedback for model improvement.

### How It Works:
1. Go to the **Disease Prediction** page.
2. Upload an X-ray image and provide demographic details (age, gender).
3. View the predicted diseases and provide feedback for further improvements.

---

### Why ThorAIx?
ThorAIx aims to assist medical professionals by providing fast and accurate predictions, reducing diagnostic times, 
and enabling better patient outcomes.

---

Click on the **Disease Prediction** tab to get started!
""")
