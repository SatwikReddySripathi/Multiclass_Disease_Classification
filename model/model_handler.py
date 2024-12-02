import torch
import json
import os
import importlib.util
from torchvision import transforms
from PIL import Image
from ts.torch_handler.base_handler import BaseHandler
import base64
import io

class MultimodalMultilabelHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        self.one_hot_encoding = {
            'No Finding': 0, 'Atelectasis': 1, 'Cardiomegaly': 2, 'Effusion': 3,
            'Infiltration': 4, 'Mass': 5, 'Nodule': 6, 'Pneumonia': 7,
            'Pneumothorax': 8, 'Consolidation': 9, 'Edema': 10, 'Emphysema': 11,
            'Fibrosis': 12, 'Pleural_Thickening': 13, 'Hernia': 14
        }
        self.device = None

    def initialize(self, context):
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")

        # Dynamically import Model.py
        model_file = os.path.join(model_dir, "Model.py")
        spec = importlib.util.spec_from_file_location("Model", model_file)
        model_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model_module)

        # Load model architecture from Model.py
        model_class_name = model_module.__dir__()[0]  # Assuming the first class is the model
        self.model_class = getattr(model_module, model_class_name)

        # Load model weights
        serialized_file = self.manifest['model'].get('serializedFile', None)
        if not serialized_file:
            raise KeyError("'serializedFile' key not found in manifest['model']")
        model_path = os.path.join(model_dir, serialized_file)

        self.model = self.model_class()  # Initialize model
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def preprocess_image(self, image):
        image = Image.open(image).convert('L')  # Convert image to grayscale
        image = self.transform(image)
        return image.unsqueeze(0)

    def encode_demographics(self, gender, age):
        gender_encoded = torch.tensor([1.0, 0.0] if gender.lower() == 'm' else [0.0, 1.0], dtype=torch.float32)
        age_tensor = torch.tensor([float(age)], dtype=torch.float32)
        return torch.cat([gender_encoded, age_tensor]).unsqueeze(0)

    def preprocess(self, data):
        images = []
        demographics = []

        for row in data:
            image_base64 = row.get("data") or row.get("body")
            image_data = base64.b64decode(image_base64)
            image = io.BytesIO(image_data)
            image_tensor = self.preprocess_image(image)
            images.append(image_tensor)

            gender = row.get("gender", "m")
            age = row.get("age", 0)
            demo_tensor = self.encode_demographics(gender, age)
            demographics.append(demo_tensor)

        return torch.cat(images), torch.cat(demographics)

    def inference(self, image_tensor, demographics_tensor):
        print("Image tensor shape:", image_tensor.shape)
        print("Demographics tensor shape:", demographics_tensor.shape)
        with torch.no_grad():
            output = self.model(image_tensor.to(self.device), demographics_tensor.to(self.device))
            probabilities = torch.sigmoid(output)
        return probabilities

    def map_probabilities_to_labels(self, probabilities, threshold=0.3):
        index_to_class = {v: k for k, v in self.one_hot_encoding.items()}
        predicted_labels = {
            index_to_class[i]: float(prob)
            for i, prob in enumerate(probabilities.squeeze().tolist())
            if prob >= threshold
        }
        return predicted_labels

    def postprocess(self, inference_output):
        results = []
        for output in inference_output:
            predicted_labels = self.map_probabilities_to_labels(output)
            results.append(predicted_labels)
        return results

    def handle(self, data, context):
        try:
            image_tensor, demographics_tensor = self.preprocess(data)
            probabilities = self.inference(image_tensor, demographics_tensor)
            return self.postprocess(probabilities)
        except Exception as e:
            print("Error during Prediction:", str(e))
            raise e
