import torch
import torch.nn as nn
from torchvision import models, transforms
from flask import Flask, request, jsonify
from PIL import Image
from flask_cors import CORS  # Import CORS
import os
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS to allow cross-origin requests

# Set device (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the pre-trained ResNet152 model
model = models.resnet152(pretrained=True)

# Modify the final fully connected layer for 5 classes
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, 5)  # 5 output classes for diabetic retinopathy stages
)

# Load the trained model weights
model.load_state_dict(torch.load(r'F:\UI_DR\backend\model\Resnet_152_rev2_final1.pth', map_location=device))
model = model.to(device)
model.eval()  # Set the model to evaluation mode

# Define the transformation for input images
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

# Create a route for model prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Handle file upload
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        # Open the image file
        img = Image.open(file.stream)

        # Apply the transformation to the image
        img = transform(img).unsqueeze(0).to(device)

        # Get the model prediction
        with torch.no_grad():
            outputs = model(img)
            _, predicted = torch.max(outputs, 1)

        # Map the prediction to the corresponding class
        class_names = ['Mild', 'Moderate','No_DR', 'Proliferate_DR', 'Severe']
        predicted_class = class_names[predicted.item()]

        # Calculate confidence scores for each class
        confidence_scores = torch.nn.functional.softmax(outputs, dim=1).squeeze().cpu().numpy()

        # Convert numpy float32 to Python float to ensure JSON serializability
        confidence_dict = {class_names[i]: float(confidence_scores[i]) for i in range(len(class_names))}

        return jsonify({
            'predicted_class': predicted_class,
            'confidence_scores': confidence_dict
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
