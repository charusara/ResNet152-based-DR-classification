import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

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
model_path = r'F:/UI_DR/backend/model/Resnet_152_rev2_final1.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()  # Set the model to evaluation mode

# Define image transformations (same as training preprocessing)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize with ImageNet stats
])

# Class names corresponding to model output
class_names = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']


def infer_image(image_path):
    """
    Perform model inference on an input image.

    Args:
        image_path (str): Path to the image file.

    Returns:
        dict: A dictionary containing the predicted class and confidence scores.
    """
    try:
        # Open and preprocess the image
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)

        # Map the predicted label to class name
        predicted_class = class_names[predicted.item()]

        # Calculate confidence scores
        confidence_scores = torch.nn.functional.softmax(outputs, dim=1).squeeze().cpu().numpy()
        confidence_dict = {class_names[i]: float(confidence_scores[i]) for i in range(len(class_names))}

        return {
            'predicted_class': predicted_class,
            'confidence_scores': confidence_dict
        }

    except Exception as e:
        print(f"Error during inference: {e}")
        return {'error': str(e)}


# Example usage
if __name__ == "__main__":
    # Provide the path to the image you want to test
    test_image_path = r"F:\UI_DR\Images for prediction\sev6.jpg"  # Replace with the actual image path

    if os.path.exists(test_image_path):
        result = infer_image(test_image_path)
        print("Prediction Results:")
        print(result)
    else:
        print("Error: Image path does not exist.")

