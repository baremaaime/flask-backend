from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import timm
import torchvision.transforms as transforms
from PIL import Image
import io
import os

# Initialize Flask app
app = Flask(__name__)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model("efficientnet_b0", pretrained=False)
num_features = model.classifier.in_features
model.classifier = nn.Linear(num_features, 4)  # 4 maize disease classes

# Load trained weights
model.load_state_dict(torch.load("efficientnet_b0_maize_disease.pth", map_location=device))
model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # Imagenet normalization
        std=[0.229, 0.224, 0.225]
    ),
])

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    img_bytes = file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)

    return jsonify({"prediction": int(predicted.item())})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)