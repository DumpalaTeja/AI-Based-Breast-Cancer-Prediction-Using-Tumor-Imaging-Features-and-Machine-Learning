from flask import Flask, render_template, request
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import json
from gradcam import GradCAM

app = Flask(__name__)

# ================== PATH SETTINGS ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "breast_image_model.pth")
METRICS_PATH = os.path.join(BASE_DIR, "metrics", "metrics.json")

print("Model Path:", MODEL_PATH)

# ================== LOAD METRICS ==================
metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

# ================== LOAD MODEL ==================
device = torch.device("cpu")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

gradcam = GradCAM(model, target_layer=model.layer4[1].conv2)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def get_risk_level(confidence):
    if confidence >= 75:
        return "High Risk"
    if confidence >= 50:
        return "Moderate Risk"
    return "Low Risk"

# ================== FRONTEND ROUTE ==================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    heatmap_path = None

    if request.method == "POST":
        image_file = request.files.get("image")
        if not image_file or image_file.filename == "":
            return render_template("index.html", error="Upload a valid image")

        uploads_folder = os.path.join(BASE_DIR, "uploads")
        os.makedirs(uploads_folder, exist_ok=True)

        img_path = os.path.join(uploads_folder, image_file.filename)
        image_file.save(img_path)

        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.softmax(output, dim=1)
            pred_class = torch.argmax(prob).item()
            confidence = prob[0][pred_class].item() * 100

        label = "Malignant" if pred_class == 1 else "Benign"
        risk = get_risk_level(confidence)

        heatmap = gradcam.generate(img_tensor, pred_class, img_path)
        heatmap_name = "heatmap.png"
        heatmap_path = os.path.join("static", heatmap_name)

        cv2_img = heatmap
        import cv2
        cv2.imwrite(os.path.join(BASE_DIR, heatmap_path), cv2_img)

        result = {
            "label": label,
            "confidence": round(confidence, 2),
            "risk": risk,
            "accuracy": metrics.get("accuracy", 0),
            "precision": metrics.get("precision", 0),
            "recall": metrics.get("recall", 0),
            "f1": metrics.get("f1", 0),
        }

    return render_template("index.html", result=result, heatmap_path=heatmap_path)

if __name__ == "__main__":
    app.run(debug=True)
