
import os
import json
import uuid
import logging
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

from gradcam import GradCAM

# 1. Application settings
app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "breast_image_model.pth"
METRICS_PATH = BASE_DIR / "metrics" / "metrics.json"
UPLOAD_FOLDER = BASE_DIR / "uploads"
HEATMAP_FOLDER = BASE_DIR / "static" / "heatmaps"

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}
MAX_IMAGE_SIZE = 5 * 1024 * 1024  # 5 MB

app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
HEATMAP_FOLDER.mkdir(parents=True, exist_ok=True)


# 2. Load model
device = torch.device("cpu")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

try:
    state_dict = torch.load(
        str(MODEL_PATH),
        map_location=device,
        weights_only=True
    )
except TypeError:
    state_dict = torch.load(
        str(MODEL_PATH),
        map_location=device
    )

model.load_state_dict(state_dict)
model.to(device)
model.eval()

gradcam = GradCAM(
    model,
    target_layer=model.layer4[1].conv2
)


# 3. Load evaluation metrics
metrics = {
    "accuracy": 0,
    "precision": 0,
    "recall": 0,
    "f1": 0
}

if METRICS_PATH.exists():
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            saved_metrics = json.load(file)

        for key in metrics:
            metrics[key] = saved_metrics.get(key, 0)

    except (OSError, ValueError, TypeError) as exc:
        logger.warning("Could not load metrics: %s", exc)


# 4. Image preprocessing
# Keep this consistent with your model's training preprocessing.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# 5. Check image file type
def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


# 6. Predict image
def predict_image(image_path):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.inference_mode():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)[0]

    predicted_class = int(probabilities.argmax().item())

    # Assumes class 0 = Benign and class 1 = Malignant.
    label = "Malignant" if predicted_class == 1 else "Benign"

    confidence = probabilities[predicted_class].item() * 100
    malignant_score = probabilities[1].item() * 100

    return (
        image_tensor,
        predicted_class,
        label,
        confidence,
        malignant_score
    )


# 7. Main page
@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    heatmap_path = None
    error = None

    if request.method == "POST":
        image_file = request.files.get("image")

        if not image_file or not image_file.filename:
            error = "Please select an image."

        elif not allowed_file(image_file.filename):
            error = "Please upload a PNG, JPG, JPEG or BMP image."

        else:
            image_id = uuid.uuid4().hex
            safe_name = secure_filename(image_file.filename)
            extension = Path(safe_name).suffix.lower()

            if extension not in {
                ".png", ".jpg", ".jpeg", ".bmp"
            }:
                error = "Invalid image filename."

            else:
                image_path = UPLOAD_FOLDER / f"{image_id}{extension}"
                heatmap_file = HEATMAP_FOLDER / f"{image_id}.png"

                try:
                    image_file.save(str(image_path))

                    # Verify that the file is a valid image.
                    with Image.open(image_path) as image:
                        image.verify()

                    (
                        image_tensor,
                        predicted_class,
                        label,
                        confidence,
                        malignant_score
                    ) = predict_image(image_path)

                    # Generate Grad-CAM visualization.
                    heatmap = gradcam.generate(
                        image_tensor,
                        predicted_class,
                        str(image_path)
                    )

                    if heatmap is None or not cv2.imwrite(
                        str(heatmap_file), heatmap
                    ):
                        raise ValueError(
                            "Could not generate the visualization."
                        )

                    result = {
                        "label": label,
                        "confidence": round(confidence, 2),
                        "malignant_probability": round(
                            malignant_score, 2
                        ),
                        "accuracy": metrics["accuracy"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1": metrics["f1"],
                        "disclaimer": (
                            "Experimental AI output only. "
                            "Not a medical diagnosis."
                        )
                    }

                    heatmap_path = (
                        f"/static/heatmaps/{image_id}.png"
                    )

                except (
                    UnidentifiedImageError,
                    OSError,
                    ValueError,
                    RuntimeError
                ):
                    logger.exception("Image processing failed.")
                    error = (
                        "Unable to process this image. "
                        "Please try another valid image."
                    )

                finally:
                    if image_path.exists():
                        try:
                            image_path.unlink()
                        except OSError:
                            logger.warning(
                                "Could not remove temporary image."
                            )

    return render_template(
        "index.html",
        result=result,
        heatmap_path=heatmap_path,
        error=error
    )


# 8. Handle images larger than 5 MB
@app.errorhandler(413)
def file_too_large(error):
    return render_template(
        "index.html",
        result=None,
        heatmap_path=None,
        error="Image is too large. Maximum size is 5 MB."
    ), 413


# 9. Start application
if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
