import streamlit as st
from PIL import Image
import numpy as np
import torch
import json
import os
from utils_gradcam import generate_heatmap_overlay
from image_validator import is_valid_histology_image


# Load model
MODEL_PATH = r"C:\Users\dumpa\OneDrive\Desktop\ML\breast-cancer-ai\backend_flask\models\breast_image_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = torch.load(MODEL_PATH, map_location=DEVICE)
model.eval()

# Load metrics
METRICS_PATH = "saved_models/metrics.json"
metrics = {}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

st.set_page_config(page_title="Breast Cancer Classifier", layout="wide")

st.title("🔬 Breast Cancer Histopathology Classifier")

uploaded_image = st.file_uploader("Upload H&E Histopathology Image", type=["jpg", "jpeg", "png"])

predict_btn = st.button("🔍 Predict")

col_main, col_metrics = st.columns([0.7, 0.3])

if uploaded_image:
    img = Image.open(uploaded_image).convert("RGB")
    col_main.image(img, caption="Uploaded Image", use_column_width=True)

# Sidebar Metrics only if available
with col_metrics:
    if metrics:
        st.markdown("### 📊 Model Validation Metrics")
        st.markdown(f"""
        **Accuracy**: {metrics.get('accuracy', 'N/A'):.4f}  
        **Precision**: {metrics.get('precision', 'N/A'):.4f}  
        **Recall**: {metrics.get('recall', 'N/A'):.4f}  
        **F1 Score**: {metrics.get('f1_score', 'N/A'):.4f}
        """)
        st.markdown("---")

if predict_btn:
    if not uploaded_image:
        st.warning("Please upload an image first!")
    else:
        # Validate histology image before prediction
        valid, score = is_valid_histology_image(np.array(img))

        if not valid:
            st.error(f"Input rejected — appears out-of-domain.\nH&E color ratio: {score:.2f}")
            st.info("Please upload a real H&E histology tissue patch.")
        else:
            st.success("✅ Histopathology features confirmed. Processing image...")

            # Inference
            transform = torch.nn.Sequential(
                torch.nn.Upsample(size=(224, 224), mode="bilinear", align_corners=False)
            )

            input_tensor = transform(torch.tensor(np.array(img)).permute(2,0,1).float().unsqueeze(0)) / 255.0
            input_tensor = input_tensor.to(DEVICE)

            with torch.no_grad():
                output = torch.softmax(model(input_tensor), dim=1)
                confidence, pred_class = torch.max(output, dim=1)

            pred_class = "Malignant" if pred_class.item() == 1 else "Benign"
            confidence = confidence.item() * 100

            # Generate Grad-CAM
            heatmap_img = generate_heatmap_overlay(model, input_tensor, img)

            col_main.subheader(f"🧠 Prediction: **{pred_class}**")
            col_main.write(f"Model Confidence: **{confidence:.2f}%**")
            col_main.image(heatmap_img, caption="Tumor Region Highlight (Grad-CAM)", use_column_width=True)

            col_main.download_button(
                "⬇ Download Explanation Image",
                data=heatmap_img.tobytes(),
                file_name="grad_cam_result.png",
                mime="image/png"
            )
