# backend_flask/streamlit_image_app.py
"""
Streamlit image app — updated:
- Risk score = malignant probability only (0-100%)
- Out-of-domain detection (confidence, entropy, texture, H&E-ish color heuristic)
- If OOD => show friendly message and DO NOT produce Grad-CAM
- Full-resolution overlay download + improved UI
"""

import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import io
import os
import json
from typing import Tuple

# ---------------------------
# CONFIG (tweakable)
# ---------------------------
MODEL_PATH = os.path.join("..", "models", "breast_image_model.pth")
METRICS_PATH = os.path.join("..", "metrics", "metrics.json")
IMG_DISPLAY_MAX_WIDTH = 560

# OOD thresholds (exposed in UI to tweak if needed)
DEFAULT_CONFIDENCE_THRESHOLD = 0.60   # minimum max-prob to accept
DEFAULT_ENTROPY_THRESHOLD = 0.90      # maximum entropy allowed (higher = uncertain)
DEFAULT_LAPLACIAN_THRESHOLD = 60.0    # minimum Laplacian variance for textured histology
DEFAULT_HE_FRACTION = 0.01            # minimum fraction of H&E-like pixels

CLASS_NAMES = ["benign", "malignant"]

# ---------------------------
# Grad-CAM utility
# ---------------------------
class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        # hooks
        def forward_hook(module, inp, out):
            self.activations = out.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        self.model.zero_grad()
        out = self.model(input_tensor)
        if class_idx is None:
            class_idx = int(out.argmax(dim=1).item())
        score = out[0, class_idx]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM failed: missing gradients/activations")

        grads = self.gradients[0] if self.gradients.dim() == 4 else self.gradients
        acts = self.activations[0] if self.activations.dim() == 4 else self.activations

        weights = grads.mean(dim=(1,2), keepdim=True)
        cam = (weights * acts).sum(dim=0).cpu().numpy()
        cam = np.maximum(cam, 0)
        if cam.max() == 0:
            return np.zeros_like(cam, dtype=np.float32)
        cam = cam - cam.min()
        cam = cam / cam.max()
        return cam.astype(np.float32)

# ---------------------------
# Load model & transforms
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_model_and_utils(model_path=MODEL_PATH):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # skeleton: resnet18 with 2 outputs
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")

    sd = torch.load(model_path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    try:
        model.load_state_dict(sd)
    except Exception:
        loaded = torch.load(model_path, map_location=device)
        if isinstance(loaded, torch.nn.Module):
            model = loaded
        else:
            raise

    model = model.to(device).eval()

    # try pick target conv (resnet typical)
    target_layer = None
    if hasattr(model, "layer4"):
        try:
            target_layer = model.layer4[-1].conv2
        except Exception:
            target_layer = None

    if target_layer is None:
        for m in reversed(list(model.modules())):
            if isinstance(m, torch.nn.Conv2d):
                target_layer = m
                break

    gradcam = GradCAM(model, target_layer) if target_layer is not None else None

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    return {"model": model, "device": device, "gradcam": gradcam, "transform": transform}

def load_metrics(metrics_path=METRICS_PATH):
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as fh:
                data = json.load(fh)
            return {k: data.get(k, "N/A") for k in ("accuracy","precision","recall","f1")}
        except Exception:
            pass
    return {"accuracy":"N/A","precision":"N/A","recall":"N/A","f1":"N/A"}

# ---------------------------
# Visual / image helpers
# ---------------------------
def pil_to_bytes(pil_img: Image.Image, fmt="PNG") -> bytes:
    buff = io.BytesIO()
    pil_img.save(buff, format=fmt)
    buff.seek(0)
    return buff.read()

def upscale_cam_to_original(cam: np.ndarray, orig_size: Tuple[int,int]) -> np.ndarray:
    # cam: h x w (small); orig_size: (width, height)
    target_w, target_h = orig_size
    cam_uint8 = np.uint8(cam * 255)
    resized = cv2.resize(cam_uint8, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    resized = resized.astype(np.float32) / 255.0
    return resized

def overlay_heatmap_on_original(orig_pil: Image.Image, cam_resized: np.ndarray, alpha=0.45) -> Image.Image:
    orig_np = np.array(orig_pil.convert("RGB"))
    heatmap_uint8 = np.uint8(cam_resized * 255)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(orig_np, 1.0 - alpha, heatmap_color, alpha, 0)
    return Image.fromarray(overlay)

# ---------------------------
# OOD detection heuristics
# ---------------------------
def laplacian_variance(img_pil: Image.Image) -> float:
    gray = cv2.cvtColor(np.array(img_pil.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def softmax_entropy(probs: np.ndarray) -> float:
    # probs: 1D np array of class probabilities
    eps = 1e-12
    ent = -np.sum(probs * np.log(probs + eps))
    # normalize entropy to [0, log(n)], we can keep raw
    return float(ent)

def he_fraction(img_pil: Image.Image) -> float:
    """
    Simple heuristic: estimate fraction of pixels that fall into H&E-ish hue/saturation ranges.
    This is not perfect but helps reject photos with no histology stain.
    """
    img = np.array(img_pil.convert("RGB"))
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h = hsv[:,:,0]  # 0..179
    s = hsv[:,:,1]
    v = hsv[:,:,2]
    # rough ranges:
    # purple/hematoxylin-ish: hue ~120-150 (in OpenCV scale) with moderate saturation
    mask_h = ((h >= 115) & (h <= 150) & (s >= 30))
    # pink/eosin-ish: hue around 160-10 (wrap-around), handle two ranges
    mask_p1 = ((h >= 150) & (h <= 179) & (s >= 20))
    mask_p2 = ((h >= 0) & (h <= 10) & (s >= 20))
    mask_pink = mask_p1 | mask_p2
    mask = mask_h | mask_pink
    frac = float(mask.sum()) / (img.shape[0] * img.shape[1])
    return frac

def is_out_of_domain(img_pil: Image.Image, probs: np.ndarray,
                     conf_thresh=DEFAULT_CONFIDENCE_THRESHOLD,
                     entropy_thresh=DEFAULT_ENTROPY_THRESHOLD,
                     lap_thresh=DEFAULT_LAPLACIAN_THRESHOLD,
                     he_frac_thresh=DEFAULT_HE_FRACTION) -> Tuple[bool,str]:
    """
    Returns (is_ood, reason_string).
    Rules:
      - if max prob < conf_thresh  -> OOD (low confidence)
      - if entropy > entropy_thresh -> OOD (uncertain)
      - if laplacian variance < lap_thresh -> OOD (too smooth)
      - if H&E fraction < he_frac_thresh -> OOD (not stained histology)
    """
    maxp = float(probs.max())
    ent = softmax_entropy(probs)
    lap = laplacian_variance(img_pil)
    he_frac = he_fraction(img_pil)

    if maxp < conf_thresh:
        return True, f"Low model confidence ({maxp:.2f} < {conf_thresh:.2f})"
    if ent > entropy_thresh:
        return True, f"Model uncertain (entropy {ent:.2f} > {entropy_thresh:.2f})"
    if lap < lap_thresh:
        return True, f"Image texture too smooth (laplacian {lap:.1f} < {lap_thresh:.1f})"
    if he_frac < he_frac_thresh:
        return True, f"H&E-like color fraction too low ({he_frac:.3f} < {he_frac_thresh:.3f})"
    return False, ""

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Breast Histology Classifier (Image)", layout="wide")
st.title("🧪 Breast Histology Classifier — Image (Benign vs Malignant)")

# load model
try:
    resources = load_model_and_utils()
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

MODEL = resources["model"]
DEVICE = resources["device"]
GRADCAM = resources["gradcam"]
TRANSFORM = resources["transform"]
metrics = load_metrics()

# Sidebar: thresholds & metrics (user can tweak)
with st.sidebar:
    st.markdown("## Model validation")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", metrics.get("accuracy","N/A"))
    c2.metric("Precision", metrics.get("precision","N/A"))
    c1, c2 = st.columns(2)
    c1.metric("Recall", metrics.get("recall","N/A"))
    c2.metric("F1", metrics.get("f1","N/A"))
    st.markdown("---")
    st.write("### OOD / safety thresholds (tweak if necessary)")
    conf_thresh = st.slider("Min confidence (max prob)", 0.40, 0.95, float(DEFAULT_CONFIDENCE_THRESHOLD), step=0.01)
    ent_thresh = st.slider("Max entropy", 0.1, 1.6, float(DEFAULT_ENTROPY_THRESHOLD), step=0.01)
    lap_thresh = st.slider("Min laplacian variance", 0.0, 500.0, float(DEFAULT_LAPLACIAN_THRESHOLD), step=1.0)
    he_frac_thresh = st.slider("Min H&E color fraction", 0.0, 0.20, float(DEFAULT_HE_FRACTION), step=0.005)
    st.markdown("---")
    st.write(f"Model device: **{DEVICE}**")
    st.write("Message on OOD: If an image is rejected, no Grad-CAM or prediction will be shown.")
    st.markdown("---")
    st.write("Tone: A — Friendly guidance will be shown for invalid images.")

# Main layout
left, right = st.columns([1.1, 1])

with left:
    st.subheader("Upload histopathology image")
    uploaded = st.file_uploader("Choose image (jpg, png, jpeg, tiff). Recommended: H&E histology patch.", type=["jpg","jpeg","png","tif","tiff"])
    if uploaded:
        try:
            pil = Image.open(uploaded).convert("RGB")
        except Exception as e:
            st.error("Could not open image: " + str(e))
            pil = None

        if pil:
            st.image(pil, caption="Uploaded image (preview)", width=min(IMG_DISPLAY_MAX_WIDTH, pil.width))
            lap = laplacian_variance(pil)
            he_frac = he_fraction(pil)
            st.markdown("**Image quick stats**")
            st.write(f"- Laplacian variance (texture): {lap:.2f}")
            st.write(f"- H&E-like color fraction: {he_frac:.4f}")

with right:
    st.subheader("Prediction & explanation")
    if not uploaded:
        st.info("Upload an image to run prediction.")
    else:
        with st.spinner("Running model and safety checks..."):
            # preprocess and forward
            input_tensor = TRANSFORM(pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                logits = MODEL(input_tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]  # e.g. [p_benign, p_malignant]
                pred_idx = int(probs.argmax())
                pred_label = CLASS_NAMES[pred_idx]
                # Malignant probability ONLY:
                malignant_prob = float(probs[CLASS_NAMES.index("malignant")])
                # entropy for display
                ent = softmax_entropy(probs)

            # OOD check using current thresholds
            ood, reason = is_out_of_domain(pil, probs, conf_thresh, ent_thresh, lap_thresh, he_frac_thresh)

            if ood:
                # Tone A: friendly guidance
                st.error("🔎 This image does not look like a breast histopathology patch.")
                st.write("Please upload an H&E-stained histology patch (biopsy patch).")
                st.write(f"Reason: {reason}")
                st.warning("No prediction or Grad-CAM will be shown for out-of-domain images.")
            else:
                # Show malignant prob as risk score and model confidence
                st.metric(label="Prediction", value=pred_label.upper(), delta=f"{malignant_prob*100:.2f}% (malignant prob)")
                st.write(f"Risk score (malignant probability): **{malignant_prob*100:.2f}%**")
                st.write(f"Model confidence (max class prob): **{float(probs.max())*100:.2f}%**")
                st.write(f"Softmax entropy: **{ent:.3f}**")

                # Grad-CAM (only if allowed)
                if GRADCAM is None:
                    st.warning("Grad-CAM not available for this model.")
                else:
                    try:
                        cam = GRADCAM.generate_cam(input_tensor, class_idx=pred_idx)  # small h x w
                        orig_w, orig_h = pil.size
                        cam_up = upscale_cam_to_original(cam, (orig_w, orig_h))
                        overlay = overlay_heatmap_on_original(pil, cam_up, alpha=0.45)
                        st.image(overlay, caption="Grad-CAM overlay (preview)", width=min(IMG_DISPLAY_MAX_WIDTH, orig_w))

                        # download full-res overlay
                        overlay_bytes = pil_to_bytes(overlay, fmt="PNG")
                        st.download_button("Download full-resolution overlay (PNG)", data=overlay_bytes, file_name="overlay_fullres.png", mime="image/png")
                    except Exception as e:
                        st.error("Grad-CAM generation failed: " + str(e))

# footer
st.markdown("---")
st.caption("Educational demo — not for clinical use.")
