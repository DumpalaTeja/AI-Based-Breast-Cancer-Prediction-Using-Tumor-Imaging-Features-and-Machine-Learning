# utils.py
import io
import base64
from typing import Tuple, Optional
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T
import cv2

IMG_SIZE = 224
normalize = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
transform = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), normalize])

def load_image_bytes(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    return transform(img).unsqueeze(0)  # 1 x C x H x W

def tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    arr = img_tensor.detach().cpu().numpy()
    arr = np.transpose(arr, (1,2,0))
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def overlay_heatmap(orig_pil: Image.Image, cam: np.ndarray, alpha=0.45):
    # cam: float HxW in 0..1
    heatmap_uint8 = np.uint8(255 * cam)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    orig = np.array(orig_pil.resize((cam.shape[1], cam.shape[0])))
    overlay = cv2.addWeighted(orig, 1-alpha, heatmap_color, alpha, 0)
    return Image.fromarray(overlay)

def pil_to_base64(pil_img: Image.Image) -> str:
    buff = io.BytesIO()
    pil_img.save(buff, format="PNG")
    return base64.b64encode(buff.getvalue()).decode("utf-8")
