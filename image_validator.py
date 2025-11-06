import numpy as np
from PIL import Image

def is_valid_histology_image(image, threshold=0.72):
    """
    Check if uploaded image resembles H&E histopathology tissue.
    Uses HSV purple-pink dominance ratio.
    """
    img = image.convert("RGB")
    hsv = np.array(img.convert("HSV"))

    H = hsv[:, :, 0]           # Hue channel
    histology_pixels = np.logical_or((H >= 260), (H <= 30))  # purple/pink hues ✅

    ratio = histology_pixels.mean()

    return ratio >= threshold, {"color_ratio": round(float(ratio), 3)}
