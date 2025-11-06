import torch
import numpy as np
import cv2
from PIL import Image

def generate_heatmap_overlay(model, input_tensor, original_img):
    model.eval()

    # Gradients storage
    gradients = []
    activations = []

    def save_gradient(grad):
        gradients.append(grad)

    # Hook into the last convolutional layer
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            target_layer = layer

    handle_activations = target_layer.register_forward_hook(
        lambda m, i, o: activations.append(o)
    )
    handle_gradients = target_layer.register_backward_hook(
        lambda m, grad_input, grad_output: save_gradient(grad_output[0])
    )

    # Forward + backward pass
    output = model(input_tensor)
    class_idx = torch.argmax(output).item()
    score = output[0, class_idx]
    model.zero_grad()
    score.backward()

    # Remove hooks
    handle_activations.remove()
    handle_gradients.remove()

    # Process Grad-CAM
    grads = gradients[-1].cpu().data.numpy()[0]
    acts = activations[-1].cpu().data.numpy()[0]

    weights = np.mean(grads, axis=(1,2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (original_img.width, original_img.height))
    cam = cam - np.min(cam)
    cam = cam / np.max(cam)

    heatmap = cv2.applyColorMap(np.uint8(cam * 255), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(np.array(original_img), 0.6, heatmap, 0.4, 0)

    return Image.fromarray(overlay)
