from PIL import Image
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision import models, transforms

# Load your trained model
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("../models/breast_image_model.pth", map_location='cpu'))
model.eval()

# Grad-CAM utility (simplified)
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.forward_hook)
        target_layer.register_full_backward_hook(self.backward_hook)

    def forward_hook(self, module, inp, out):
        self.activations = out.detach()
    def backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()
    def generate_cam(self, input_tensor):
        out = self.model(input_tensor)
        class_idx = int(out.argmax(dim=1))
        out[0, class_idx].backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze().numpy()
        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        return cam

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# Load one test image
img_path = "../test_images/sample_histology.jpg"  # Replace with your image path
img = Image.open(img_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0)

# Grad-CAM generation
target_layer = model.layer4[-1].conv2
cam_generator = GradCAM(model, target_layer)
cam = cam_generator.generate_cam(input_tensor)

# Overlay
orig_np = np.array(img)
cam_resized = cv2.resize(cam, (orig_np.shape[1], orig_np.shape[0]))
heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(orig_np, 0.6, heatmap, 0.4, 0)

plt.figure(figsize=(5, 5))
plt.imshow(overlay)
plt.axis('off')
plt.title('Fig. 7. Grad-CAM Visualization Highlighting Malignant Regions')
plt.tight_layout()
plt.savefig('fig7_gradcam_overlay.png', dpi=300)
plt.show()
