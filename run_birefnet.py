import sys, os
sys.path.append(os.path.dirname(__file__))

import torch
from PIL import Image
import numpy as np
import cv2

from models.birefnet import BiRefNet

# ----- CONFIG -----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "checkpoints/birefnet_general.pth"
INPUT_IMAGE = "test.png"   # replace with your test image name if different
OUTPUT_MASK = "mask.png"
OUTPUT_IMAGE = "output.png"
# ------------------

def load_model():
    model = BiRefNet()
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

def preprocess(image):
    img = np.array(image)
    # Resize to 1024x1024 for inference
    img_resized = cv2.resize(img, (1024, 1024))
    img_norm = img_resized / 255.0
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float()
    return img_tensor.to(DEVICE), image.size  # returns (width, height)

def postprocess(mask_tensor, orig_size):
    mask = mask_tensor.squeeze().cpu().numpy()
    mask = cv2.resize(mask, orig_size, interpolation=cv2.INTER_LINEAR)
    mask = (mask * 255).astype(np.uint8)
    return mask

def remove_background():
    model = load_model()
    image = Image.open(INPUT_IMAGE).convert("RGB")

    inp_tensor, (orig_w, orig_h) = preprocess(image)

    with torch.no_grad():
        outs = model(inp_tensor)
        pred = outs[-1]
        pred = torch.sigmoid(pred)

    mask = postprocess(pred, (orig_w, orig_h))

    # Save mask
    Image.fromarray(mask).save(OUTPUT_MASK)

    # Apply mask as alpha channel
    img_np = np.array(image)
    rgba = cv2.cvtColor(img_np, cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = mask
    Image.fromarray(rgba).save(OUTPUT_IMAGE)

    print(f"✅ Done. Saved {OUTPUT_IMAGE} and {OUTPUT_MASK}")

if __name__ == "__main__":
    remove_background()

