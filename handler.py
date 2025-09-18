import base64
import torch
from PIL import Image
import numpy as np
import cv2
from io import BytesIO

from models.birefnet import BiRefNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "checkpoints/birefnet_general.pth"

# Load model at cold start
model = BiRefNet()
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
model.load_state_dict(state_dict)
model.to(DEVICE)
model.eval()

def preprocess(image):
    img = np.array(image)
    img_resized = cv2.resize(img, (1024, 1024))
    img_norm = img_resized / 255.0
    img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float()
    return img_tensor.to(DEVICE), image.size

def postprocess(mask_tensor, orig_size):
    mask = mask_tensor.squeeze().cpu().numpy()
    mask = cv2.resize(mask, orig_size, interpolation=cv2.INTER_LINEAR)
    mask = (mask * 255).astype(np.uint8)
    return mask

def run_inference(image: Image.Image):
    inp_tensor, (w, h) = preprocess(image)
    with torch.no_grad():
        pred = model(inp_tensor)[-1]
        pred = torch.sigmoid(pred)
    mask = postprocess(pred, (w, h))
    rgba = np.array(image.convert("RGBA"))
    rgba[:, :, 3] = mask
    return Image.fromarray(rgba)

def handler(event):
    """
    Input: {"input": {"image": "<base64 string>"}}
    Output: {"output": {"image": "<base64 string>"}}
    """
    try:
        b64_img = event["input"]["image"]
        image_bytes = base64.b64decode(b64_img)
        image = Image.open(BytesIO(image_bytes)).convert("RGB")

        result = run_inference(image)

        buf = BytesIO()
        result.save(buf, format="PNG")
        out_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {"output": {"image": out_b64}}
    except Exception as e:
        return {"error": str(e)}

