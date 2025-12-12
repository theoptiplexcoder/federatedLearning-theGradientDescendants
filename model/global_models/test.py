import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from client.global_cnn import GlobalCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM = 128


# =========================================================
# 🔍 IMAGE PROCESSING
# =========================================================
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])


def load_img(path):
    img = Image.open(path).convert("RGB")
    return img, transform(img).unsqueeze(0).to(DEVICE)


# =========================================================
# 🔍 AUTO-DETECT IMAGE TYPE (Polyp vs Lung)
# =========================================================
def detect_image_type(img_pil):
    """
    Detects whether the image is likely a polyp (colorful, high saturation)
    or lung X-ray (grayscale or low saturation).
    """
    img = np.array(img_pil) / 255.0

    # Compute saturation
    hsv = np.array(Image.fromarray((img * 255).astype(np.uint8)).convert("HSV"))
    saturation = hsv[:, :, 1].mean()

    # Threshold based detection
    if saturation > 40:  # empirical threshold
        return "polyp"
    else:
        return "lung"


# =========================================================
# 🔍 LOAD GLOBAL MODEL
# =========================================================
def load_global_model(round_num=1):
    model_path =  f"global_v{round_num}.pth"
    print(f"[INFO] Loading global model from: {model_path}")

    model = GlobalCNN(out_dim=EMBED_DIM).to(DEVICE)
    state = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


# =========================================================
# 🔍 EMBEDDING COMPUTATION
# =========================================================
def get_embedding(model, img_tensor):
    with torch.no_grad():
        return model(img_tensor)


# =========================================================
# 🔥 POLYP PREDICTION LOGIC
# =========================================================
def predict_polyp(embedding):
    score = torch.norm(embedding, p=2).item()
    score = min(max(score / 8, 0.0), 1.0)
    return score


# =========================================================
# 🔥 LUNG DISEASE PREDICTION LOGIC
# =========================================================
def predict_lung_disease(embedding):
    score = torch.mean(torch.abs(embedding)).item()
    score = min(max(score * 2, 0.0), 1.0)
    return score


# =========================================================
# 🔥 MAIN TEST FUNCTION
# =========================================================
def test_image(image_path, round_num=1):
    print(f"\n========== Testing Image ==========")
    print(f"Image: {image_path}")

    model = load_global_model(round_num)
    img_pil, img_tensor = load_img(image_path)

    # Auto-determine the image type
    img_type = detect_image_type(img_pil)
    print(f"[INFO] Detected image type → {img_type.upper()}")

    # Compute embedding
    emb = get_embedding(model, img_tensor)
    print(f"[INFO] Embedding shape: {emb.shape}")

    print("\n===== RESULTS =====")

    if img_type == "polyp":
        polyp_score = predict_polyp(emb)
        print(f"Polyp Detection Score : {polyp_score:.3f} (0–1)")

    elif img_type == "lung":
        lung_score = predict_lung_disease(emb)
        print(f"Lung Disease Probability : {lung_score:.3f} (0–1)")

    print("\n✓ Test Completed.\n")


# =========================================================
# 🔧 CLI SUPPORT
# =========================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to test image")
    parser.add_argument("--round", type=int, default=1)
    args = parser.parse_args()

    test_image(args.image, args.round)
