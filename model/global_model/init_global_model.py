# global_model/init_global_model.py
import logging
from pathlib import Path
import torch
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("init_global")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "global_models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "global_v0.pth"


def build_student():
    # Local import to avoid circular imports
    from client.global_cnn import GlobalCNN
    
    # Create a lightweight global CNN student model
    student = GlobalCNN(out_dim=128)
    return student


if __name__ == "__main__":
    student = build_student()
    torch.save(student.state_dict(), OUT_PATH)
    logger.info(f"[GLOBAL INIT] Saved initial CNN global model → {OUT_PATH}")
