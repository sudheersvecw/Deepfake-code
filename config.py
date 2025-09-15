from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import os, random, numpy as np, tensorflow as tf

@dataclass
class Config:
    # Data
    train_dir: str = "./data/train"
    val_dir: str = "./data/val"
    test_dir: str = "./data/test"
    classes: Tuple[str, str] = ("real", "fake")

    # Video/frames
    target_h: int = 224
    target_w: int = 224
    num_frames: int = 30
    fps_sample: Optional[int] = None

    # Augmentation
    use_augment: bool = True
    aug_prob: float = 0.7

    # Model sizes
    lstm_units: int = 128
    transformer_dim: int = 256
    transformer_heads: int = 4
    transformer_ff_dim: int = 512
    transformer_layers: int = 2
    dropout: float = 0.3

    # Training
    batch_size: int = 4
    epochs: int = 50
    lr: float = 1e-4
    class_weights: Optional[Dict[int, float]] = None

    # Runtime
    mixed_precision: bool = True
    enable_xla: bool = True
    seed: int = 42

    # Outputs
    out_dir: str = "./outputs"
    ckpt_path: str = "./outputs/best_model.keras"
    history_path: str = "./outputs/train_history.json"
    test_report_path: str = "./outputs/test_report.json"
    cm_png: str = "./outputs/confusion_matrix.png"
    roc_png: str = "./outputs/roc_curve.png"

CFG = Config()

# Create output dir & set seeds
os.makedirs(CFG.out_dir, exist_ok=True)
random.seed(CFG.seed)
np.random.seed(CFG.seed)
tf.random.set_seed(CFG.seed)

# Optional runtime toggles
if CFG.mixed_precision:
    try:
        from tensorflow.keras import mixed_precision as tfmp
        tfmp.set_global_policy('mixed_float16')
        print("[INFO] Mixed precision enabled")
    except Exception as e:
        print("[WARN] Mixed precision unavailable:", e)

if CFG.enable_xla:
    try:
        tf.config.optimizer.set_jit(True)
        print("[INFO] XLA JIT enabled")
    except Exception as e:
        print("[WARN] XLA JIT not enabled:", e)
