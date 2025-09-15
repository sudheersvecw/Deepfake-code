import json, glob, os
import numpy as np
from tensorflow import keras
from .model import TransformerEncoder
from .metrics import F1Score
from .data import read_video_uniform


def load_model(path: str):
    return keras.models.load_model(path, custom_objects={"F1Score": F1Score, "TransformerEncoder": TransformerEncoder})


def infer_on_video(model, video_path: str, cfg) -> float:
    arr = read_video_uniform(video_path, cfg.num_frames, cfg.target_h, cfg.target_w, do_aug=False)
    arr = np.expand_dims(arr, 0)
    prob = float(model.predict(arr, verbose=0).ravel()[0])
    return prob


def infer_on_folder(model, folder: str, cfg, exts=(".mp4", ".avi", ".mov")):
    results = {}
    for p in glob.glob(os.path.join(folder, "**", "*"), recursive=True):
        if os.path.splitext(p)[1].lower() in exts:
            results[p] = infer_on_video(model, p, cfg)
    return results
