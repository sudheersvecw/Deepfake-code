# Hybrid CNN–BiLSTM–Transformer Deepfake Detector (TensorFlow 2.x)

A production‑ready video deepfake detection pipeline with:

- **MobileNetV2** spatial backbone (frozen by default)
- **BiLSTM** + **Transformer Encoder** for temporal reasoning (parallel branches)
- End‑to‑end **tf.data** pipelines from raw videos (uniform frame sampling)
- Optional **Albumentations** augmentations
- Mixed precision + XLA ready
- Metrics: Accuracy, Precision, Recall, AUC, **F1** (custom)
- Evaluation: **confusion matrix**, **ROC/AUC**, and classification report
- Inference utilities for single video or whole folder



## Project Structure


hybrid_deepfake_detector/
├── config.py          # Config dataclass, runtime toggles (mixed precision, XLA)
├── utils.py           # Small helpers (uniform index sampling)
├── data.py            # Video I/O, augmentations, tf.data pipelines
├── model.py           # TransformerEncoder layer + model builder
├── metrics.py         # Custom F1 metric (tf.keras Metric)
├── callbacks.py       # Eval+save callback based on validation AUC
├── evaluate.py        # Testing, classification report, ROC & confusion matrix plots
├── infer.py           # Load model, single-video & folder inference
├── train.py           # Dataset discovery, training orchestration
└── main.py            # CLI entry point

## Requirements

- Python **3.9–3.11**
- TensorFlow **>=2.12,<2.17** (GPU build recommended)
- OpenCV‑Python
- Albumentations (optional but recommended)
- scikit‑learn
- matplotlib

### Install

python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install --upgrade pip
pip install "tensorflow>=2.12,<2.17" opencv-python albumentations scikit-learn matplotlib


