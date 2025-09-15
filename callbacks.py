import numpy as np
from sklearn.metrics import roc_curve, auc
from tensorflow import keras

class EvalAndSaveCallback(keras.callbacks.Callback):
    def __init__(self, val_data, test_data, cfg):
        super().__init__()
        self.val_data = val_data
        self.test_data = test_data
        self.cfg = cfg
        self.best_val_auc = 0.0

    def on_epoch_end(self, epoch, logs=None):
        y_true, y_prob = [], []
        for xb, yb in self.val_data:
            pr = self.model.predict(xb, verbose=0).ravel()
            y_prob.append(pr)
            y_true.append(yb.numpy())
        y_true = np.concatenate(y_true)
        y_prob = np.concatenate(y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        val_auc = auc(fpr, tpr)
        print(f"\n[Eval] Epoch {epoch+1}: val AUC = {val_auc:.4f}")
        if val_auc > self.best_val_auc:
            self.best_val_auc = val_auc
            self.model.save(self.cfg.ckpt_path)
            print(f"[Eval] New best model saved -> {self.cfg.ckpt_path}")
        if self.test_data is not None and (epoch+1) % 5 == 0:
            self.model.evaluate(self.test_data, verbose=0)
