import json, itertools
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt


def plot_confusion_matrix(cm: np.ndarray, classes, save_path: str):
    fig = plt.figure(figsize=(4,4))
    ax = plt.gca()
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title('Confusion Matrix')
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, save_path: str):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    fig = plt.figure(figsize=(4,4))
    ax = plt.gca()
    ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    ax.plot([0,1], [0,1], linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def test_and_report(model, ds_test, cfg):
    y_true_all, y_prob_all = [], []
    for xb, yb in ds_test:
        pr = model.predict(xb, verbose=0).ravel()
        y_prob_all.append(pr)
        y_true_all.append(yb.numpy())
    y_true = np.concatenate(y_true_all)
    y_prob = np.concatenate(y_prob_all)
    y_pred = (y_prob >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=list(cfg.classes), output_dict=True)

    with open(cfg.test_report_path, 'w') as f:
        json.dump(report, f, indent=2)

    plot_confusion_matrix(cm, list(cfg.classes), cfg.cm_png)
    plot_roc(y_true, y_prob, cfg.roc_png)
    return report, cm
