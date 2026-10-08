"""Evaluation metrics and logging helpers."""

import csv
import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve


def compute_metrics(y_true, y_prob) -> dict:
    """Compute ACC, AUC, sensitivity and specificity.

    ``y_prob`` holds the predicted probability of the positive class; the
    binary decision threshold is 0.5.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    return {
        'acc': (tp + tn) / max(len(y_true), 1),
        'auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float('nan'),
        'sen': tp / (tp + fn) if (tp + fn) else 0.0,
        'spe': tn / (tn + fp) if (tn + fp) else 0.0,
    }


def plot_roc(y_true, y_prob, save_path: str, title: str = 'ROC curve') -> float:
    """Plot and save the ROC curve; returns the AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.rcParams['font.size'] = 14
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return auc


class CsvLogger:
    """Append rows to a CSV log file with a header written once."""

    def __init__(self, path: str, fieldnames):
        self.path = path
        self.fieldnames = list(fieldnames)
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        if not os.path.exists(path):
            with open(path, 'w', newline='') as f:
                csv.writer(f).writerow(self.fieldnames)

    def log(self, *values):
        with open(self.path, 'a', newline='') as f:
            csv.writer(f).writerow(values)


def save_predictions(path: str, y_true, y_prob, y_pred):
    """Persist per-sample test predictions for later analysis."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    pd.DataFrame({
        'true_label': np.asarray(y_true),
        'pred_prob': np.asarray(y_prob),
        'pred_label': np.asarray(y_pred),
    }).to_csv(path, index=False)
