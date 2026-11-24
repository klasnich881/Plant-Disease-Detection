import os
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.features import extract_single_image_features


def evaluate_classifier(model, X_eval, y_eval, class_names):
    y_pred = model.predict(X_eval)

    acc = float(accuracy_score(y_eval, y_pred))
    report = classification_report(
        y_eval, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_eval, y_pred)

    per_class = {}
    for cls in class_names:
        per_class[cls] = {
            "precision": float(report[cls]["precision"]),
            "recall": float(report[cls]["recall"]),
            "f1_score": float(report[cls]["f1-score"]),
            "support": int(report[cls]["support"]),
        }

    macro_avg = {
        "precision": float(report["macro avg"]["precision"]),
        "recall": float(report["macro avg"]["recall"]),
        "f1_score": float(report["macro avg"]["f1-score"]),
    }

    return {
        "accuracy": acc,
        "per_class": per_class,
        "macro_avg": macro_avg,
        "confusion_matrix": cm.tolist(),
    }


def measure_inference_time(test_image_paths, scaler, model):
    total_time = 0.0

    for path in test_image_paths:
        start = time.perf_counter()

        # time the full pipeline per image: preprocessing, segmentation, features, scaling, prediction
        feats = extract_single_image_features(path).reshape(1, -1)
        feats_scaled = scaler.transform(feats)
        model.predict(feats_scaled)

        total_time += time.perf_counter() - start

    return total_time / len(test_image_paths)


def plot_confusion_matrix(conf_matrix, class_names, save_path):
    cm = np.array(conf_matrix)
    labels = [c.replace("Tomato___", "") for c in class_names]

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", ax=ax, xticks_rotation=30)
    ax.set_title("Approach A (RBF-SVM) Confusion Matrix")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)