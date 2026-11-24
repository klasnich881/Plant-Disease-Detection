import json
import os
import sys
import time

import joblib
import numpy as np

# add approach_a/ and repo root to path so imports work from anywhere
APPROACH_A_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APPROACH_A_DIR, ".."))
sys.path.insert(0, APPROACH_A_DIR)
sys.path.insert(0, PROJECT_ROOT)

from shared.dataset import create_and_save_splits, load_splits
from src.config import (
    CLASS_NAMES,
    DATA_RAW_DIR,
    DATA_SPLITS_PATH,
    OUTPUT_FEATURES_DIR,
    OUTPUT_MODELS_DIR,
    OUTPUT_RESULTS_DIR,
    RANDOM_SEED,
    SAMPLES_PER_CLASS,
    SVM_PARAM_GRID,
)
from src.evaluate import (
    evaluate_classifier,
    measure_inference_time,
    plot_confusion_matrix,
)
from src.features import extract_single_image_features, standardize_features
from src.model import train_svm, tune_svm_hyperparameters


def extract_split_features(image_paths, split_name):
    print(f"Extracting {split_name} features ({len(image_paths)} images)...")
    features = [extract_single_image_features(p) for p in image_paths]
    return np.vstack(features)


def run_approach_a_pipeline():
    os.makedirs(OUTPUT_FEATURES_DIR, exist_ok=True)
    os.makedirs(OUTPUT_MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_RESULTS_DIR, exist_ok=True)

    if not os.path.isfile(DATA_SPLITS_PATH):
        create_and_save_splits(
            DATA_RAW_DIR,
            DATA_SPLITS_PATH,
            class_names=CLASS_NAMES,
            samples_per_class=SAMPLES_PER_CLASS,
            random_seed=RANDOM_SEED,
        )
    splits = load_splits(DATA_SPLITS_PATH, project_root=PROJECT_ROOT)

    y_train = np.array(splits["train"]["labels"])
    y_val = np.array(splits["val"]["labels"])
    y_test = np.array(splits["test"]["labels"])

    # extract features and time feature prep separately from SVM training
    prep_start = time.perf_counter()
    X_train_raw = extract_split_features(splits["train"]["paths"], "train")
    X_val_raw = extract_split_features(splits["val"]["paths"], "val")
    X_test_raw = extract_split_features(splits["test"]["paths"], "test")

    X_train, X_val, X_test, scaler = standardize_features(X_train_raw, X_val_raw, X_test_raw)
    prep_time = time.perf_counter() - prep_start
    print(f"Feature extraction + scaling took {prep_time:.2f}s")

    np.savez_compressed(
        os.path.join(OUTPUT_FEATURES_DIR, "standardized_features.npz"),
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )
    joblib.dump(scaler, os.path.join(OUTPUT_FEATURES_DIR, "standard_scaler.joblib"))

    # tune C and gamma on validation set, then train final SVM
    best_params = tune_svm_hyperparameters(X_train, y_train, X_val, y_val, SVM_PARAM_GRID)
    svm_model, train_time = train_svm(X_train, y_train, best_params)
    print(f"SVM training took {train_time:.2f}s")

    joblib.dump(svm_model, os.path.join(OUTPUT_MODELS_DIR, "rbf_svm_model.joblib"))

    # evaluate on test set
    test_metrics = evaluate_classifier(svm_model, X_test, y_test, CLASS_NAMES)
    avg_img_time = measure_inference_time(splits["test"]["paths"], scaler, svm_model)

    cm_path = os.path.join(OUTPUT_RESULTS_DIR, "approach_a_confusion_matrix.png")
    plot_confusion_matrix(test_metrics["confusion_matrix"], CLASS_NAMES, cm_path)

    results = {
        "approach": "Approach A (Classical CV + RBF-SVM)",
        "best_hyperparameters": best_params,
        "timings_seconds": {
            "feature_preparation_time": float(prep_time),
            "svm_training_time": float(train_time),
            "avg_processing_time_per_test_image": float(avg_img_time),
        },
        "test_metrics": test_metrics,
    }

    results_path = os.path.join(OUTPUT_RESULTS_DIR, "approach_a_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n--- Approach A Results ---")
    print(f"Test Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"Macro F1-Score:   {test_metrics['macro_avg']['f1_score']:.4f}")
    print(f"Avg Time / Image: {avg_img_time * 1000:.2f} ms")
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    run_approach_a_pipeline()