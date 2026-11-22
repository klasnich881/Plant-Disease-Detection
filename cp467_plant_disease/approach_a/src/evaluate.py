# Evaluates the trained SVM on a dataset split, returning overall accuracy,
# per-class and macro-averaged precision, recall, F1-score, and the confusion matrix.
def evaluate_classifier(model, X_eval, y_eval, class_names: list) -> dict:
    pass


# Measures the average end-to-end processing time per test image, timing image
# preprocessing, leaf segmentation, feature extraction, scaling, and SVM prediction.
def measure_inference_time(test_image_paths: list, scaler, model) -> float:
    pass


# Generates and saves a Matplotlib confusion matrix plot to visualize misclassifications
# across the five tomato leaf classes.
def plot_confusion_matrix(conf_matrix, class_names: list, save_path: str) -> None:
    pass