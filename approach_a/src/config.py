import os

APPROACH_A_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_ROOT = os.path.abspath(os.path.join(APPROACH_A_DIR, ".."))

DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_SPLITS_PATH = os.path.join(PROJECT_ROOT, "data", "splits", "dataset_splits.json")

OUTPUT_FEATURES_DIR = os.path.join(APPROACH_A_DIR, "outputs", "features")
OUTPUT_MODELS_DIR = os.path.join(APPROACH_A_DIR, "outputs", "models")
OUTPUT_RESULTS_DIR = os.path.join(APPROACH_A_DIR, "outputs", "results")

RANDOM_SEED = 42
SAMPLES_PER_CLASS = 1000

CLASS_NAMES = [
    "Tomato___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Septoria_leaf_spot",
]

IMAGE_SIZE = (256, 256)
GAUSSIAN_KERNEL_SIZE = (5, 5)

# OpenCV HSV ranges: H is 0-179, S and V are 0-255
LEAF_HSV_LOWER = (10, 30, 20)
LEAF_HSV_UPPER = (95, 255, 255)
MORPH_KERNEL_SIZE = (5, 5)

# hue band for normal green leaf tissue; pixels outside this inside the leaf are treated as lesions
HEALTHY_GREEN_HUE_LOWER = 35
HEALTHY_GREEN_HUE_UPPER = 85

SVM_PARAM_GRID = {
    "C": [0.1, 1.0, 10.0, 100.0],
    "gamma": ["scale", 0.001, 0.01, 0.1],
}