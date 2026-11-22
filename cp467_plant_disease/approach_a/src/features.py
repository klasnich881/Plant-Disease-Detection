# Computes normalized HSV color histograms (16 bins for H, 8 for S, 8 for V),
# the mean and standard deviation of each HSV channel within the leaf mask,
# and the percentage of total leaf area marked as diseased.
def extract_color_features(hsv_image, leaf_mask, disease_mask):
    pass


# Converts the masked leaf region to grayscale and computes GLCM texture properties
# (contrast, correlation, energy, homogeneity) averaged across 0, 45, 90, and 135 degrees,
# alongside a uniform Local Binary Pattern (LBP) histogram (8 neighbors, radius 1).
def extract_texture_features(bgr_image, leaf_mask):
    pass


# Runs preprocessing, segmentation, and color/texture extraction for a single image,
# concatenating all extracted values into one 1D feature vector.
def extract_single_image_features(image_path: str):
    pass


# Fits a scikit-learn StandardScaler exclusively on the training feature matrix
# and applies that transformation to the training, validation, and test feature sets.
def standardize_features(X_train, X_val, X_test):
    pass