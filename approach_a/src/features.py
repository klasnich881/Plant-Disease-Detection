import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.preprocessing import StandardScaler

from src.preprocessing import preprocess_image
from src.segmentation import detect_diseased_regions, segment_leaf

# Computes normalized HSV color histograms (16 bins for H, 8 for S, 8 for V),
# the mean and standard deviation of each HSV channel within the leaf mask,
# and the percentage of total leaf area marked as diseased.
def extract_color_features(hsv_image, leaf_mask, disease_mask)-> np.ndarray:
    # Normalized HSV histograms (16 bins for H [0, 180], 8 for S [0, 256], 8 for V [0, 256])
    hist_h = cv2.calcHist([hsv_image], [0], leaf_mask, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv_image], [1], leaf_mask, [8], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv_image], [2], leaf_mask, [8], [0, 256]).flatten()

    hist_h = hist_h / (hist_h.sum() + 1e-8)
    hist_s = hist_s / (hist_s.sum() + 1e-8)
    hist_v = hist_v / (hist_v.sum() + 1e-8)

    # Mean and standard deviation of each HSV channel within the leaf mask
    leaf_pixels = hsv_image[leaf_mask > 0]
    if leaf_pixels.size > 0:
        channel_means = np.mean(leaf_pixels, axis=0)
        channel_stds = np.std(leaf_pixels, axis=0)
    else:
        channel_means = np.zeros(3, dtype=np.float64)
        channel_stds = np.zeros(3, dtype=np.float64)

    # Percentage of total leaf area marked as diseased
    total_leaf_area = np.count_nonzero(leaf_mask)
    diseased_area = np.count_nonzero(cv2.bitwise_and(disease_mask, leaf_mask))
    diseased_pct = (diseased_area / total_leaf_area) * 100.0 if total_leaf_area > 0 else 0.0

    color_features = np.concatenate([
        hist_h,
        hist_s,
        hist_v,
        channel_means,
        channel_stds,
        np.array([diseased_pct], dtype=np.float64),
    ])
    return color_features.astype(np.float64)


# Converts the masked leaf region to grayscale and computes GLCM texture properties
# (contrast, correlation, energy, homogeneity) averaged across 0, 45, 90, and 135 degrees,
# alongside a uniform Local Binary Pattern (LBP) histogram (8 neighbors, radius 1).
def extract_texture_features(bgr_image, leaf_mask)-> np.ndarray:
    masked_bgr = cv2.bitwise_and(bgr_image, bgr_image, mask=leaf_mask)
    gray = cv2.cvtColor(masked_bgr, cv2.COLOR_BGR2GRAY)

    # Crop to the leaf bounding box so background zeros don't dominate GLCM statistics
    ys, xs = np.where(leaf_mask > 0)
    if ys.size > 0 and xs.size > 0:
        gray_crop = gray[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    else:
        gray_crop = gray

    # GLCM properties averaged over 0, 45, 90, and 135 degrees
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(
        gray_crop,
        distances=[1],
        angles=angles,
        levels=256,
        symmetric=True,
        normed=True,
    )
    properties = ["contrast", "correlation", "energy", "homogeneity"]
    glcm_features = np.array(
        [np.mean(graycoprops(glcm, prop)) for prop in properties],
        dtype=np.float64,
    )

    # Uniform LBP histogram (8 neighbors, radius 1 -> 10 uniform bins: 0..9)
    n_points = 8
    radius = 1
    lbp = local_binary_pattern(gray, P=n_points, R=radius, method="uniform")
    leaf_lbp = lbp[leaf_mask > 0]

    n_bins = n_points + 2  # 10 bins for 'uniform' method with P=8
    if leaf_lbp.size > 0:
        lbp_hist, _ = np.histogram(leaf_lbp, bins=n_bins, range=(0, n_bins), density=True)
    else:
        lbp_hist = np.zeros(n_bins, dtype=np.float64)

    texture_features = np.concatenate([glcm_features, lbp_hist.astype(np.float64)])
    return texture_features


# Runs preprocessing, segmentation, and color/texture extraction for a single image,
# concatenating all extracted values into one 1D feature vector.
def extract_single_image_features(image_path: str)-> np.ndarray:
    preprocessed_bgr = preprocess_image(image_path)
    hsv_image = cv2.cvtColor(preprocessed_bgr, cv2.COLOR_BGR2HSV)

    leaf_mask = segment_leaf(hsv_image)
    disease_mask = detect_diseased_regions(hsv_image, leaf_mask)

    color_vec = extract_color_features(hsv_image, leaf_mask, disease_mask)
    texture_vec = extract_texture_features(preprocessed_bgr, leaf_mask)

    return np.concatenate([color_vec, texture_vec])


# Fits a scikit-learn StandardScaler exclusively on the training feature matrix
# and applies that transformation to the training, validation, and test feature sets.
def standardize_features(X_train, X_val, X_test)-> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler