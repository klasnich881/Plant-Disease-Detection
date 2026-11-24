import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.preprocessing import StandardScaler

from src.preprocessing import preprocess_image
from src.segmentation import detect_diseased_regions, segment_leaf


def extract_color_features(hsv_image, leaf_mask, disease_mask):
    # normalized HSV histograms (16 bins for H, 8 for S, 8 for V)
    hist_h = cv2.calcHist([hsv_image], [0], leaf_mask, [16], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv_image], [1], leaf_mask, [8], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv_image], [2], leaf_mask, [8], [0, 256]).flatten()

    if hist_h.sum() > 0:
        hist_h /= hist_h.sum()
    if hist_s.sum() > 0:
        hist_s /= hist_s.sum()
    if hist_v.sum() > 0:
        hist_v /= hist_v.sum()

    # mean and std of H, S, V inside the leaf mask
    leaf_pixels = hsv_image[leaf_mask > 0]
    if len(leaf_pixels) > 0:
        means = np.mean(leaf_pixels, axis=0)
        stds = np.std(leaf_pixels, axis=0)
    else:
        means = np.zeros(3)
        stds = np.zeros(3)

    # percentage of leaf area flagged as diseased
    leaf_area = np.count_nonzero(leaf_mask)
    diseased_area = np.count_nonzero(cv2.bitwise_and(disease_mask, leaf_mask))
    diseased_pct = (diseased_area / leaf_area) * 100.0 if leaf_area > 0 else 0.0

    return np.concatenate([hist_h, hist_s, hist_v, means, stds, [diseased_pct]])


def extract_texture_features(bgr_image, leaf_mask):
    masked = cv2.bitwise_and(bgr_image, bgr_image, mask=leaf_mask)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    # crop to leaf bounding box so black background doesn't skew GLCM
    ys, xs = np.where(leaf_mask > 0)
    if len(ys) > 0 and len(xs) > 0:
        gray_crop = gray[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
    else:
        gray_crop = gray

    # GLCM averaged across 0, 45, 90, and 135 degrees
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(gray_crop, distances=[1], angles=angles, levels=256, symmetric=True, normed=True)

    glcm_props = []
    for prop in ["contrast", "correlation", "energy", "homogeneity"]:
        glcm_props.append(np.mean(graycoprops(glcm, prop)))

    # uniform LBP histogram (P=8, R=1 gives 10 bins)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    leaf_lbp = lbp[leaf_mask > 0]

    if len(leaf_lbp) > 0:
        lbp_hist, _ = np.histogram(leaf_lbp, bins=10, range=(0, 10), density=True)
    else:
        lbp_hist = np.zeros(10)

    return np.concatenate([glcm_props, lbp_hist])


def extract_single_image_features(image_path):
    bgr = preprocess_image(image_path)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    leaf_mask = segment_leaf(hsv)
    disease_mask = detect_diseased_regions(hsv, leaf_mask)

    color_feats = extract_color_features(hsv, leaf_mask, disease_mask)
    texture_feats = extract_texture_features(bgr, leaf_mask)

    return np.concatenate([color_feats, texture_feats])


def standardize_features(X_train, X_val, X_test):
    # fit scaler only on training data to avoid data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler