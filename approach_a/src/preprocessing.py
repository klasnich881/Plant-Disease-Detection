import cv2


def preprocess_image(image_path, target_size=(256, 256), kernel_size=(5, 5)):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(resized, kernel_size, 0)
    return blurred