# Experiment 16: Image Segmentation using Thresholding and Morphology

import numpy as np
import cv2
import matplotlib.pyplot as plt

# Use the image you uploaded in Experiment 15
image_path = "dog.jpg.png"

# Read image
img = cv2.imread(image_path)

# Check if image loaded successfully
if img is None:
    print("Error: Could not load image. Please check the file name.")
else:
    # Convert BGR to RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Otsu Thresholding
    ret, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Morphological Closing
    kernel = np.ones((2, 2), np.uint8)
    closing = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # Dilation
    sure_bg = cv2.dilate(
        closing,
        kernel,
        iterations=3
    )

    # Display Results
    plt.figure(figsize=(12,8))
    plt.subplot(231)
    plt.imshow(rgb_img)
    plt.title("Original Image")
    plt.axis("off")
    plt.subplot(232)
    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale Image")
    plt.axis("off")
    plt.subplot(233)
    plt.imshow(thresh, cmap="gray")
    plt.title("Otsu's Threshold")
    plt.axis("off")
    plt.subplot(234)
    plt.imshow(closing, cmap="gray")
    plt.title("MorphologyEx: Closing (2x2)")
    plt.axis("off")
    plt.subplot(235)
    plt.imshow(sure_bg, cmap="gray")
    plt.title("Dilation")
    plt.axis("off")
    plt.tight_layout()
    plt.show()
    # Save Output
    plt.imsave("dilation.png", sure_bg, cmap="gray")
