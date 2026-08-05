# Experiment 15: Image Segmentation using K-Means

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Image file name (uploaded in Colab)
image_path = "dog.jpg.png"

# Read the image
img = cv2.imread(image_path)

# Check if image is loaded
if img is None:
    print("Error: Could not load image. Please check the file name.")
else:
    # Convert BGR to RGB
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert image to pixel values
    pixels = np.float32(rgb_img.reshape((-1, 3)))

    # Define K-Means criteria
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        100,
        0.2
    )

    # Number of clusters
    K = 3

    # Apply K-Means clustering
    _, labels, centers = cv2.kmeans(
        pixels,
        K,
        None,
        criteria,
        10,
        cv2.KMEANS_RANDOM_CENTERS
    )

    # Convert centers back to uint8
    centers = np.uint8(centers)

    # Create segmented image
    segmented_img = centers[labels.flatten()]
    segmented_img = segmented_img.reshape(rgb_img.shape)

    # Display original and segmented images
    plt.figure(figsize=(10,5))

    plt.subplot(121)
    plt.imshow(rgb_img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(122)
    plt.imshow(segmented_img)
    plt.title("Segmented Image (K-Means)")
    plt.axis("off")

    plt.tight_layout()
    plt.show()
