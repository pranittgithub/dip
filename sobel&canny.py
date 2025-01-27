import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image (original color image)
image = cv2.imread('example.jpg')

# Check if the image is loaded correctly
if image is None:
    print("Error: Could not load image.")
    exit()

# Convert the image to grayscale (required for edge detection)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 1. Sobel Edge Detection (Horizontal and Vertical Gradients)
sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)  # Horizontal gradient
sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)  # Vertical gradient
sobel_edge = cv2.magnitude(sobel_x, sobel_y)  # Combine the two gradients

# 2. Canny Edge Detection
canny_edge = cv2.Canny(gray_image, 150, 250)  # Increase the thresholds


# Plot the results
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(2, 2, 2), plt.imshow(sobel_edge, cmap='gray'), plt.title('Sobel Edge Detection')
plt.subplot(2, 2, 3), plt.imshow(canny_edge, cmap='gray'), plt.title('Canny Edge Detection')
plt.tight_layout()
plt.savefig("edge_detection.png")
