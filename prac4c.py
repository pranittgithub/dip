import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image = cv2.imread('example.jpg', cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded correctly
if image is None:
    print("Error: Could not load image. Please check the file path.")
    exit()

# Apply Histogram Equalization
equalized_image = cv2.equalizeHist(image)

# Plot the original and equalized images along with their histograms
plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Histogram of Original Image
plt.subplot(2, 2, 2)
plt.hist(image.ravel(), bins=256, range=(0, 255))
plt.title('Histogram of Original Image')

# Equalized Image
plt.subplot(2, 2, 3)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

# Histogram of Equalized Image
plt.subplot(2, 2, 4)
plt.hist(equalized_image.ravel(), bins=256, range=(0, 255))
plt.title('Histogram of Equalized Image')

# Save the comparison plot
plt.savefig("histogram_equalization.png")
print("Histogram equalization result saved.")
