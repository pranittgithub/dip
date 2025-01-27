import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.util import random_noise

# Load the image (original color image)
image = cv2.imread('example.jpg')

# Check if the image is loaded correctly
if image is None:
    print("Error: Could not load image.")
    exit()

# Add salt-and-pepper noise to the color image using skimage's random_noise
noisy_image = random_noise(image, mode='s&p', amount=0.5)  # Increased noise to 50%

# Convert back to 8-bit image (since random_noise outputs float in [0, 1])
noisy_image = (255 * noisy_image).astype(np.uint8)

# Linear smoothing: Gaussian blur
gaussian_blur = cv2.GaussianBlur(noisy_image, (5, 5), 0)

# Nonlinear smoothing: Median filter
median_blur = cv2.medianBlur(noisy_image, 5)

# Plot the results
plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original')
plt.subplot(2, 2, 2), plt.imshow(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)), plt.title('Noisy')
plt.subplot(2, 2, 3), plt.imshow(cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2RGB)), plt.title('Gaussian Blur')
plt.subplot(2, 2, 4), plt.imshow(cv2.cvtColor(median_blur, cv2.COLOR_BGR2RGB)), plt.title('Median Filter')
plt.tight_layout()
plt.savefig("prac6.png")
