import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image (binary image or thresholded image)
image = cv2.imread(r"C:\Users\PRANIT\Desktop\dip\source.png", cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded correctly
if image is None:
    print("Error: Could not load image.")
    exit()

# Threshold the image to get a binary image (just for demonstration)
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Define a kernel for morphological operations (3x3 kernel)
kernel = np.ones((5, 5), np.uint8)

# 1. Erosion (shrinks white regions)
erosion = cv2.erode(binary_image, kernel, iterations=1)

# 2. Dilation (expands white regions)
dilation = cv2.dilate(binary_image, kernel, iterations=1)

# 3. Opening (erosion followed by dilation)
opening = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel)

# 4. Closing (dilation followed by erosion)
closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)


# Plot the results
plt.figure(figsize=(10, 8))

# Plot each image
plt.subplot(3, 3, 1), plt.imshow(binary_image, cmap='gray'), plt.title('Original Binary')
plt.subplot(3, 3, 2), plt.imshow(erosion, cmap='gray'), plt.title('Erosion')
plt.subplot(3, 3, 3), plt.imshow(dilation, cmap='gray'), plt.title('Dilation')
plt.subplot(3, 3, 4), plt.imshow(opening, cmap='gray'), plt.title('Opening')
plt.subplot(3, 3, 5), plt.imshow(closing, cmap='gray'), plt.title('Closing')

plt.tight_layout()
plt.savefig(r"C:\Users\PRANIT\Desktop\dip\morphologicals.png")
