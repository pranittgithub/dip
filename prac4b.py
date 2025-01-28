import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the color image
image = cv2.imread('example.jpg')

# Check if the image is loaded correctly
if image is None:
    print("Error: Could not load image. Please check the file path.")
    exit()

# Split the image into R, G, B channels
b, g, r = cv2.split(image)

# Function for linear contrast adjustment
def contrast_adjustment(channel):
    min_val = np.min(channel)
    max_val = np.max(channel)
    return ((channel - min_val) / (max_val - min_val)) * 255

# Apply contrast adjustment to each channel
r_contrast = contrast_adjustment(r)
g_contrast = contrast_adjustment(g)
b_contrast = contrast_adjustment(b)

# Convert the contrast-adjusted channels back to uint8
r_contrast = np.uint8(r_contrast)
g_contrast = np.uint8(g_contrast)
b_contrast = np.uint8(b_contrast)

# Merge the adjusted channels back
contrast_adjusted_image = cv2.merge([b_contrast, g_contrast, r_contrast])

# Plot the results
plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Contrast Adjusted Image
plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(contrast_adjusted_image, cv2.COLOR_BGR2RGB))
plt.title('Contrast Adjusted Image')
plt.axis('off')

# Save the comparison plot
plt.savefig("color_contrast_adjustment.png")
print("Contrast adjustment result saved.")
