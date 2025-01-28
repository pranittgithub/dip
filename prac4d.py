import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in color
image = cv2.imread('example.jpg')

# Check if the image is loaded correctly
if image is None:
    print("Error: Could not load image. Please check the file path.")
    exit()

# --- Thresholding (on each color channel) ---
# Convert the image to a binary thresholded image
thresholded_image = cv2.inRange(image, (0, 0, 0), (127, 127, 127))

# --- Halftoning ---
# We'll simplify the halftoning by applying a threshold on blocks using NumPy
def halftone(image, block_size=8):
    # Convert to grayscale for simplicity in halftoning (for demonstration)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create a halftoned image using thresholding
    halftoned_image = np.where(gray_image > 127, 255, 0)
    
    # Convert back to color (3 channels) to match the original image's shape
    return cv2.cvtColor(halftoned_image.astype(np.uint8), cv2.COLOR_GRAY2BGR)

# Apply halftoning
halftoned_image = halftone(image)

# --- Plotting Results ---
plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Thresholded Image
plt.subplot(1, 3, 2)
plt.imshow(thresholded_image, cmap='gray')
plt.title('Thresholded Image')
plt.axis('off')

# Halftoned Image
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(halftoned_image, cv2.COLOR_BGR2RGB))
plt.title('Halftoned Image')
plt.axis('off')

# Save the comparison plot
plt.savefig("thresholding_halftoning_color.png")
print("Thresholding and Halftoning result saved.")
