import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread(r"C:\Users\PRANIT\Desktop\dip\source.png")

# Check if the image is loaded correctly
if image is None:
    print("Error: Could not load image.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --- Edge-based Segmentation (Canny Edge Detection) ---
edges = cv2.Canny(gray, 100, 200)

# --- Region-based Segmentation (Thresholding and Contour Detection) ---
# Apply binary thresholding
_, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# --- Detecting Circles using Hough Transform ---
# Detect circles in the image using HoughCircles
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                           param1=50, param2=30, minRadius=20, maxRadius=100)

# Convert the circle coordinates to integers
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")

# --- Detecting Lines using Hough Line Transform ---
# Detect lines in the image using Hough Line Transform
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)

# --- Drawing the results ---
# Draw contours
image_contours = image.copy()
cv2.drawContours(image_contours, contours, -1, (0, 255, 0), 2)

# Draw circles
image_circles = image.copy()
if circles is not None:
    for (x, y, r) in circles:
        cv2.circle(image_circles, (x, y), r, (0, 255, 0), 4)

# Draw lines
image_lines = image.copy()
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image_lines, (x1, y1), (x2, y2), (0, 0, 255), 3)

# --- Save all images in one figure ---
plt.figure(figsize=(12, 10))

# Plot original image
plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Plot edge detection (Canny)
plt.subplot(2, 3, 2)
plt.imshow(edges, cmap='gray')
plt.title('Edge Detection (Canny)')
plt.axis('off')

# Plot contours
plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(image_contours, cv2.COLOR_BGR2RGB))
plt.title('Contours Detection')
plt.axis('off')

# Plot circles detection
plt.subplot(2, 3, 4)
plt.imshow(cv2.cvtColor(image_circles, cv2.COLOR_BGR2RGB))
plt.title('Circles Detection')
plt.axis('off')

# Plot lines detection
plt.subplot(2, 3, 5)
plt.imshow(cv2.cvtColor(image_lines, cv2.COLOR_BGR2RGB))
plt.title('Lines Detection')
plt.axis('off')

# Save the result as a single image
plt.tight_layout()
plt.savefig('segmentation_results.png')


print(r"C:\Users\PRANIT\Desktop\dip\segmentations.png")
