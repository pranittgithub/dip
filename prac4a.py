import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image = cv2.imread('example.jpg', cv2.IMREAD_GRAYSCALE)

# Apply Log transformation
log_transformed = cv2.log(1 + np.float32(image))

# Apply Power-law transformation (Gamma correction)
gamma = 2.0
power_transformed = np.power(image / 255.0, gamma) * 255.0

# Save the results
cv2.imwrite('log_transformed_image.png', log_transformed)
cv2.imwrite('power_transformed_image.png', power_transformed)

# Plot the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Original')
plt.subplot(1, 3, 2), plt.imshow(log_transformed, cmap='gray'), plt.title('Log Transformation')
plt.subplot(1, 3, 3), plt.imshow(power_transformed, cmap='gray'), plt.title('Power-law Transformation')
plt.savefig("transformations_comparison.png")
print("Results saved as images and comparison plot.")

"""
1. Logarithmic Transformation (Log Transform)
A logarithmic transformation enhances the dark areas of an image while compressing the bright areas. It is useful when we want to highlight low-intensity details in an image (like shadows or dark objects) while preventing bright regions from becoming too exaggerated.

Formula:
output
=
𝑐
⋅
log
⁡
(
1
+
input
)
output=c⋅log(1+input)

Where:

input: Original pixel value.
output: Transformed pixel value.
c: Constant used for scaling.
log(1 + input): This part of the formula applies the logarithm function to the pixel values.
What it does:
Dark areas (low pixel values) become brighter.
Bright areas (high pixel values) are compressed, i.e., they lose some of their brightness.
Why use it?
If you have an image where you want to bring out details in the darker regions (like shadowy areas), log transformation can be very helpful.

2. Power-law Transformation (Gamma Correction)
Power-law transformation is a technique used to adjust the brightness and contrast of an image. This is done using a parameter called gamma.

Formula:
output
=
𝑐
⋅
(
input
)
𝛾
output=c⋅(input) 
γ
 

Where:

input: Original pixel value.
output: Transformed pixel value.
gamma: A constant (usually between 0 and 5).
c: Scaling constant.
What it does:
Gamma > 1: Brightens the image, especially for the brighter parts.
Gamma < 1: Darkens the image, especially for the darker parts.
Why use it?
If you want to make an image appear brighter or darker (without just changing its exposure), you can adjust the gamma value. For example, setting gamma to 0.5 makes dark areas lighter (good for enhancing shadows), and setting it to 2.0 can make bright areas even brighter.
"""