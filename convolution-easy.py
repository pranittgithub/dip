import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from skimage import data

def convolve_image(kernel):
    image = data.camera()  # Load example image
    convolved_image = signal.convolve2d(image, kernel, mode="same", boundary="symm")
    
    # Plot original and convolved images
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap="gray")
    plt.title("Original")

    plt.subplot(1, 2, 2)
    plt.imshow(convolved_image, cmap="gray")
    plt.title("Convolved")
    plt.show()

# Create a simple kernel and apply convolution
kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
convolve_image(kernel)
