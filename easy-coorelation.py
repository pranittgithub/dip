import numpy as np
import matplotlib.pyplot as plt

def plot_and_correlate(signal1, signal2, length):
    # Plot signals
    plt.subplot(2, 1, 1)
    plt.plot(signal1, label="Signal 1")
    plt.plot(signal2, label="Signal 2")
    plt.legend()

    # Compute correlation
    correlation = [np.correlate(signal1[i:i+length], signal2[i:i+length])[0] 
                   for i in range(len(signal1) - length + 1)]

    # Plot correlation
    plt.subplot(2, 1, 2)
    plt.plot(correlation, label="Correlation")
    plt.legend()
    plt.tight_layout()
    plt.show()

# Example usage
fs = 100  # Sampling frequency
t = np.linspace(0, 1, fs, endpoint=False)
signal1 = np.sin(2 * np.pi * 5 * t)
signal2 = np.sin(2 * np.pi * 5 * t + np.pi / 2)  # Shifted sine wave
plot_and_correlate(signal1, signal2, 10)
