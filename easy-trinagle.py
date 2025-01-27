import numpy as np
import matplotlib.pyplot as plt

# Create a time array that goes from 0 to 3 seconds, with 100 points per second
time = np.linspace(0, 3, 300)

# Create the triangle wave: absolute value of the time within each period (1 second)
triangle_wave = np.abs(2 * (time % 1 - 0.5))

# Plot the triangle wave
plt.plot(time, triangle_wave)
plt.title("Triangle Wave - 3 Periods")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
