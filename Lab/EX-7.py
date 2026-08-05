# Experiment 7: Visualization of Sigmoid Function

# Import required libraries
import numpy as np
import matplotlib.pyplot as plt

# Define the Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Generate values from -5 to 5
x = np.arange(-5, 5, 0.1)

# Calculate sigmoid values
y = sigmoid(x)

# Plot the sigmoid curve
plt.figure(figsize=(8,5))
plt.plot(x, y, color='pink', linewidth=3)

# Add labels and title
plt.title("Visualization of the Sigmoid Function", fontsize=14)
plt.xlabel("Input (z)")
plt.ylabel("Sigmoid(z)")
plt.grid(True)

# Display the plot
plt.show()
