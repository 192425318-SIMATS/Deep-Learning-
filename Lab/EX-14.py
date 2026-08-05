# Experiment 14: Gradient Descent

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Mean Squared Error Function
def mean_squared_error(y_true, y_predicted):
    cost = np.sum((y_true - y_predicted) ** 2) / len(y_true)
    return cost

# Gradient Descent Function
def gradient_descent(x, y, iterations=1000, learning_rate=0.01,
                     stopping_threshold=1e-6):

    current_weight = 0.0
    current_bias = 0.0

    n = float(len(x))
    costs = []
    previous_cost = None

    for i in range(iterations):

        # Prediction
        y_predicted = current_weight * x + current_bias

        # Calculate Cost
        current_cost = mean_squared_error(y, y_predicted)

        # Stop if improvement is very small
        if previous_cost is not None and abs(previous_cost - current_cost) <= stopping_threshold:
            break

        previous_cost = current_cost
        costs.append(current_cost)

        # Compute Gradients
        weight_derivative = -(2 / n) * np.sum(x * (y - y_predicted))
        bias_derivative = -(2 / n) * np.sum(y - y_predicted)

        # Update Parameters
        current_weight = current_weight - learning_rate * weight_derivative
        current_bias = current_bias - learning_rate * bias_derivative

        # Print every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i+1}: Cost = {current_cost:.4f}, "
                  f"Weight = {current_weight:.4f}, "
                  f"Bias = {current_bias:.4f}")

    return current_weight, current_bias, costs

# -----------------------------
# Sample Dataset
# -----------------------------
x = np.array([1,2,3,4,5,6,7,8,9,10], dtype=float)
y = np.array([2,4,5,4,5,7,8,9,10,12], dtype=float)

# Standardize Features
scaler = StandardScaler()
x = scaler.fit_transform(x.reshape(-1,1)).flatten()

# Train using Gradient Descent
weight, bias, costs = gradient_descent(x, y)

print("\nFinal Weight:", weight)
print("Final Bias:", bias)

# Plot Cost vs Iterations
plt.figure(figsize=(8,6))
plt.plot(range(len(costs)), costs, 'r-')
plt.title("Cost vs Iterations")
plt.xlabel("Iterations")
plt.ylabel("Cost (MSE)")
plt.grid(True)
plt.show()

# Predictions
y_pred = weight * x + bias

# Plot Regression Line
plt.figure(figsize=(8,6))
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, y_pred, color='red', linewidth=2, label='Regression Line')
plt.title("Gradient Descent Regression")
plt.xlabel("Standardized X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()
