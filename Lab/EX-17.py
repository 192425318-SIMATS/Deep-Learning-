# Experiment 17: Linear Separability using Logistic Regression
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Generate dataset
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

# Create Logistic Regression model
model = LogisticRegression()
# Train the model
model.fit(X, y)
# Predict
y_pred = model.predict(X)
# Accuracy
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy:.2f}")
# Plot data points
plt.figure(figsize=(10,6))

plt.scatter(
    X[:,0],
    X[:,1],
    c=y,
    cmap='viridis',      # Corrected from 'berlin'
    edgecolor='black',
    s=60
)

# Decision Boundary
coef = model.coef_[0]
intercept = model.intercept_[0]

x_vals = np.linspace(X[:,0].min(), X[:,0].max(), 100)
y_vals = -(coef[0] * x_vals + intercept) / coef[1]

plt.plot(
    x_vals,
    y_vals,
    'k--',
    linewidth=2,
    label="Decision Boundary"
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Linear Separability Demonstration")
plt.legend()
plt.grid(True)

plt.show()
