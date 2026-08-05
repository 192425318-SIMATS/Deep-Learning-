# Experiment 1: Confusion Matrix

# Import required libraries
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Actual labels
actual = np.array([
    'Dog', 'Dog', 'Dog', 'Not Dog', 'Dog',
    'Not Dog', 'Dog', 'Dog', 'Not Dog', 'Not Dog'
])

# Predicted labels
predicted = np.array([
    'Dog', 'Not Dog', 'Dog', 'Not Dog', 'Dog',
    'Dog', 'Dog', 'Dog', 'Not Dog', 'Not Dog'
])

# Create confusion matrix
cm = confusion_matrix(actual, predicted)

# Display confusion matrix values
print("Confusion Matrix:")
print(cm)

# Plot heatmap
plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='RdPu',
    xticklabels=['Dog', 'Not Dog'],
    yticklabels=['Dog', 'Not Dog']
)

plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Confusion Matrix")
plt.show()
