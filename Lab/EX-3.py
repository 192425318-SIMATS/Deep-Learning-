# Experiment 3: Multi-Class Confusion Matrix

# Import required libraries
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# Load Digits dataset
X, y = load_digits(return_X_y=True)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=23
)

# Train Random Forest Classifier
model = RandomForestClassifier(random_state=23)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Create Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Display Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='winter'
)

plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Multi-Class Confusion Matrix")
plt.show()

# Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
