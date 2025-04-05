import joblib
import numpy as np
import tensorflow as tf
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from common.preprocessing import load_and_preprocess_data
import json
import matplotlib.pyplot as plt

print("Loading TON_IoT dataset...")
X, y, features = load_and_preprocess_data("./data/TON_IoT/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv")

print("Feature headers:", features)
print("Sample data:")
print(X[:5])

# Load saved scaler and label encoder
scaler = joblib.load("models/scaler.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

print("Evaluating RandomForest model...")
model_rf = joblib.load("models/random_forest.pkl")
pred_rf = model_rf.predict(X)
accuracy_rf = accuracy_score(y, pred_rf)
print(f"RandomForest Accuracy: {accuracy_rf:.4f}")

# Print Confusion Matrix & Classification Report
print("\n[RandomForest Confusion Matrix]")
print(confusion_matrix(y, pred_rf))

print("\n[RandomForest Classification Report]")
print(classification_report(y, pred_rf, target_names=label_encoder.classes_))

print("Loading CNN TensorFlow Lite model...")
interpreter = tf.lite.Interpreter(model_path="models/cnn_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
print(f"CNN Model expects input shape: {input_shape}")

# Reshape data accordingly if model expects 3D input (e.g., [batch_size, features, 1])
if len(input_shape) == 3:
    X_cnn = np.expand_dims(X, axis=-1)
else:
    X_cnn = X

print("Evaluating CNN model...")
correct_predictions = 0
y_pred_cnn = []

for i in range(len(X)):
    input_data = np.expand_dims(X_cnn[i], axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred = 1 if output_data[0][0] >= 0.5 else 0
    y_pred_cnn.append(pred)
    if pred == y[i]:
        correct_predictions += 1

accuracy_cnn = correct_predictions / len(y)
print(f"CNN Model Accuracy: {accuracy_cnn:.4f}")

# Print Confusion Matrix & Classification Report for CNN
print("\n[CNN Confusion Matrix]")
print(confusion_matrix(y, y_pred_cnn))

print("\n[CNN Classification Report]")
print(classification_report(y, y_pred_cnn, target_names=label_encoder.classes_))

# Load logged metrics
with open("metrics/metrics.json", "r") as f:
    metrics = json.load(f)

# Extract and sort
models = list(metrics.keys())
accuracies = [metrics[m]["accuracy"] for m in models]

# Visualization
os.makedirs("visualizations", exist_ok=True)

plt.figure(figsize=(8, 5))
plt.bar(models, accuracies, color='mediumseagreen')
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1.0)
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Add % labels on top of bars
for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, height + 0.02, f"{acc*100:.1f}%", ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig("visualizations/accuracy_comparison.png")
plt.show()