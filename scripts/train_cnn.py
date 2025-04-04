import numpy as np
import tensorflow as tf
import joblib
import sys, os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from common.preprocessing import load_and_preprocess_data

from sklearn.metrics import confusion_matrix
from common.utils import plot_confusion_matrix
from common.utils import save_metric

print("[*] Loading TON_IoT dataset...")
X, y, features = load_and_preprocess_data("./data/TON_IoT/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv")

print("[*] Feature headers:", features)
print("[*] Sample data:\n", X[:5])

# Reshape input for CNN: [samples, features, 1]
X = np.expand_dims(X, axis=-1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("[*] Training CNN model...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X.shape[1], 1)),
    tf.keras.layers.Conv1D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # 10 attack types
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

print("[*] Evaluating CNN on test set...")
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
accuracy = accuracy_score(y_test, y_pred)
print(f"CNN Test Accuracy: {accuracy:.4f}")
print("\n[*] Classification Report:\n", classification_report(y_test, y_pred))


# Confusion Matrix
# Assuming y_test and y_pred are defined
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, class_names=["Normal", "Anomaly"], filename="visualizations/confusion_matrix_CNN.png")

# Save model
model.save("models/cnn_model.h5")

# Convert to TensorFlow Lite
print("[*] Converting to TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("models/cnn_model.tflite", "wb") as f:
    f.write(tflite_model)

print("CNN Model training and conversion complete.")

# Save the report in a json file
save_metric("CNN", accuracy)