import tensorflow as tf
import joblib
from sklearn.model_selection import train_test_split
from common.preprocessing import load_and_preprocess_data

# Load data
X, y, features = load_and_preprocess_data()

# Reshape for CNN (Conv1D expects 3D input)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build CNN Model
print("[*] Building CNN model...")
model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(64, kernel_size=3, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(set(y)), activation='softmax')
])

# Compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
print("[*] Training CNN model...")
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

# Convert to TensorFlow Lite
print("[*] Converting CNN model to TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save TFLite Model
with open("./models/cnn_tinyml_model.tflite", "wb") as f:
    f.write(tflite_model)

print("[✓] CNN model saved at './models/cnn_tinyml_model.tflite'")
