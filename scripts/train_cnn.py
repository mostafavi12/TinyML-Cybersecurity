import argparse
import logging
import sys
import numpy as np
import tensorflow as tf
import joblib
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.utils import (
    setup_logging,
    plot_confusion_matrix,
    plot_classification_metrics,
    generate_model_name,
    save_metrics_with_all_scores,
)
from common.preprocessing import load_and_preprocess_data

# ------------------------------
# Argument Parsing
# ------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--model_type", default="Base", help="Type of CNN model to build")
args = parser.parse_args()
model_type = args.model_type

# ------------------------------
# Logging Setup
# ------------------------------
model_name = generate_model_name("CNN", model_type)
setup_logging(model_name)
logging.info(f"[*] Training CNN model: {model_type}")

# ------------------------------
# Load Data
# ------------------------------
X, y, features, class_names = load_and_preprocess_data(
    "./data/TON_IoT/Train_Test_datasets/train_test_network.csv"
)
X = np.expand_dims(X, axis=-1)  # Shape: (samples, 6, 1)

# ------------------------------
# Train-Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
unique, train_counts = np.unique(y_train, return_counts=True)
logging.info("Train label distribution: " + str([f"{class_names[i]}={c}" for i, c in zip(unique, train_counts)]))
unique, test_counts = np.unique(y_test, return_counts=True)
logging.info("Test label distribution: " + str([f"{class_names[i]}={c}" for i, c in zip(unique, test_counts)]))

# ------------------------------
# Class Weights for Imbalance
# ------------------------------
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
logging.info(f"Computed class weights: {class_weight_dict}")

# ------------------------------
# Build CNN Model
# ------------------------------
def build_model(model_type, input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))

    if model_type == "Base":
        model.add(tf.keras.layers.Conv1D(32, 3, activation="relu"))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation="relu"))
    elif model_type == "Tiny":
        model.add(tf.keras.layers.Conv1D(8, 3, activation='relu'))
        model.add(tf.keras.layers.Flatten())
    elif model_type == "Deep":
        model.add(tf.keras.layers.Conv1D(64, 3, activation="relu"))
        model.add(tf.keras.layers.Conv1D(128, 3, activation="relu"))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation="relu"))
    elif model_type == "Wider":
        model.add(tf.keras.layers.Conv1D(128, 3, activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
    elif model_type == "Shallow":
        model.add(tf.keras.layers.Conv1D(16, 3, activation="relu"))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(32, activation="relu"))
    elif model_type == "Dropout":
        model.add(tf.keras.layers.Conv1D(32, 3, activation="relu"))
        model.add(tf.keras.layers.Dropout(0.3))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation="relu"))
    elif model_type == "BatchNorm":
        model.add(tf.keras.layers.Conv1D(32, 3, activation="relu"))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation="relu"))
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

model = build_model(model_type, input_shape=(X.shape[1], 1))

# ------------------------------
# Training
# ------------------------------
start_time = time.time()
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    verbose=2,
    class_weight=class_weight_dict
)
train_duration = time.time() - start_time
logging.info(f"Training completed in {train_duration:.2f} seconds.")

# ------------------------------
# Evaluation Helper
# ------------------------------
def evaluate_model(X_eval, y_eval, split_name):
    y_pred_probs = model.predict(X_eval)
    y_pred = np.argmax(y_pred_probs, axis=1)
    accuracy = accuracy_score(y_eval, y_pred)
    report = classification_report(y_eval, y_pred, output_dict=True, zero_division=0)

    logging.info(f"{split_name} Accuracy: {accuracy:.4f}")
    logging.info(f"{split_name} Classification Report:\n{classification_report(y_eval, y_pred, zero_division=0)}")

    # Plot confusion matrix
    cm_file = f"confusion_matrix_{model_name}_{split_name}.png"
    plot_confusion_matrix(
        confusion_matrix(y_eval, y_pred),
        class_names=class_names,
        filename=cm_file
    )

    # Plot per-class metrics
    metrics_file = f"metrics_{model_name}_{split_name}.png"
    precision = [report[str(i)]['precision'] for i in range(len(class_names))]
    recall = [report[str(i)]['recall'] for i in range(len(class_names))]
    f1 = [report[str(i)]['f1-score'] for i in range(len(class_names))]
    plot_classification_metrics(precision, recall, f1, class_names, metrics_file, title=f"Per-Class Metrics - {model_name} ({split_name})")

    return accuracy, precision, recall, f1

# ------------------------------
# Evaluate on Test, Train, and Full Data
# ------------------------------
acc_test, prec_test, rec_test, f1_test = evaluate_model(X_test, y_test, "test")
# acc_train, prec_train, rec_train, f1_train = evaluate_model(X_train, y_train, "train")
# acc_all, prec_all, rec_all, f1_all = evaluate_model(X, y, "all")

# ------------------------------
# Save Metrics
# ------------------------------
save_metrics_with_all_scores(
    model_name=model_name,
    split_name="Test",
    y_true=y_test,
    y_pred=np.argmax(model.predict(X_test), axis=1)
)

# ------------------------------
# Save Model
# ------------------------------
model.save(f"models/{model_name}.keras")

# ------------------------------
# Convert to Fully Integer Quantized TFLite
# ------------------------------
def representative_dataset_gen():
    for i in range(min(100, len(X_train))):
        # Ensure input shape matches (1, 6, 1) and dtype is float32
        yield [X_train[i:i+1].astype(np.float32)]

# Initialize converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Set optimization and representative dataset
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen

# Force full integer quantization for ops and tensors
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Enforce integer-only weights
converter.experimental_new_converter = True

# Convert and save
tflite_model = converter.convert()
int8_model_path = f"models/{model_name}_int8.tflite"
with open(int8_model_path, "wb") as f:
    f.write(tflite_model)

logging.info(f"Fully Integer Quantized TFLite model saved to {int8_model_path}")

def convert_tflite_to_c_array(tflite_path, output_path, array_name="model_data"):
    with open(tflite_path, "rb") as f:
        data = f.read()

    with open(output_path, "w") as f:
        f.write(f"const unsigned char {array_name}[] = {{\n")
        for i in range(0, len(data), 12):
            line = ", ".join(f"0x{byte:02x}" for byte in data[i:i+12])
            f.write(f"  {line},\n")
        f.write("};\n")
        f.write(f"const unsigned int {array_name}_len = {len(data)};\n")

# convert quantized model into C arrays
C_model_path = f"models/{model_name}_int8.cc"
convert_tflite_to_c_array(int8_model_path, C_model_path, "cnn_model")