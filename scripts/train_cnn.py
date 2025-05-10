import logging
import numpy as np
import tensorflow as tf
import joblib
import sys, os, argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.utils import (
    save_metrics_with_all_scores, setup_logging, plot_confusion_matrix, save_metric, generate_model_name, plot_classification_metrics
)
from common.preprocessing import load_and_preprocess_data


def build_model(model_type, input_shape):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))

    if model_type == "Base":
        model.add(tf.keras.layers.Conv1D(32, 3, activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation='relu'))
    elif model_type == "Deep":
        model.add(tf.keras.layers.Conv1D(64, 3, activation='relu'))
        model.add(tf.keras.layers.Conv1D(64, 3, activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
    elif model_type == "Wider":
        model.add(tf.keras.layers.Conv1D(128, 3, activation='relu'))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
    elif model_type == "Shallow":
        model.add(tf.keras.layers.Conv1D(16, 3, activation='relu'))
        model.add(tf.keras.layers.Flatten())
    elif model_type == "Tiny":
        model.add(tf.keras.layers.Conv1D(8, 3, activation='relu'))
        model.add(tf.keras.layers.Flatten())
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.add(tf.keras.layers.Dense(10, activation='softmax'))  # 10 attack types
    return model


# Argument Parsing
parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default="Base", help="Model type: Base, Deep, Wider, Shallow, Tiny")
args = parser.parse_args()
model_type = args.model_type
model_name = generate_model_name("CNN", model_type)

# Setup logging
setup_logging(model_name)
logging.info(f"[*] Training CNN model: {model_type}")

# Load Data
X, y, features, class_labels = load_and_preprocess_data(
    "./data/TON_IoT/Train_Test_datasets/train_test_network.csv"
)

# Reshape for CNN
X = np.expand_dims(X, axis=-1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))
logging.info(f"[*] Computed class weights: {class_weights}")

# Build Model
model = build_model(model_type, input_shape=(X.shape[1], 1))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2,
          class_weight=class_weights, verbose=2)

# Evaluate
for dataset_name, data, labels in [
    #("train", X_train, y_train),
    ("test", X_test, y_test)#,
    #("all", X, y)
]:
    logging.info(f"[*] Evaluating on {dataset_name} set...")
    y_pred_probs = model.predict(data)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(labels, y_pred)
    logging.info(f"{dataset_name.capitalize()} Accuracy: {acc:.4f}")
    # save_metric(f"{model_name}_{dataset_name}", acc)
    save_metrics_with_all_scores(model_name, dataset_name, labels, y_pred)

    report = classification_report(labels, y_pred, target_names=class_labels, zero_division=0)
    logging.info(f"\nClassification Report ({dataset_name}):\n{report}")

    # Confusion matrix
    cm = confusion_matrix(labels, y_pred)
    plot_confusion_matrix(cm, class_labels,
                          filename=f"confusion_matrix_{model_name}_{dataset_name}.png")

    # Per-class metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, y_pred, zero_division=0
    )
    plot_classification_metrics(
        precision, recall, f1,
        class_labels,
        f"per_class_metrics_{model_name}_{dataset_name}.png",
        title=f"{model_name} - {dataset_name.capitalize()} Set"
    )

# Save
model.save(f"models/{model_name}.keras")

# Export TFLite
logging.info("[*] Converting to TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open(f"models/{model_name}.tflite", "wb") as f:
    f.write(tflite_model)

logging.info(f"[✓] Training and conversion for {model_name} completed.")