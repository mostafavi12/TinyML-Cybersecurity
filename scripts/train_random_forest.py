# Updated train_random_forest.py with dynamic feature handling, class distribution logging, and sample prediction
import argparse
import json
import logging
import os
import joblib
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.utils import (
    save_metrics_with_all_scores,
    setup_logging,
    plot_confusion_matrix,
    plot_classification_metrics,
    generate_model_name
)

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, default="RF_Base")
args = parser.parse_args()
model_type = args.model_type

model_name = generate_model_name("RF", model_type)
setup_logging(model_name)
logging.info("[*] Training Random Forest model: %s", model_type)

# Load preprocessed data
X = np.load("./data/processed/X.npy")
y = np.load("./data/processed/y.npy")

# Load metadata
with open("./models/selected_features.json") as f:
    selected_features = json.load(f)
with open("./models/class_names.json") as f:
    class_names = json.load(f)
label_encoder = joblib.load("./models/label_encoder.pkl")

# Log selected features
logging.info("[*] Feature headers: %s", selected_features)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Log class distribution
unique, train_counts = np.unique(y_train, return_counts=True)
logging.info("Train label distribution: %s", [f"{class_names[i]}={c}" for i, c in zip(unique, train_counts)])
unique, test_counts = np.unique(y_test, return_counts=True)
logging.info("Test label distribution: %s", [f"{class_names[i]}={c}" for i, c in zip(unique, test_counts)])

# --- Define models via string keys
def build_model(model_type):
    config_map = {
        "RF_Base": {"n_estimators": 50, "max_depth": 10},
        "RF_ManyTrees": {"n_estimators": 200, "max_depth": 20},
        "RF_DeepTrees": {"n_estimators": 100, "max_depth": None},
        "RF_Conservative": {"n_estimators": 100, "min_samples_split": 10, "min_samples_leaf": 4},
        "RF_Balanced": {"n_estimators": 100, "class_weight": "balanced"},
    }
    config = config_map.get(model_type, config_map["RF_Base"])
    return RandomForestClassifier(**config), config

model, config = build_model(model_type)
logging.info("[*] RF config: %s", config)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
logging.info(f"Accuracy: {acc:.4f}")
logging.info(f"Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")

# Confusion Matrix
cm_file = f"confusion_matrix_{model_name}.png"
plot_confusion_matrix(confusion_matrix(y_test, y_pred), class_names=class_names, filename=cm_file)

# Per-class metrics
metrics_file = f"metrics_{model_name}.png"
precision = [report[str(i)]['precision'] for i in range(len(class_names))]
recall = [report[str(i)]['recall'] for i in range(len(class_names))]
f1 = [report[str(i)]['f1-score'] for i in range(len(class_names))]
plot_classification_metrics(precision, recall, f1, class_names, metrics_file, title=f"Per-Class Metrics - {model_name}")

# Save model
os.makedirs("./models", exist_ok=True)
joblib.dump(model, f"./models/{model_name}.pkl")

# Save metrics
save_metrics_with_all_scores(
    model_name=model_name,
    split_name="Test",
    y_true=y_test,
    y_pred=y_pred
)

# Predict one sample
sample_idx = 0
sample_pred = model.predict([X_test[sample_idx]])[0]
logging.info(f"[*] Single sample prediction: {class_names[sample_pred]}")

# === Export to C++ header using micromlgen ===
try:
    from micromlgen import port

    header_code = port(model)
    header_filename = f"./models/{model_name}.h"
    with open(header_filename, "w") as f:
        f.write(header_code)
    logging.info(f"[*] Exported model to C header: {header_filename}")
except ImportError:
    logging.warning("[!] micromlgen is not installed. Skipping model export to C.")
except Exception as e:
    logging.error(f"[!] Failed to export model to C: {e}")