import argparse
import json
import logging
import os
import joblib
import sys
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.utils import (
    save_metrics_with_all_scores,
    setup_logging,
    plot_confusion_matrix,
    plot_classification_metrics,
    generate_model_name
)

MODEL_TYPES = ["RF_Base", "RF_DeepTrees"]
INPUT_TYPES = ["float32", "int32"]
os.makedirs("./metrics", exist_ok=True)
os.makedirs("./models", exist_ok=True)

# Load data
X = np.load("./data/processed/X.npy")
y = np.load("./data/processed/y.npy")
with open("./models/selected_features.json") as f:
    selected_features = json.load(f)
with open("./models/class_names.json") as f:
    class_names = json.load(f)
label_encoder = joblib.load("./models/label_encoder.pkl")

def build_model(model_type):
    config_map = {
        "RF_Base": {"n_estimators": 50, "max_depth": 10},
        "RF_DeepTrees": {"n_estimators": 100, "max_depth": None},
    }
    config = config_map.get(model_type, config_map["RF_Base"])
    return RandomForestClassifier(**config), config

# Init CSV summary
summary_rows = []
summary_csv_path = "./metrics/model_comparison.csv"
with open(summary_csv_path, "w") as f:
    f.write("model,input_type,accuracy,precision,recall,f1,train_time,inference_avg,inference_1sample,model_size_kb\n")

for model_type in MODEL_TYPES:
    for input_type in INPUT_TYPES:
        suffix = "int32" if input_type == "int32" else "float32"
        model_name = f"{model_type}_{suffix}"
        setup_logging(model_name)
        logging.info(f"[START] Training model {model_name}")

        # Preprocess input
        if input_type == "int32":
            scaler = MinMaxScaler(feature_range=(0, 255))
            X_scaled = scaler.fit_transform(X).astype(np.int32)
            joblib.dump(scaler, f"./models/{model_name}_scaler.pkl")
        else:
            X_scaled = X.astype(np.float32)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, stratify=y, random_state=42)

        # Train model
        model, config = build_model(model_type)
        t_start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t_start

        # Predict
        t_inf_start = time.time()
        y_pred = model.predict(X_test)
        inf_time_total = time.time() - t_inf_start
        inf_time_avg = inf_time_total / len(X_test)

        # Single-sample inference time
        t_single = time.time()
        _ = model.predict([X_test[0]])
        inf_time_single = time.time() - t_single

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        precision = report["weighted avg"]["precision"]
        recall = report["weighted avg"]["recall"]
        f1 = report["weighted avg"]["f1-score"]

        # Generate per-class metric plot (Precision, Recall, F1-score)
        metrics_file = f"metrics_{model_name}.png"
        precision_per_class = [report[str(i)]['precision'] for i in range(len(class_names))]
        recall_per_class = [report[str(i)]['recall'] for i in range(len(class_names))]
        f1_per_class = [report[str(i)]['f1-score'] for i in range(len(class_names))]

        plot_classification_metrics(
            precision_per_class,
            recall_per_class,
            f1_per_class,
            class_names,
            metrics_file,
            title=f"Per-Class Metrics - {model_name}"
        )
        
        # Save model
        model_path = f"./models/{model_name}.pkl"
        joblib.dump(model, model_path)
        model_size_kb = os.path.getsize(model_path) / 1024

        # Export to C
        try:
            from micromlgen import port
            header_code = port(model)
            with open(f"./models/{model_name}.h", "w") as f:
                f.write(header_code)
        except ImportError:
            logging.warning("micromlgen not installed. Skipping export.")
        except Exception as e:
            logging.error(f"Header export failed: {e}")

        # Save metrics JSON
        metrics = {
            "metrics": {
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1
            },
            "info": {
                "training_time_sec": train_time,
                "inference_time_avg_ms": inf_time_avg * 1000,
                "inference_time_single_ms": inf_time_single * 1000,
                "model_size_kb": model_size_kb,
                "input_type": input_type
            }
        }

        with open(f"./metrics/{model_name}.json", "w") as f:
            json.dump(metrics, f, indent=2)

        # Append to CSV
        with open(summary_csv_path, "a") as f:
            f.write(f"{model_type},{input_type},{acc:.4f},{precision:.4f},{recall:.4f},{f1:.4f},"
                    f"{train_time:.4f},{inf_time_avg*1000:.4f},{inf_time_single*1000:.4f},{model_size_kb:.2f}\n")

        logging.info(f"[DONE] Saved model and metrics for {model_name}\n")