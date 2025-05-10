import argparse
import time
import joblib
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.preprocessing import load_and_preprocess_data
from common.utils import (
    save_metrics_with_all_scores,
    setup_logging,
    save_metric,
    plot_confusion_matrix,
    plot_classification_metrics,
    generate_model_name
)


# --- New: Define models via string keys
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

# --- Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="Base", help="Specify model type")
    args = parser.parse_args()

    model_type = args.model_type
    model_name = generate_model_name("RF", model_type)
    setup_logging(model_name)

    logging.info(f"[*] Training RF model: {model_type}")

    # Load and preprocess
    X, y, features, class_names = load_and_preprocess_data(
        "./data/TON_IoT/Train_Test_datasets/train_test_network.csv"
    )

    logging.info("[*] Feature headers: %s", features)
    logging.info("[*] Sample data:\n%s", X[:5])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    logging.info("Train shape: %s, Test shape: %s", X_train.shape, X_test.shape)
    logging.info("Train label distribution: %s", np.unique(y_train, return_counts=True))
    logging.info("Test label distribution: %s", np.unique(y_test, return_counts=True))

    # Build model
    model, config = build_model(model_type)
    logging.info(f"Model config: {config}")

    # Train model
    start_train = time.time()
    model.fit(X_train, y_train)
    end_train = time.time()

    training_duration = end_train - start_train
    logging.info(f"Training completed in {training_duration:.2f} seconds")

    # Save model
    joblib.dump(model, f"models/{model_name}.joblib")

    # Inference timing
    start_inf = time.time()
    _ = model.predict([X_test[0]])
    end_inf = time.time()
    single_inf = end_inf - start_inf
    logging.info(f"Single inference time: {single_inf:.6f} seconds")

    # Predictions
    for split_name, X_eval, y_eval in [
        #("train", X_train, y_train),
        ("test", X_test, y_test)#,
        #("all", X, y),
    ]:
        y_pred = model.predict(X_eval)

        acc = accuracy_score(y_eval, y_pred)
        logging.info(f"[{split_name.upper()}] Accuracy: {acc:.4f}")
        logging.info(f"[{split_name.upper()}] Classification report:\n{classification_report(y_eval, y_pred, target_names=class_names)}")

        # Save metrics
        save_metrics_with_all_scores(model_name, split_name, y_eval, y_pred)

        # Compute metrics
        precision, recall, f1, _ = precision_recall_fscore_support(y_eval, y_pred, zero_division=0)

        # Plot Confusion Matrix and per-class metrics
        cm = confusion_matrix(y_eval, y_pred)
        plot_confusion_matrix(cm, class_names=class_names, filename=f"confusion_matrix_{model_name}_{split_name}.png")
        plot_classification_metrics(precision, recall, f1, class_names=class_names, filename=f"metrics_{model_name}_{split_name}.png")