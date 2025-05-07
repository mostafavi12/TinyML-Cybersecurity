import os
import json
import warnings
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def setup_logging(log_filename="run.log", log_dir="logs", level=logging.INFO):
    log_path = os.path.join(os.path.dirname(__file__), "..", log_dir)
    os.makedirs(log_path, exist_ok=True)

    full_log_path = os.path.join(log_path, log_filename)

    logging.basicConfig(
        filename=full_log_path,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=level
    )

    logging.captureWarnings(True)  # Redirect warnings to logging
    warnings.simplefilter("ignore", DeprecationWarning)

    logging.info("=== Logging Initialized ===")


def save_metric(model_name, accuracy):
    os.makedirs("metrics", exist_ok=True)
    metrics_file = "metrics/metrics.json"

    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            data = json.load(f)
    else:
        data = {}

    data[model_name] = {"accuracy": accuracy}

    with open(metrics_file, "w") as f:
        json.dump(data, f, indent=4)


def plot_confusion_matrix(cm, class_names, filename, title="Confusion Matrix"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label"
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=11)
    plt.setp(ax.get_yticklabels(), fontsize=11)

    # Add text annotations with bold labels
    cm_sum = cm.sum()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            percent = (value / cm_sum) * 100 if cm_sum else 0
            ax.text(j, i, f"{value}\n{percent:.1f}%",
                    ha="center", va="center",
                    fontsize=10, color="black", weight="bold")

    fig.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
    plt.close()
