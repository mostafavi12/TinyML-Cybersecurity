import os
import json
import sys
import warnings
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

def setup_logging(log_name="default", log_dir="logs", level=logging.INFO, tee_console=False):
    log_dir_path = os.path.join(os.path.dirname(__file__), "..", log_dir)
    os.makedirs(log_dir_path, exist_ok=True)

    full_log_path = os.path.join(log_dir_path, f"{log_name}.log")

    # Setup logging
    handlers = [logging.FileHandler(full_log_path, mode="w")]
    if tee_console:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers
    )

    # Redirect stdout and stderr to log file
    sys.stdout = open(full_log_path, "a")
    sys.stderr = open(full_log_path, "a")

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
    BASE_DIR = os.path.dirname(__file__)
    VIS_DIR = os.path.join(BASE_DIR, "..","visualizations")
    os.makedirs(VIS_DIR, exist_ok=True)

    plot_file_path = os.path.join(VIS_DIR, filename)

    fig, ax = plt.subplots(figsize=(9, 8))
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

    #plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=11)
    #plt.setp(ax.get_yticklabels(), fontsize=11)

    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(fontsize=10)

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
    plt.savefig(plot_file_path, bbox_inches="tight")
    plt.close()
