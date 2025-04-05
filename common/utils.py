import os
import json
import numpy as np
import matplotlib.pyplot as plt

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

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True label",
        xlabel="Predicted label"
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add percentage annotations
    cm_sum = cm.sum()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            percent = (value / cm_sum) * 100 if cm_sum else 0
            ax.text(j, i, f"{value}\n{percent:.1f}%", ha="center", va="center", color="black", fontsize=9)

    fig.tight_layout()
    plt.savefig(filename)
    plt.close()
