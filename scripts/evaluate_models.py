import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Directory containing all model metric JSON files
METRICS_DIR = "./metrics"

def load_metrics(metrics_dir):
    models = {}
    for filename in os.listdir(metrics_dir):
        if filename.endswith(".json"):
            model_name = filename.replace(".json", "")
            with open(os.path.join(metrics_dir, filename), "r") as f:
                data = json.load(f)
                models[model_name] = data
    return models

def plot_accuracy_comparison(models):
    datasets = ["train", "test", "full"]
    model_names = list(models.keys())

    # Data preparation
    bar_width = 0.2
    x = np.arange(len(model_names))

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, dataset in enumerate(datasets):
        accuracies = [
            models[model].get(f"{dataset}_accuracy", 0.0)
            for model in model_names
        ]
        ax.bar(x + i * bar_width, accuracies, bar_width, label=f"{dataset.capitalize()}")

    ax.set_xlabel("Model")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Comparison of Models")
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    os.makedirs("visualizations", exist_ok=True)
    plt.tight_layout()
    plt.savefig("visualizations/model_accuracy_comparison.png")
    plt.close()

if __name__ == "__main__":
    models = load_metrics(METRICS_DIR)
    if not models:
        print("No metrics found in the directory.")
    else:
        plot_accuracy_comparison(models)
        print("Saved: visualizations/model_accuracy_comparison.png")
