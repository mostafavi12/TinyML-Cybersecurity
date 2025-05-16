import json
import os
import matplotlib.pyplot as plt

# Directory containing all model metric JSON files
METRICS_DIR = "./metrics"
VIS_DIR = "./visualizations"
os.makedirs(VIS_DIR, exist_ok=True)

def load_metrics(metrics_file):
    with open(metrics_file, "r") as f:
        return json.load(f)

def plot_comparison(models, metric_name, output_file):
    model_names = []
    metric_values = []

    for model_name, result in models.items():
        if "metrics" in result and metric_name in result["metrics"]:
            model_names.append(model_name)
            metric_values.append(result["metrics"][metric_name])

    if not model_names:
        print(f"No valid data for {metric_name}. Skipping plot.")
        return

    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, metric_values)
    plt.title(f"Model {metric_name.capitalize()} Comparison")
    plt.xlabel("Model")
    plt.ylabel(metric_name.capitalize())
    plt.xticks(rotation=45)
    plt.ylim(0, 1)

    for bar, val in zip(bars, metric_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val * 100:.1f}%",
         ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, f"model_{metric_name}_comparison.png"))
    plt.close()

if __name__ == "__main__":
    metrics_file = "metrics/metrics.json"
    models = load_metrics(metrics_file)

    for metric in ["accuracy", "precision", "recall", "f1"]:
        plot_comparison(models, metric, f"model_{metric}_comparison.png")
