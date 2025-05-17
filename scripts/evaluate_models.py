import json
import os
import matplotlib.pyplot as plt

METRICS_DIR = "./metrics"
VIS_DIR = "./visualizations"
os.makedirs(VIS_DIR, exist_ok=True)

def load_all_metrics():
    models = {}
    for fname in os.listdir(METRICS_DIR):
        if fname.endswith(".json") and not fname.startswith("metrics"):  # skip any summary files
            path = os.path.join(METRICS_DIR, fname)
            with open(path, "r") as f:
                data = json.load(f)
            model_name = fname.replace(".json", "")
            models[model_name] = data
    return models

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
    models = load_all_metrics()

    for metric in ["accuracy", "precision", "recall", "f1"]:
        plot_comparison(models, metric, f"model_{metric}_comparison.png")

    # Optional: log extended info to console
    print("\n[INFO] Summary of extended metrics:")
    for model_name, data in models.items():
        info = data.get("info", {})
        print(f"{model_name}: size={info.get('model_size_kb', 0):.1f}KB, "
              f"train_time={info.get('training_time_sec', 0):.2f}s, "
              f"infer_avg={info.get('inference_time_avg_ms', 0):.3f}ms, "
              f"infer_1sample={info.get('inference_time_single_ms', 0):.3f}ms")
