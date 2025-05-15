# prepare_data.py — one-time preprocessing and save to disk
import numpy as np
import joblib
import json
from preprocessing import load_and_preprocess_data
import os
from utils import setup_logging

# Setup logging
setup_logging(log_name="PrepareData", tee_console=True)

# Create models directory if not exists
os.makedirs("./models", exist_ok=True)
os.makedirs("./data/processed", exist_ok=True)

# Run preprocessing once
X, y, selected_features, class_names = load_and_preprocess_data(
    "./data/TON_IoT/Train_Test_datasets/train_test_network.csv")

# Save X and y
np.save("./data/processed/X.npy", X)
np.save("./data/processed/y.npy", y)

# Save class names for CNN/metrics
with open("./models/class_names.json", "w") as f:
    json.dump(list(class_names), f, indent=4)

print("[*] Preprocessing complete. Data saved to ./data/processed")
