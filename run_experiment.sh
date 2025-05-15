#!/bin/bash


# Updated run_experiment.sh
set -e

# 1. Prepare data only once
python3 common/prepare_data.py

<<comment
# 2. Run all RF models
for model_type in Base ManyTrees DeepTrees Conservative Balanced; do
    echo "[*] Training Random Forest model: $model_type"
    python3 scripts/train_random_forest.py --model_type $model_type
done

# 3. Run all CNN models
for model_type in Base Tiny Deep Wider Shallow Dropout BatchNorm; do
    echo "[*] Training CNN model: $model_type"
    python3 scripts/train_cnn.py --model_type $model_type
done
comment 

python3 scripts/train_random_forest.py --model_type Balanced

# python3 scripts/train_cnn.py --model_type Base

echo "Evaluating Models..."
# python3 scripts/evaluate_models.py