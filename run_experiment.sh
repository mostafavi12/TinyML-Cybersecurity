#!/bin/bash
echo "Training CNN..."
python3 scripts/train_cnn.py

echo "Training RandomForest..."
python3 scripts/train_random_forest.py

echo "Evaluating Models..."
python3 scripts/evaluate_models.py
