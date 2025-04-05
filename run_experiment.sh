#!/bin/bash

echo "Training RandomForest..."
python3 scripts/train_random_forest.py

echo "Training CNN..."
python3 scripts/train_cnn.py

echo "Training RandomForest Tuned..."
python3 scripts/train_random_forest_tuned.py

echo "Evaluating Models..."
python3 scripts/evaluate_models.py
