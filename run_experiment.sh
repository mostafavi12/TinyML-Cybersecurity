#!/bin/bash
# Run Random Forest Base
echo "Training RandomForest (Base)..."
python3 scripts/train_random_forest.py --model_type Base > logs/RFExp_Base.log 2>&1

# Run CNN Deep
echo "Training CNN (Deep)..."
python3 scripts/train_cnn.py --model_type Base

echo "Evaluating Models..."
python3 scripts/evaluate_models.py > logs/EvaluationExp.log 2>&1