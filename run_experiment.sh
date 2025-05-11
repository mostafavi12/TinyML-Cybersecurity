#!/bin/bash

# <<comment
# Run Random Forest Base
echo "Training RandomForest (Base)..."
python3 scripts/train_random_forest.py --model_type Base

# Run Random Forest ManyTrees
echo "Training RandomForest (ManyTrees)..."
python3 scripts/train_random_forest.py --model_type ManyTrees

# Run Random Forest DeepTrees
echo "Training RandomForest (DeepTrees)..."
python3 scripts/train_random_forest.py --model_type DeepTrees

# Run Random Forest Conservative
echo "Training RandomForest (Conservative)..."
python3 scripts/train_random_forest.py --model_type Conservative

# Run Random Forest Balanced
echo "Training RandomForest (Balanced)..."
python3 scripts/train_random_forest.py --model_type Balanced

# Run CNN Base
echo "Training CNN (Base)..."
python3 scripts/train_cnn.py --model_type Base

# Run CNN Tiny
echo "Training CNN (Tiny)..."
python3 scripts/train_cnn.py --model_type Tiny

# Run CNN Deep
echo "Training CNN (Deep)..."
python3 scripts/train_cnn.py --model_type Deep

# Run CNN Wider
echo "Training CNN (Wider)..."
python3 scripts/train_cnn.py --model_type Wider

# Run CNN Shallow
echo "Training CNN (Shallow)..."
python3 scripts/train_cnn.py --model_type Shallow

# Run CNN Dropout
echo "Training CNN (Dropout)..."
python3 scripts/train_cnn.py --model_type Dropout

# comment

# Run CNN BatchNorm
echo "Training CNN (BatchNorm)..."
python3 scripts/train_cnn.py --model_type BatchNorm



echo "Evaluating Models..."
python3 scripts/evaluate_models.py