import joblib
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from common.preprocessing import load_and_preprocess_data

print("[*] Loading TON_IoT dataset...")
X, y, features = load_and_preprocess_data("./data/TON_IoT/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv")

print("[*] Feature headers:", features)
print("[*] Sample data:\n", X[:5])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("[*] Performing Hyperparameter Tuning for Random Forest...")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

print("[*] Evaluating Tuned Random Forest model...")
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Tuned RandomForest Test Accuracy: {accuracy:.4f}")

# Save the best model
joblib.dump(best_rf, "models/random_forest_tuned.pkl")
print("Tuned RandomForest model saved at models/random_forest_tuned.pkl")