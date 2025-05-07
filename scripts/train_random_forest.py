import logging
import joblib
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from common.preprocessing import load_and_preprocess_data

from sklearn.metrics import confusion_matrix
from common.utils import plot_confusion_matrix
from common.utils import save_metric

from common.utils import setup_logging
setup_logging("RF_Tuned.log")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
BASE_DIR = os.path.dirname(__file__)
VIS_DIR = os.path.join(BASE_DIR, "visualizations")
os.makedirs(VIS_DIR, exist_ok=True)

logging.info("[*] Loading TON_IoT dataset...")
X, y, features = load_and_preprocess_data("./data/TON_IoT/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv")

logging.info("[*] Feature headers:", features)
logging.info("[*] Sample data:\n", X[:5])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

logging.info("[*] Performing Hyperparameter Tuning for Random Forest...")
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
logging.info(f"Best Parameters: {grid_search.best_params_}")

logging.info("[*] Evaluating Tuned Random Forest model...")
y_pred = best_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Tuned RandomForest Test Accuracy: {accuracy:.4f}")

# Classification Report
logging.info("\nClassification Report:")
logging.info(classification_report(y_test, y_pred, zero_division=0))

# Confusion Matrix
# Assuming y_test and y_pred are defined
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, class_names=["Normal", "Anomaly"], filename=os.path.join(VIS_DIR, "confusion_matrix_rfTuned.png"))

# Save the best model
joblib.dump(best_rf, "models/random_forest_tuned.pkl")
logging.info("Tuned RandomForest model saved at models/random_forest_tuned.pkl")

# Save the report in a json file
save_metric("Tuned RF", accuracy)