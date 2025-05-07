import logging
import joblib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from common.preprocessing import load_and_preprocess_data

from sklearn.metrics import confusion_matrix
from common.utils import plot_confusion_matrix
from common.utils import save_metric

from common.utils import setup_logging
setup_logging("random_forest.log")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
BASE_DIR = os.path.dirname(__file__)
VIS_DIR = os.path.join(BASE_DIR, "visualizations")
os.makedirs(VIS_DIR, exist_ok=True)

# Create output directory if it doesn't exist
os.makedirs("visualizations", exist_ok=True)

logging.info("Loading TON_IoT dataset...")
X, y, features = load_and_preprocess_data("./data/TON_IoT/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv")

logging.info("Feature headers:", features)
logging.info("Sample data:\n", X[:5])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

logging.info("Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

logging.info("Evaluating Random Forest model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
logging.info(f"Random Forest Test Accuracy: {accuracy:.4f}")

# Classification Report
logging.info("\nClassification Report:")
logging.info(classification_report(y_test, y_pred, zero_division=0))

# Confusion Matrix
# Assuming y_test and y_pred are defined
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, class_names=["Normal", "Anomaly"], filename=os.path.join(VIS_DIR, "confusion_matrix_rf.png"))


# Save model
joblib.dump(model, "models/random_forest.pkl")
logging.info("Random Forest model saved at models/random_forest.pkl")

# Save the report in a json file
save_metric("Random Forest", accuracy)