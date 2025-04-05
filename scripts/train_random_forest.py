import joblib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from common.preprocessing import load_and_preprocess_data

from sklearn.metrics import confusion_matrix
from common.utils import plot_confusion_matrix
from common.utils import save_metric

# Create output directory if it doesn't exist
os.makedirs("visualizations", exist_ok=True)

print("[*] Loading TON_IoT dataset...")
X, y, features = load_and_preprocess_data("./data/TON_IoT/Train_Test_datasets/Train_Test_Network_dataset/train_test_network.csv")

print("[*] Feature headers:", features)
print("[*] Sample data:\n", X[:5])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("[*] Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("[*] Evaluating Random Forest model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Test Accuracy: {accuracy:.4f}")

print("[*] Evaluating Random Forest model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Test Accuracy: {accuracy:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
# Assuming y_test and y_pred are defined
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, class_names=["Normal", "Anomaly"], filename="visualizations/confusion_matrix_rf.png")


# Save model
joblib.dump(model, "models/random_forest.pkl")
print("Random Forest model saved at models/random_forest.pkl")

# Save the report in a json file
save_metric("Random Forest", accuracy)