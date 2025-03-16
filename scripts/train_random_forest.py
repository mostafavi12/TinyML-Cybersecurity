import joblib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from common.preprocessing import load_and_preprocess_data

# Load data
X, y, features = load_and_preprocess_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train RandomForest model
print("[*] Training RandomForest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"[✓] RandomForest Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "./models/random_forest.pkl")
print("[✓] Model saved at './models/random_forest.pkl'")
