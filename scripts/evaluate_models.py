import joblib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from common.preprocessing import load_and_preprocess_data

# Load data
X, y, _ = load_and_preprocess_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Load and evaluate RandomForest
"""
rf_model = joblib.load("./models/random_forest.pkl")
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"RandomForest Accuracy: {rf_accuracy:.4f}")
"""

# Load and evaluate CNN (TensorFlow Lite)
import tensorflow.lite as tflite

print("Loading CNN TensorFlow Lite model...")
interpreter = tflite.Interpreter(model_path="./models/cnn_tinyml_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

correct_predictions = 0
for i in range(len(X_test)):
    input_data = X_test[i:i+1].astype('float32')  # Convert to float32
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    if output_data.argmax() == y_test[i]:
        correct_predictions += 1

cnn_accuracy = correct_predictions / len(y_test)
print(f"CNN (TinyML) Accuracy: {cnn_accuracy:.4f}")
