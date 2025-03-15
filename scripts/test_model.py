# Test the Model Using the Remaining 30% of the Dataset
# Load the TensorFlow Lite runtime
import tensorflow.lite as tflite

# Load the trained TinyML model
interpreter = tflite.Interpreter(model_path="model_ton_iot.tflite")
interpreter.allocate_tensors()

# Get input/output tensor indices
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Run inference on the test dataset
correct_predictions = 0
total_predictions = len(X_test)

print("[*] Testing TinyML model with the remaining 30% of TON_IoT dataset...")

for i in range(len(X_test)):
    input_data = np.array([X_test[i]], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_class = 1 if output_data[0] > 0.5 else 0
    actual_class = y_test.iloc[i]

    if predicted_class == actual_class:
        correct_predictions += 1

accuracy = correct_predictions / total_predictions * 100
print(f"[✓] Model Accuracy on Test Data: {accuracy:.2f}%") 