#include "include/tensorflow/lite/micro/all_ops_resolver.h"
#include "include/tensorflow/lite/micro/micro_error_reporter.h"
#include "include/tensorflow/lite/micro/micro_interpreter.h"
#include "include/tensorflow/lite/schema/schema_generated.h"
#include "include/tensorflow/lite/version.h"
#include "model_data.h"
#include "include/tensorflow/lite/micro/micro_mutable_op_resolver.h"

// Constants
constexpr int kTensorArenaSize = 10 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

int main() {
  // Set up logging
  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Load model
  const tflite::Model* model = tflite::GetModel(cnn_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report("Model schema mismatch!");
    return 1;
  }

  // Set up op resolver
  static tflite::AllOpsResolver resolver;

  // Set up interpreter
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize, error_reporter);

  tflite::MicroInterpreter* interpreter = &static_interpreter;

  // Allocate memory
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    error_reporter->Report("Tensor allocation failed");
    return 1;
  }

  // Set up input tensor
  TfLiteTensor* input = interpreter->input(0);
  if (input->type != kTfLiteInt8) {
    error_reporter->Report("Expected int8 input");
    return 1;
  }

  // Fill dummy input
  int8_t input_data[6] = {12, -45, 30, -100, 45, 0};
  for (int i = 0; i < 6; i++) {
    input->data.int8[i] = input_data[i];
  }

  // Run inference
  if (interpreter->Invoke() != kTfLiteOk) {
    error_reporter->Report("Inference failed");
    return 1;
  }

  // Get output tensor
  TfLiteTensor* output = interpreter->output(0);

  // Find max scoring class
  int max_index = 0;
  int8_t max_score = output->data.int8[0];
  for (int i = 1; i < output->dims->data[1]; i++) {
    if (output->data.int8[i] > max_score) {
      max_score = output->data.int8[i];
      max_index = i;
    }
  }

  MicroPrintf("Predicted class: %d\n", max_index);

  return 0;
}