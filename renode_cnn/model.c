#include "model.h"
#include "cnn_model_data.h" // contains cnn_model_tflite[]

#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Memory area for tensors
#define TENSOR_ARENA_SIZE 8 * 1024
static uint8_t tensor_arena[TENSOR_ARENA_SIZE];

int predict(const int* features) {
    // Load model
    const tflite::Model* model = tflite::GetModel(cnn_model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        return -1;
    }

    static tflite::AllOpsResolver resolver;
    static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, TENSOR_ARENA_SIZE);

    interpreter.AllocateTensors();

    TfLiteTensor* input = interpreter.input(0);

    // Assuming 6 features, float32 model
    for (int i = 0; i < 6; i++) {
        input->data.f[i] = (float)features[i] / 100.0f;  // Normalize if needed
    }

    interpreter.Invoke();

    TfLiteTensor* output = interpreter.output(0);

    int max_index = 0;
    float max_val = output->data.f[0];
    for (int i = 1; i < output->dims->data[1]; ++i) {
        if (output->data.f[i] > max_val) {
            max_val = output->data.f[i];
            max_index = i;
        }
    }

    return max_index;
}