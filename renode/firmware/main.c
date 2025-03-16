#include <stdio.h>
#include "tensorflow/lite/c/c_api.h"

int main() {
    printf("Loading TinyML Model...\n");

    TfLiteModel* model = TfLiteModelCreateFromFile("cnn_model.tflite");
    if (!model) {
        printf("Failed to load TFLite model.\n");
        return 1;
    }

    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);
    TfLiteInterpreterAllocateTensors(interpreter);

    TfLiteTensor* input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
    float input_data[6] = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    memcpy(input_tensor->data.f, input_data, sizeof(input_data));

    TfLiteInterpreterInvoke(interpreter);
    TfLiteTensor* output_tensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);
    float result = output_tensor->data.f[0];

    printf("Inference Result: %f\n", result);

    TfLiteInterpreterDelete(interpreter);
    TfLiteInterpreterOptionsDelete(options);
    TfLiteModelDelete(model);

    return 0;
}
