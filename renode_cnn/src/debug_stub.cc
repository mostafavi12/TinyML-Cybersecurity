#include <cstdarg>
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/core/api/error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"  // for TensorType

extern "C" int TfLiteIntArrayGetSizeInBytes(int size) {
  return size * sizeof(int);
}

namespace tflite {

TfLiteStatus ConvertTensorType(TensorType, TfLiteType*, ErrorReporter*) {
  return kTfLiteOk;
}

class StubErrorReporter : public ErrorReporter {
 public:
  int Report(const char* format, va_list args) override { return 0; }
};

ErrorReporter* GetStubErrorReporter() {
  static StubErrorReporter reporter;
  return &reporter;
}

// This overload is also used in some builds
int ErrorReporter::Report(const char* format, ...) {
  return 0;
}

extern "C" int ReportError(void*, const char*, ...) {
  return 0;
}

}  // namespace tflite