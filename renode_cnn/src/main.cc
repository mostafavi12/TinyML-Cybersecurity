#include "model_data.c"
#include "model.h"

#include "include/tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "include/tensorflow/lite/micro/micro_interpreter.h"
#include "include/tensorflow/lite/schema/schema_generated.h"
#include "include/tensorflow/lite/version.h"

constexpr int kTensorArenaSize = 20 * 1024;
uint8_t tensor_arena[kTensorArenaSize];

#define UART ((volatile char *)0x10000000)

__attribute__((used))
void uart_puts(const char *str) {
    while (*str) {
        UART[0] = *str++;
    }
}

__attribute__((used))
void uart_putc(char c) {
    UART[0] = c;
}

__attribute__((used))
void uart_put_hex(unsigned int val) {
    const char *hex = "0123456789ABCDEF";
    for (int i = 28; i >= 0; i -= 4)
        uart_putc(hex[(val >> i) & 0xF]);
}

__attribute__((used)) __attribute__((section(".data")))
volatile int input_features[6] = {50, 30, 10, 70, 60, 90};

__attribute__((used))
int mock_infer(volatile int *features, int len) {
    int sum = 0;
    for (int i = 0; i < len; i++) {
        sum += features[i];
    }
    return (sum > 300) ? 1 : 0;
}

__attribute__((used))
int main(void) {
   
    int score = 0;
    // score = predict((const int* )input_features);

    uart_puts("Score (x1000): 0x");
    uart_put_hex(score);
    uart_puts("\n");

    if (score > 0)
        uart_puts("Prediction: ANOMALY\n");
    else
        uart_puts("Prediction: NORMAL\n");

    while (1);
}
void _start() {
    __asm__ volatile("lui sp, 0x00400");  // Load upper immediate for 0x00400000
    __asm__ volatile("addi sp, sp, -4");  // Adjust to 0x003FFFFC
    main();
    while (1);
}