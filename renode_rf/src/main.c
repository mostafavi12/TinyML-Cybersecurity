
#include "model.h"

// UART output address (adjust as needed)
#define UART_ADDR ((volatile char *)0x10000000)

__attribute__((used))
void uart_putc(char c)
{
    UART_ADDR[0] = c;
}

// Basic UART print function
__attribute__((used))
void uart_puts(const char *str)
{
    while (*str)
    {
        UART_ADDR[0] = *str++;
    }
}

__attribute__((used))
void uart_put_hex(unsigned int val)
{
    const char *hex = "0123456789ABCDEF";
    for (int i = 28; i >= 0; i -= 4)
        uart_putc(hex[(val >> i) & 0xF]);
}

__attribute__((used)) __attribute__((section(".data")))
float input_data[][28] = {
    {0.842100f, -0.212300f, 0.512300f, 0.702300f, -0.004000f, 1.102300f, 1.034500f, 0.313400f, -0.402100f, 0.824400f, 0.915600f, 0.624300f, 1.221400f, 0.455100f, 0.632000f, -0.332400f, 0.722100f, 0.610200f, 0.422300f, -0.312400f, 0.232100f, 0.843200f, -0.432200f, 0.822100f, -0.124300f, 0.713400f, 0.954300f, 0.582300f},
    {-0.221000f, 1.234500f, 0.512300f, -0.302300f, -0.214000f, -0.413400f, -0.623500f, 0.012300f, -0.402100f, -0.212400f, -0.145600f, -0.324300f, -0.154600f, 0.455100f, 0.632000f, -0.332400f, 0.722100f, -0.489800f, -0.302300f, 0.823200f, 0.432100f, 1.123200f, -0.332200f, 0.712300f, 0.843200f, -0.723400f, 0.213400f, -0.128300f},
    {1.432100f, -0.142300f, 0.712300f, 0.602300f, 0.143200f, 1.243000f, 1.384500f, 0.613400f, 0.342100f, 1.124400f, 1.715600f, 1.024300f, 1.821400f, 0.455100f, 0.632000f, -0.332400f, 0.722100f, 0.610200f, 0.422300f, -0.312400f, 0.632100f, 0.743200f, 0.732200f, -0.112300f, 0.912300f, 0.834200f, 1.234300f, 1.082300f},
    {-0.431000f, -0.412300f, 1.712300f, 1.202300f, -0.304000f, -0.713400f, -0.823500f, 1.012300f, -0.402100f, -0.512400f, -0.845600f, -0.724300f, -0.954600f, 0.455100f, 1.132000f, -0.332400f, -0.722100f, -0.810200f, -0.622300f, 1.212400f, 0.632100f, 0.843200f, 1.132200f, 1.212300f, -0.524300f, 0.913400f, -0.484300f, -0.782300f},
    {-0.521000f, -0.512300f, 0.512300f, -0.302300f, 0.244000f, 1.034000f, 1.204500f, 0.312300f, 0.402100f, 0.424400f, 1.215600f, 0.824300f, 1.421400f, -0.545100f, -0.432000f, -0.332400f, 0.722100f, 0.610200f, 0.422300f, 1.012400f, 0.832100f, 1.143200f, 0.632200f, -0.812300f, 0.123400f, -0.623400f, 0.734300f, 0.882300f}};

const int NUM_SAMPLES = sizeof(input_data) / sizeof(input_data[0]);

__attribute__((used))
int main()
{
    uart_puts("Booting RF IDS firmware...\n");
    
    for (int i = 0; i < NUM_SAMPLES; ++i)
    {
        int prediction = predict(input_data[i]);

        uart_puts("Score (x1000): 0x");
        uart_put_hex(prediction);
        uart_puts("\n");

        if (prediction == 5)
            uart_puts("Prediction: NORMAL\n");
        else
            uart_puts("Prediction: ANOMALY\n");
    }

    while (1)
        ; // idle

    return 0;
}

void _start() {
    __asm__ volatile("lui sp, 0x00400");  // Load upper immediate for 0x00400000
    __asm__ volatile("addi sp, sp, -4");  // Adjust to 0x003FFFFC
    main();
    while (1);
}