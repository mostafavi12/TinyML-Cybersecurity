volatile unsigned int *uart = (unsigned int *)0x10000000;
#define UART_THR (uart + 0)
#define UART_LSR (uart + 5)
#define UART_LSR_THRE 0x20

void uart_putc(char c) {
    while (!(UART_LSR[0] & UART_LSR_THRE));
    UART_THR[0] = c;
}

void uart_puts(const char *s) {
    while (*s) uart_putc(*s++);
}

int main() {
    uart_puts("Hello TinyML!\n");
    while (1); // loop forever

    return 0;
}

void _start(){
    main();
    while(1);
}