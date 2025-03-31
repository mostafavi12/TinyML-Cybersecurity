// main.c
volatile unsigned char *uart = (unsigned char *)0x10000000;
int main();

void _start() {
    main();
    while (1); // trap CPU if main returns
}

int main() {
    const char *msg = "Hello, UART from RISC-V!\n";
    while (*msg) {
        uart[0] = *msg++;
    }

    while (1);  // Loop forever
}