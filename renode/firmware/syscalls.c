// Minimal syscalls.c without newlib dependencies

int _write(int file, char *ptr, int len) {
    volatile char *uart = (char *)0x10000000;
    for (int i = 0; i < len; i++) {
        uart[0] = ptr[i];
    }
    return len;
}

void *_sbrk(int incr) {
    static char heap[4096];
    static char *heap_end = heap;
    char *prev_heap_end = heap_end;
    heap_end += incr;
    return (void *)prev_heap_end;
}

void _exit(int status) {
    while (1);
}