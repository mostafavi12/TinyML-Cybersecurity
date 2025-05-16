int _write(int file, const char *ptr, int len) {
    volatile char *uart = (char *)0x10000000;
    for (int i = 0; i < len; i++) {
        uart[0] = ptr[i];
    }
    return len;
}

int _close(int file) { return -1; }
int _fstat(int file, void *st) { return 0; }
int _isatty(int file) { return 1; }
int _lseek(int file, int ptr, int dir) { return 0; }
int _read(int file, char *ptr, int len) { return 0; }
int _kill(int pid, int sig) { return -1; }
int _getpid(void) { return 1; }
void _exit(int status) { while (1); }

void *_sbrk(int incr) {
    extern char _end;
    static char *heap_end;
    char *prev_heap_end;

    if (heap_end == 0)
        heap_end = &_end;

    prev_heap_end = heap_end;
    heap_end += incr;
    return (void *) prev_heap_end;
}