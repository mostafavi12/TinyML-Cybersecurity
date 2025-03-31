# Makefile
CC=riscv64-unknown-elf-gcc
CFLAGS=-O2 -Wall -nostdlib -nostartfiles -march=rv32imac -mabi=ilp32 -Ttext=0x0
OBJS=main.o syscalls.o

all: firmware.elf

firmware.elf: $(OBJS)
	$(CC) $(CFLAGS) -o firmware.elf $(OBJS)

%.o: %.c
	$(CC) $(CFLAGS) -c $<

clean:
	rm -f *.o firmware.elf