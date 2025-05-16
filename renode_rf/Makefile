# Makefile
CC=riscv64-unknown-elf-gcc
CFLAGS=-O2 -Wall -nostdlib -nostartfiles -march=rv32imac -mabi=ilp32 -Ttext=0x0 -Iinclude 
SRC_DIR=src
BUILD_DIR = build

OBJS=$(SRC_DIR)/main.o $(SRC_DIR)/model.o $(SRC_DIR)/syscalls.o

all: $(BUILD_DIR)/rf_model.elf

$(BUILD_DIR)/rf_model.elf: $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) -lgcc

$(SRC_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(SRC_DIR)/*.o $(BUILD_DIR)/rf_model.elf
