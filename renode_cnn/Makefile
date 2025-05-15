# Toolchain
CC = riscv-none-elf-gcc
CXX = riscv-none-elf-g++

# Flags
CFLAGS_C = -Wall -Os -g -std=c99 -Wno-nonnull -Wno-unused-variable -Wno-maybe-uninitialized
CFLAGS_CPP = -Wall -Os -g -std=c++11 -Wno-nonnull -Wno-unused-variable -Wno-maybe-uninitialized

# Paths
BUILD_DIR = build
SRC_DIR = src
INCLUDE_DIR = include

# Includes
INCLUDES = \
 -I. \
 -I$(INCLUDE_DIR) \
 -I$(INCLUDE_DIR)/tensorflow \
 -I$(INCLUDE_DIR)/tensorflow/lite \
 -I$(INCLUDE_DIR)/tensorflow/lite/c \
 -I$(INCLUDE_DIR)/tensorflow/lite/core/api \
 -I$(INCLUDE_DIR)/tensorflow/lite/schema \
 -I$(INCLUDE_DIR)/tensorflow/lite/micro \
 -I$(INCLUDE_DIR)/tensorflow/lite/micro/kernels \
 -I/home/ahmad/flatbuffers-1.12.0/include

# Source files
C_SRCS := model.c model_data.c syscalls.c
C_SRCS += \
	$(SRC_DIR)/tensorflow/lite/c/common.c

CPP_SRCS := \
		$(SRC_DIR)/main.cc \
		$(SRC_DIR)/tensorflow/lite/core/api/error_reporter.cc \
		$(SRC_DIR)/tensorflow/lite/core/api/flatbuffer_conversions.cc \
		$(SRC_DIR)/tensorflow/lite/micro/all_ops_resolver.cc \
		$(SRC_DIR)/tensorflow/lite/micro/micro_allocator.cc \
		$(SRC_DIR)/tensorflow/lite/micro/micro_interpreter.cc \
		$(SRC_DIR)/tensorflow/lite/micro/memory_helpers.cc \
		$(SRC_DIR)/tensorflow/lite/micro/debug_log.cc \
		$(SRC_DIR)/tensorflow/lite/micro/micro_graph.cc \
		$(SRC_DIR)/tensorflow/lite/micro/memory_planner/greedy_memory_planner.cc \
		$(SRC_DIR)/tensorflow/lite/micro/simple_memory_allocator.cc \
		$(SRC_DIR)/tensorflow/lite/micro/micro_error_reporter.cc \
		$(SRC_DIR)/tensorflow/lite/micro/micro_string.cc

CPP_SRCS += $(wildcard $(SRC_DIR)/tensorflow/lite/micro/kernels/*.cc)		   

CPP_OBJS := $(patsubst %.cc, $(BUILD_DIR)/%.o, $(CPP_SRCS))
C_OBJS := $(patsubst %.c, $(BUILD_DIR)/%.o, $(C_SRCS))

OBJS := $(CPP_OBJS) $(C_OBJS)

# Pattern rules for building object files
$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_C) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CFLAGS_CPP) $(INCLUDES) -c $< -o $@

# Object file rules
$(BUILD_DIR)/%.o: %.cc
	@mkdir -p $(dir $@)
	$(CXX) $(CFLAGS_CPP) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS_C) $(INCLUDES) -c $< -o $@

# Link
$(BUILD_DIR)/firmware.elf: $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $^

# Build rule for all
all: $(BUILD_DIR)/firmware.elf

clean:
	rm -rf $(BUILD_DIR)