
TARGET := llm_infer$(shell python3-config --extension-suffix)
SRC_DIR := src 
CU_DIR := kernel
INC_DIR := include
BUILD_DIR := build
CPPFLAGS := -Wall -fPIC -I$(INC_DIR) -I$(INC_DIR)/layer -I$(INC_DIR)/utils -I$(INC_DIR)/model -I$(INC_DIR)/kernel \
	$(shell python3 -m pybind11 --includes)
NVCCFLAGS := -O2 -Xcompiler -fPIC -I$(INC_DIR) -I$(INC_DIR)/kernel
LDFLAGS := -shared
PYTHON := python3
LDLIBS := $(shell $(PYTHON)-config --ldflags)
CXX := g++
NVCC := nvcc

SRCS_CPP = $(shell find src -name "*.cpp")
SRCS_CU = $(shell find kernel -name "*.cu")
OBJS_CPP = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(SRCS_CPP))
OBJS_CU = $(patsubst $(CU_DIR)/%.cu,$(BUILD_DIR)/%.o,$(SRCS_CU))


$(TARGET): $(OBJS_CPP) $(OBJS_CU)
	$(CXX) $(LDFLAGS) -o $@ $(OBJS_CPP) $(OBJS_CU) $(LDLIBS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) -c $< -o $@

$(BUILD_DIR)/%.o: $(CU_DIR)/%.cu
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

all: $(TARGET)

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(TARGET)