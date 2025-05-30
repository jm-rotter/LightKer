# Compiler commands
CXX = g++
NVCC = nvcc

# Flags
CXXFLAGS = -std=c++17 -Iinclude -O2 -Wall -Wextra
NVCCFLAGS = -std=c++17 -Iinclude -O2 -Xcompiler "-Wall -Wextra"

# Output executable
TARGET = matrix_multiplication

# Source files
CPU_SRCS = src/cpu_matmul.cpp src/utils.cpp 
GPU_SRCS = src/gpu_matmul.cu src/main.cpp

# Object files
CPU_OBJS = $(CPU_SRCS:.cpp=.o)

GPU_OBJS = $(filter %.cu,$(GPU_SRCS))
GPU_OBJS := $(GPU_OBJS:.cu=.o)
GPU_OBJS += $(filter %.cpp,$(GPU_SRCS))
GPU_OBJS := $(GPU_OBJS:.cpp=.o)

# Default rule
all: $(TARGET)

# Link all objects to create the final executable
$(TARGET): $(CPU_OBJS) $(GPU_OBJS)
	$(NVCC) -o $@ $^

# Compile C++ source files (except main.cpp which is compiled with nvcc)
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

src/main.o: src/main.cpp
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile CUDA source files
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean build files
clean:
	rm -f $(CPU_OBJS) $(GPU_OBJS) $(TARGET)
