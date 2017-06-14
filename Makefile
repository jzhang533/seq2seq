CXX := g++
NVCC := /usr/local/cuda/bin/nvcc

CUDA_DIR := /usr/local/cuda
CUDNN_DIR := ./deps/cudnn
MKL_DIR := ./deps/mkl
OMP_DIR := ./deps/omp

CUDA_INCLUDE = "$(CUDA_DIR)/include"
CUDA_LIB = "$(CUDA_DIR)/lib64"

CUDNN_INCLUDE = "$(CUDNN_DIR)/cuda/include"
CUDNN_LIB = "$(CUDNN_DIR)/cuda/lib64"

MKL_INCLUDE = "$(MKL_DIR)/include"
MKL_LIB = "$(MKL_DIR)/lib/intel64"

OMP_LIB = "$(OMP_DIR)/lib"

#CFLAGS := -g -Wall -std=c++11 -pthread -O0 -ffast-math
CFLAGS := -Wall -g -std=c++11 -pthread -O3 -ffast-math

NVCCFLAGS = -ccbin $(CXX)
NVCCFLAGS += -g -G -Xcompiler -fPIC -arch=sm_30

INCLUDES := -I./include \
    -I${HOME}/.jumbo/include \
    -I$(CUDA_INCLUDE) \
    -I$(CUDNN_INCLUDE) \
    -I$(MKL_INCLUDE)

LIBFLAGS := -lpthread \
    -L${HOME}/.jumbo/lib -lgflags \
    -L$(CUDA_LIB) -lcublas -lcudart \
    -L$(CUDNN_LIB) -lcudnn \
    -L$(OMP_LIB) -liomp5 \
    -L$(MKL_LIB) -lmkl_rt

LIBNAME := seq2seq
BIN_DIR := bin
SRC_DIR := src
LIB_DIR := lib
TOOL_DIR := tools
TEST_DIR := test
INCLUDE_DIR := include
OUTPUT_DIR := output

SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJS = $(patsubst %.cpp, %.o, $(SOURCES))

KERNEL_SOURCES = $(wildcard $(SRC_DIR)/*.cu)
KERNEL_OBJS = $(patsubst %.cu,%.o,$(KERNEL_SOURCES))

TEST_SOURCES = $(wildcard $(TEST_DIR)/*.cpp)
TEST_OBJS = $(patsubst %.cpp, %.o, $(TEST_SOURCES))
TEST_EXES = $(patsubst %.cpp, %, $(TEST_SOURCES))

TOOL_SOURCES = $(wildcard $(TOOL_DIR)/*.cpp)
TOOL_OBJS = $(patsubst %.cpp, %.o, $(TOOL_SOURCES))
TOOL_EXES = $(patsubst %.cpp, %, $(TOOL_SOURCES))

BIN_EXES = $(patsubst $(TOOL_DIR)/%, $(BIN_DIR)/%, $(TOOL_EXES))

CC := $(CXX) $(CFLAGS) $(LIBFLAGS)

#-----------------------------------------------------------------#

.PHONY: all clean lib tool test

#-----------------------------------------------------------------#
# make all
#-----------------------------------------------------------------#
all: lib tool
	if [ ! -d lib ]; then mkdir -p lib; fi
	if [ ! -d $(OUTPUT_DIR)/lib ]; then mkdir -p $(OUTPUT_DIR)/lib; fi
	    cp -r lib/* $(OUTPUT_DIR)/lib
	if [ ! -d $(OUTPUT_DIR)/bin ]; then mkdir -p $(OUTPUT_DIR)/bin; fi
		cp bin/* $(OUTPUT_DIR)/bin	

#-----------------------------------------------------------------#
# make library
#-----------------------------------------------------------------#
lib: $(OBJS) $(KERNEL_OBJS)
	if [ ! -d $(LIB_DIR) ]; then mkdir $(LIB_DIR); fi
	ar -ruv $(LIB_DIR)/lib$(LIBNAME).a $(OBJS) $(KERNEL_OBJS)

$(OBJS): %.o: %.cpp
	$(CC) -c $< -o $@ $(INCLUDES)

$(KERNEL_OBJS): %.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(LIBFLAGS) -c $< -o $@ $(INCLUDES)

#-----------------------------------------------------------------#
# make tool	
#-----------------------------------------------------------------#
tool: 	$(TOOL_EXES)

$(TOOL_EXES): %: %.cpp 
	$(CC) $< -o $@  $(INCLUDES) -Xlinker "-(" -L$(LIB_DIR) -l$(LIBNAME) -Xlinker "-)"
	if [ ! -d bin ]; then mkdir -p bin; fi
	cp $@ $(BIN_DIR)/    
	if [ ! -d bin ]; then mkdir -p bin; fi

test: $(TEST_EXES)

$(TEST_EXES): %: %.cpp 
	$(CC) $< -o $@  $(INCLUDES) -Xlinker "-(" -L$(LIB_DIR) -l$(LIBNAME) -Xlinker "-)"
	if [ ! -d bin ]; then mkdir -p bin; fi
	cp $@ $(BIN_DIR)/    
	if [ ! -d bin ]; then mkdir -p bin; fi

#-----------------------------------------------------------------#
# make clean
#-----------------------------------------------------------------#
clean:
	rm -f $(OBJS) $(TOOL_EXES) $(TEST_EXES) $(BIN_EXES) $(LIB_DIR)/*.a ./test/*.o ./src/*.o
	rm -f $(BIN_DIR)/core.* $(BIN_DIR)/gmon.* $(TEST_DIR)/gmon.* ./core.* 
	if [ -d $(OUTPUT_DIR) ]; then rm -r $(OUTPUT_DIR); fi
	if [ -d $(LIB_DIR)/so ]; then rm -rf $(LIB_DIR)/so/*.so; fi
	rm -f test/so/*.o  test/so/*.so
