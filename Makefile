SRC_DIR = ./src
INC_DIR = ./include
OBJ_DIR = ./obj
BIN_DIR = ./bin
DEDISP_DIR = ./dedisp_paf
CC=clang++
NVCC=/Developer/NVIDIA/CUDA-7.5/bin/nvcc
DEBUG= -g -G

INCLUDE = -I${INC_DIR}
LIBS = -L${DEDISP_DIR}/lib -lstdc++ -ldedisp

CFLAGS = -Wall -Wextra -std=c++11 -stdlib=libc++
NVCC_FLAG = -gencode=arch=compute_52,code=sm_52 --std=c++11 -lcufft -Xcompiler ${DEBUG}

# CPPOBJECTS = ${OBJ_DIR}/

CUDAOBJECTS = ${OBJ_DIR}/threads.o ${OBJ_DIR}/pool.o ${OBJ_DIR}/kernels.o

all: pafrb

pafrb: ${CUDAOBJECTS}
	${NVCC} ${NVCC_FLAG} ${INCLUDE} ${LIBS} ${CUDAOBJECTS} -o ${BIN_DIR}/pafrb

${OBJ_DIR}/%.o: ${SRC_DIR}/%.cu
	${NVCC} -c ${NVCC_FLAG} ${INCLUDE} $< -o $@

.PHONY: clean

clean:
	rm -f ${OBJ_DIR}/*.o ${BIN_DIR}/*
