SRC_DIR = ./src
INC_DIR = ./include
OBJ_DIR = ./obj
BIN_DIR = ./bin
DEDISP_DIR = ./dedisp_paf
CC=g++
NVCC=/Developer/NVIDIA/CUDA-7.5/bin/nvcc
DEBUG= -g -G

INCLUDE = -I${INC_DIR} -I${DEDISP_DIR}/include
LIBS =

CFLAGS = -Wall -Wextra -std=c++11
NVCC_FLAG = -gencode=arch=compute_52,code=sm_52 --std=c++11 -lcufft -Xcompiler ${DEBUG}

all: pafrb

pafrb: ${OBJ_DIR}/threads.o
	${NVCC} ${NVCC_FLAG} ${INCLUDE} ${LIBS} $< -o ${BIN_DIR}/$@
${OBJ_DIR}/threads.o: ${SRC_DIR}/threads.cu
	${NVCC} -c ${NVCC_FLAG} ${INCLUDE} $< -o $@

.PHONY: clean

clean:
	rm -f ${OBJ_DIR}/*.o ${BIN_DIR}/*
