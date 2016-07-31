SRC_DIR = ./src
INC_DIR = ./include
OBJ_DIR = ./obj
BIN_DIR = ./bin
DEDISP_DIR = ./dedisp_paf
CC=g++
NVCC=/usr/bin/nvcc
DEBUG= #-g -G

INCLUDE = -I${INC_DIR}
LIBS = -L${DEDISP_DIR}/lib -lstdc++ -lboost_system -lpthread -lcudart -lcuda

CFLAGS = -Wall -Wextra -std=c++11
NVCC_FLAG = -gencode=arch=compute_52,code=sm_52 --std=c++11 -lcufft -Xcompiler ${DEBUG} #--default-stream per-thread

CPPOBJECTS = ${OBJ_DIR}/DedispPlan.o ${OBJ_DIR}/paf_metadata.o

CUDAOBJECTS = ${OBJ_DIR}/threads.o ${OBJ_DIR}/pool_multi.o ${OBJ_DIR}/kernels.o ${OBJ_DIR}/dedisp.o

all: pafrb

pafrb: ${CUDAOBJECTS} ${CPPOBJECTS}
	${NVCC} ${NVCC_FLAG} ${INCLUDE} ${LIBS} ${CUDAOBJECTS} ${CPPOBJECTS} -o ${BIN_DIR}/pafrb

${OBJ_DIR}/%.o: ${SRC_DIR}/%.cu
	${NVCC} -c ${NVCC_FLAG} ${INCLUDE} $< -o $@

${OBJ_DIR}/%.o: ${SRC_DIR}/%.cpp
	${CC} -c ${CFLAGS} ${INCLUDE} $< -o $@

.PHONY: clean

clean:
	rm -f ${OBJ_DIR}/*.o ${BIN_DIR}/*
