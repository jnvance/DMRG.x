
TARGETS = UnitTests_DMRGBlock.x

CXXFLAGS += -O3 -std=c++11 -Wall -I${PWD}/.. -I${PWD}/../..

UNITTESTS_DMRGBLOCK_DEPS = ../DMRGBlock.o ../DMRGKron.o ../QuantumNumbers.o ../MiscTools.o ../../linalg_tools.o
# ../linalg_tools.o ../kron.o

all: ${TARGETS}

UnitTests_DMRGBlock.x: UnitTests_DMRGBlock.o ${UNITTESTS_DMRGBLOCK_DEPS} chkopts
	-${CLINKER} -o UnitTests_DMRGBlock.x UnitTests_DMRGBlock.o ${UNITTESTS_DMRGBLOCK_DEPS} ${SLEPC_EPS_LIB}
	${RM} UnitTests_DMRGBlock.o

flush: clean
	${RM} -rf ${TARGETS} ${UNITTESTS_DMRGBLOCK_DEPS} *.dat *.dat.info *.x
	${RM} -rf test_kron/*
	${RM} -rf test_reshape/*
	${RM} -rf data/*

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common
