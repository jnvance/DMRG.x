
TARGETS = main.x

CXXFLAGS += -O3 -std=c++11 -Wall -I${PWD}/..

# CXXFLAGS += -D__KRON_TIMINGS
# CXXFLAGS += -D__KRON_PS_TIMINGS

MAIN_DEPS = DMRGBlockContainer.o DMRGBlock.o DMRGKron.o QuantumNumbers.o MiscTools.o ../linalg_tools.o
# ../linalg_tools.o ../kron.o

all: ${TARGETS}

main.x: main.o ${MAIN_DEPS} chkopts
	-${CLINKER} -o main.x main.o ${MAIN_DEPS} ${SLEPC_EPS_LIB}
	${RM} main.o

flush: clean
	${RM} ${TARGETS} *.dat *.dat.info *.x
	${RM} -rf test_kron/*
	${RM} -rf test_reshape/*
	${RM} -rf data/*

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common
