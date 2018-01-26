
TARGETS = 	UnitTests_DMRGBlock.x \
			UnitTests_DMRGKron.x

CXXFLAGS += -O3 -std=c++11 -Wall -I${PWD}/.. -I${PWD}/../..

UnitTests_DMRGBlock_DEPS = UnitTests_Misc.o \
	../DMRGBlock.o ../DMRGKron.o ../QuantumNumbers.o ../MiscTools.o \
	../../linalg_tools.o

UnitTests_DMRGBlockContainer_DEPS = ../DMRGBlockContainer.o ${UnitTests_DMRGBlock_DEPS}

UnitTests_DMRGKron_DEPS = ${UnitTests_DMRGBlock_DEPS}

all: ${TARGETS}

UnitTests_DMRGBlock.x: UnitTests_DMRGBlock.o ${UnitTests_DMRGBlock_DEPS} chkopts
	-${CLINKER} -o UnitTests_DMRGBlock.x UnitTests_DMRGBlock.o ${UnitTests_DMRGBlock_DEPS} ${SLEPC_EPS_LIB}
	${RM} UnitTests_DMRGBlock.o

UnitTests_DMRGBlockContainer.x: UnitTests_DMRGBlockContainer.o ${UnitTests_DMRGBlockContainer_DEPS} chkopts
	-${CLINKER} -o UnitTests_DMRGBlockContainer.x UnitTests_DMRGBlockContainer.o ${UnitTests_DMRGBlockContainer_DEPS} ${SLEPC_EPS_LIB}
	${RM} UnitTests_DMRGBlockContainer.o

UnitTests_DMRGKron.x: UnitTests_DMRGKron.o ${UnitTests_DMRGKron_DEPS} chkopts
	-${CLINKER} -o UnitTests_DMRGKron.x UnitTests_DMRGKron.o ${UnitTests_DMRGKron_DEPS} ${SLEPC_EPS_LIB}
	${RM} UnitTests_DMRGKron.o

flush: clean
	${RM} -rf ${TARGETS} ${UnitTests_DMRGBlockContainer_DEPS} *.dat *.dat.info *.x
	${RM} -rf test_kron/*
	${RM} -rf test_reshape/*
	${RM} -rf data/*

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common
