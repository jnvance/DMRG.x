
TARGETS = 	UnitTests_DMRGBlock.x \
			UnitTests_DMRGBlock_SaveInfo.x \
			UnitTests_DMRGBlock_SetDiskStorage.x \
			UnitTests_DMRGBlock_InitializeFromDisk.x \
			UnitTests_DMRGKron.x \
			UnitTests_DMRGBlockContainer.x

CXXFLAGS += -O3 -std=c++11 -Wall -I${PWD}/../include -I${PWD}/../old

UnitTests_DMRGBlock_DEPS = UnitTests_Misc.o \
	../src/DMRGBlock.o ../src/DMRGKron.o ../src/QuantumNumbers.o ../src/MiscTools.o \
	../old/linalg_tools.o

UnitTests_DMRGBlockContainer_DEPS = ${UnitTests_DMRGBlock_DEPS} ../src/Hamiltonians.o

UnitTests_DMRGKron_DEPS = ${UnitTests_DMRGBlock_DEPS}

all: ${TARGETS}

UnitTests_DMRGBlock.x: UnitTests_DMRGBlock.o ${UnitTests_DMRGBlock_DEPS} chkopts
	-${CLINKER} -o UnitTests_DMRGBlock.x UnitTests_DMRGBlock.o ${UnitTests_DMRGBlock_DEPS} ${SLEPC_EPS_LIB}
	${RM} UnitTests_DMRGBlock.o

UnitTests_DMRGBlock_SaveInfo.x: UnitTests_DMRGBlock_SaveInfo.o ${UnitTests_DMRGBlock_DEPS} chkopts
	-${CLINKER} -o UnitTests_DMRGBlock_SaveInfo.x UnitTests_DMRGBlock_SaveInfo.o ${UnitTests_DMRGBlock_DEPS} ${SLEPC_EPS_LIB}
	${RM} UnitTests_DMRGBlock_SaveInfo.o

UnitTests_DMRGBlock_InitializeFromDisk.x: UnitTests_DMRGBlock_InitializeFromDisk.o ${UnitTests_DMRGBlock_DEPS} chkopts
	-${CLINKER} -o UnitTests_DMRGBlock_InitializeFromDisk.x UnitTests_DMRGBlock_InitializeFromDisk.o ${UnitTests_DMRGBlock_DEPS} ${SLEPC_EPS_LIB}
	${RM} UnitTests_DMRGBlock_InitializeFromDisk.o

UnitTests_DMRGBlock_SetDiskStorage.x: UnitTests_DMRGBlock_SetDiskStorage.o ${UnitTests_DMRGBlock_DEPS} chkopts
	-${CLINKER} -o UnitTests_DMRGBlock_SetDiskStorage.x UnitTests_DMRGBlock_SetDiskStorage.o ${UnitTests_DMRGBlock_DEPS} ${SLEPC_EPS_LIB}
	${RM} UnitTests_DMRGBlock_SetDiskStorage.o

UnitTests_DMRGBlockContainer.x: UnitTests_DMRGBlockContainer.o ${UnitTests_DMRGBlockContainer_DEPS} ../include/DMRGBlockContainer.hpp chkopts
	-${CLINKER} -o UnitTests_DMRGBlockContainer.x UnitTests_DMRGBlockContainer.o ${UnitTests_DMRGBlockContainer_DEPS} ${SLEPC_EPS_LIB}
	${RM} UnitTests_DMRGBlockContainer.o

UnitTests_DMRGKron.x: UnitTests_DMRGKron.o ${UnitTests_DMRGKron_DEPS} chkopts
	-${CLINKER} -o UnitTests_DMRGKron.x UnitTests_DMRGKron.o ${UnitTests_DMRGKron_DEPS} ${SLEPC_EPS_LIB}
	${RM} UnitTests_DMRGKron.o

docs:
	cd .. && make docs

flush: clean
	${RM} -rf ${TARGETS} ${UnitTests_DMRGBlockContainer_DEPS} *.dat *.dat.info *.x
	${RM} -rf test_kron/*
	${RM} -rf test_reshape/*
	${RM} -rf data/*

clear: FORCE
	${RM} -rf trash_block_test
	${RM} -rf trash_block_test_*
	${RM} -rf data_dir scratch_dir

FORCE:

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common
