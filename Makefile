TARGET     = bin/DMRG-SpinOneHalf-J1J2XYSquare.x
TARGET_OBJ = src/DMRG-SpinOneHalf-J1J2XYSquare.o

override CXXFLAGS += -O3 -std=c++11 -Wall -I include/ -I old/

TARGET_DEPS = src/DMRGBlock.o src/DMRGKronShell.o src/DMRGKron.o src/QuantumNumbers.o src/MiscTools.o src/Hamiltonians.o old/linalg_tools.o

all: ${TARGET}

${TARGET}: ${TARGET_OBJ} ${TARGET_DEPS} chkopts
	mkdir -p bin
	-${CLINKER} -o ${TARGET} ${TARGET_OBJ} ${TARGET_DEPS} ${SLEPC_EPS_LIB}
	${RM} ${TARGET_OBJ}

docs:
	doxygen Doxyfile

flush: clean
	${RM} ${TARGET} ${TARGET_OBJ} ${TARGET_DEPS}
	${RM} src/*.optrpt

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common
