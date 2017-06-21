
TARGETS = dmrg.x test_kron.x

CXXFLAGS=-O3

all: ${TARGETS}

test_dmrg.x: test_dmrg.o chkopts
	-${CLINKER} -o test_dmrg.x test_dmrg.o ${SLEPC_EPS_LIB}
	${RM} test_dmrg.o

test_kron.x: test_kron.o chkopts
	-${CLINKER} -o test_kron.x test_kron.o ${SLEPC_EPS_LIB}
	${RM} test_kron.o

flush: clean
	${RM} ${TARGETS} *.dat *.dat.info *.x
	${RM} -r test_kron

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common
