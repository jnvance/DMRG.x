
TARGETS = dmrg.x test_kron.x

CXXFLAGS="-O3"

all: ${TARGETS}

dmrg.x: dmrg.o chkopts
	-${CLINKER} -o dmrg.x dmrg.o ${SLEPC_EPS_LIB}
	${RM} dmrg.o

test_kron.x: test_kron.o chkopts
	-${CLINKER} -o test_kron.x test_kron.o ${SLEPC_EPS_LIB}
	${RM} test_kron.o

flush: clean
	${RM} ${TARGETS} *.dat *.dat.info

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common
