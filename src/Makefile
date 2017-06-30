
TARGETS = test_dmrg.x test_kron.x test_reshape.x

CXXFLAGS=-O3 -std=c++11

all: ${TARGETS}

test_dmrg.x: test_dmrg.o dmrgblock.o idmrg.o linalg_tools.o chkopts
	-${CLINKER} -o test_dmrg.x test_dmrg.o dmrgblock.o idmrg.o linalg_tools.o ${SLEPC_EPS_LIB}
	${RM} test_dmrg.o

test_kron.x: test_kron.o linalg_tools.o chkopts
	-${CLINKER} -o test_kron.x test_kron.o linalg_tools.o ${SLEPC_EPS_LIB}
	${RM} test_kron.o

test_reshape.x: test_reshape.o linalg_tools.o chkopts
	-${CLINKER} -o test_reshape.x test_reshape.o linalg_tools.o ${SLEPC_EPS_LIB}
	${RM} test_reshape.o

tests: test_dmrg.x test_kron.x test_reshape.x
	mpirun -np 2 ./test_dmrg.x
	python test_kron.py
	mpirun -np 2 ./test_reshape.x

flush: clean
	${RM} ${TARGETS} *.dat *.dat.info *.x
	${RM} -r test_kron

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common
