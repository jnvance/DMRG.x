TARGET     = bin/DMRG-SpinOneHalf-J1J2XXZSquare.x
TARGET_OBJ = src/DMRG-SpinOneHalf-J1J2XXZSquare.o

override CXXFLAGS += -O3 -std=c++11 -Wall -I include/ -I old/

TARGET_DEPS = src/DMRGBlock.o src/DMRGKron.o src/QuantumNumbers.o src/MiscTools.o src/Hamiltonians.o old/linalg_tools.o

all: ${TARGET}

${TARGET}: ${TARGET_OBJ} ${TARGET_DEPS} chkopts
	mkdir -p bin
	-${CLINKER} -o ${TARGET} ${TARGET_OBJ} ${TARGET_DEPS} ${SLEPC_EPS_LIB}
	${RM} ${TARGET_OBJ}

docs: docs-generate-files FORCE
	doxygen Doxyfile
	cp assets/html/dynsections.js.in docs/html/dynsections.js

docs-default: docs-generate-files FORCE
	echo "HTML_HEADER=\n" \
		"HTML_FOOTER=\n" \
		"HTML_STYLESHEET=\n" \
		"HTML_EXTRA_STYLESHEET=\n" \
		"HTML_EXTRA_FILES=\n "\
		"LAYOUT_FILE=\n" \
		"OUTPUT_DIRECTORY = ./docs/default\n" \
		"GENERATE_TREEVIEW = YES\n" | \
	(cat Doxyfile && cat) | doxygen -

docs-generate-files: FORCE
	./docs/docs_generate_files.sh

flush: clean
	${RM} ${TARGET} ${TARGET_OBJ} ${TARGET_DEPS}
	${RM} src/*.optrpt
	${RM} -rf docs/html docs/latex docs/doc_00_overview.dox docs/default

FORCE:

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common
