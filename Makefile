TARGET     = bin/DMRG-SquareLattice.x
TARGET_OBJ = src/DMRG-SquareLattice.o

override CXXFLAGS += -O3 -std=c++11 -Wall -I include/ -I old/

TARGET_DEPS = src/DMRGBlock.o src/DMRGKron.o src/QuantumNumbers.o src/MiscTools.o src/Hamiltonians.o old/linalg_tools.o

all: ${TARGET}

${TARGET}: ${TARGET_OBJ} ${TARGET_DEPS} chkopts
	mkdir -p bin
	-${CLINKER} -o ${TARGET} ${TARGET_OBJ} ${TARGET_DEPS} ${SLEPC_EPS_LIB}
	${RM} ${TARGET_OBJ}

# Note: These commands should match the subprocess.call in docs/sphinx/conf.py
docs: docs-generate-files FORCE
	doxygen Doxyfile
	cp assets/html/dynsections.js.in docs/html/dynsections.js
	cp assets/html/doc_postproc_01.html docs/html/doc_postproc_01.html

docs-default: docs-generate-files FORCE
	echo "HTML_HEADER=\n" \
		"HTML_FOOTER=\n" \
		"HTML_STYLESHEET=\n" \
		"HTML_EXTRA_STYLESHEET=\n" \
		"HTML_EXTRA_FILES=\n "\
		"OUTPUT_DIRECTORY = ./docs/default\n" \
		"GENERATE_TREEVIEW = YES\n" | \
	(cat Doxyfile && cat) | doxygen -
	cp assets/html/doc_postproc_01.html docs/default/html/doc_postproc_01.html

docs-man: docs-generate-files FORCE
	echo "GENERATE_HTML = NO\n" \
		"GENERATE_MAN = YES\n" | \
	(cat Doxyfile && cat) | doxygen -

docs-generate-files: FORCE
	./docs/docs_generate_files.sh

flush: clean
	${RM} ${TARGET} ${TARGET_OBJ} ${TARGET_DEPS}
	${RM} src/*.optrpt

flush-docs:
	${RM} -rf docs/html docs/latex docs/doc_00_overview.dox docs/default docs/man

FORCE:

include ${SLEPC_DIR}/lib/slepc/conf/slepc_common
