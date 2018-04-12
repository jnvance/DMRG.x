#!/bin/zsh

    TARGET_DIR_REL="."
    TARGET_EXE="test_dmrg.x"
    JOB_DIR="${PWD}/trash_test_eigvals"

    NSITES=30
    PPN_LIST=(2 4)
    NNODES_LIST=(1)
    # MSTATES_LIST=(20 64 128)
    MSTATES_LIST=(20)
    DO_TARGET_SZ_LIST=(0)

    TOL="1e-30"

    cd ${TARGET_DIR_REL}
    TARGET_DIR=${PWD}
    TARGET_PATH=${TARGET_DIR}/${TARGET_EXE}
    echo "TARGET_PATH=${TARGET_PATH}"
    if [ -e "${TARGET_PATH}" ]
    then
        echo -e "TARGET_PATH exists \n"
    else
        echo -e "TARGET_PATH does not exist \n"
        cd ${TARGET_DIR} && make flush && make ${TARGET_EXE}
    fi

    ADD_FLAGS="-eps_type krylovschur -eps_hermitian -eps_tol 1e-16 -svd_tol 1e-14 -log_summary"
    echo "ADD_FLAGS=${ADD_FLAGS}"

    mkdir -p ${JOB_DIR}
    echo "JOB_DIR=${JOB_DIR}"

    echo

    for DO_TARGET_SZ in "${DO_TARGET_SZ_LIST[@]}"; do
        for MSTATES in "${MSTATES_LIST[@]}"; do
            for PPN in "${PPN_LIST[@]}"; do
                for NNODES in "${NNODES_LIST[@]}"; do

                    NPROCS=$((NNODES*PPN))
                    DATA_FOLDER="heis1d"
                    DATA_FOLDER+="__dosz_$(printf %1d ${DO_TARGET_SZ})"
                    DATA_FOLDER+="__mstates_$(printf %05d ${MSTATES})"
                    DATA_FOLDER+="__ppn_$(printf %05d ${PPN})"
                    DATA_FOLDER+="__nnodes_$(printf %05d ${NNODES})"

                    DATA_DIR=${JOB_DIR}/${DATA_FOLDER}
                    # DATA_DIR=${JOB_DIR}

                    FLAGS="-nsites ${NSITES} -mstates ${MSTATES} -do_target_Sz ${DO_TARGET_SZ}"
                    # COMMAND="mpirun -np ${NPROCS} -map-by ppr:${PPN}:node \${TARGET_PATH} ${FLAGS} \${ADD_FLAGS}"
                    COMMAND="mpirun -np ${NPROCS} \${TARGET_PATH} ${FLAGS} \${ADD_FLAGS}"

                    echo "    COMMAND=${COMMAND}"
                    echo "    PWD=\${JOB_DIR}/${DATA_FOLDER}"
                    # echo "    PWD=${PWD}"

                    cd ${JOB_DIR}
                    mkdir -p ${DATA_DIR}
                    cd ${DATA_DIR}
                    mkdir -p "data"
                    eval ${COMMAND} |& tee output.dat

                    cd ${JOB_DIR}
                    # check data after run with python script
                    # compare against data generated in subfolder correct
                    COMMAND="\${TARGET_DIR}/test_eigvals.py \${PWD}/correct/\${DATA_FOLDER} \${PWD}/\${DATA_FOLDER} ${TOL}"
                    echo "    COMMAND=${COMMAND}"
                    eval ${COMMAND}
                    ret=$?
                    if [ $ret -ne 0 ]; then
                        echo -e "\n    FAILED"
                        break 4
                    fi
                    echo
                done
            done
        done
    done

