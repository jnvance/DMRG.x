#!/usr/bin/env bash
#
#   Automated timings for test_dmrg.x
#
#   Store current working directory
    curr_dir=`pwd`
    label="timings"
    data_dir="${curr_dir}/${label}"
#
#   Compile program
    cd ..
    # make flush
    # make test_dmrg.x
#
#   Setup variables to inspect
#
#   1.  Number of MPI Processes
#   2.  Number of states to retain
#   3.  Number of sites
#
#       Macbook
        nprocslist=( 2 1 )
        mstateslist=( 12 16 20 24 )
        nsites=20
#
#       Cluster (1 node)
        # nprocslist=( 18 16 12 8 4 2 1 )
        # mstateslist=( 12 16 20 24 )
        # nsites=20
#
#
#
        mkdir ${data_dir}
        cd ${data_dir}
        for nprocs in "${nprocslist[@]}"
        do
            for mstates in "${mstateslist[@]}"
            do
                subfolder="${data_dir}/data_${nprocs}_${mstates}"
                mkdir ${subfolder}
                cd ${subfolder}
                mkdir "data"
                mpirun -np ${nprocs} ../../../test_dmrg.x -mstates ${mstates} -nsites ${nsites}
            done
        done
#
#
#
