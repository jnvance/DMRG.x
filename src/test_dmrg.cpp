static char help[] =
    "Test Code for Kronecker product\n";

#include <iostream>
#include <slepceps.h>

#include "dmrg.hpp"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
    MPI_Comm        comm;
    PetscMPIInt     nprocs, rank;
    PetscErrorCode  ierr = 0;

    SlepcInitialize(&argc, &argv, (char*)0, help);
    comm = PETSC_COMM_WORLD;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);


    DMRGBlock block(comm);
    block.init();

    /* Do something */

    block.destroy();
    SlepcFinalize();
    return ierr;
}
