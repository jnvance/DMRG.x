static char help[] =
    "Test Code for DMRG\n";

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

    DMRGBlock block;
    block.init();

    Mat Sz = block.Sz();
    Mat Sp = block.Sp();

    Mat eye2;
    MatEyeCreate(comm, eye2, 10);

    Mat H_temp;
    MatKron(eye2, Sp, H_temp, comm);

    MatPeek(comm,Sz, "Sz");
    MatPeek(comm,Sp, "Sp");
    MatPeek(comm,eye2, "eye2");
    MatPeek(comm,H_temp, "H_temp");

    block.destroy();

    SlepcFinalize();
    return ierr;
}
