static char help[] =
    "Test Code for reshape\n";

#include <iostream>
#include <slepceps.h>
#include "linalg_tools.hpp"

#define CHK(FUNCTION) ierr = FUNCTION; CHKERRQ(ierr);

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

    Vec         vec;
    PetscInt    vec_size = 36;
    PetscInt    Istart, Iend, Isize;

    VecCreate(comm, &vec);
    VecSetType(vec, VECMPI);
    VecSetSizes(vec, PETSC_DECIDE, vec_size);
    VecGetOwnershipRange(vec, &Istart, &Iend);
    Isize = Iend - Istart;

    PetscScalar     *vals;
    VecGetArray(vec, &vals);
    for (PetscInt i = 0; i < Isize; ++i)
    {
        vals[i] = Istart + i;
    }
    VecRestoreArray(vec, &vals);


    PetscViewer fd = nullptr;
    PetscPrintf(comm, "Original Vector\n");
    ierr = VecView(vec, fd); CHKERRQ(ierr);
    PetscViewerDestroy(&fd);

    Mat         mat;

    VecReshapeToMat(comm, vec, mat, 12, 3);

    PetscPrintf(comm, "Matrix\n");
    ierr = MatView(mat, fd); CHKERRQ(ierr);
    PetscViewerDestroy(&fd);


    VecDestroy(&vec);
    MatDestroy(&mat);
    SlepcFinalize();
    return ierr;
}
