static char help[] =
    "Test Code for Kronecker product\n";

#include <iostream>
#include <slepceps.h>
#include "kron.hpp"

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

    // Create two square matrices A and B w/in petsc
    Mat A, B;
    Mat C = NULL;

    const PetscInt size_A = 12;  // size of matrix A
    const PetscInt size_B =  8;  // size of matrix B

    #define INIT_AND_ZERO(MAT,MATSIZE) \
        MatCreate(comm, &MAT); \
        MatSetSizes(MAT, PETSC_DECIDE, PETSC_DECIDE, MATSIZE, MATSIZE); \
        MatSetFromOptions(MAT); \
        MatSetUp(MAT); \
        MatZeroEntries(MAT);

        INIT_AND_ZERO(A,size_A)
        INIT_AND_ZERO(B,size_B)
    #undef INIT_AND_ZERO


    // Fill matrices with values
    PetscInt Istart,Iend;
    MatGetOwnershipRange(B,&Istart,&Iend);

    // Matrix B is tridiagonal with constant diagonal and subdiagonal entries
    MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY);
    for (PetscInt i = Istart; i < Iend; ++i){
        MatSetValue(B, i, i, 2, INSERT_VALUES);
        // if (i<size_B-1) MatSetValue(B, i, i+1, -1, INSERT_VALUES);
        if (i>0)        MatSetValue(B, i, i-1, -1, INSERT_VALUES);
    }
    MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY);

    // Matrix A is diagonal with varying entries
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(A, &Istart, &Iend); CHKERRQ(ierr);
    for (PetscInt i = Istart; i < Iend; ++i){
        ierr = MatSetValue(A, i, i, i, INSERT_VALUES); CHKERRQ(ierr);
        if (i%2 == 0) {ierr = MatSetValue(A, i, i+1, -1, INSERT_VALUES); CHKERRQ(ierr);}
        if (i%2 == 1) {ierr = MatSetValue(A, i, i-1, -1, INSERT_VALUES); CHKERRQ(ierr);}
    }
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    MatKron(A, B, C, comm);

    // Peek into values
    // #define __PEEK__
    #ifdef __PEEK__
        PetscViewer fd = nullptr;
        #define PEEK(MAT) \
            MatAssemblyBegin(MAT, MAT_FLUSH_ASSEMBLY); \
            MatAssemblyEnd(MAT, MAT_FINAL_ASSEMBLY); \
            MatView(MAT, fd); \
            PetscViewerDestroy(&fd)

        if(A && rank==0) printf("\nA\n");
        if(A){ PEEK(A); }
        if(B && rank==0) printf("\nB\n");
        if(B){ PEEK(B); }
        if(C && rank==0) printf("\nC\n");
        if(C){ PEEK(C); }

        #undef PEEK
        PetscViewerDestroy(&fd);
    #endif
    #undef __PEEK__

    // Write to file
    #define __WRITE__
    #ifdef __WRITE__
        PetscViewer writer = nullptr;
        #define WRITE(MAT,FILE) \
            MatAssemblyBegin(MAT, MAT_FLUSH_ASSEMBLY);\
            MatAssemblyEnd(MAT, MAT_FINAL_ASSEMBLY);\
            PetscViewerBinaryOpen(PETSC_COMM_WORLD,FILE,FILE_MODE_WRITE,&writer);\
            MatView(MAT, writer);\
            PetscViewerDestroy(&writer);

        WRITE(A,"test_kron/A.dat")
        WRITE(B,"test_kron/B.dat")
        WRITE(C,"test_kron/C.dat")
        #undef WRITE
        PetscViewerDestroy(&writer);
    #endif
    #undef __WRITE__

    MatDestroy(&A);
    MatDestroy(&B);
    MatDestroy(&C);
    SlepcFinalize();
    return ierr;

}
