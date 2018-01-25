static char help[] =
    "Test code for DMRGKron routines\n";

#include <slepceps.h>
#include "DMRGBlock.hpp"
#include "linalg_tools.hpp"
#include <iostream>

PETSC_EXTERN PetscErrorCode SetRow(const Mat& A, const PetscInt& row, const std::vector<PetscInt>& idxn);
PETSC_EXTERN PetscErrorCode Kron_Explicit(
    const Block_SpinOneHalf& LeftBlock,
    const Block_SpinOneHalf& RightBlock,
    Block_SpinOneHalf& BlockOut,
    PetscBool BuildHamiltonian);
PETSC_EXTERN const char hborder[];

/* TODO: Comparison function for each row to check expected KronProd

PETSC_EXTERN PetscErrorCode CheckRow(
    const Mat& A,
    const PetscInt& row,
    const std::vector<PetscInt>& idxn,
    const std::vector<PetscScalar>& v);

 */


#define PrintHeader(COMM,TEXT)  PetscPrintf((COMM), "%s\n%s\n%s\n", hborder, (TEXT), hborder)

int main(int argc, char **argv)
{
    PetscErrorCode  ierr = 0;
    PetscMPIInt     nprocs, rank;
    MPI_Comm&       comm = PETSC_COMM_WORLD;

    /*  Initialize MPI  */
    ierr = SlepcInitialize(&argc, &argv, (char*)0, help); CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm, &nprocs); CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);

    Block_SpinOneHalf LeftBlock, RightBlock, BlockOut;

    ierr = LeftBlock.Initialize(PETSC_COMM_WORLD, 2, {+1.,0.,-1.}, {1,2,1}); CHKERRQ(ierr);

    SetRow(LeftBlock.Sz[0],0, {0         });
    SetRow(LeftBlock.Sz[0],1, {   1      });
    SetRow(LeftBlock.Sz[0],2, {   1, 2   });
    SetRow(LeftBlock.Sz[0],3, {         3});

    SetRow(LeftBlock.Sz[1],0, {0         });
    SetRow(LeftBlock.Sz[1],1, {   1, 2   });
    SetRow(LeftBlock.Sz[1],2, {      2   });
    SetRow(LeftBlock.Sz[1],3, {          });

    SetRow(LeftBlock.Sp[0],0, {   1, 2   });
    SetRow(LeftBlock.Sp[0],1, {         3});
    SetRow(LeftBlock.Sp[0],2, {         3});
    SetRow(LeftBlock.Sp[0],3, {          });

    SetRow(LeftBlock.Sp[1],0, {   1, 2   });
    SetRow(LeftBlock.Sp[1],1, {         3});
    SetRow(LeftBlock.Sp[1],2, {         3});
    SetRow(LeftBlock.Sp[1],3, {          });

    /* Initialize to single-site defaults */
    ierr = RightBlock.Initialize(PETSC_COMM_WORLD, 1, PETSC_DEFAULT); CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD, "Left: %lu  Right %lu\n", LeftBlock.NumStates(), RightBlock.NumStates()); CHKERRQ(ierr);

    /* Calculate the Kronecker product of the blocks */
    ierr = Kron_Explicit(LeftBlock, RightBlock, BlockOut, PETSC_FALSE); CHKERRQ(ierr);

    ierr = LeftBlock.Destroy(); CHKERRQ(ierr);
    ierr = RightBlock.Destroy(); CHKERRQ(ierr);
    ierr = BlockOut.Destroy(); CHKERRQ(ierr);

    ierr = SlepcFinalize(); CHKERRQ(ierr);
    return ierr;
}
