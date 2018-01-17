static char help[] =
    "Test code for DMRGBlock module\n";

#include <slepceps.h>
#include "DMRGBlock.hpp"
#include "linalg_tools.hpp"
#include <iostream>

PETSC_EXTERN PetscErrorCode SetRow(const Mat& A, const PetscInt& row, const std::vector<PetscInt>& idxn);

/** Tests the CheckOperatorBlocks method */
PetscErrorCode Test_MatCheckOperatorBlocks()
{
    PetscErrorCode ierr = 0;
    /*  Create an artificial block object */
    Block_SpinOneHalf blk;
    ierr = blk.Initialize(PETSC_COMM_WORLD, 2, 8);CHKERRQ(ierr);

    /*  Initialize the block's magnetization sectors */
    ierr = blk.Magnetization.Initialize(PETSC_COMM_WORLD, {1.5,0.5,-0.5,-1.5}, {2,3,2,1}); CHKERRQ(ierr);
    ierr = blk.CheckSectors(); CHKERRQ(ierr);

    /*  Set the entries of Sz[0] following the correct sectors */
    ierr = SetRow(blk.Sz[0], 0, {0,1}); CHKERRQ(ierr);
    ierr = SetRow(blk.Sz[0], 1, {1}); CHKERRQ(ierr);
    ierr = SetRow(blk.Sz[0], 2, {2,3,4}); CHKERRQ(ierr);
    ierr = SetRow(blk.Sz[0], 3, {3}); CHKERRQ(ierr);
    ierr = SetRow(blk.Sz[0], 4, {2,4}); CHKERRQ(ierr);
    ierr = SetRow(blk.Sz[0], 5, {5,6}); CHKERRQ(ierr);
    ierr = SetRow(blk.Sz[0], 7, {7}); CHKERRQ(ierr);
    ierr = MatPeek(blk.Sz[0], "blk.Sz[0]"); CHKERRQ(ierr);
    ierr = blk.MatCheckOperatorBlocks(OpSz, 0); CHKERRQ(ierr);

    /*  Set the entries of Sp[0] following the correct sectors */
    ierr = SetRow(blk.Sp[0], 0, {2,3,4}); CHKERRQ(ierr);
    ierr = SetRow(blk.Sp[0], 1, {2,4}); CHKERRQ(ierr);
    ierr = SetRow(blk.Sp[0], 2, {5,6}); CHKERRQ(ierr);
    ierr = SetRow(blk.Sp[0], 3, {5}); CHKERRQ(ierr);
    ierr = SetRow(blk.Sp[0], 4, {6}); CHKERRQ(ierr);
    ierr = SetRow(blk.Sp[0], 6, {7}); CHKERRQ(ierr);
    ierr = MatPeek(blk.Sp[0], "blk.Sp[0]"); CHKERRQ(ierr);
    ierr = blk.MatCheckOperatorBlocks(OpSp, 0); CHKERRQ(ierr);

    /*  Set the entries of Sz[1] following the INCORRECT sectors */
    ierr = SetRow(blk.Sz[1], 0, {0,1}); CHKERRQ(ierr);
    ierr = SetRow(blk.Sz[1], 1, {1}); CHKERRQ(ierr);
    ierr = SetRow(blk.Sz[1], 2, {2,3,4}); CHKERRQ(ierr);
    ierr = SetRow(blk.Sz[1], 3, {3}); CHKERRQ(ierr);
    ierr = SetRow(blk.Sz[1], 4, {2,4}); CHKERRQ(ierr);
    ierr = SetRow(blk.Sz[1], 5, {5,6}); CHKERRQ(ierr);
    ierr = SetRow(blk.Sz[1], 7, {1,7}); CHKERRQ(ierr); /* Error here */
    ierr = MatPeek(blk.Sz[1], "blk.Sz[1]"); CHKERRQ(ierr);
    ierr = blk.MatCheckOperatorBlocks(OpSz, 1);
    /*  Verify that error has been caught */
    {
        PetscInt rstart, rend;
        MatGetOwnershipRange(blk.Sz[1], &rstart, &rend);
        if(rstart <= 7 && 7 < rend){
            if(ierr!=PETSC_ERR_ARG_OUTOFRANGE){
                SETERRQ(PETSC_COMM_SELF, 1, "Failed test");
            } else {
                printf("\n    Exception caught. :)\n\n");
            }
        }
    }
    ierr = blk.Destroy(); CHKERRQ(ierr);
    return ierr;
}

int main(int argc, char **argv)
{
    PetscErrorCode  ierr = 0;
    PetscMPIInt     nprocs, rank;
    MPI_Comm&       comm = PETSC_COMM_WORLD;

    /*  Initialize MPI  */
    ierr = SlepcInitialize(&argc, &argv, (char*)0, help); CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm, &nprocs); CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);

    ierr = Test_MatCheckOperatorBlocks(); CHKERRQ(ierr);

    ierr = SlepcFinalize(); CHKERRQ(ierr);
    return ierr;
}
