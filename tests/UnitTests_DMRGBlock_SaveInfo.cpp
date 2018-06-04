static char help[] =
    "Test code for DMRGBlock module on saving and retrieving blocks from disk \n";

#include <slepceps.h>
#include "DMRGBlock.hpp"
#include "linalg_tools.hpp"
#include <iostream>

PETSC_EXTERN PetscErrorCode SetRow(const Mat& A, const PetscInt& row, const std::vector<PetscInt>& idxn);
PETSC_EXTERN PetscErrorCode Makedir(const std::string& dir_name);
PETSC_EXTERN PetscErrorCode SetSz0(const Mat& Sz);
PETSC_EXTERN PetscErrorCode SetSp0(const Mat& Sp);
PETSC_EXTERN PetscErrorCode SetSz1(const Mat& Sz);
PETSC_EXTERN PetscErrorCode SetSp1(const Mat& Sp);
PETSC_EXTERN PetscErrorCode Makedir(const std::string& dir_name);

PetscErrorCode CreateAndSaveBlock(Block::SpinOneHalf& blk)
{
    PetscErrorCode ierr;

    ierr = blk.Initialize(PETSC_COMM_WORLD, 2, {1.5,0.5,-0.5,-1.5}, {2,3,2,1});CHKERRQ(ierr);
    ierr = blk.CheckSectors(); CHKERRQ(ierr);
    ierr = Makedir("trash_block_test_save"); CHKERRQ(ierr);
    ierr = blk.InitializeSave("trash_block_test_save"); CHKERRQ(ierr);

    /*  Set the entries of the operators following the correct sectors */
    ierr = SetSz0(blk.Sz(0)); CHKERRQ(ierr);
    ierr = SetSp0(blk.Sp(0)); CHKERRQ(ierr);
    ierr = SetSz1(blk.Sz(1)); CHKERRQ(ierr);
    ierr = SetSp1(blk.Sp(1)); CHKERRQ(ierr);
    ierr = blk.AssembleOperators(); CHKERRQ(ierr);
    ierr = blk.EnsureSaved(); CHKERRQ(ierr);

    return(0);
}

PetscErrorCode RetrieveBlockFromDisk(Block::SpinOneHalf& blk)
{
    PetscErrorCode ierr;
    ierr = blk.InitializeFromDisk(PETSC_COMM_WORLD,"trash_block_test_save/"); CHKERRQ(ierr);

    return(0);
}

PetscErrorCode CompareBlocks(
    Block::SpinOneHalf& blk1,
    Block::SpinOneHalf& blk2
    )
{
    return(0);
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

    Block::SpinOneHalf blk1, blk2;

    ierr = CreateAndSaveBlock(blk1); CHKERRQ(ierr);
    ierr = RetrieveBlockFromDisk(blk2); CHKERRQ(ierr);
    ierr = CompareBlocks(blk1, blk2); CHKERRQ(ierr);

    ierr = blk1.Destroy(); CHKERRQ(ierr);
    ierr = blk2.Destroy(); CHKERRQ(ierr);

    ierr = SlepcFinalize(); CHKERRQ(ierr);
    return ierr;
}
