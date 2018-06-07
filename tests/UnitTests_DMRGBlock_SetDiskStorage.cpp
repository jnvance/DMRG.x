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

PetscErrorCode Test()
{
    PetscErrorCode ierr;
    PetscMPIInt     nprocs, rank;
    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &nprocs); CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);

    Block::SpinOneHalf blk;
    ierr = blk.Initialize(PETSC_COMM_WORLD, 2, {1.5,0.5,-0.5,-1.5}, {2,3,2,1});CHKERRQ(ierr);
    ierr = blk.CheckSectors(); CHKERRQ(ierr);
    ierr = Makedir("trash_block_test_save01"); CHKERRQ(ierr);
    ierr = Makedir("trash_block_test_save02"); CHKERRQ(ierr);
    ierr = Makedir("trash_block_test_save03"); CHKERRQ(ierr);

    ierr = blk.SetDiskStorage("trash_block_test_save01","trash_block_test_save01"); CHKERRQ(ierr);

    /*  Set the entries of the operators */
    ierr = SetSz0(blk.Sz(0)); CHKERRQ(ierr);
    ierr = SetSp0(blk.Sp(0)); CHKERRQ(ierr);
    ierr = SetSz1(blk.Sz(1)); CHKERRQ(ierr);
    ierr = SetSp1(blk.Sp(1)); CHKERRQ(ierr);
    ierr = blk.AssembleOperators(); CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,"Writing data to:   %s\n", blk.SaveDir().c_str()); CHKERRQ(ierr);
    ierr = blk.EnsureSaved(); CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nSetting read_dir as: \"%s\" and write_dir as: \"%s\"\n",
        "trash_block_test_save01","trash_block_test_save02"); CHKERRQ(ierr);
    ierr = blk.SetDiskStorage("trash_block_test_save01","trash_block_test_save02"); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Reading data from: %s\n", blk.SaveDir().c_str()); CHKERRQ(ierr);
    ierr = blk.EnsureRetrieved(); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Writing data to:   %s\n", blk.SaveDir().c_str()); CHKERRQ(ierr);
    ierr = blk.EnsureSaved(); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Reading data from: %s\n", blk.SaveDir().c_str()); CHKERRQ(ierr);
    ierr = blk.EnsureRetrieved(); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Writing data to:   %s\n", blk.SaveDir().c_str()); CHKERRQ(ierr);
    ierr = blk.EnsureSaved(); CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nSetting read_dir as: \"%s\" and write_dir as: \"%s\"\n",
        "trash_block_test_save02","trash_block_test_save03"); CHKERRQ(ierr);
    ierr = blk.SetDiskStorage("trash_block_test_save02","trash_block_test_save03"); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Reading data from: %s\n", blk.SaveDir().c_str()); CHKERRQ(ierr);
    ierr = blk.EnsureRetrieved(); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Writing data to:   %s\n", blk.SaveDir().c_str()); CHKERRQ(ierr);
    ierr = blk.EnsureSaved(); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Reading data from: %s\n", blk.SaveDir().c_str()); CHKERRQ(ierr);
    ierr = blk.EnsureRetrieved(); CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Writing data to:   %s\n", blk.SaveDir().c_str()); CHKERRQ(ierr);
    ierr = blk.EnsureSaved(); CHKERRQ(ierr);

    ierr = blk.Destroy(); CHKERRQ(ierr);
    return(0);
}

int main(int argc, char **argv)
{
    PetscErrorCode  ierr = 0;
    PetscMPIInt     nprocs, rank;
    ierr = SlepcInitialize(&argc, &argv, (char*)0, help); CHKERRQ(ierr);
    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &nprocs); CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);

    ierr = Test(); CHKERRQ(ierr);

    ierr = SlepcFinalize(); CHKERRQ(ierr);
    return ierr;
}
