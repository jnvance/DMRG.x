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

PetscErrorCode CreateAndSaveBlock(Block::SpinBase& blk)
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
    ierr = blk.EnsureRetrieved(); CHKERRQ(ierr);

    return(0);
}

PetscErrorCode RetrieveBlockFromDisk(Block::SpinBase& blk)
{
    PetscErrorCode ierr;
    ierr = blk.InitializeFromDisk(PETSC_COMM_WORLD,"trash_block_test_save/"); CHKERRQ(ierr);
    ierr = blk.AssembleOperators(); CHKERRQ(ierr);

    return(0);
}

PetscErrorCode CompareBlocks(
    Block::SpinBase& blk1,
    Block::SpinBase& blk2
    )
{
    PetscErrorCode ierr;
    PetscBool flg;

    if(blk1.NumSites()!=blk2.NumSites()) SETERRQ3(PETSC_COMM_WORLD,1,"Unequal %s. "
        "Block1: %lld. Block2: %lld.", "NumSites", blk1.NumSites(), blk2.NumSites());
    if(blk1.NumStates()!=blk2.NumStates()) SETERRQ3(PETSC_COMM_WORLD,1,"Unequal %s. "
        "Block1: %lld. Block2: %lld.", "NumStates", blk1.NumStates(), blk2.NumStates());
    if(blk1.Magnetization.NumStates()!=blk2.Magnetization.NumStates())
        SETERRQ3(PETSC_COMM_WORLD,1,"Unequal %s. " "Block1: %lld. Block2: %lld.",
            "Magnetization.NumStates", blk1.Magnetization.NumStates(), blk2.Magnetization.NumStates());
    if(blk1.Magnetization.NumSectors()!=blk2.Magnetization.NumSectors())
        SETERRQ3(PETSC_COMM_WORLD,1,"Unequal %s. " "Block1: %lld. Block2: %lld.",
            "Magnetization.NumSectors", blk1.Magnetization.NumSectors(), blk2.Magnetization.NumSectors());

    for(PetscInt idx=0; idx<blk1.Magnetization.NumSectors(); ++idx)
    {
        if(blk1.Magnetization.List(idx)!=blk2.Magnetization.List(idx))
            SETERRQ4(PETSC_COMM_WORLD,1,"Unequal %s at idx %lld. Block1: %g. Block2: %g.",
                "Magnetization.List", idx, blk1.Magnetization.List(idx), blk2.Magnetization.List(idx));

        if(blk1.Magnetization.Sizes(idx)!=blk2.Magnetization.Sizes(idx))
            SETERRQ4(PETSC_COMM_WORLD,1,"Unequal %s at idx %lld. Block1: %lld. Block2: %lld.",
                "Magnetization.List", idx, blk1.Magnetization.Sizes(idx), blk2.Magnetization.Sizes(idx));
    }

    /* Compare the operators with each other */
    ierr = MatEqual(blk1.Sz(0), blk2.Sz(0), &flg); CHKERRQ(ierr);
    if(!flg) SETERRQ(PETSC_COMM_WORLD,1,"Wrong retrieval of matrix Sz0");
    ierr = MatEqual(blk1.Sp(0), blk2.Sp(0), &flg); CHKERRQ(ierr);
    if(!flg) SETERRQ(PETSC_COMM_WORLD,1,"Wrong retrieval of matrix Sp0");
    ierr = MatEqual(blk1.Sz(1), blk2.Sz(1), &flg); CHKERRQ(ierr);
    if(!flg) SETERRQ(PETSC_COMM_WORLD,1,"Wrong retrieval of matrix Sz1");
    ierr = MatEqual(blk1.Sp(1), blk2.Sp(1), &flg); CHKERRQ(ierr);
    if(!flg) SETERRQ(PETSC_COMM_WORLD,1,"Wrong retrieval of matrix Sp1");

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

    Block::SpinBase blk1, blk2;

    ierr = CreateAndSaveBlock(blk1); CHKERRQ(ierr);
    ierr = RetrieveBlockFromDisk(blk2); CHKERRQ(ierr);
    ierr = CompareBlocks(blk1, blk2); CHKERRQ(ierr);

    ierr = blk1.Destroy(); CHKERRQ(ierr);
    ierr = blk2.Destroy(); CHKERRQ(ierr);

    ierr = SlepcFinalize(); CHKERRQ(ierr);
    return ierr;
}
