static char help[] =
    "Test code for DMRGBlock module which retrieves blocks from disk "
    "";

#include <slepceps.h>
#include "DMRGBlock.hpp"
#include "linalg_tools.hpp"
#include <iostream>

PETSC_EXTERN PetscErrorCode SetRow(const Mat& A, const PetscInt& row, const std::vector<PetscInt>& idxn);
PETSC_EXTERN PetscErrorCode Makedir(const std::string& dir_name);

PetscErrorCode RetrieveBlockFromDisk(Block::SpinOneHalf& blk)
{
    PetscErrorCode ierr;
    char           file[PETSC_MAX_PATH_LEN];
    PetscBool      flg;
    ierr = PetscOptionsGetString(NULL,NULL,"-blockdir",file,PETSC_MAX_PATH_LEN,&flg); CHKERRQ(ierr);
    if(!flg) SETERRQ(PETSC_COMM_WORLD,1,"-blockdir <str> must be specified.");
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Reading from: %s\n",file); CHKERRQ(ierr);
    ierr = blk.InitializeFromDisk(PETSC_COMM_WORLD,std::string(file)); CHKERRQ(ierr);
    ierr = blk.AssembleOperators(); CHKERRQ(ierr);
    ierr = blk.CheckOperatorBlocks(); CHKERRQ(ierr);
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

    Block::SpinOneHalf blk1;

    ierr = RetrieveBlockFromDisk(blk1); CHKERRQ(ierr);

    ierr = blk1.Destroy(); CHKERRQ(ierr);

    ierr = SlepcFinalize(); CHKERRQ(ierr);
    return ierr;
}
