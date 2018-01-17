static char help[] =
    "Test code for DMRGBlockContainer module\n";

#include "DMRGBlockContainer.hpp"

PETSC_EXTERN PetscErrorCode SetRow(const Mat& A, const PetscInt& row, const std::vector<PetscInt>& idxn);

/** Tests the EnlargeBlock method */
PetscErrorCode Test_EnlargeBlock()
{
    PetscErrorCode ierr = 0;

    /*  Initialize a block-container object */
    Heisenberg_SpinOneHalf_SquareLattice Lattice;
    ierr = Lattice.Initialize(); CHKERRQ(ierr);

    /*  Check the matrix operator blocks */
    ierr = Lattice.SysBlock(0).CheckOperators(); CHKERRQ(ierr);
    ierr = Lattice.EnvBlock().CheckOperators(); CHKERRQ(ierr);

    /* TODO: Insert some operations here with EnlargeBlock() */

    ierr = Lattice.Destroy(); CHKERRQ(ierr);
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

    ierr = Test_EnlargeBlock(); CHKERRQ(ierr);

    ierr = SlepcFinalize(); CHKERRQ(ierr);
    return ierr;
}
