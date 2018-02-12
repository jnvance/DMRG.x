static char help[] =
    "Test code for DMRGBlockContainer module\n";

#include "DMRGBlock.hpp"
#include "Hamiltonians.hpp"
#include "DMRGBlockContainer.hpp"

PETSC_EXTERN PetscErrorCode SetRow(const Mat& A, const PetscInt& row, const std::vector<PetscInt>& idxn);
PETSC_EXTERN PetscErrorCode CatchErrorCode(const MPI_Comm& comm, const PetscInt& ierr_in, const PetscInt& ierr_exp);

PetscErrorCode Test()
{
    PetscErrorCode ierr = 0;

    DMRGBlockContainer<Block::SpinOneHalf, Hamiltonian::J1J2XYModel_SquareLattice> DMRG(PETSC_COMM_WORLD);

    ierr = DMRG.Warmup(10); CHKERRQ(ierr);

    /* Peek at the last created envblock */
    ierr = MatPeek(DMRG.EnvBlock().Sz(0),"DMRG.EnvBlock().Sz(0)"); CHKERRQ(ierr);

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

    ierr = Test(); CHKERRQ(ierr);

    ierr = SlepcFinalize(); CHKERRQ(ierr);
    return ierr;
}
