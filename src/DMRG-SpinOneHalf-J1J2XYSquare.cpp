static char help[] =
    "DMRG executable for the Spin-1/2 J1-J2 XY Model on a two-dimensional square lattice.\n";

#include "DMRGBlock.hpp"
#include "Hamiltonians.hpp"
#include "DMRGBlockContainer.hpp"

#define MAX_SWEEPS 100

int main(int argc, char **argv)
{
    PetscErrorCode  ierr;
    PetscMPIInt     nprocs, rank;
    ierr = SlepcInitialize(&argc, &argv, (char*)0, help); CHKERRQ(ierr);
    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &nprocs); CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);
    {
        DMRGBlockContainer<Block::SpinOneHalf, Hamiltonians::J1J2XYModel_SquareLattice> DMRG(PETSC_COMM_WORLD);

        /* Default behavior: Use the same number of states for warmup and sweeps */
        PetscInt nsweeps = 1, mstates = 8;
        ierr = PetscOptionsGetInt(NULL,NULL,"-mstates",&mstates,NULL); CHKERRQ(ierr);
        ierr = PetscOptionsGetInt(NULL,NULL,"-nsweeps",&nsweeps,NULL); CHKERRQ(ierr);

        /* Optional behavior: Specify the number of states for warmup, and the number of states for successive sweeps */
        PetscBool use_msweeps = PETSC_FALSE;
        PetscInt num_msweeps = MAX_SWEEPS;
        std::vector<PetscInt> msweeps(MAX_SWEEPS);
        ierr = PetscOptionsGetIntArray(NULL,NULL,"-msweeps",&msweeps[0],&num_msweeps,&use_msweeps); CHKERRQ(ierr);
        msweeps.resize(num_msweeps);
        if(!rank){
            printf( "WARMUP\n");
            printf( "  m_states:                %d\n",mstates);
            printf( "SWEEPS\n");
            printf( "  Use msweeps array:       %s\n",use_msweeps?"yes":"no");
            printf( "  Number of sweeps:        %d\n",use_msweeps?num_msweeps:nsweeps);
            printf( "  m_states:               ");
            if(use_msweeps) for(const PetscInt& m: msweeps) printf(" %d ",m);
            else printf(" %d ",mstates);
            printf("\n");
            printf("=========================================\n");
        }

        ierr = DMRG.Warmup(mstates); CHKERRQ(ierr);
        if(use_msweeps){
            for(const PetscInt& mstates: msweeps){
                ierr = DMRG.Sweep(mstates); CHKERRQ(ierr);
            }
        } else {
            for(PetscInt isweep = 0; isweep < nsweeps; ++isweep){
                ierr = DMRG.Sweep(mstates); CHKERRQ(ierr);
            }
        }
    }
    ierr = SlepcFinalize(); CHKERRQ(ierr);
    return(0);
}
