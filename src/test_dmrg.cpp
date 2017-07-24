static char help[] =
    "Test Code for DMRG\n";

#include <iostream>
#include "idmrg_1d_heisenberg.hpp"


#if defined(__PRINT_SIZES) || defined(__PRINT_TRUNCATION_ERROR)
    #define PRINT_EMPTY_LINE PetscPrintf(comm,"\n");
#else
    #define PRINT_EMPTY_LINE
#endif


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

    /*
        Determine from options the target number of sites
        and number of states retained at each truncation
    */
    PetscInt nsites = 12;
    PetscInt mstates = 15;

    ierr = PetscOptionsGetInt(NULL,NULL,"-nsites",&nsites,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-mstates",&mstates,NULL); CHKERRQ(ierr);

    iDMRG_Heisenberg heis;
    heis.init(comm, nsites, mstates);

    ierr = PetscPrintf(comm,   "\n"
                        "iDMRG of the 1D Heisenberg model\n"
                        "Target number of sites  : %-10d\n"
                        "Number of states to keep: %-10d\n\n", nsites, mstates); CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,
            "   iter     nsites   gs energy   gs energy /site   rel error   ||Ax-kx||/||kx||\n"
            "  -------- -------- ----------- ----------------- ----------- ------------------\n"); CHKERRQ(ierr);
    PetscReal gse_r, gse_i, error;

    double gse_site_theor =  -0.4431471805599;
    PetscInt superblocklength = 0;

    while(heis.TotalLength() < heis.TargetLength() && heis.iter() < heis.TargetLength())
    {
        PRINT_EMPTY_LINE;

        ierr = heis.BuildBlockLeft(); CHKERRQ(ierr);
        ierr = heis.BuildBlockRight(); CHKERRQ(ierr);

        if (heis.TotalBasisSize() >= heis.mstates()*heis.mstates())
        {
            ierr = heis.BuildSuperBlock(); CHKERRQ(ierr);
            ierr = heis.SolveGroundState(gse_r, gse_i, error); CHKERRQ(ierr);
            superblocklength = heis.LengthBlockLeft() + heis.LengthBlockRight();

            PRINT_EMPTY_LINE;

            if (gse_i!=0.0) {
                // TODO: Implement error printing for complex values
                SETERRQ(comm,1,"Not implemented for complex ground state energy.\n");
                ierr = PetscPrintf(PETSC_COMM_WORLD," %6d    %9f%+9fi %12g\n", superblocklength, (double)gse_r/((double)(superblocklength)), (double)gse_i/((double)(superblocklength)),(double)error); CHKERRQ(ierr);
            } else {
                double gse_site  = (double)gse_r/((double)(superblocklength));
                double error_rel = (gse_site - gse_site_theor) / gse_site_theor;
                ierr = PetscPrintf(PETSC_COMM_WORLD,"   %6d   %6d%12f    %12f     %9f    %12g\n", heis.iter(), superblocklength, (double)gse_r, gse_site,  error_rel, (double)(error)); CHKERRQ(ierr);
            }

            PRINT_EMPTY_LINE;

            ierr = heis.BuildReducedDensityMatrices(); CHKERRQ(ierr);
            ierr = heis.GetRotationMatrices(); CHKERRQ(ierr);
            ierr = heis.TruncateOperators(); CHKERRQ(ierr);
        } else {
            #if defined(__PRINT_SIZES) || defined(__PRINT_TRUNCATION_ERROR)
                PetscPrintf(PETSC_COMM_WORLD,"   %6d\n", heis.iter());
            #endif
        }

        heis.iter()++;
    }

    ierr = heis.destroy(); CHKERRQ(ierr);

    SlepcFinalize();
    return ierr;
}
