static char help[] =
    "Test Code for DMRG\n";

#include <iostream>
#include "idmrg_1d_heisenberg.hpp"


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

    PetscPrintf(comm, "nsites  %-10d\n" "mstates %-10d \n\n", nsites, mstates);

    ierr = PetscPrintf(PETSC_COMM_WORLD,
            "   iter     nsites   gs energy   gs energy /site   rel error   ||Ax-kx||/||kx||\n"
            "  -------- -------- ----------- ----------------- ----------- ------------------\n");CHKERRQ(ierr);
    PetscReal gse_r, gse_i, error;

    double gse_site_theor =  -0.4431471805599;

    while(heis.TotalLength() < heis.TargetLength() && heis.iter() < heis.TargetLength())
    {
        heis.BuildBlockLeft();
        heis.BuildBlockRight();

        if (heis.TotalBasisSize() >= heis.mstates()*heis.mstates())
        {
            heis.BuildSuperBlock();
            heis.SolveGroundState(gse_r, gse_i, error);
            PetscInt superblocklength = heis.LengthBlockLeft() + heis.LengthBlockRight();

            if (gse_i!=0.0) {
                // TODO: Implement error printing for complex values
                SETERRQ(comm,1,"Not implemented for complex ground state energy.\n");
                ierr = PetscPrintf(PETSC_COMM_WORLD," %6d    %9f%+9fi %12g\n", superblocklength, (double)gse_r/((double)(superblocklength)), (double)gse_i/((double)(superblocklength)),(double)error);CHKERRQ(ierr);
            } else {
                double gse_site  = (double)gse_r/((double)(superblocklength));
                double error_rel = (gse_site - gse_site_theor) / gse_site_theor;
                ierr = PetscPrintf(PETSC_COMM_WORLD,"   %6d   %6d%12f    %12f     %9f    %12g\n", heis.iter(), superblocklength, (double)gse_r, gse_site,  error_rel, (double)(error)); CHKERRQ(ierr);
            }

            heis.BuildReducedDensityMatrices();
            heis.GetRotationMatrices();
            heis.TruncateOperators();
        }

        PetscPrintf(comm, "Total sites: %d\n", heis.TotalLength());
        heis.iter()++;
    }

    // heis.MatSaveOperators();
    // heis.MatPeekOperators();
    heis.destroy();

    SlepcFinalize();
    return ierr;
}
