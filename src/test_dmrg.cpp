static char help[] =
    "Test Code for DMRG\n";

#include <iostream>
#include "idmrg_1d_heisenberg.hpp"


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
    PetscErrorCode  ierr = 0;
    SlepcInitialize(&argc, &argv, (char*)0, help);

    MPI_Comm        comm = PETSC_COMM_WORLD;
    PetscMPIInt     nprocs, rank;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    /*
        Determine from options the target number of sites,
        the number of states retained at each truncation, and
        the coupling constants
    */
    PetscInt nsites = 12;
    PetscInt mstates = 15;
    PetscScalar J  = 1.0;
    PetscScalar Jz = 1.0;
    PetscScalar target_Sz = 0.0;
    PetscBool do_target_Sz = PETSC_FALSE;
    PetscBool do_save_operators = PETSC_FALSE;

    ierr = PetscOptionsGetInt(NULL,NULL,"-nsites",&nsites,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-mstates",&mstates,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-J", &J,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-Jz",&Jz,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(NULL,NULL,"-do_target_Sz",&do_target_Sz,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(NULL,NULL,"-do_save_operators",&do_save_operators,NULL); CHKERRQ(ierr);

    iDMRG_Heisenberg heis;
    heis.init(comm, nsites, mstates);
    heis.SetParameters(J, Jz);
    heis.SetTargetSz(target_Sz, do_target_Sz);

    /*
        Timings
    */
    PetscLogDouble total_time0, total_time;
    ierr = PetscTime(&total_time0); CHKERRQ(ierr);

    PetscBool petsc_use_complex = PETSC_FALSE;
    #ifdef PETSC_USE_COMPLEX
        petsc_use_complex = PETSC_TRUE;
    #endif

    char scalar_type[80];
    sprintf(scalar_type, (petsc_use_complex == PETSC_TRUE) ? "COMPLEX" : "REAL");

    ierr = PetscPrintf(comm,
                        "iDMRG of the 1D Heisenberg model\n"
                        "Coupling J              : %-f\n"
                        "Anisotropy Jz           : %-f\n"
                        "PetscScalar type        : %-s\n"
                        "Target number of sites  : %-d\n"
                        "Number of states to keep: %-d\n"
                        "Number of MPI processes : %-d\n"
                        "Do target magnetization : %-s\n"
                        "Target magnetization    : %-f\n"
                        "\n",
                        J, Jz, scalar_type, nsites, mstates, nprocs,
                        do_target_Sz==PETSC_TRUE ? "yes" : "no", target_Sz); CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,
            "   iter     nsites   gs energy   gs energy /site   rel error   ||Ax-kx||/||kx||     Truncation Errors\n"
            "  -------- -------- ----------- ----------------- ----------- ------------------ -------------------------\n"); CHKERRQ(ierr);
    PetscReal gse_r, gse_i, error;
    PetscReal truncerr_left, truncerr_right;

    double gse_site_theor =  -0.4431471805599;
    PetscInt superblocklength = 0;

    FILE *fp;
    ierr = PetscFOpen(PETSC_COMM_WORLD, "eigvals.dat", "w", &fp); CHKERRQ(ierr);

    while(heis.TotalLength() < heis.TargetLength())
    {
        heis.iter()++;
        /*
            Grow the left and right blocks by adding one site in the junction
        */
        ierr = heis.BuildBlockLeft(); CHKERRQ(ierr);
        ierr = heis.BuildBlockRight(); CHKERRQ(ierr);
        /*
            As long as the basis size is less than
            the number of kept states, continue adding sites
        */
        if (heis.TotalBasisSize() <= heis.mstates()*heis.mstates())
            continue;

        ierr = heis.BuildSuperBlock(); CHKERRQ(ierr);
        if (do_save_operators){
            ierr = heis.MatSaveOperators(); CHKERRQ(ierr);
        }
        ierr = heis.SolveGroundState(gse_r, gse_i, error); CHKERRQ(ierr);
        /*
            Printout data on ground state energy and wavevector
        */
        superblocklength = heis.LengthBlockLeft() + heis.LengthBlockRight();

        /*
            From the ground state wavevector get the reduced density matrices of the left
            and right blocks, and construct the rectangular rotation matrices to perform the
            truncation of site and block operators.
         */
        ierr = heis.BuildReducedDensityMatrices(); CHKERRQ(ierr);
        ierr = heis.GetRotationMatrices(truncerr_left, truncerr_right); CHKERRQ(ierr);
        ierr = heis.TruncateOperators(); CHKERRQ(ierr);

        if (gse_i!=0.0) {
            SETERRQ(comm,1,"Not implemented for complex ground state energy.\n");
        } else {
            double gse_site  = (double)gse_r/((double)(superblocklength));
            double error_rel = (gse_site - gse_site_theor) / std::abs(gse_site_theor);
            ierr = PetscPrintf(PETSC_COMM_WORLD,"   %6d   %6d%12f    %12f     %9f     %12g     %+8.5g  %+8.5g\n",
                heis.iter(), superblocklength, (double)gse_r, gse_site,
                error_rel, (double)(error), (double)(truncerr_left), (double)(truncerr_right)); CHKERRQ(ierr);
            ierr = PetscFPrintf(PETSC_COMM_WORLD, fp,"   %6d   %6d    %.20g    %.20g    %.20g    %.20g    %.20g    %.20g\n",
                heis.iter(), superblocklength, (double)gse_r, gse_site,
                error_rel, (double)(error), (double)(truncerr_left), (double)(truncerr_right)); CHKERRQ(ierr);
        }
    }

    ierr = PetscTime(&total_time); CHKERRQ(ierr); \
    total_time = total_time - total_time0; \
    ierr = PetscPrintf(PETSC_COMM_WORLD, "%10s      %-50s %.20g\n", " ", "TotalTime", total_time);
    #ifdef __TIMINGS
    ierr = PetscFPrintf(PETSC_COMM_WORLD, heis.fp_timings, "%10d      %-50s %.20g\n", heis.iter(), "TotalTime", total_time);
    #endif //__TIMINGS


    #ifdef __TESTING
    ierr = heis.MatSaveOperators(); CHKERRQ(ierr);
    #endif

    ierr = PetscFClose(PETSC_COMM_WORLD, fp); CHKERRQ(ierr);

    ierr = heis.destroy(); CHKERRQ(ierr);

    SlepcFinalize();
    return ierr;
}
