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

    ierr = PetscOptionsGetInt(NULL,NULL,"-nsites",&nsites,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-mstates",&mstates,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-J", &J, NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-Jz",&Jz,NULL); CHKERRQ(ierr);

    iDMRG_Heisenberg heis;

    heis.init(comm, nsites, mstates);
    heis.SetParameters(J, Jz, 0.0);

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

    ierr = PetscPrintf(comm,   "\n"
                        "iDMRG of the 1D Heisenberg model\n"
                        "J = %f    Jz = %f\n"
                        "PetscScalar type        : %-s\n"
                        "Target number of sites  : %-d\n"
                        "Number of states to keep: %-d\n"
                        "Number of MPI processes : %-d\n\n",
                        J, Jz, scalar_type, nsites, mstates, nprocs); CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,
            "   iter     nsites   gs energy   gs energy /site   rel error   ||Ax-kx||/||kx||\n"
            "  -------- -------- ----------- ----------------- ----------- ------------------\n"); CHKERRQ(ierr);
    PetscReal gse_r, gse_i, error;

    double gse_site_theor =  -0.4431471805599;
    PetscInt superblocklength = 0;

    /*
        Opens file to save eigenenergies
        TODO: File may also be used to save other data
     */
    FILE *fp;

    ierr = PetscFOpen(PETSC_COMM_WORLD, "eigvals.dat", "w", &fp); CHKERRQ(ierr);

    while(heis.TotalLength() < heis.TargetLength() && heis.iter() < heis.TargetLength())
    {
        /*
            Grow the left and right blocks by adding one site in the junction
         */
        ierr = heis.BuildBlockLeft(); CHKERRQ(ierr);
        ierr = heis.BuildBlockRight(); CHKERRQ(ierr);
        ierr = heis.BuildSuperBlock(); CHKERRQ(ierr);
        ierr = heis.MatSaveOperators(); CHKERRQ(ierr);
        ierr = heis.SolveGroundState(gse_r, gse_i, error); CHKERRQ(ierr);
        /*
                Printout data on ground state energy and wavevector
        */
        PRINT_EMPTY_LINE;
        superblocklength = heis.LengthBlockLeft() + heis.LengthBlockRight();
        if (gse_i!=0.0) {
            /*
                TODO: Implement error printing for complex values
             */
            SETERRQ(comm,1,"Not implemented for complex ground state energy.\n");
            // ierr = PetscPrintf(PETSC_COMM_WORLD," %6d    %9f%+9fi %12g\n",
            //     superblocklength, (double)gse_r/((double)(superblocklength)),
            //     (double)gse_i/((double)(superblocklength)),(double)error); CHKERRQ(ierr);
        } else {
            double gse_site  = (double)gse_r/((double)(superblocklength));
            double error_rel = (gse_site - gse_site_theor) / gse_site_theor;
            ierr = PetscPrintf(PETSC_COMM_WORLD,"   %6d   %6d%12f    %12f     %9f    %12g\n",
                heis.iter(), superblocklength, (double)gse_r, gse_site,
                error_rel, (double)(error)); CHKERRQ(ierr);
            ierr = PetscFPrintf(PETSC_COMM_WORLD, fp,"   %6d   %6d    %.20g    %.20g    %.20g    %.20g\n",
                heis.iter(), superblocklength, (double)gse_r, gse_site,
                error_rel, (double)(error)); CHKERRQ(ierr);
        }
        PRINT_EMPTY_LINE;
        /*
            From the ground state wavevector get the reduced density matrices of the left
            and right blocks, and construct the rectangular rotation matrices to perform the
            truncation of site and block operators.
         */
        ierr = heis.BuildReducedDensityMatrices(); CHKERRQ(ierr);
        ierr = heis.GetRotationMatrices(); CHKERRQ(ierr);
        ierr = heis.TruncateOperators(); CHKERRQ(ierr);

        PetscPrintf(PETSC_COMM_WORLD,"   %6d\n", heis.iter());

        heis.iter()++;
    }


#if 0
    while(heis.TotalLength() < heis.TargetLength() && heis.iter() < heis.TargetLength())
    {
        PRINT_EMPTY_LINE;
        #ifdef __TESTING
        ierr = heis.MatSaveOperators(); CHKERRQ(ierr);
        #endif
        /*
            Grow the left and right blocks by adding one site in the junction
         */
        ierr = heis.BuildBlockLeft(); CHKERRQ(ierr);
        ierr = heis.BuildBlockRight(); CHKERRQ(ierr);
        /*
            Save operator states and increment
         */
        #ifdef __TESTING
        ierr = heis.BuildSuperBlock(); CHKERRQ(ierr);
        ierr = heis.MatSaveOperators(); CHKERRQ(ierr);
        ierr = heis.SolveGroundState(gse_r, gse_i, error); CHKERRQ(ierr);
        #endif
        /*
            Check whether the basis set already need truncation
         */
        #ifndef __TESTING
        if (heis.TotalBasisSize() > heis.mstates()*heis.mstates())
        {
            /*
                The superblock Hamiltonian is constructed and the ground state is solved
             */
            // #ifndef __TESTING

            ierr = heis.BuildSuperBlock(); CHKERRQ(ierr);
            ierr = heis.SolveGroundState(gse_r, gse_i, error); CHKERRQ(ierr);

        #endif
            /*
                Printout data on ground state energy and wavevector
             */
            PRINT_EMPTY_LINE;
            superblocklength = heis.LengthBlockLeft() + heis.LengthBlockRight();
            if (gse_i!=0.0) {
                /*
                    TODO: Implement error printing for complex values
                 */
                SETERRQ(comm,1,"Not implemented for complex ground state energy.\n");
                // ierr = PetscPrintf(PETSC_COMM_WORLD," %6d    %9f%+9fi %12g\n",
                //     superblocklength, (double)gse_r/((double)(superblocklength)),
                //     (double)gse_i/((double)(superblocklength)),(double)error); CHKERRQ(ierr);
            } else {
                double gse_site  = (double)gse_r/((double)(superblocklength));
                double error_rel = (gse_site - gse_site_theor) / gse_site_theor;
                ierr = PetscPrintf(PETSC_COMM_WORLD,"   %6d   %6d%12f    %12f     %9f    %12g\n",
                    heis.iter(), superblocklength, (double)gse_r, gse_site,
                    error_rel, (double)(error)); CHKERRQ(ierr);
                ierr = PetscFPrintf(PETSC_COMM_WORLD, fp,"   %6d   %6d    %.20g    %.20g    %.20g    %.20g\n",
                    heis.iter(), superblocklength, (double)gse_r, gse_site,
                    error_rel, (double)(error)); CHKERRQ(ierr);
            }
            PRINT_EMPTY_LINE;

        #ifdef __TESTING
        if (heis.TotalBasisSize() > heis.mstates()*heis.mstates())
        {
        #endif
            /*
                From the ground state wavevector get the reduced density matrices of the left
                and right blocks, and construct the rectangular rotation matrices to perform the
                truncation of site and block operators.
             */
            ierr = heis.BuildReducedDensityMatrices(); CHKERRQ(ierr);
            ierr = heis.GetRotationMatrices(); CHKERRQ(ierr);
            ierr = heis.TruncateOperators(); CHKERRQ(ierr);
            #ifdef __TESTING
                ierr = PetscPrintf(PETSC_COMM_WORLD,"%9s>> Truncation \n"," "); CHKERRQ(ierr);
            #endif
        } else {
            #ifndef __TESTING
                #if defined(__PRINT_SIZES) || defined(__PRINT_TRUNCATION_ERROR)
                    PetscPrintf(PETSC_COMM_WORLD,"   %6d\n", heis.iter());
                #endif
            #endif
        }
        heis.iter()++;
    }
#endif

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
