static char help[] =
    "Test Code for DMRG\n";

#include <iostream>
#include <slepceps.h>

#include "dmrg.hpp"

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

    DMRGBlock block;
    block.init();

    // Build the Hamiltonian for a two-spin system
    // B(L,1) + S(1)

    Mat eye1, Sz1, Sp1, Sm1;
    MatEyeCreate(comm, eye1, 2);
    MatSzCreate(comm, Sz1);
    MatSpCreate(comm, Sp1);
    MatTranspose(Sp1, MAT_INITIAL_MATRIX, &Sm1);


    Mat H_temp;
    Mat block_Sm;

    ierr = MatAssemblyBegin(block.Sp(), MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(block.Sp(), MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    MatTranspose(block.Sp(), MAT_INITIAL_MATRIX, &block_Sm);

    MatKron(block.H(), eye1, H_temp, comm);
    MatKronAdd(block.Sz(), Sz1, H_temp, comm);

    MatKronScaleAdd(0.5, block.Sp(), Sm1, H_temp, comm);
    MatKronScaleAdd(0.5, block_Sm, Sp1, H_temp, comm);

    MatPeek(comm, eye1,   "eye1");
    MatPeek(comm, Sp1, "Sp1");
    MatPeek(comm, Sm1, "Sm1");
    MatPeek(comm, H_temp, "H_temp");

    /*------ Solver ------*/

    EPS eps;
    EPSCreate(comm,&eps);
    EPSSetOperators(eps,H_temp,NULL);
    EPSSetProblemType(eps,EPS_HEP);

    /*
        Set solver parameters at runtime
    */
    ierr = EPSSetFromOptions(eps);CHKERRQ(ierr);
    ierr = EPSSolve(eps);CHKERRQ(ierr);


    /*
        Get number of converged approximate eigenpairs
    */
    PetscInt nconv;
    ierr = EPSGetConverged(eps,&nconv);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD," Number of converged eigenpairs: %D\n\n",nconv);CHKERRQ(ierr);

    PetscReal re, im, error;
    PetscScalar    kr,ki;
    PetscInt i;
    Vec xr, xi;


    ierr = MatCreateVecs(H_temp,NULL,&xr);CHKERRQ(ierr);
    ierr = MatCreateVecs(H_temp,NULL,&xi);CHKERRQ(ierr);

    if (nconv>0) {
    /*
       Display eigenvalues and relative errors
    */
    ierr = PetscPrintf(PETSC_COMM_WORLD,
         "           k          ||Ax-kx||/||kx||\n"
         "   ----------------- ------------------\n");CHKERRQ(ierr);

        for (i=0;i<nconv;i++) {
            /*
              Get converged eigenpairs: i-th eigenvalue is stored in kr (real part) and
              ki (imaginary part)
            */
            ierr = EPSGetEigenpair(eps,i,&kr,&ki,xr,xi);CHKERRQ(ierr);
            /*
               Compute the relative error associated to each eigenpair
            */
            ierr = EPSComputeError(eps,i,EPS_ERROR_RELATIVE,&error);CHKERRQ(ierr);

            #if defined(PETSC_USE_COMPLEX)
                re = PetscRealPart(kr);
                im = PetscImaginaryPart(kr);
            #else
                re = kr;
                im = ki;
            #endif

            if (im!=0.0) {
                ierr = PetscPrintf(PETSC_COMM_WORLD," %9f%+9fi %12g\n",(double)re,(double)im,(double)error);CHKERRQ(ierr);
            } else {
                ierr = PetscPrintf(PETSC_COMM_WORLD,"   %12f       %12g\n",(double)re,(double)error);CHKERRQ(ierr);
            }
        }
        ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRQ(ierr);
    }



    MatDestroy(&eye1);
    MatDestroy(&Sz1);
    MatDestroy(&Sp1);
    MatDestroy(&Sm1);
    MatDestroy(&H_temp);
    MatDestroy(&block_Sm);

    block.destroy();

    SlepcFinalize();
    return ierr;
}
