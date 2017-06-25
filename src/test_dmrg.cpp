static char help[] =
    "Test Code for DMRG\n";

#include <iostream>
#include "dmrg.hpp"

/*
    The DMRG Class is inherited by a class that specifies the Hamiltonian
*/
class iDMRG_Heisenberg: public iDMRG
{

public:

    /*
        Overload base class implementation
        with the Heisenberg Hamiltonian
    */
    PetscErrorCode BuildBlockLeft();
    PetscErrorCode BuildBlockRight();
    PetscErrorCode BuildSuperBlock();

};


/* Implementation of the Heisenberg Hamiltonian */

PetscErrorCode iDMRG_Heisenberg::BuildBlockLeft()
{
    PetscErrorCode  ierr = 0;

    /*
        Prepare Sm as explicit Hermitian conjugate of Sp
        TODO: Implement as part of Kronecker product
    */
    Mat BlockLeft_Sm;
    ierr = MatAssemblyBegin(BlockLeft_.Sp(), MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(BlockLeft_.Sp(), MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatTranspose(BlockLeft_.Sp(), MAT_INITIAL_MATRIX, &BlockLeft_Sm); CHKERRQ(ierr);
    ierr = MatConjugate(BlockLeft_Sm); CHKERRQ(ierr);

    /*
        Update the Hamiltonian
    */
    Mat Mat_temp;
    ierr = MatKron(BlockLeft_.H(), eye1_, Mat_temp, comm_); CHKERRQ(ierr);
    ierr = MatKronAdd(BlockLeft_.Sz(), Sz1_, Mat_temp, comm_); CHKERRQ(ierr);
    ierr = MatKronScaleAdd(0.5, BlockLeft_.Sp(), Sm1_, Mat_temp, comm_); CHKERRQ(ierr);
    ierr = MatKronScaleAdd(0.5, BlockLeft_Sm, Sp1_, Mat_temp, comm_); CHKERRQ(ierr);
    ierr = BlockLeft_.update_H(Mat_temp); /* H_temp is destroyed here */ CHKERRQ(ierr);
    ierr = MatDestroy(&BlockLeft_Sm); CHKERRQ(ierr);
    Mat_temp = NULL;

    /*
        Update the Sz operator
    */
    ierr = MatKron(eye1_, BlockLeft_.Sz(), Mat_temp, comm_); CHKERRQ(ierr);
    ierr = BlockLeft_.update_Sz(Mat_temp); CHKERRQ(ierr);
    Mat_temp = NULL;

    /*
        Update the Sp operator
    */
    ierr = MatKron(eye1_, BlockLeft_.Sp(), Mat_temp, comm_); CHKERRQ(ierr);
    ierr = BlockLeft_.update_Sp(Mat_temp); CHKERRQ(ierr);
    Mat_temp = NULL;

    BlockLeft_.length(BlockLeft_.length() + 1);

    ierr = MatDestroy(&BlockLeft_Sm); CHKERRQ(ierr);

    return ierr;
}


PetscErrorCode iDMRG_Heisenberg::BuildBlockRight()
{
    PetscErrorCode  ierr = 0;

    /*
        Prepare Sm as explicit Hermitian conjugate of Sp
        TODO: Implement as part of Kronecker product
    */
    Mat BlockRight_Sm;
    ierr = MatAssemblyBegin(BlockRight_.Sp(), MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(BlockRight_.Sp(), MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatTranspose(BlockRight_.Sp(), MAT_INITIAL_MATRIX, &BlockRight_Sm); CHKERRQ(ierr);
    ierr = MatConjugate(BlockRight_Sm); CHKERRQ(ierr);

    /*
        Update the Hamiltonian
    */
    Mat Mat_temp;
    ierr = MatKron(eye1_, BlockRight_.H(), Mat_temp, comm_); CHKERRQ(ierr);
    ierr = MatKronAdd(Sz1_, BlockRight_.Sz(), Mat_temp, comm_); CHKERRQ(ierr);
    ierr = MatKronScaleAdd(0.5, Sm1_, BlockRight_.Sp(), Mat_temp, comm_); CHKERRQ(ierr);
    ierr = MatKronScaleAdd(0.5, Sp1_, BlockRight_Sm, Mat_temp, comm_); CHKERRQ(ierr);
    ierr = BlockRight_.update_H(Mat_temp); /* H_temp is destroyed here */ CHKERRQ(ierr);
    ierr = MatDestroy(&BlockRight_Sm); CHKERRQ(ierr);

    /*
        Update the Sz operator
    */
    ierr = MatKron(BlockRight_.Sz(), eye1_, Mat_temp, comm_); CHKERRQ(ierr);
    ierr = BlockRight_.update_Sz(Mat_temp); CHKERRQ(ierr);

    /*
        Update the Sp operator
    */
    ierr = MatKron(BlockRight_.Sp(), eye1_, Mat_temp, comm_); CHKERRQ(ierr);
    ierr = BlockRight_.update_Sp(Mat_temp); CHKERRQ(ierr);

    BlockRight_.length(BlockRight_.length() + 1);

    ierr = MatDestroy(&BlockRight_Sm); CHKERRQ(ierr);

    return ierr;
}



PetscErrorCode iDMRG_Heisenberg::BuildSuperBlock()
{
    PetscErrorCode  ierr = 0;
    Mat             mat_temp;
    PetscInt        M_left, M_right;

    /*
        TODO: Impose a checkpoint correctness of blocks
    */

    /*
        Update the Hamiltonian

        First term:  H_{L,i+1} \otimes 1_{DR×2}    ???? DRx2 ????

        Prepare mat_temp = Identity corresponding to right block
    */
    ierr = MatGetSize(BlockRight_.H(), &M_right, NULL); CHKERRQ(ierr);
    ierr = MatEyeCreate(comm_, mat_temp, M_right); CHKERRQ(ierr);
    ierr = MatKron(BlockLeft_.H(), mat_temp, superblock_H_, comm_); CHKERRQ(ierr);

    /*
        If the left and right sizes are the same, re-use the identity.
        Otherwise, create a new identity matrix with the correct size.
    */
    ierr = MatGetSize(BlockLeft_.H(), &M_left, NULL); CHKERRQ(ierr);
    if(M_left != M_right){
        ierr = MatDestroy(&mat_temp); CHKERRQ(ierr);
        ierr = MatEyeCreate(comm_, mat_temp, M_left); CHKERRQ(ierr);
    }

    /*
        Second term: 1_{DL×2} \otimes H_{R,i+2}
    */
    ierr = MatKronAdd(mat_temp, BlockRight_.H(), superblock_H_, comm_); CHKERRQ(ierr);

    /*
        Third term: S^z_{L,i+1} \otimes S^z_{R,i+2}
    */
    ierr = MatKronAdd(BlockLeft_.Sz(), BlockRight_.Sz(), superblock_H_, comm_); CHKERRQ(ierr);

    /*
        Fourth term: 1/2 S^+_{L,i+1} \otimes S^-_{R,i+2}

        Prepare mat_temp = BlockRight_.Sm
    */
    ierr = MatDestroy(&mat_temp); CHKERRQ(ierr);
    ierr = MatAssemblyBegin(BlockRight_.Sp(), MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(BlockRight_.Sp(), MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatTranspose(BlockRight_.Sp(), MAT_INITIAL_MATRIX, &mat_temp); CHKERRQ(ierr);
    ierr = MatConjugate(mat_temp); CHKERRQ(ierr);
    ierr = MatKronScaleAdd(0.5, BlockLeft_.Sp(), mat_temp, superblock_H_, comm_); CHKERRQ(ierr);

    /*
        Fifth term: 1/2 S^-_{L,i+1} \otimes S^+_{R,i+2}

        Prepare mat_temp = BlockLeft_.Sm
    */
    ierr = MatDestroy(&mat_temp); CHKERRQ(ierr);
    ierr = MatAssemblyBegin(BlockLeft_.Sp(), MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(BlockLeft_.Sp(), MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatTranspose(BlockLeft_.Sp(), MAT_INITIAL_MATRIX, &mat_temp); CHKERRQ(ierr);
    ierr = MatConjugate(mat_temp); CHKERRQ(ierr);
    ierr = MatKronScaleAdd(0.5, mat_temp, BlockRight_.Sp(), superblock_H_, comm_); CHKERRQ(ierr);

    ierr = MatDestroy(&mat_temp); CHKERRQ(ierr);

    superblock_set_ = PETSC_TRUE;

    return ierr;
}


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

    iDMRG_Heisenberg heis;
    heis.init(comm);

    ierr = PetscPrintf(PETSC_COMM_WORLD,
            "   gs energy /site    ||Ax-kx||/||kx||\n"
            "   ----------------- ------------------\n");CHKERRQ(ierr);

    PetscReal gse_r, gse_i, error;

    PetscInt superblocklength;
    for (PetscInt i = 0; i < 4; ++i)
    {
        heis.BuildBlockRight();
        heis.BuildBlockLeft();
        heis.BuildSuperBlock();
        heis.SolveGroundState(gse_r, gse_i, error);

        superblocklength = heis.LengthBlockLeft() + heis.LengthBlockRight();

        if (gse_i!=0.0) {
            ierr = PetscPrintf(PETSC_COMM_WORLD," %9f%+9fi %12g\n",(double)gse_r,(double)gse_i,(double)error);CHKERRQ(ierr);
        } else {
            ierr = PetscPrintf(PETSC_COMM_WORLD,"   %12f       %12g\n", (double)gse_r/((double)(superblocklength)),(double)(error)); CHKERRQ(ierr);
        }

    }

    heis.MatSaveOperators();
    // heis.MatPeekOperators();
    heis.destroy();





    // #define __TEST_01__
    #ifdef __TEST_01__

    DMRGBlock block;
    block.init();
    /*
        Build the Hamiltonian for a two-spin system
        B(L,1) + S(1)
    */
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
    #endif // __TEST_01__
    #undef __TEST_01__

    SlepcFinalize();
    return ierr;
}
