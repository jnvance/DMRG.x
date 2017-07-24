static char help[] =
    "Test Code for DMRG\n";

#include <iostream>
#include "dmrg.hpp"

/*
    The DMRG Class is inherited by a class that specifies the Hamiltonian
*/
class iDMRG_Heisenberg: public iDMRG
{
    // PetscInt local_dim_ = 2;

public:

    /*
        Overload base class implementation
        with the Heisenberg Hamiltonian
    */
    PetscErrorCode BuildBlockLeft() final;
    PetscErrorCode BuildBlockRight() final;
    PetscErrorCode BuildSuperBlock() final;

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

    if(!BlockLeft_.is_valid()) SETERRQ(comm_, 1, "Invalid left block");
    #ifdef __TESTING
        PetscPrintf(comm_, "Left       basis size: %-5d nsites: %-5d \n", BlockLeft_.basis_size(), BlockLeft_.length());
    #endif
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
    if(!BlockRight_.is_valid()) SETERRQ(comm_, 1, "Invalid right block");
    #ifdef __TESTING
        PetscPrintf(comm_, "Right      basis size: %-5d nsites: %-5d \n", BlockRight_.basis_size(), BlockRight_.length());
    #endif
    ierr = MatDestroy(&BlockRight_Sm); CHKERRQ(ierr);

    return ierr;
}



PetscErrorCode iDMRG_Heisenberg::BuildSuperBlock()
{
    PetscErrorCode  ierr = 0;
    Mat             mat_temp;
    PetscInt        M_left, M_right, M_superblock;

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

    ierr = MatGetSize(superblock_H_, &M_superblock, nullptr); CHKERRQ(ierr);
    if(M_superblock != TotalBasisSize()) SETERRQ(comm_, 1, "Basis size mismatch.\n");
    #ifdef __TESTING
        PetscPrintf(comm_, "Superblock basis size: %-5d nsites: %-5d \n", M_superblock, BlockLeft_.length() + BlockRight_.length());
    #endif


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
            "   nsites   gs energy   gs energy /site    rel error      ||Ax-kx||/||kx||\n"
            "  -------- ----------- -----------------  -----------     ------------------\n");CHKERRQ(ierr);

    PetscReal gse_r, gse_i, error;

    double gse_site_theor =  -0.4431471805599;


    PetscInt iter = 0;
    while(heis.TotalLength() < heis.TargetLength() && iter < heis.TargetLength()){
        heis.BuildBlockLeft();
        heis.BuildBlockRight();

        heis.BuildSuperBlock();
        heis.SolveGroundState(gse_r, gse_i, error);

        if (heis.TotalBasisSize() >= heis.mstates()*heis.mstates())
        {
            // heis.BuildSuperBlock();
            // heis.SolveGroundState(gse_r, gse_i, error);


            PetscInt superblocklength = heis.LengthBlockLeft() + heis.LengthBlockRight();

            if (gse_i!=0.0) {
                // TODO: Implement error printing for complex values
                ierr = PetscPrintf(PETSC_COMM_WORLD," %6d    %9f%+9fi %12g\n", superblocklength, (double)gse_r/((double)(superblocklength)), (double)gse_i/((double)(superblocklength)),(double)error);CHKERRQ(ierr);
            } else {
                double gse_site  = (double)gse_r/((double)(superblocklength));
                double error_rel = (gse_site - gse_site_theor) / gse_site_theor;
                ierr = PetscPrintf(PETSC_COMM_WORLD,"   %6d%12f    %12f       %9f    %12g\n", superblocklength, (double)gse_r, gse_site,  error_rel, (double)(error)); CHKERRQ(ierr);
            }

            heis.BuildReducedDensityMatrices();
            heis.GetRotationMatrices();
            heis.TruncateOperators();
        }


        PetscPrintf(comm, "Total sites: %d\n", heis.TotalLength());
        ++iter;
    }


    #ifdef __TRASH
    for (PetscInt i = 0; i < n_pre; ++i){
        heis.BuildBlockLeft();
        heis.BuildBlockRight();
    }

    PetscInt superblocklength;
    for (PetscInt i = 0; i < n_solve; ++i)
    {
        heis.BuildBlockRight();
        heis.BuildBlockLeft();
        heis.BuildSuperBlock();
        // heis.MatSaveOperators();
        heis.SolveGroundState(gse_r, gse_i, error);

        superblocklength = heis.LengthBlockLeft() + heis.LengthBlockRight();

        if (gse_i!=0.0) {
            // TODO: Implement error printing for complex values
            ierr = PetscPrintf(PETSC_COMM_WORLD," %6d    %9f%+9fi %12g\n", superblocklength, (double)gse_r/((double)(superblocklength)), (double)gse_i/((double)(superblocklength)),(double)error);CHKERRQ(ierr);
        } else {
            double gse_site  = (double)gse_r/((double)(superblocklength));
            double error_rel = (gse_site - gse_site_theor) / gse_site_theor;
            ierr = PetscPrintf(PETSC_COMM_WORLD,"   %6d%12f    %12f       %9f    %12g\n", superblocklength, (double)gse_r, gse_site,  error_rel, (double)(error)); CHKERRQ(ierr);
        }

        heis.BuildReducedDensityMatrices();
        // heis.SVDReducedDensityMatrices();
        heis.GetRotationMatrices();
        heis.TruncateOperators();

    }
    #endif

    // heis.MatSaveOperators();
    // heis.MatPeekOperators();
    heis.destroy();

    SlepcFinalize();
    return ierr;
}
