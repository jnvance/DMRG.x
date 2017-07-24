#include "idmrg_1d_heisenberg.hpp"

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
