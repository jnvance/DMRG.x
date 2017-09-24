#include "dmrgblock.hpp"


#undef __FUNCT__
#define __FUNCT__ "DMRGBlock::init"
PetscErrorCode DMRGBlock::init( MPI_Comm comm, PetscInt length, PetscInt basis_size)
{
    PetscErrorCode  ierr = 0;
    comm_ = comm;
    length_ = length;
    basis_size_ = basis_size;

    PetscInt sqmatrixdim = pow(basis_size_,length_);

    /*
        initialize the matrices
    */
    #define INIT_AND_ZERO(mat) \
        ierr = MatCreate(comm_, &mat); CHKERRQ(ierr); \
        ierr = MatSetSizes(mat, PETSC_DECIDE, PETSC_DECIDE, sqmatrixdim, sqmatrixdim); CHKERRQ(ierr); \
        ierr = MatSetFromOptions(mat); CHKERRQ(ierr); \
        ierr = MatSetUp(mat); CHKERRQ(ierr); \
        ierr = MatZeroEntries(mat); CHKERRQ(ierr);

        INIT_AND_ZERO(H_)
        INIT_AND_ZERO(Sz_)
        INIT_AND_ZERO(Sp_)
    #undef INIT_AND_ZERO

    /*
        Operators are constructed explicitly in this section
        For the simple infinite-system DMRG, the calculations begin with a
        block of 2x2 matrices explictly constructed in 1-2 processor implementations
    */

    /*
        fill the operator values
        matrix assembly assumes block length = 1, basis_size = 2
        TODO: generalize!
    */
    if(!(length_==1 && basis_size==2)) SETERRQ(comm,1,"Matrix assembly assumes block length = 1, basis_size = 2\n");
    ierr = MatSetValue(Sz_, 0, 0, +0.5, INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatSetValue(Sz_, 1, 1, -0.5, INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatSetValue(Sp_, 0, 1, +1.0, INSERT_VALUES); CHKERRQ(ierr);

    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "DMRGBlock::destroy"
PetscErrorCode DMRGBlock::destroy()
{
    PetscErrorCode  ierr = 0;

    /* All matrices created in init() must be destroyed here */
    if(H_) ierr = MatDestroy(&H_); CHKERRQ(ierr);
    if(Sz_) ierr = MatDestroy(&Sz_); CHKERRQ(ierr);
    if(Sp_) ierr = MatDestroy(&Sp_); CHKERRQ(ierr);

    length_ = 0;
    basis_size_ = 0;

    H_ = NULL;
    Sz_ = NULL;
    Sp_ = NULL;

    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "DMRGBlock::update_operators"
PetscErrorCode DMRGBlock::update_operators(Mat& H_new, Mat& Sz_new, Mat& Sp_new)
{
    PetscErrorCode  ierr = 0;
    ierr = update_H(H_new); CHKERRQ(ierr);
    ierr = update_Sz(Sz_new); CHKERRQ(ierr);
    ierr = update_Sp(Sp_new); CHKERRQ(ierr);
    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "DMRGBlock::update_H"
PetscErrorCode DMRGBlock::update_H(Mat& H_new)
{
    PetscErrorCode  ierr = 0;
    if (H_ == H_new)
        return ierr;
    Mat H_temp = H_;
    H_ = H_new;
    H_new = nullptr;
    ierr = MatDestroy(&H_temp); CHKERRQ(ierr);
    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "DMRGBlock::update_Sz"
PetscErrorCode DMRGBlock::update_Sz(Mat& Sz_new)
{
    PetscErrorCode  ierr = 0;
    if (Sz_ == Sz_new)
        return ierr;
    Mat Sz_temp = Sz_;
    Sz_ = Sz_new;
    Sz_new = nullptr;
    ierr = MatDestroy(&Sz_temp); CHKERRQ(ierr);
    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "DMRGBlock::update_Sp"
PetscErrorCode DMRGBlock::update_Sp(Mat& Sp_new)
{
    PetscErrorCode  ierr = 0;
    if (Sp_ == Sp_new)
        return ierr;
    Mat Sp_temp = Sp_;
    Sp_ = Sp_new;
    Sp_new = nullptr;
    ierr = MatDestroy(&Sp_temp); CHKERRQ(ierr);
    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "DMRGBlock::is_valid"
PetscBool DMRGBlock::is_valid()
{
    PetscInt size1, size2, size3;

    MatGetSize(H_, &size1, &size2);
    if(size1 != size2) return PETSC_FALSE;

    MatGetSize(Sz_, &size2, &size3);
    if(size1 != size2) return PETSC_FALSE;
    if(size2 != size3) return PETSC_FALSE;

    MatGetSize(Sp_, &size2, &size3);
    if(size1 != size2) return PETSC_FALSE;
    if(size2 != size3) return PETSC_FALSE;

    /* TODO: put this back later on */
    // if (basis_sector_array.size() != size1) return PETSC_FALSE;

    basis_size_ = size1;

    return PETSC_TRUE;
}





