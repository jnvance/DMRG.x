#include "dmrgblock.hpp"


// PetscErrorCode DMRGBlock::init( MPI_Comm comm = DMRG_DEFAULT_MPI_COMM,
//                                 PetscInt length = DMRGBLOCK_DEFAULT_LENGTH,
//                                 PetscInt basis_size = DMRGBLOCK_DEFAULT_BASIS_SIZE)
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


PetscErrorCode DMRGBlock::destroy()
{
    PetscErrorCode  ierr = 0;

    // All matrices created in init() must be destroyed here
    ierr = MatDestroy(&H_); CHKERRQ(ierr);
    ierr = MatDestroy(&Sz_); CHKERRQ(ierr);
    ierr = MatDestroy(&Sp_); CHKERRQ(ierr);

    length_ = 0;
    basis_size_ = 0;

    H_ = NULL;
    Sz_ = NULL;
    Sp_ = NULL;

    return ierr;
}

PetscErrorCode DMRGBlock::update_operators(Mat H_new, Mat Sz_new, Mat Sp_new)
{
    PetscErrorCode  ierr = 0;
    ierr = update_H(H_new); CHKERRQ(ierr);
    ierr = update_Sz(Sz_new); CHKERRQ(ierr);
    ierr = update_Sp(Sp_new); CHKERRQ(ierr);
    return ierr;
}

#define UPDATE_OPERATOR(MATRIX) \
PetscErrorCode DMRGBlock::update_ ## MATRIX(Mat MATRIX ## _new)         \
{                                                                       \
    PetscErrorCode  ierr = 0;                                           \
    if (MATRIX ## _ == MATRIX ## _new) /* Check whether same matrix */  \
        return ierr;                                                    \
    Mat MATRIX ## _temp = MATRIX ## _;                                  \
    MATRIX ## _ = MATRIX ## _new;                                       \
    ierr = MatDestroy(&MATRIX ## _temp); CHKERRQ(ierr);                 \
    return ierr;                                                        \
}

UPDATE_OPERATOR(H)
UPDATE_OPERATOR(Sz)
UPDATE_OPERATOR(Sp)

#undef UPDATE_OPERATOR
