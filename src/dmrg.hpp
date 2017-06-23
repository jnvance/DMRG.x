#ifndef __DMRG_HPP__
#define __DMRG_HPP__

/** @defgroup DMRG

    @brief Implementation of the DMRGBlock and iDMRG classes

 */


#include <slepceps.h>
#include <petscmat.h>
#include "kron.hpp"
#include "matrix_tools.hpp"


#define     DMRGBLOCK_DEFAULT_LENGTH        1
#define     DMRGBLOCK_DEFAULT_BASIS_SIZE    2
#define     DMRG_DEFAULT_MPI_COMM           PETSC_COMM_WORLD

/** @defgroup DMRG

 */
class DMRGBlock
{
    Mat             H_;     /* Hamiltonian of entire block */
    Mat             Sz_;    /* Operators of rightmost/leftmost spin */
    Mat             Sp_;

    PetscInt        length_;
    PetscInt        basis_size_;

    MPI_Comm        comm_;

public:

    DMRGBlock() :   length_(DMRGBLOCK_DEFAULT_LENGTH),
                    basis_size_(DMRGBLOCK_DEFAULT_BASIS_SIZE),
                    comm_(DMRG_DEFAULT_MPI_COMM)
                    {}  // constructor with spin 1/2 defaults

    ~DMRGBlock(){}

    PetscErrorCode  init(MPI_Comm, PetscInt, PetscInt);     /* explicit initializer */
    PetscErrorCode  destroy();                              /* explicit destructor */

    const Mat  H()        {return H_;}
    const Mat  Sz()       {return Sz_;}
    const Mat  Sp()       {return Sp_;}

    PetscErrorCode  update_operators(Mat H_new, Mat Sz_new, Mat Sp_new);

    PetscErrorCode  update_H(Mat H_new);
    PetscErrorCode  update_Sz(Mat Sz_new);
    PetscErrorCode  update_Sp(Mat Sp_new);


    PetscInt        length(){return length_;}
    PetscInt        basis_size(){return basis_size_;}

    void            length(PetscInt _length){length_ = _length;}

};


PetscErrorCode DMRGBlock::init( MPI_Comm comm = DMRG_DEFAULT_MPI_COMM,
                                PetscInt length = DMRGBLOCK_DEFAULT_LENGTH,
                                PetscInt basis_size = DMRGBLOCK_DEFAULT_BASIS_SIZE)
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


/*--------------------------------------------------------------------------------*/


class iDMRG
{

protected:

    DMRGBlock BlockLeft_;
    DMRGBlock BlockRight_;

    Mat       superblock_H_ = NULL;

    MPI_Comm    comm_;

    /* Matrices of the single-site operators */
    Mat eye1_, Sz1_, Sp1_, Sm1_;

public:

    PetscErrorCode init(MPI_Comm);
    PetscErrorCode destroy();

    /* Block enlargement to be implemented in inherited classes */
    virtual PetscErrorCode BuildBlockLeft(){
        SETERRQ(comm_, 1, "BuildBlockLeft() is not implemented in the base class.\n");
    }

    virtual PetscErrorCode BuildBlockRight(){
        SETERRQ(comm_, 1, "BuildBlockRight() is not implemented in the base class.\n");
    }

    virtual PetscErrorCode BuildSuperBlock(){
        SETERRQ(comm_, 1, "BuildSuperBlock() is not implemented in the base class.\n");
    }

    /* Miscellaneous functions */
    PetscErrorCode MatPeekOperators();

};


PetscErrorCode iDMRG::init(MPI_Comm comm=DMRG_DEFAULT_MPI_COMM)
{
    PetscErrorCode  ierr = 0;
    comm_ = comm;

    /* Initialize block objects */
    ierr = BlockLeft_.init(comm_); CHKERRQ(ierr);
    ierr = BlockRight_.init(comm_); CHKERRQ(ierr);

    /* Initialize single-site operators */
    MatEyeCreate(comm, eye1_, 2);
    MatSzCreate(comm, Sz1_);
    MatSpCreate(comm, Sp1_);
    MatTranspose(Sp1_, MAT_INITIAL_MATRIX, &Sm1_);

    return ierr;
}


PetscErrorCode iDMRG::destroy()
{
    PetscErrorCode  ierr = 0;

    /* Destroy block objects */
    ierr = BlockLeft_.destroy(); CHKERRQ(ierr);
    ierr = BlockRight_.destroy(); CHKERRQ(ierr);

    /* Destroy single-site operators */
    MatDestroy(&eye1_);
    MatDestroy(&Sz1_);
    MatDestroy(&Sp1_);
    MatDestroy(&Sm1_);
    MatDestroy(&superblock_H_); /* Do a check whether matrix is in the correct state */

    eye1_ = NULL;
    Sz1_ = NULL;
    Sp1_ = NULL;
    Sm1_ = NULL;
    superblock_H_ = NULL;

    return ierr;
}


PetscErrorCode iDMRG::MatPeekOperators()
{
    PetscErrorCode  ierr = 0;

    PetscPrintf(comm_, "\nLeft Block Operators\nBlock Length = %d\n", BlockLeft_.length());
    ierr = MatPeek(comm_, BlockLeft_.H(), "H (left)");
    ierr = MatPeek(comm_, BlockLeft_.Sz(), "Sz (left)");
    ierr = MatPeek(comm_, BlockLeft_.Sp(), "Sp (left)");

    PetscPrintf(comm_, "\nRight Block Operators\nBlock Length = %d\n", BlockRight_.length());
    ierr = MatPeek(comm_, BlockRight_.H(), "H (right)");
    ierr = MatPeek(comm_, BlockRight_.Sz(), "Sz (right)");
    ierr = MatPeek(comm_, BlockRight_.Sp(), "Sp (right)");

    if (superblock_H_){
        PetscPrintf(comm_, "\nSuperblock\nBlock Length = %d\n", BlockLeft_.length() + BlockRight_.length());
        ierr = MatPeek(comm_, superblock_H_, "H (superblock)"); CHKERRQ(ierr);
    }

    return ierr;
}


#endif // __DMRG_HPP__
