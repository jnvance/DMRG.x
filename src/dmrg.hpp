#ifndef __DMRG_HPP__
#define __DMRG_HPP__

/** @defgroup DMRG

    @brief Implementation of the DMRGBlock and iDMRG classes

 */


#include <slepceps.h>
#include <petscmat.h>
#include "kron.hpp"
#include "linalg_tools.hpp"


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

    Mat  H()        {return H_;}
    Mat  Sz()       {return Sz_;}
    Mat  Sp()       {return Sp_;}

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

    PetscInt    final_nsites_;
    PetscInt    nsteps_;
    PetscInt    iter_;

    DMRGBlock   BlockLeft_;
    DMRGBlock   BlockRight_;

    Mat         superblock_H_ = NULL;
    PetscBool   superblock_set_ = PETSC_FALSE;

    /* Ground state */
    PetscScalar *gse_r_list_, *gse_i_list_;
    Vec         gsv_r_, gsv_i_;
    PetscBool   groundstate_solved_ = PETSC_FALSE;

    Mat         dm_left;
    Mat         dm_right;

    MPI_Comm    comm_;

    /* Matrices of the single-site operators */
    Mat eye1_, Sz1_, Sp1_, Sm1_;

public:

    PetscErrorCode init(MPI_Comm);
    PetscErrorCode destroy();

    PetscInt LengthBlockLeft(){ return BlockLeft_.length(); }
    PetscInt LengthBlockRight(){ return BlockRight_.length(); }

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

    /* Solve states */
    PetscErrorCode SolveGroundState(PetscReal& gse_r, PetscReal& gse_i, PetscReal& error);

    /* Miscellaneous functions */
    PetscErrorCode MatPeekOperators();
    PetscErrorCode MatSaveOperators();

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


PetscErrorCode iDMRG::SolveGroundState(PetscReal& gse_r, PetscReal& gse_i, PetscReal& error)
{
    PetscErrorCode ierr = 0;

    /*
        Checkpoint whether superblock Hamiltonian has been set and assembled
    */
    if (superblock_set_ == PETSC_FALSE)
        SETERRQ(comm_, 1, "Superblock Hamiltonian has not been set with BuildSuperBlock().");

    PetscBool assembled;
    ierr = MatAssembled(superblock_H_, &assembled); CHKERRQ(ierr);
    if (assembled == PETSC_FALSE){
        ierr = MatAssemblyBegin(superblock_H_, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(superblock_H_, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }

    /*
        Solve the eigensystem using SLEPC EPS
    */

    EPS eps;
    ierr = EPSCreate(comm_, &eps); CHKERRQ(ierr);
    ierr = EPSSetOperators(eps, superblock_H_, NULL); CHKERRQ(ierr);
    ierr = EPSSetProblemType(eps, EPS_HEP); CHKERRQ(ierr);
    ierr = EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL);
    ierr = EPSSetDimensions(eps, 1, PETSC_DECIDE, PETSC_DECIDE);

    ierr = EPSSetFromOptions(eps); CHKERRQ(ierr);
    ierr = EPSSolve(eps); CHKERRQ(ierr);

    PetscInt nconv;
    ierr = EPSGetConverged(eps,&nconv);CHKERRQ(ierr);

    ierr = MatCreateVecs(superblock_H_,NULL,&gsv_r_); CHKERRQ(ierr);

    /* TODO: Verify that this works */
    #if defined(PETSC_USE_COMPLEX)
        gsv_i_ = NULL;
    #else
        ierr = MatCreateVecs(superblock_H_,NULL,&gsv_i_); CHKERRQ(ierr);
    #endif

    PetscScalar kr, ki;

    if (nconv>0)
    {
        /*
            Get converged eigenpairs: 0-th eigenvalue is stored in gse_r (real part) and
            gse_i (imaginary part)

            Note on EPSGetEigenpair():

            If the eigenvalue is real, then eigi and Vi are set to zero. If PETSc is configured
            with complex scalars the eigenvalue is stored directly in eigr (eigi is set to zero)
            and the eigenvector in Vr (Vi is set to zero).
        */

        #if defined(PETSC_USE_COMPLEX)
            ierr = EPSGetEigenpair(eps, 0, &kr, &ki, gsv_r_, NULL); CHKERRQ(ierr);
            gse_r = PetscRealPart(kr);
            gse_i = PetscImaginaryPart(kr);
        #else
            ierr = EPSGetEigenpair(eps, 0, &kr, &ki, gsv_r_, gsv_i_); CHKERRQ(ierr);
            gse_r = kr;
            gse_i = ki;
        #endif

        ierr = EPSComputeError(eps, 0, EPS_ERROR_RELATIVE, &error);CHKERRQ(ierr);
        groundstate_solved_ = PETSC_TRUE;

    }
    else
    {
        PetscPrintf(PETSC_COMM_WORLD,"Warning: EPS did not converge.");
    }

    superblock_set_ = PETSC_FALSE;
    MatDestroy(&superblock_H_);
    EPSDestroy(&eps);

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

    if (superblock_H_ && (superblock_set_ == PETSC_TRUE)){
        PetscPrintf(comm_, "\nSuperblock\nBlock Length = %d\n", BlockLeft_.length() + BlockRight_.length());
        ierr = MatPeek(comm_, superblock_H_, "H (superblock)"); CHKERRQ(ierr);
    }

    return ierr;
}


PetscErrorCode iDMRG::MatSaveOperators()
{
    PetscErrorCode  ierr = 0;

    ierr = MatWrite(comm_, BlockLeft_.H(), "data/H_left.dat"); CHKERRQ(ierr);
    ierr = MatWrite(comm_, BlockLeft_.Sz(), "data/Sz_left.dat"); CHKERRQ(ierr);
    ierr = MatWrite(comm_, BlockLeft_.Sp(), "data/Sp_left.dat"); CHKERRQ(ierr);

    ierr = MatWrite(comm_, BlockRight_.H(), "data/H_right.dat"); CHKERRQ(ierr);
    ierr = MatWrite(comm_, BlockRight_.Sz(), "data/Sz_right.dat"); CHKERRQ(ierr);
    ierr = MatWrite(comm_, BlockRight_.Sp(), "data/Sp_right.dat"); CHKERRQ(ierr);

    if (superblock_H_ && (superblock_set_ == PETSC_TRUE)){
        ierr = MatWrite(comm_, superblock_H_, "data/H_superblock.dat"); CHKERRQ(ierr);
    }

    return ierr;
}


#endif // __DMRG_HPP__
