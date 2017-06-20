#ifndef __DMRG_H__
#define __DMRG_H__

/** @defgroup DMRG

    @brief Implementation of the DMRGBlock and DMRGSystem classes

 */

#include <slepceps.h>
#include <petscmat.h>
#include "kron.hpp"

/** @defgroup DMRG

 */
class DMRGBlock
{
    public:

        DMRGBlock(MPI_Comm comm=PETSC_COMM_WORLD, PetscInt length=1, PetscInt basis_size=2) :
            length_(length),
            basis_size_(basis_size),
            comm_(comm)
            {} // constructor with spin 1/2 defaults

        ~DMRGBlock(){}

        PetscErrorCode init(); // explicit initializer
        PetscErrorCode destroy(); // explicit destructor

        PetscInt length(){return length_;}
        PetscInt basis_size(){return basis_size_;}

    private:

        Mat         H_;
        Mat         Sz_;
        Mat         Sp_;

        PetscInt    length_;
        PetscInt    basis_size_;

        MPI_Comm    comm_;
};


// We employ an explicit function for initializing the system
// since we want to catch and check errors through ierr, which produces a return.
// This cannot be done with default constructors since return values are not allowed.

PetscErrorCode DMRGBlock::init()
{
    PetscErrorCode  ierr = 0;

    PetscInt sqmatrixdim = pow(basis_size_,length_);

    // initialize the matrices
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

    // Operators are constructed explicitly in this section
    // For the simple infinite-system DMRG, the calculations begin with a
    // block of 2x2 matrices explictly constructed in 1-2 processor implementations

    // This section is model-specific code for the Heisenberg XXZ chain
    // TODO: Modify and refactor later to be model-independent

    // fill the operator values
    ierr = MatSetValue(Sz_, 0, 0, +0.5, INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatSetValue(Sz_, 1, 1, -0.5, INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatSetValue(Sp_, 0, 1, +1.0, INSERT_VALUES); CHKERRQ(ierr);

    #define __PEEK__
    #ifdef __PEEK__
        // Peek into values
        PetscViewer fd = nullptr;
    #define PEEK(mat) \
        ierr = MatAssemblyBegin(mat, MAT_FLUSH_ASSEMBLY); CHKERRQ(ierr); \
        ierr = MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr); \
        ierr = MatView(mat, fd); CHKERRQ(ierr);

        PEEK(Sz_)
        PEEK(Sp_)
    #undef PEEK
    #endif


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

    return ierr;
}


/*--------------------------------------------------------------------------------*/


class DMRGSystem
{
    public:

    private:

};

#endif // __DMRG_H__
