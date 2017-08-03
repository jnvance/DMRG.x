#ifndef __KRON_HPP__
#define __KRON_HPP__

#include <slepceps.h>
#include <stdlib.h>
#include <petsctime.h>

/* Inspect the timings inside matkron */
#ifdef __KRON_TIMINGS

    #define KRON_TIMINGS_PRINT(SOMETEXT) \
        ierr = PetscPrintf(PETSC_COMM_WORLD, "%s\n",SOMETEXT);

    /* Inspect timings for a full block of code */

    #define KRON_TIMINGS_INIT(SECTION_LABEL) \
        PetscLogDouble funct_time0 ## SECTION_LABEL, funct_time ## SECTION_LABEL;

    #define KRON_TIMINGS_START(SECTION_LABEL) \
        ierr = PetscTime(&funct_time0 ## SECTION_LABEL); CHKERRQ(ierr);

    #define KRON_TIMINGS_END(SECTION_LABEL) \
        ierr = PetscTime(&funct_time ## SECTION_LABEL); CHKERRQ(ierr); \
        funct_time ## SECTION_LABEL = funct_time ## SECTION_LABEL - funct_time0 ## SECTION_LABEL; \
        ierr = PetscPrintf(PETSC_COMM_WORLD, "%8s %-50s %.20g\n", "", SECTION_LABEL, funct_time ## SECTION_LABEL);

    /* Inspect accumulated timings for a section of code inside a loop */

    #define KRON_TIMINGS_ACCUM_INIT(SECTION_LABEL) \
        PetscLogDouble funct_time0 ## SECTION_LABEL, funct_time1 ## SECTION_LABEL, funct_time ## SECTION_LABEL = 0.0;

    #define KRON_TIMINGS_ACCUM_START(SECTION_LABEL) \
        ierr = PetscTime(&funct_time0 ## SECTION_LABEL); CHKERRQ(ierr);

    #define KRON_TIMINGS_ACCUM_END(SECTION_LABEL) \
        ierr = PetscTime(&funct_time1 ## SECTION_LABEL); CHKERRQ(ierr); \
        funct_time ## SECTION_LABEL += funct_time1 ## SECTION_LABEL - funct_time0 ## SECTION_LABEL; \

    #define KRON_TIMINGS_ACCUM_PRINT(SECTION_LABEL) \
        ierr = PetscPrintf(PETSC_COMM_WORLD, "%8s %-50s %.20g\n", "", SECTION_LABEL, funct_time ## SECTION_LABEL);

#else

    #define KRON_TIMINGS_PRINT(SOMETEXT)

    #define KRON_TIMINGS_INIT(SECTION_LABEL)
    #define KRON_TIMINGS_START(SECTION_LABEL)
    #define KRON_TIMINGS_END(SECTION_LABEL)
    #define KRON_TIMINGS_ACCUM_INIT(SECTION_LABEL)
    #define KRON_TIMINGS_ACCUM_START(SECTION_LABEL)
    #define KRON_TIMINGS_ACCUM_END(SECTION_LABEL)
    #define KRON_TIMINGS_ACCUM_PRINT(SECTION_LABEL)

#endif

/**
    @defgroup   kron    Kronecker Product
    @brief      Implementation of the Kronecker product with distributed sparse matrices

    Given an \f$m_A \times n_A\f$ matrix \f$\mathsf{A}\f$ and \f$m_B \times n_B\f$ matrix \f$\mathsf{B}\f$,
    their Kronecker product \f$ \mathsf{C} = \mathsf{A} \otimes \mathsf{B} \f$ is an \f$(m_A m_B) \times (n_A n_B)\f$ matrix with elements
    \f[
         c_{\alpha\beta} = a_{ij} b_{kl}
    \f]
    where
    \f{eqnarray*}{
        \alpha &\equiv& m_B \cdot i + k \\
        \beta  &\equiv& n_B \cdot j + l
    \f}

    In this implementation, we require that each process have a local copy of the parallel matrix B.
    This is achieved by obtaining \f$ \mathsf{B} \f$ on each process using a submatrix with [MatGetSubMatrix](http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/Mat/MatGetSubMatrix.html).

    #### TODO:
        - Implement overloading for when one or more arguments is the identity matrix
        - Implement overloading for when one or more arguments is the Sz, Sp, or Sm
        - Reduce communication.
        - Write a test suite to check whether this works for small and large matrices
        - Test for the case when A is small and B is large.
        - Check if this works for sparse x dense (m~10000s)
        - Check if it works well with other PetscScalar datatypes (complex/real)

    #### Reference:
    Weisstein, Eric W. "Kronecker Product." From MathWorld--A Wolfram Web Resource.
    [http://mathworld.wolfram.com/KroneckerProduct.html](http://mathworld.wolfram.com/KroneckerProduct.html)

 */


/**
    @addtogroup kron
    @{
 */


/**
    Implements the Kronecker product between matrices A and B and puts the result in a new matrix C.

    @param[in]   A      Input matrix
    @param[in]   B      Input matrix
    @param[out]  C      Output matrix
    @param[in]   comm   MPI communicator (usually `PETSC_COMM_WORLD` or `PETSC_COMM_SELF`)

    This function performs
    \f[
        \mathsf{C} = \mathsf{A} \otimes \mathsf{B}.
    \f]
    The resultant matrix C is created inside this function so the input must be uninitialized.
 */
PetscErrorCode MatKron(const Mat& A, const Mat& B, Mat& C, const MPI_Comm& comm);


/**
    Implements the Kronecker product between matrices A and B and adds the result to matrix C.

    @param[in]   A      Input matrix
    @param[in]   B      Input matrix
    @param[out]  C      Output matrix
    @param[in]   comm   MPI communicator (usually `PETSC_COMM_WORLD` or `PETSC_COMM_SELF`)

    This function performs
    \f[
        \mathsf{C} = \mathsf{C} + \mathsf{A} \otimes \mathsf{B}.
    \f]
    The resultant matrix C must already be initialized and have the correct dimensions.
 */
PetscErrorCode MatKronAdd(const Mat& A, const Mat& B, Mat& C, const MPI_Comm& comm);


/**
    Implements the Kronecker product between matrices `A` and `B` multiplied by a scale factor `a`
    and adds the result to matrix `C`.

    @param[in]   a      Scaling factor
    @param[in]   A      Input matrix
    @param[in]   B      Input matrix
    @param[out]  C      Output matrix
    @param[in]   comm   MPI communicator (usually `PETSC_COMM_WORLD` or `PETSC_COMM_SELF`)


    This function performs
    \f[
        \mathsf{C} = \mathsf{C} + a \cdot (\mathsf{A} \otimes \mathsf{B}).
    \f]
    The resultant matrix C must already be initialized and have the correct dimensions.
 */
PetscErrorCode MatKronScaleAdd(const PetscScalar a, const Mat& A, const Mat& B, Mat& C, const MPI_Comm& comm);


/** @} */
#endif // __KRON_HPP__
