#ifndef __KRON_HPP__
#define __KRON_HPP__

/** @defgroup Kronecker
 *  @brief Implementation of the kronecker product with distributed sparse matrices in PETSc and SLEPc (v3.7)
 */

#include <slepceps.h>
#include <stdlib.h>

/**
    In this implementation of the kronecker product $ C = A \otimes B $ we
    require that each process have a local copy of the parallel matrix B. This
    is achieved by getting the submatrices of A and C

    TODO:
        * Implement MatKronAdd $ C += A \otimes B $
        * Implement MatKronScaleAdd $ C += a * A \otimes B $
        * Implement overloading for when one or more arguments is the identity matrix
        * Implement overloading for when one or more arguments is the Sz, Sp, or Sm
        * Reduce communication.
        * Write a test suite to check whether this works for small and
            large matrices
        * Test for the case when A is small and B is large.
        * Check if this works for sparse x dense (m~10000s)
        * Check if it works well with other PetscScalar datatypes (complex/real)
 */


PetscErrorCode MatKronScaleAdd(const PetscScalar a, const Mat& A, const Mat& B, Mat& C, const MPI_Comm& comm);


PetscErrorCode MatKronAdd(const Mat& A, const Mat& B, Mat& C, const MPI_Comm& comm);


PetscErrorCode MatKron(const Mat& A, const Mat& B, Mat& C, const MPI_Comm& comm);


#endif // __KRON_HPP__
