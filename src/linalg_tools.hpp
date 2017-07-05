#ifndef __LINALG_TOOLS_HPP__
#define __LINALG_TOOLS_HPP__

#include <slepceps.h>


PetscErrorCode MatEyeCreate(MPI_Comm comm, Mat& eye, PetscInt dim);


PetscErrorCode MatSzCreate(MPI_Comm comm, Mat& Sz);


PetscErrorCode MatSpCreate(MPI_Comm comm, Mat& Sp);


PetscErrorCode MatPeek(MPI_Comm comm, Mat mat, const char* label);


PetscErrorCode MatWrite(const MPI_Comm comm, const Mat mat, const char* filename);


PetscErrorCode VecWrite(const MPI_Comm& comm, const Vec& vec, const char* filename);


PetscErrorCode VecPeek(const MPI_Comm& comm, const Vec& vec, const char* label);


/**
 * Reshape m*n vector to m x n array
 */
PetscErrorCode VecReshapeToMat(
    const MPI_Comm& comm, const Vec& vec, Mat& mat,
    const PetscInt M, const PetscInt N, const PetscBool mat_is_local);


/**
 * Reshape m*n vector to m x n array and store array locally
 */
PetscErrorCode VecReshapeToLocalMat(
    const MPI_Comm& comm, const Vec& vec, Mat& mat, const PetscInt M, const PetscInt N);


/**
 *  Reshapes a vector into a matrix and multiplies the matrix to its own Hermitian transpose
 *
 *  This function reshapes the vector vec = vec_r + i*vec_i into matrix A with shape (M,N).
 *  With transpose_right = PETSC_TRUE, the output is mat = A * A^dag with shape (M,M), where dag is
 *  the Hermitian conjugate. Otherwise, mat = A^dag * A with shape (N,N)
 *
 *  Note: This function is implemented only for complex scalars so that vec_i is ignored.
 */
PetscErrorCode VecToMatMultHC(const MPI_Comm& comm, const Vec& vec_r, const Vec& vec_i,
    Mat& mat, const PetscInt M, const PetscInt N, const PetscBool hc_right);


#endif // __LINALG_TOOLS_HPP__
