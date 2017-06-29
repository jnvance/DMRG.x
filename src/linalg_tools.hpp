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


/* Reshape m*n vector to m x n array */
PetscErrorCode VecReshapeToMat(const MPI_Comm& comm, const Vec& vec, Mat& mat, const PetscInt M, const PetscInt N);

#endif // __LINALG_TOOLS_HPP__
