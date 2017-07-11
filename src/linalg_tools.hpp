#ifndef __LINALG_TOOLS_HPP__
#define __LINALG_TOOLS_HPP__

// #include <slepceps.h>
#include <slepcsvd.h>

/**
    @defgroup   linalg_tools    Linear Algebra Tools

    @brief      Collection of useful functions for the input, output and manipulation of
                Petsc Vector and Matrix objects.

    In the following functions, the input parameter **comm** is the MPI communicator in which the matrices and vectors reside.
    For distributed matrices this is usually `PETSC_COMM_WORLD`, and for sequential matrices this is `PETSC_COMM_SELF`.

    TODO:
    - For `Mat___Create` operations, if the matrices have been initialized, destroy them.


    @addtogroup linalg_tools
    @{
 */


/**
    Creates an identity matrix of size `dim`\f$\times\f$`dim`.

    @param[in]   comm   the MPI communicator
    @param[out]  eye    the output matrix
    @param[in]   dim    dimensions of the identity matrix

 */
PetscErrorCode MatEyeCreate(const MPI_Comm& comm, Mat& eye, PetscInt dim);


/**
    Creates the two-by-two matrix representation of a single-site \f$ S_z \f$ operator.

    @param[in]   comm   the MPI communicator
    @param[out]  Sz     the output matrix

    The operator has the matrix form
    \f[
        S_z =   \frac{1}{2}
                \begin{pmatrix}
                    1 & 0  \\
                    0 & -1
                \end{pmatrix}
    \f]
 */
PetscErrorCode MatSzCreate(const MPI_Comm& comm, Mat& Sz);


/**
    Creates the two-by-two matrix representation of a single-site \f$ S_+ \f$ operator.

    @param[in]   comm   the MPI communicator
    @param[out]  Sp     the output matrix

    The operator has the matrix form
    \f[
        S_+ =   \begin{pmatrix}
                    0 & 1  \\
                    0 & 0
                \end{pmatrix}
    \f]

    The \f$ S_- \f$ operator may be obtained as the transpose of this matrix.
 */
PetscErrorCode MatSpCreate(const MPI_Comm& comm, Mat& Sp);


/**
    Prints a matrix to standard output

    @param[in]   mat    Input matrix
    @param[in]   label  Label or title of the matrix
 */
PetscErrorCode MatPeek(Mat mat, const char* label);


/**
    Saves a matrix to file

    @param[in]   mat        Input matrix
    @param[in]   filename   filename/location of output file
 */
PetscErrorCode MatWrite(const Mat mat, const char* filename);


/**
    Prints a vector to standard output

    @param[in]   vec    Input vector
    @param[in]   label  Label or title of the vector
 */
PetscErrorCode VecPeek(const Vec& vec, const char* label);


/**
    Saves a vector to file

    @param[in]   vec        Input vector
    @param[in]   filename   filename/location of output file
 */
PetscErrorCode VecWrite(const Vec& vec, const char* filename);


/**
    Reshape an \f$ (M \cdot N) \f$-length vector to an \f$ M \times N \f$ matrix

    @param[in]   vec            Input vector
    @param[out]  mat            Output matrix
    @param[in]   M              number of rows of output matrix
    @param[in]   N              number of columns of output matrix
    @param[in]   mat_is_local   whether to create a local sequential matrix

    Using this function with `mat_is_local = PETSC_TRUE` to create a sequential matrix is inefficient.
    Use VecReshapeToLocalMat() instead.
 */
PetscErrorCode VecReshapeToMat(
    const Vec& vec, Mat& mat,
    const PetscInt M, const PetscInt N, const PetscBool mat_is_local = PETSC_FALSE);


/**
    Reshape an \f$ (M \cdot N) \f$-length distributed vector to an \f$ M \times N \f$ sequential
    matrix with a full copy on each MPI process.
 */
PetscErrorCode VecReshapeToLocalMat(
    const Vec& vec, Mat& mat, const PetscInt M, const PetscInt N);


/**
    Reshapes a vector into a matrix and multiplies the matrix to its own Hermitian conjugate.

    @param[in]   vec_r          Real part of the input vector
    @param[in]   vec_i          Imaginary part of the input vector
    @param[out]  mat            Output matrix
    @param[in]   M              number of rows of output matrix
    @param[in]   N              number of columns of output matrix
    @param[in]   hc_right       whether the Hermitian conjugate is applied to the right matrix

    This function reshapes the \f$ M \cdot N \f$-length vector
    \f$\mathsf{vec} = \mathsf{vec}_r + i\mathsf{vec}_i \f$ into matrix A with shape \f$ M \times N \f$ and calculates
    its Hermitian conjugate \f$\mathsf{A}^\dag = \mathsf{A}^{T*} \f$.

    If `hc_right = PETSC_TRUE`, the output is \f$ \mathsf{mat} = \mathsf{A} * \mathsf{A}^\dag \f$
    with shape \f$ M \times M \f$. Otherwise, mat = A^dag * A with shape \f$ N \times N \f$ (not yet implemented).

    _TODO:_ Implement the case `hc_right = PETSC_FALSE` (_DONE_ using MatMultSelfHC)

    Note: This function is implemented only for complex scalars so that vec_i is ignored.

 */
PetscErrorCode VecToMatMultHC(const Vec& vec_r, const Vec& vec_i,
    Mat& mat, const PetscInt M, const PetscInt N, const PetscBool hc_right);


/**
    Multiplies a local duplicate seqdense matrix to its own Hermitian conjugate and constructs a global result

    @param[in]   mat_in         Input matrix  (must be sequential dense matrix)
    @param[out]  mat            Output matrix (w)
    @param[in]   hc_right       whether the Hermitian conjugate is applied to the right matrix

    This function takes a matrix \f$ \mathsf{A} \f$ with shape \f$ M \times N \f$ and calculates
    its Hermitian conjugate \f$\mathsf{A}^\dag = \mathsf{A}^{T*} \f$.

    If `hc_right = PETSC_TRUE`, the output is \f$ \mathsf{mat} = \mathsf{A} * \mathsf{A}^\dag \f$
    with shape \f$ M \times M \f$. Otherwise, mat = A^dag * A with shape \f$ N \times N \f$ (not yet implemented).

    Note: This function is implemented only for complex scalars so that vec_i is ignored.

 */
PetscErrorCode MatMultSelfHC(const Mat& mat_in, Mat& mat, const PetscBool hc_right);


/**
    Calculates the singular value decomposition of a matrix


 */
PetscErrorCode MatGetSVD(const Mat& mat);
// PetscErrorCode MatGetSVD(const MPI_Comm& comm, const Mat& mat);


/** @} */

#endif // __LINALG_TOOLS_HPP__
