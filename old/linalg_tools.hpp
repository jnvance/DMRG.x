#ifndef __LINALG_TOOLS_HPP__
#define __LINALG_TOOLS_HPP__

#include <slepceps.h>
#include <slepcsvd.h>
#include <vector>
#include <map>
#include <unordered_map>
#include <petsctime.h>
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

/* For back-compatibility with versions that use single precision integers */
#if !defined(LLD)
    #if defined(PETSC_USE_64BIT_INDICES)
        #define LLD(INT) (INT)
    #else
        #define LLD(INT) ((long long)(INT))
    #endif
#endif

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
PetscErrorCode MatPeek(const Mat& mat, const char* label);


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
    Reshape an \f$ (M \cdot N) \f$-length local vector to an \f$ M \times N \f$ sequential
    matrix with a full copy on each MPI process.
 */
PetscErrorCode LocalVecReshapeToLocalMat(
    const Vec& vec_seq, Mat& mat, const PetscInt M, const PetscInt N,
    const std::vector<PetscInt> idx = std::vector<PetscInt>());


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
PetscErrorCode MatMultSelfHC_AIJ(const MPI_Comm comm, const Mat& mat_in, Mat& mat, const PetscBool hc_right);


/**

 */
PetscErrorCode SVDLargestStates(const Mat& mat_in, const PetscInt mstates_in, PetscScalar& error, Mat& mat, FILE *fp);

PetscErrorCode SVDLargestStates_split(const Mat& mat_in, const PetscInt mstates_in, PetscScalar& error, Mat& mat, FILE *fp);

/**

 */
PetscErrorCode MatGetSVD(const Mat& mat_in, SVD& svd, PetscInt& nconv, PetscScalar& error, FILE *fp);


/**
    Takes the first mstates eigenpairs of mat_in with highest eigenvalues and calculates it with eps,
    places them as columns of a matrix, and calculates the truncation error.

    @param[in]  mat_in   The matrix in
    @param[in]  mstates  The mstates
    @param      error    The error
    @param      mat      The matrix

    @return     { description_of_the_return_value }
 */
PetscErrorCode EPSLargestEigenpairs(const Mat& mat_in, const PetscInt mstates, PetscScalar& error, Mat& mat, FILE *fp);


/**
    Mimics the python operation: np.add.outer(A, B).flatten()
 */
std::vector<PetscScalar> OuterSumFlatten(std::vector<PetscScalar> A, std::vector<PetscScalar> B);


/**
    Index map
 */
std::unordered_map<PetscScalar,std::vector<PetscInt>> IndexMap(std::vector<PetscScalar> array);


/**
    Creates an MPI matrix from nested vectors corresponding to the column indices and values of the resulting matrix object
 */
PetscErrorCode MatCreateAIJ_FromSeqList(
    const MPI_Comm comm,
    const std::vector<std::vector<PetscInt>>& cols_list,
    const std::vector<std::vector<PetscScalar>>& vals_list,
    const PetscInt M,
    const PetscInt N,
    Mat*& p_mat_out);

#define LINALG_TOOLS__MATASSEMBLY_INIT() \
    PetscBool assembled;

#define LINALG_TOOLS__MATASSEMBLY_FINAL(MATRIX) \
    {\
        PetscBool assembled;\
        ierr = MatAssembled(MATRIX, &assembled); CHKERRQ(ierr);\
        if (assembled == PETSC_FALSE){\
            ierr = MatAssemblyBegin(MATRIX, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);\
            ierr = MatAssemblyEnd(MATRIX, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);\
        }\
    }

#define LINALG_TOOLS__MATASSEMBLY_FINAL_FORCED(MATRIX) \
    ierr = MatAssemblyBegin(MATRIX, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);\
    ierr = MatAssemblyEnd(MATRIX, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

#define LINALG_TOOLS__MATASSEMBLY_FLUSH(MATRIX) \
    ierr = MatAssembled(MATRIX, &assembled); CHKERRQ(ierr);\
    if (assembled == PETSC_FALSE){\
        ierr = MatAssemblyBegin(MATRIX, MAT_FLUSH_ASSEMBLY); CHKERRQ(ierr);\
        ierr = MatAssemblyEnd(MATRIX, MAT_FLUSH_ASSEMBLY); CHKERRQ(ierr);\
    }

#define LINALG_TOOLS__MATDESTROY(MATRIX) \
    if(MATRIX){ierr = MatDestroy(&MATRIX); CHKERRQ(ierr); MATRIX = nullptr;}
    /* Requires ierr to be defined */

#define LINALG_TOOLS__VECDESTROY(VECTOR) \
    if(VECTOR){ierr = VecDestroy(&VECTOR); CHKERRQ(ierr); VECTOR = nullptr;}
    /* Requires ierr to be defined */

#ifdef __LINALG_TOOLS_TIMINGS

    #include <petsctime.h>

    #define LINALG_TOOLS_TIMINGS_START(SECTION_LABEL) \
        PetscLogDouble funct_time0 ## SECTION_LABEL, funct_time ## SECTION_LABEL; \
        ierr = PetscTime(&funct_time0 ## SECTION_LABEL); CHKERRQ(ierr);

    #define LINALG_TOOLS_TIMINGS_END(SECTION_LABEL) \
        ierr = PetscTime(&funct_time ## SECTION_LABEL); CHKERRQ(ierr); \
        funct_time ## SECTION_LABEL = funct_time ## SECTION_LABEL - funct_time0 ## SECTION_LABEL; \
        ierr = PetscPrintf(PETSC_COMM_WORLD, "%8s %-50s %.20g\n", "", SECTION_LABEL, funct_time ## SECTION_LABEL);

    /* Inspect accumulated timings for a section of code inside a loop */

    #define LINALG_TOOLS_TIMINGS_ACCUM_INIT(SECTION_LABEL) \
        PetscLogDouble funct_time0 ## SECTION_LABEL, funct_time1 ## SECTION_LABEL, funct_time ## SECTION_LABEL = 0.0;

    #define LINALG_TOOLS_TIMINGS_ACCUM_START(SECTION_LABEL) \
        ierr = PetscTime(&funct_time0 ## SECTION_LABEL); CHKERRQ(ierr);

    #define LINALG_TOOLS_TIMINGS_ACCUM_END(SECTION_LABEL) \
        ierr = PetscTime(&funct_time1 ## SECTION_LABEL); CHKERRQ(ierr); \
        funct_time ## SECTION_LABEL += funct_time1 ## SECTION_LABEL - funct_time0 ## SECTION_LABEL; \

    #define LINALG_TOOLS_TIMINGS_ACCUM_PRINT(SECTION_LABEL) \
        ierr = PetscPrintf(PETSC_COMM_WORLD, "%8s %-50s %.20g\n", "", SECTION_LABEL, funct_time ## SECTION_LABEL);

#else
    #define LINALG_TOOLS_TIMINGS_INIT(SECTION_LABEL)
    #define LINALG_TOOLS_TIMINGS_START(SECTION_LABEL)
    #define LINALG_TOOLS_TIMINGS_END(SECTION_LABEL)
    #define LINALG_TOOLS_TIMINGS_ACCUM_INIT(SECTION_LABEL)
    #define LINALG_TOOLS_TIMINGS_ACCUM_START(SECTION_LABEL)
    #define LINALG_TOOLS_TIMINGS_ACCUM_END(SECTION_LABEL)
    #define LINALG_TOOLS_TIMINGS_ACCUM_PRINT(SECTION_LABEL)
#endif


/* TODO: Transfer to tools.hpp */
#define CPP_CHKERRQ(err) \
    do { \
        if (PetscUnlikely(err)){\
            PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, err, PETSC_ERROR_IN_CXX, 0);\
            throw std::runtime_error("Error.");}}\
    while(0)


#define CPP_CHKERRQ_MSG(err, msg) \
  if (err) { \
    PetscError(PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, err, PETSC_ERROR_IN_CXX, 0, msg); \
    throw std::runtime_error(msg); }


/** @} */

#endif // __LINALG_TOOLS_HPP__
