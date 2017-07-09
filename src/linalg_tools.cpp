
#include "linalg_tools.hpp"

#undef __FUNCT__
#define __FUNCT__ "MatEyeCreate"
PetscErrorCode MatEyeCreate(const MPI_Comm& comm, Mat& eye, PetscInt dim)
{
    PetscErrorCode  ierr = 0;

    ierr = MatCreate(comm, &eye); CHKERRQ(ierr);
    ierr = MatSetSizes(eye, PETSC_DECIDE, PETSC_DECIDE, dim, dim); CHKERRQ(ierr);
    ierr = MatSetFromOptions(eye); CHKERRQ(ierr);
    ierr = MatSetUp(eye); CHKERRQ(ierr);
    ierr = MatZeroEntries(eye); CHKERRQ(ierr);

    ierr = MatAssemblyBegin(eye, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(eye, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    MatShift(eye, 1.00);

    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "MatSzCreate"
PetscErrorCode MatSzCreate(const MPI_Comm& comm, Mat& Sz)
{
    PetscErrorCode  ierr = 0;

    ierr = MatCreate(comm, &Sz); CHKERRQ(ierr);
    ierr = MatSetSizes(Sz, PETSC_DECIDE, PETSC_DECIDE, 2, 2); CHKERRQ(ierr);
    ierr = MatSetFromOptions(Sz); CHKERRQ(ierr);
    ierr = MatSetUp(Sz); CHKERRQ(ierr);
    ierr = MatZeroEntries(Sz); CHKERRQ(ierr);

    ierr = MatAssemblyBegin(Sz, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Sz, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    ierr = MatSetValue(Sz, 0, 0, +0.5, INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatSetValue(Sz, 1, 1, -0.5, INSERT_VALUES); CHKERRQ(ierr);

    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "MatSpCreate"
PetscErrorCode MatSpCreate(const MPI_Comm& comm, Mat& Sp)
{
    PetscErrorCode  ierr = 0;

    ierr = MatCreate(comm, &Sp); CHKERRQ(ierr);
    ierr = MatSetSizes(Sp, PETSC_DECIDE, PETSC_DECIDE, 2, 2); CHKERRQ(ierr);
    ierr = MatSetFromOptions(Sp); CHKERRQ(ierr);
    ierr = MatSetUp(Sp); CHKERRQ(ierr);
    ierr = MatZeroEntries(Sp); CHKERRQ(ierr);

    ierr = MatAssemblyBegin(Sp, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Sp, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    ierr = MatSetValue(Sp, 0, 1, +1.0, INSERT_VALUES); CHKERRQ(ierr);

    ierr = MatAssemblyBegin(Sp, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Sp, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "MatPeek"
PetscErrorCode MatPeek(const MPI_Comm& comm, const Mat mat, const char* label)
{
    PetscErrorCode  ierr = 0;

    // Peek into values
    PetscViewer fd = nullptr;
    ierr = MatAssemblyBegin(mat, MAT_FLUSH_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    PetscPrintf(comm, "\n%s\n", label);
    ierr = MatView(mat, fd); CHKERRQ(ierr);

    PetscViewerDestroy(&fd);
    fd = nullptr;

    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "MatWrite"
PetscErrorCode MatWrite(const MPI_Comm& comm, const Mat mat, const char* filename)
{
    PetscErrorCode  ierr = 0;

    PetscViewer writer = nullptr;

    MatAssemblyBegin(mat, MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY);

    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    PetscBool flg;
    ierr = PetscObjectTypeCompare((PetscObject)mat,MATMPIDENSE,&flg);CHKERRQ(ierr);

    char filename_[80];
    if(comm==PETSC_COMM_SELF || flg==PETSC_TRUE){
        sprintf(filename_, "%s.%d", filename, rank);
    }
    else {
        sprintf(filename_, "%s", filename);
    }

    if(flg==PETSC_TRUE) {

        #ifdef __IMPLEMENTATION01
        /*
            Get indices for submatrix, with process 0 taking all elements
        */
        PetscInt M, N;
        Mat submat = nullptr, submat_loc = nullptr;
        MatGetSize(mat, &M, &N);
        if(rank!=0){ M = 0; N = 0; }
        PetscInt id_rows[M], id_cols[N];
        for (PetscInt Irow = 0; Irow < M; ++Irow)
            id_rows[Irow] = Irow;
        for (PetscInt Icol = 0; Icol < N; ++Icol)
            id_cols[Icol] = Icol;
        IS isrow, iscol;

        ISCreateGeneral(comm, M, id_rows, PETSC_COPY_VALUES, &isrow);
        ISCreateGeneral(comm, N, id_cols, PETSC_COPY_VALUES, &iscol);
        MatGetSubMatrix(mat, isrow, iscol, MAT_INITIAL_MATRIX, &submat);

        if(rank==0)
        {
            MatDenseGetLocalMatrix(submat, &submat_loc);
            PetscViewerBinaryOpen(PETSC_COMM_SELF, filename_, FILE_MODE_WRITE, &writer);
            MatView(submat_loc, writer);
        }
        if(submat) MatDestroy(&submat);

        #else

        Mat mat_loc;
        MatDenseGetLocalMatrix(mat, &mat_loc);
        PetscViewerBinaryOpen(PETSC_COMM_SELF, filename_, FILE_MODE_WRITE, &writer);
        MatView(mat_loc, writer);

        /*
            Read in python using:

                M = []
                for i in range(4):
                    with open('<filename>.'+str(i),'r') as fh:
                        A = io.readBinaryFile(fh,complexscalars=True,mattype='dense')[0]
                    M.append(A.copy())
                M = np.vstack(M)
        */
        #endif
    }
    else
    {
        PetscViewerBinaryOpen(comm, filename_, FILE_MODE_WRITE, &writer);
        MatView(mat, writer);
    }

    if(writer) PetscViewerDestroy(&writer);
    writer = nullptr;

    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "VecWrite"
PetscErrorCode VecWrite(const MPI_Comm& comm, const Vec& vec, const char* filename)
{
    PetscErrorCode  ierr = 0;

    PetscViewer writer = nullptr;
    // MatAssemblyBegin(mat, MAT_FLUSH_ASSEMBLY);
    // MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY);
    PetscViewerBinaryOpen(comm, filename, FILE_MODE_WRITE, &writer);
    VecView(vec, writer);
    PetscViewerDestroy(&writer);
    writer = nullptr;

    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "VecPeek"
PetscErrorCode VecPeek(const MPI_Comm& comm, const Vec& vec, const char* label)
{
    PetscErrorCode  ierr = 0;

    // PetscViewer viewer = nullptr;
    // MatAssemblyBegin(mat, MAT_FLUSH_ASSEMBLY);
    // MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY);
    // PetscViewerBinaryOpen(comm, filename, FILE_MODE_WRITE, &viewer);
    ierr = PetscPrintf(comm, "\n%s\n", label); CHKERRQ(ierr);
    ierr = VecView(vec, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    // PetscViewerDestroy(&viewer);
    // viewer = nullptr;

    return ierr;
}


/* Reshape m*n vector to m x n array */
#undef __FUNCT__
#define __FUNCT__ "VecReshapeToMat"
PetscErrorCode VecReshapeToMat(const MPI_Comm& comm, const Vec& vec, Mat& mat, const PetscInt M, const PetscInt N, const PetscBool mat_is_local)
{

    PetscErrorCode  ierr = 0;

    /*
        Get the size of vec and determine whether the size of the ouput matrix
        is compatible with this size, i.e. vec_size == M*N
     */
    PetscInt    vec_size;
    ierr = VecGetSize(vec, &vec_size); CHKERRQ(ierr);
    if( M * N != vec_size ) SETERRQ(comm, 1, "Size mismatch");


    PetscInt    mat_Istart, mat_Iend, mat_nrows;
    PetscInt    subvec_Istart, subvec_Iend, subvec_nitems;
    PetscInt*   vec_idx;
    IS          vec_is;
    Vec         subvec;

    /*
        Matrix may be created locally as sequential or globally with MPI
    */
    if (mat_is_local == PETSC_TRUE)
    {
        MatCreateSeqDense(PETSC_COMM_SELF, M, N, NULL, &mat);
    }
    else
    {
        MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, M, N, NULL, &mat);
    }

    MatGetOwnershipRange(mat, &mat_Istart, &mat_Iend);

    mat_nrows  = mat_Iend - mat_Istart;
    subvec_Istart = mat_Istart*N;
    subvec_Iend   = mat_Iend*N;
    subvec_nitems = subvec_Iend - subvec_Istart;

    PetscMalloc1(subvec_nitems, &vec_idx);
    for (int i = 0; i < subvec_nitems; ++i)
        vec_idx[i] = subvec_Istart + i;
    ISCreateGeneral(comm, subvec_nitems, vec_idx, PETSC_OWN_POINTER, &vec_is);
    /* vec_idx is now owned by vec_is */

    VecGetSubVector(vec, vec_is, &subvec);

    PetscScalar*    subvec_array;
    VecGetArray(subvec, &subvec_array);


    PetscInt*   col_idx;
    PetscMalloc1(N, &col_idx);
    for (PetscInt i = 0; i < N; ++i) col_idx[i] = i;

    for (PetscInt Irow = mat_Istart; Irow < mat_Iend; ++Irow)
    {
        MatSetValues(mat, 1, &Irow, N, col_idx, &subvec_array[(Irow-mat_Istart)*N], INSERT_VALUES);
    }

    PetscFree(col_idx);
    VecRestoreSubVector(vec, vec_is, &subvec);
    ISDestroy(&vec_is);

    ierr = MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    return ierr;
}


PetscErrorCode VecReshapeToLocalMat(const MPI_Comm& comm, const Vec& vec,
    Mat& mat, const PetscInt M, const PetscInt N)
{
    PetscErrorCode ierr = 0;

    Vec         vec_seq;
    VecScatter  ctx;
    PetscScalar *vec_vals;
    PetscInt    *col_idx;

    ierr = VecScatterCreateToAll(vec, &ctx, &vec_seq); CHKERRQ(ierr);
    ierr = VecScatterBegin(ctx, vec, vec_seq, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecScatterEnd(ctx, vec, vec_seq, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecGetArray(vec_seq, &vec_vals); CHKERRQ(ierr);

    ierr = MatCreateSeqDense(PETSC_COMM_SELF, M, N, NULL, &mat); CHKERRQ(ierr);

    ierr = PetscMalloc1(N, &col_idx); CHKERRQ(ierr);

    for (PetscInt i = 0; i < N; ++i)
        col_idx[i] = i;

    for (PetscInt Irow = 0; Irow < M; ++Irow)
    {
        ierr = MatSetValues(mat, 1, &Irow, N, col_idx, &vec_vals[Irow*N], INSERT_VALUES);
        CHKERRQ(ierr);
    }
    ierr = PetscFree(col_idx); CHKERRQ(ierr);

    ierr = MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    ierr = VecRestoreArray(vec_seq, &vec_vals); CHKERRQ(ierr);
    ierr = VecScatterDestroy(&ctx); CHKERRQ(ierr);
    ierr = VecDestroy(&vec_seq); CHKERRQ(ierr);

    return ierr;
};


#undef __FUNCT__
#define __FUNCT__ "VecToMatMultHC"
PetscErrorCode VecToMatMultHC(const MPI_Comm& comm, const Vec& vec_r, const Vec& vec_i,
    Mat& mat, const PetscInt M, const PetscInt N, const PetscBool hc_right = PETSC_TRUE)
{

    PetscErrorCode  ierr = 0;

    #ifndef PETSC_USE_COMPLEX
        SETERRQ(comm, 1, "Not implemented for real scalars.");
    #endif

    /*
        Get the size of vec and determine whether the size of the ouput matrix
        is compatible with this size, i.e. vec_size == M*N
     */
    PetscInt vec_size;
    ierr = VecGetSize(vec_r, &vec_size); CHKERRQ(ierr);
    if( M * N != vec_size ) SETERRQ(comm, 1, "Size mismatch");

    /*
        Collect entire vector into sequential matrices residing in each process
    */
    Mat gsv_mat_seq = nullptr;
    Mat gsv_mat_hc  = nullptr;

    #ifdef __BUILD_SEQUENTIAL
        ierr = VecReshapeToMat(comm, vec_r, gsv_mat_seq, M, N); CHKERRQ(ierr);
        ierr = MatHermitianTranspose(gsv_mat_seq, MAT_INITIAL_MATRIX, &gsv_mat_hc);
        ierr = MatMatMult(gsv_mat_seq, gsv_mat_hc, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &mat); CHKERRQ(ierr);
        if(gsv_mat_seq) {ierr = MatDestroy(&gsv_mat_seq); CHKERRQ(ierr);}
        if(gsv_mat_hc ) {ierr = MatDestroy(&gsv_mat_hc ); CHKERRQ(ierr);}
        return ierr;
    #endif // __BUILD_SEQUENTIAL


    ierr = VecReshapeToLocalMat(comm, vec_r, gsv_mat_seq, M, N); CHKERRQ(ierr);

    /*
        Create the resultant matrix mat with the correct dimensions
    */
    PetscInt    mat_dim;
    if (hc_right == PETSC_TRUE){
        mat_dim = M;
    }
    else {
        SETERRQ(comm, 1, "Hermitian conjugate on left matrix not yet supported.");
        mat_dim = N;
    }
    ierr = MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, mat_dim, mat_dim, NULL, &mat); CHKERRQ(ierr);
    /*
        Get ownership info
    */
    PetscInt    Istart, Iend, nrows;
    ierr = MatGetOwnershipRange(mat, &Istart, &Iend); CHKERRQ(ierr);
    nrows = Iend - Istart;

    Mat gsv_mat_loc = nullptr;
    Mat mat_local = nullptr;

    /*
        Some processes may not have been assigned any rows.
        Otherwise, "Intel MKL ERROR: Parameter 8 was incorrect on entry to ZGEMM ." is produced
    */
    if(nrows > 0)
    {
        /*
            Create a matrix object that handles the local portion of mat
        */
        ierr = MatDenseGetLocalMatrix(mat, &mat_local); CHKERRQ(ierr);
        /*
            Create a copy of the portion of gsv_mat that mimics the local row partition of mat
        */
        ierr = MatCreateSeqDense(PETSC_COMM_SELF, nrows, N, NULL, &gsv_mat_loc); CHKERRQ(ierr);
        /*
            Fill gsv_mat_loc with column slices of gsv_mat_seq that belong to the local row partition of mat
        */
        PetscScalar *vals_gsv_mat_seq, *vals_gsv_mat_loc;
        ierr = MatDenseGetArray(gsv_mat_seq, &vals_gsv_mat_seq); CHKERRQ(ierr);
        ierr = MatDenseGetArray(gsv_mat_loc, &vals_gsv_mat_loc); CHKERRQ(ierr);
        for (PetscInt Icol = 0; Icol < N; ++Icol)
        {
            ierr = PetscMemcpy(&vals_gsv_mat_loc[Icol*nrows],&vals_gsv_mat_seq[Istart+Icol*M], nrows*sizeof(PetscScalar));
            CHKERRQ(ierr);
        }
        ierr = MatDenseRestoreArray(gsv_mat_seq, &vals_gsv_mat_seq); CHKERRQ(ierr);
        ierr = MatDenseRestoreArray(gsv_mat_loc, &vals_gsv_mat_loc); CHKERRQ(ierr);

        ierr = MatHermitianTranspose(gsv_mat_seq, MAT_INITIAL_MATRIX, &gsv_mat_hc); CHKERRQ(ierr);
        ierr = MatMatMult(gsv_mat_loc,gsv_mat_hc,MAT_REUSE_MATRIX,PETSC_DEFAULT,&mat_local); CHKERRQ(ierr);
    }

    if(gsv_mat_loc) {ierr = MatDestroy(&gsv_mat_loc); CHKERRQ(ierr);}
    if(gsv_mat_seq) {ierr = MatDestroy(&gsv_mat_seq); CHKERRQ(ierr);}
    if(gsv_mat_hc ) {ierr = MatDestroy(&gsv_mat_hc ); CHKERRQ(ierr);}

    return ierr;
}
