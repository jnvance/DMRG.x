
#include "linalg_tools.hpp"

/*
    TODO: Move these definitions to spin-dependent class
 */

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

    ierr = MatShift(eye, 1.00); CHKERRQ(ierr);

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
PetscErrorCode MatPeek(const Mat mat, const char* label)
{
    PetscErrorCode  ierr = 0;
    const MPI_Comm comm = PetscObjectComm((PetscObject) mat);
    PetscViewer fd = nullptr;

    ierr = MatAssemblyBegin(mat, MAT_FLUSH_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    ierr = PetscPrintf(comm, "\n%s\n", label); CHKERRQ(ierr);
    ierr = MatView(mat, fd); CHKERRQ(ierr);

    ierr = PetscViewerDestroy(&fd); CHKERRQ(ierr);
    fd = nullptr;

    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "MatWrite"
PetscErrorCode MatWrite(const Mat mat, const char* filename)
{
    PetscErrorCode  ierr = 0;
    const MPI_Comm comm = PetscObjectComm((PetscObject) mat);
    PetscViewer writer = nullptr;

    MatAssemblyBegin(mat, MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY);

    PetscMPIInt rank;
    MPI_Comm_rank(comm, &rank);

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
PetscErrorCode VecWrite(const Vec& vec, const char* filename)
{
    PetscErrorCode  ierr = 0;
    const MPI_Comm comm = PetscObjectComm((PetscObject) vec);
    PetscViewer writer = nullptr;
    PetscViewerBinaryOpen(comm, filename, FILE_MODE_WRITE, &writer);
    VecView(vec, writer);
    PetscViewerDestroy(&writer);
    writer = nullptr;

    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "VecPeek"
PetscErrorCode VecPeek(const Vec& vec, const char* label)
{
    PetscErrorCode  ierr = 0;
    const MPI_Comm comm = PetscObjectComm((PetscObject) vec);
    ierr = PetscPrintf(comm, "\n%s\n", label); CHKERRQ(ierr);
    ierr = VecView(vec, PETSC_VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "VecReshapeToMat"
PetscErrorCode VecReshapeToMat(const Vec& vec, Mat& mat, const PetscInt M, const PetscInt N, const PetscBool mat_is_local)
{
    PetscErrorCode  ierr = 0;
    const MPI_Comm comm = PetscObjectComm((PetscObject) vec);

    /*
        Get the size of vec and determine whether the size of the ouput matrix
        is compatible with this size, i.e. vec_size == M*N
     */
    PetscInt    vec_size;
    ierr = VecGetSize(vec, &vec_size); CHKERRQ(ierr);
    if( M * N != vec_size ) SETERRQ(comm, 1, "Size mismatch");


    // PetscInt    mat_Istart, mat_Iend, mat_nrows;
    PetscInt    mat_Istart, mat_Iend;
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

    // mat_nrows  = mat_Iend - mat_Istart;
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


#undef __FUNCT__
#define __FUNCT__ "VecReshapeToLocalMat"
PetscErrorCode VecReshapeToLocalMat(const Vec& vec, Mat& mat, const PetscInt M, const PetscInt N)
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

/* FIXME */
#undef __FUNCT__
#define __FUNCT__ "LocalVecReshapeToLocalMat"
PetscErrorCode LocalVecReshapeToLocalMat(
    const Vec& vec_seq, Mat& mat_seq,
    const PetscInt M, const PetscInt N,
    const std::vector<PetscInt> idx)
{
    PetscErrorCode ierr = 0;

    /* Checkpoints */

    PetscBool flg;
    ierr = PetscObjectTypeCompare((PetscObject)vec_seq,VECSEQ,&flg);CHKERRQ(ierr);
    if(!flg) SETERRQ(PETSC_COMM_SELF,1,"Argument 1 vec_seq must be a sequential vector (VECSEQ) object.");

    if(idx.size()>0 && idx.size() != M*N)
        SETERRQ2(PETSC_COMM_SELF,1,"Index size and M*N mismatch. Expected M*N size %d. Got %d.",M*N, idx.size());

    PetscScalar *vec_vals;
    ierr = VecGetArray(vec_seq, &vec_vals); CHKERRQ(ierr);
    ierr = MatCreateSeqDense(PETSC_COMM_SELF, M, N, NULL, &mat_seq); CHKERRQ(ierr);

    /* Prepare column indices*/
    PetscInt    *col_idx;
    ierr = PetscMalloc1(N, &col_idx); CHKERRQ(ierr);
    for (PetscInt i = 0; i < N; ++i)
        col_idx[i] = i;

    /* With idx: copy only values of vec_vals indexed by idx in row-order */
    if(idx.size()>0)
    {
        PetscScalar *row_vals;
        ierr = PetscMalloc1(N, &row_vals); CHKERRQ(ierr);
        for (PetscInt Irow = 0; Irow < M; ++Irow)
        {
            for (PetscInt Icol = 0; Icol < N; ++Icol)
                row_vals[Icol] = vec_vals[idx[Irow*N + Icol]];
            ierr = MatSetValues(mat_seq, 1, &Irow, N, col_idx, row_vals, INSERT_VALUES);
            CHKERRQ(ierr);
        }
        ierr = PetscFree(row_vals); CHKERRQ(ierr);
    }
    /* Without idx: copy all values of vec_vals in row-order */
    else
    {
        for (PetscInt Irow = 0; Irow < M; ++Irow)
        {
            ierr = MatSetValues(mat_seq, 1, &Irow, N, col_idx, &vec_vals[Irow*N], INSERT_VALUES);
            CHKERRQ(ierr);
        }
    }

    ierr = MatAssemblyBegin(mat_seq, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(mat_seq, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    ierr = PetscFree(col_idx); CHKERRQ(ierr);
    ierr = VecRestoreArray(vec_seq, &vec_vals); CHKERRQ(ierr);

    return ierr;
};


#undef __FUNCT__
#define __FUNCT__ "VecToMatMultHC"
PetscErrorCode VecToMatMultHC(const Vec& vec_r, const Vec& vec_i, Mat& mat,
    const PetscInt M, const PetscInt N, const PetscBool hc_right = PETSC_TRUE)
{
    PetscErrorCode  ierr = 0;

    const MPI_Comm comm = PetscObjectComm((PetscObject) vec_r);

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
        ierr = VecReshapeToMat(vec_r, gsv_mat_seq, M, N); CHKERRQ(ierr);
        ierr = MatHermitianTranspose(gsv_mat_seq, MAT_INITIAL_MATRIX, &gsv_mat_hc); CHKERRQ(ierr);
        ierr = MatMatMult(gsv_mat_seq, gsv_mat_hc, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &mat); CHKERRQ(ierr);
        if(gsv_mat_seq) {ierr = MatDestroy(&gsv_mat_seq); CHKERRQ(ierr);}
        if(gsv_mat_hc ) {ierr = MatDestroy(&gsv_mat_hc ); CHKERRQ(ierr);}
        return ierr;
    #endif // __BUILD_SEQUENTIAL


    ierr = VecReshapeToLocalMat(vec_r, gsv_mat_seq, M, N); CHKERRQ(ierr);

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

    Mat mat_in_loc = nullptr;
    Mat mat_out_loc = nullptr;

    /*
        Some processes may not have been assigned any rows.
        Otherwise, "Intel MKL ERROR: Parameter 8 was incorrect on entry to ZGEMM ." is produced
    */
    if(nrows > 0)
    {
        /*
            Create a matrix object that handles the local portion of mat
        */
        ierr = MatDenseGetLocalMatrix(mat, &mat_out_loc); CHKERRQ(ierr);
        /*
            Create a copy of the portion of gsv_mat that mimics the local row partition of mat
        */
        ierr = MatCreateSeqDense(PETSC_COMM_SELF, nrows, N, NULL, &mat_in_loc); CHKERRQ(ierr);
        /*
            Fill mat_in_loc with column slices of gsv_mat_seq that belong to the local row partition of mat
        */
        PetscScalar *vals_gsv_mat_seq, *vals_mat_in_loc;
        ierr = MatDenseGetArray(gsv_mat_seq, &vals_gsv_mat_seq); CHKERRQ(ierr);
        ierr = MatDenseGetArray(mat_in_loc, &vals_mat_in_loc); CHKERRQ(ierr);
        for (PetscInt Icol = 0; Icol < N; ++Icol)
        {
            ierr = PetscMemcpy(&vals_mat_in_loc[Icol*nrows],&vals_gsv_mat_seq[Istart+Icol*M], nrows*sizeof(PetscScalar));
            CHKERRQ(ierr);
        }
        ierr = MatDenseRestoreArray(gsv_mat_seq, &vals_gsv_mat_seq); CHKERRQ(ierr);
        ierr = MatDenseRestoreArray(mat_in_loc, &vals_mat_in_loc); CHKERRQ(ierr);

        ierr = MatHermitianTranspose(gsv_mat_seq, MAT_INITIAL_MATRIX, &gsv_mat_hc); CHKERRQ(ierr);
        ierr = MatMatMult(mat_in_loc,gsv_mat_hc,MAT_REUSE_MATRIX,PETSC_DEFAULT,&mat_out_loc); CHKERRQ(ierr);
    }

    ierr = MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    if(mat_in_loc) {ierr = MatDestroy(&mat_in_loc); CHKERRQ(ierr);}
    if(gsv_mat_seq) {ierr = MatDestroy(&gsv_mat_seq); CHKERRQ(ierr);}
    if(gsv_mat_hc ) {ierr = MatDestroy(&gsv_mat_hc ); CHKERRQ(ierr);}

    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "MatMultSelfHC_AIJ"
PetscErrorCode MatMultSelfHC_AIJ(const Mat& mat_in, Mat& mat, const PetscBool hc_right)
{
    PetscErrorCode  ierr = 0;
    /*
        The resulting matrix will be created in PETSC_COMM_WORLD
    */
    MPI_Comm comm = PETSC_COMM_WORLD;
    #ifndef PETSC_USE_COMPLEX
        // SETERRQ(comm, 1, "Not implemented for real scalars.");
    #endif
    /*
        Impose that the input matrix be of type seqdense
    */
    PetscBool flg;
    ierr = PetscObjectTypeCompare((PetscObject)mat_in,MATSEQDENSE,&flg);CHKERRQ(ierr);
    if(!flg) SETERRQ(comm, 1, "Input matrix must be of type seqdense.\n");
    /*
        Create the resultant matrix mat with the correct dimensions
    */
    PetscInt    mat_dim, M, N;
    MatGetSize(mat_in, &M, &N);
    if (hc_right == PETSC_TRUE){
        mat_dim = M;
    }
    else {
        mat_dim = N;
    }
    ierr = MatCreate(PETSC_COMM_WORLD, &mat); CHKERRQ(ierr);
    ierr = MatSetType(mat, MATAIJ);
    ierr = MatSetSizes(mat, PETSC_DECIDE, PETSC_DECIDE, mat_dim, mat_dim); CHKERRQ(ierr);
    ierr = MatSetFromOptions(mat); CHKERRQ(ierr);

    PetscMPIInt nprocs, rank;
    MPI_Comm_size(PETSC_COMM_WORLD, &nprocs);
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    PetscInt remrows = mat_dim % nprocs;
    PetscInt locrows = mat_dim / nprocs + (rank < remrows ? 1 : 0 );

    ierr = MatMPIAIJSetPreallocation(mat, locrows, NULL, mat_dim - locrows, NULL); CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(mat, mat_dim, NULL); CHKERRQ(ierr);
    /*
        Get ownership info
    */
    PetscInt    Istart, Iend, nrows;
    ierr = MatGetOwnershipRange(mat, &Istart, &Iend); CHKERRQ(ierr);
    nrows = Iend - Istart;

    if(nrows != locrows)
        SETERRQ(PETSC_COMM_WORLD, 1, "Matrix layout different from expected.");

    Mat mat_in_loc = nullptr;
    Mat mat_out_loc = nullptr;
    Mat mat_in_hc  = nullptr;
    PetscScalar *vals, *vals_loc;

    /*
        Some processes may not have been assigned any rows.
        Otherwise, "Intel MKL ERROR: Parameter 8 was incorrect on entry to ZGEMM ." is produced
    */
    if(nrows > 0)
    {
        /*
            Create a matrix object that handles the local portion of mat
        */
        // ierr = MatDenseGetLocalMatrix(mat_dense, &mat_out_loc); CHKERRQ(ierr);
        ierr = MatCreateSeqDense(PETSC_COMM_SELF, nrows, mat_dim, NULL, &mat_out_loc); CHKERRQ(ierr);
        /*
            Create a copy of the portion of mat_in that mimics the local row partition of mat
        */
        ierr = MatCreateSeqDense(PETSC_COMM_SELF, nrows, hc_right ? N : M , NULL, &mat_in_loc); CHKERRQ(ierr);
        /*
            Get the Hermitian conjugate of mat_in
        */
        ierr = MatHermitianTranspose(mat_in, MAT_INITIAL_MATRIX, &mat_in_hc); CHKERRQ(ierr);
        /*
            Fill mat_in_loc with column slices of mat_in that belong to the local row partition of mat_in/mat_in_hc
        */
        ierr = MatDenseGetArray(mat_in_loc, &vals_loc); CHKERRQ(ierr);

        if (hc_right == PETSC_TRUE){
            ierr = MatDenseGetArray(mat_in, &vals); CHKERRQ(ierr);
            for (PetscInt Icol = 0; Icol < N; ++Icol)
            {
                ierr = PetscMemcpy(&vals_loc[Icol*nrows],&vals[Istart+Icol*M], nrows*sizeof(PetscScalar));
                CHKERRQ(ierr);
            }
            ierr = MatDenseRestoreArray(mat_in, &vals); CHKERRQ(ierr);
            ierr = MatMatMult(mat_in_loc,mat_in_hc,MAT_REUSE_MATRIX,PETSC_DEFAULT,&mat_out_loc); CHKERRQ(ierr);
        } else
        {
            ierr = MatDenseGetArray(mat_in_hc, &vals); CHKERRQ(ierr);
            for (PetscInt Icol = 0; Icol < M; ++Icol)
            {
                ierr = PetscMemcpy(&vals_loc[Icol*nrows],&vals[Istart+Icol*N], nrows*sizeof(PetscScalar));
                CHKERRQ(ierr);
            }
            ierr = MatDenseRestoreArray(mat_in, &vals); CHKERRQ(ierr);
            ierr = MatMatMult(mat_in_loc,mat_in,MAT_REUSE_MATRIX,PETSC_DEFAULT,&mat_out_loc); CHKERRQ(ierr);
        }
        ierr = MatDenseRestoreArray(mat_in_loc, &vals_loc); CHKERRQ(ierr);
        ierr = MatDenseGetArray(mat_out_loc, &vals_loc); CHKERRQ(ierr);

        /* Prepare row indices and follow matrix layout */
        PetscInt idxm[nrows];
        for (PetscInt Irow = Istart; Irow < Iend; ++Irow)
            idxm[Irow - Istart] = Irow;

        /* Copy values from dense matrix to output sparse matrix */
        for (PetscInt Icol = 0; Icol < mat_dim; ++Icol)
        {
            ierr = MatSetValues(mat, nrows, idxm, 1, &Icol, vals_loc+(Icol*nrows), INSERT_VALUES); CHKERRQ(ierr);
        }

        ierr = MatDenseRestoreArray(mat_out_loc, &vals_loc); CHKERRQ(ierr);
    }

    ierr = MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    if(mat_in_loc ) {ierr = MatDestroy(&mat_in_loc ); CHKERRQ(ierr);}
    if(mat_in_hc  ) {ierr = MatDestroy(&mat_in_hc  ); CHKERRQ(ierr);}
    if(mat_out_loc) {ierr = MatDestroy(&mat_out_loc); CHKERRQ(ierr);}
    // SETERRQ(PETSC_COMM_WORLD,1,"stahp");

    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "SVDLargestStates"
PetscErrorCode SVDLargestStates(const Mat& mat_in, const PetscInt mstates_in, PetscScalar& error, Mat& mat, FILE *fp)
{
    PetscErrorCode  ierr = 0;

    MPI_Comm comm = PetscObjectComm((PetscObject)mat_in);

    #ifndef PETSC_USE_COMPLEX
        // SETERRQ(comm, 1, "Not implemented for real scalars.");
    #endif

    /*********************TIMINGS**********************/
    #ifdef __DMRG_SUB_TIMINGS
        PetscLogDouble svd_total_time0, svd_total_time;
        ierr = PetscTime(&svd_total_time0); CHKERRQ(ierr);
    #endif
    /**************************************************/

    PetscBool assembled;
    ierr = MatAssembled(mat_in, &assembled); CHKERRQ(ierr);
    if (assembled == PETSC_FALSE){
        ierr = MatAssemblyBegin(mat_in, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(mat_in, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }

    PetscInt mat_in_nrows, mat_in_ncols;
    ierr = MatGetSize(mat_in, &mat_in_nrows, &mat_in_ncols);
    if(mat_in_nrows != mat_in_ncols)
    {
        char errormsg[80];
        sprintf(errormsg,"Matrix dimension mismatch. "
                         "Number of rows (%d) is not equal to number of columns (%d).",
                         mat_in_nrows, mat_in_ncols);
        SETERRQ(comm, 1, errormsg);
    }

    #ifdef __DMRG_SUB_TIMINGS
        PetscPrintf(comm,"%16s SVD size: %lu\n","",(unsigned long int)(mat_in_nrows));
    #endif

    PetscInt mstates = mstates_in;

    if(mat_in_nrows < mstates)
    {
        char errormsg[80];
        sprintf(errormsg,"Matrix dimension too small. "
                         "Matrix size (%d) must at least be equal to mstates (%d).",
                         mat_in_nrows, mstates);
        SETERRQ(comm, 1, errormsg);
    }

    SVD svd = nullptr;
    ierr = SVDCreate(comm, &svd); CHKERRQ(ierr);
    ierr = SVDSetOperator(svd, mat_in); CHKERRQ(ierr);
    ierr = SVDSetWhichSingularTriplets(svd,SVD_LARGEST); CHKERRQ(ierr);
    ierr = SVDSetType(svd, SVDTRLANCZOS); CHKERRQ(ierr);
    // ierr = SVDSetDimensions(svd, mstates, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);
    ierr = SVDSetDimensions(svd, mstates, mat_in_ncols, PETSC_DEFAULT); CHKERRQ(ierr);
    // ierr = SVDSetTolerances(svd, 1e-20, 200); CHKERRQ(ierr);
    ierr = SVDSetFromOptions(svd); CHKERRQ(ierr);


    /*********************TIMINGS**********************/
    #ifdef __DMRG_SUB_TIMINGS
        ierr = PetscTime(&svd_total_time); CHKERRQ(ierr);
        svd_total_time = svd_total_time - svd_total_time0;
        ierr = PetscPrintf(PETSC_COMM_WORLD, "%16s %-42s %.20f\n", "","SVD prep:", svd_total_time);

        PetscLogDouble svd_solve_time0, svd_solve_time;
        ierr = PetscTime(&svd_solve_time0); CHKERRQ(ierr);
    #endif
    /**************************************************/

    #define __SVD_SOLVE "        SVDSolve"
    LINALG_TOOLS_TIMINGS_START(__SVD_SOLVE)

    ierr = SVDSolve(svd);CHKERRQ(ierr);

    LINALG_TOOLS_TIMINGS_END(__SVD_SOLVE)
    #undef __SVD_SOLVE

    /*********************TIMINGS**********************/
    #ifdef __DMRG_SUB_TIMINGS
        ierr = PetscTime(&svd_solve_time); CHKERRQ(ierr);
        svd_solve_time = svd_solve_time - svd_solve_time0;
        ierr = PetscPrintf(PETSC_COMM_WORLD, "%16s %-42s %.20f\n", "","SVD solve:", svd_solve_time);

        PetscLogDouble svd_post_time0, svd_post_time;
        ierr = PetscTime(&svd_post_time0); CHKERRQ(ierr);
    #endif
    /**************************************************/

    #define __SVD_LOAD  "        SVDLoad"
    LINALG_TOOLS_TIMINGS_START(__SVD_LOAD)


    PetscInt nconv;
    ierr = SVDGetConverged(svd, &nconv); CHKERRQ(ierr);
    if (nconv < mstates)
    {
        char errormsg[80];
        sprintf(errormsg,"Number of converged singular values (%d) is less than mstates (%d).", nconv, mstates);
        SETERRQ(comm, 1, errormsg);
    }

    #ifdef __PRINT_SVD_CONVERGENCE
        PetscPrintf(comm, "%12sSVD requested mstates: %d\n","",mstates);
        PetscPrintf(comm, "%12sSVD no of conv states: %d\n","",nconv);
    #endif

    /**
        The output matrix is a dense matrix but stored as SPARSE.
     */
    Vec Vr;
    PetscInt    Istart, Iend, Istart_mat, Iend_mat;
    ierr = MatCreate(comm, &mat); CHKERRQ(ierr);
    ierr = MatSetSizes(mat, PETSC_DECIDE, PETSC_DECIDE, mat_in_nrows, mstates); CHKERRQ(ierr);
    ierr = MatCreateVecs(mat_in, &Vr, nullptr); CHKERRQ(ierr);
    ierr = MatSetFromOptions(mat); CHKERRQ(ierr);
    ierr = MatSetUp(mat); CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(Vr,  &Istart, &Iend); CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(mat, &Istart_mat, &Iend_mat); CHKERRQ(ierr);

    if (!(Istart == Istart_mat && Iend == Iend_mat))
        SETERRQ(comm, 1, "Matrix and vector layout do not match.");

    /* Prepare row indices */
    PetscInt mrows = Iend - Istart;
    PetscInt idxm[mrows];
    for (PetscInt Irow = Istart; Irow < Iend; ++Irow) idxm[Irow - Istart] = Irow;

    PetscReal sum_first_mstates = 0;
    PetscReal eigr;
    const PetscScalar *vals;
    for (PetscInt Istate = 0; Istate < mstates; ++Istate)
    {
        ierr = SVDGetSingularTriplet(svd, Istate, &eigr, Vr, nullptr); CHKERRQ(ierr);
        sum_first_mstates += eigr;
        #ifdef __TESTING
            ierr = PetscFPrintf(comm, fp, "%.20g+0.0j\n",eigr);
        #endif
        ierr = VecGetArrayRead(Vr, &vals); CHKERRQ(ierr);
        ierr = MatSetValues(mat, mrows, idxm, 1, &Istate, vals, INSERT_VALUES); CHKERRQ(ierr);
        ierr = VecRestoreArrayRead(Vr, &vals); CHKERRQ(ierr);
    }
    error = 1.0 - sum_first_mstates;

    ierr = MatAssembled(mat, &assembled); CHKERRQ(ierr);
    if (assembled == PETSC_FALSE){
        ierr = MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }

    #ifdef __PRINT_SVD_LARGEST
        SVDType        type;
        PetscReal      tol;
        PetscInt       maxit,its,nsv;
        PetscBool      terse;

        ierr = SVDGetIterationNumber(svd,&its);CHKERRQ(ierr);
        ierr = PetscPrintf(comm," Number of iterations of the method: %D\n",its);CHKERRQ(ierr);

        ierr = SVDGetType(svd,&type);CHKERRQ(ierr);
        ierr = PetscPrintf(comm," Solution method: %s\n\n",type);CHKERRQ(ierr);
        ierr = SVDGetDimensions(svd,&nsv,NULL,NULL);CHKERRQ(ierr);
        ierr = PetscPrintf(comm," Number of requested singular values: %D\n",nsv);CHKERRQ(ierr);
        ierr = SVDGetTolerances(svd,&tol,&maxit);CHKERRQ(ierr);
        ierr = PetscPrintf(comm," Stopping condition: tol=%.4g, maxit=%D\n",(double)tol,maxit);CHKERRQ(ierr);
        /*
            Show detailed info unless -terse option is given by user
         */
        ierr = PetscOptionsHasName(NULL,NULL,"-terse",&terse);CHKERRQ(ierr);
        if (terse) {
            ierr = SVDErrorView(svd,SVD_ERROR_RELATIVE,NULL);CHKERRQ(ierr);
        } else {
            ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
            ierr = SVDReasonView(svd,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
            ierr = SVDErrorView(svd,SVD_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
            ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        }
    #endif

    LINALG_TOOLS_TIMINGS_END(__SVD_LOAD)

    ierr = SVDDestroy(&svd);

    /*********************TIMINGS**********************/
    #ifdef __DMRG_SUB_TIMINGS
        ierr = PetscTime(&svd_post_time); CHKERRQ(ierr);
        svd_post_time = svd_post_time - svd_post_time0;
        ierr = PetscPrintf(PETSC_COMM_WORLD, "%16s %-42s %.20f\n", "","SVD post:", svd_post_time);

        ierr = PetscTime(&svd_total_time); CHKERRQ(ierr);
        svd_total_time = svd_total_time - svd_total_time0;
        ierr = PetscPrintf(PETSC_COMM_WORLD, "%16s %-42s %.20f\n\n", "","SVD TOTAL:", svd_total_time);
    #endif
    /**************************************************/

    return ierr;
}


std::vector<PetscScalar> OuterSumFlatten(std::vector<PetscScalar> A, std::vector<PetscScalar> B)
{
    std::vector<PetscScalar> C(A.size()*B.size(), 0);

    for (PetscInt i = 0; i < (PetscInt)(A.size()); ++i)
        for (PetscInt j = 0; j < (PetscInt)(B.size()); ++j)
            C[i*(PetscInt)(B.size()) + j] = A[i] + B[j];

    return C;
}


std::unordered_map<PetscScalar,std::vector<PetscInt>> IndexMap(std::vector<PetscScalar> array)
{
    std::unordered_map<PetscScalar,std::vector<PetscInt>> map;

    for (PetscInt i = 0; i < (PetscInt)(array.size()); ++i)
        map[array[i]].push_back(i);

    return map;
}


#undef __FUNCT__
#define __FUNCT__ "MatGetSVD"
PetscErrorCode MatGetSVD(const Mat& mat_in, SVD& svd, PetscInt& nconv, PetscScalar& error, FILE *fp)
{
    PetscErrorCode  ierr = 0;

    MPI_Comm comm = PETSC_COMM_WORLD;
    PetscBool assembled;
    ierr = MatAssembled(mat_in, &assembled); CHKERRQ(ierr);
    if (assembled == PETSC_FALSE){
        ierr = MatAssemblyBegin(mat_in, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(mat_in, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }

    PetscInt mat_in_nrows, mat_in_ncols;
    ierr = MatGetSize(mat_in, &mat_in_nrows, &mat_in_ncols);
    if(mat_in_nrows != mat_in_ncols)
    {
        char errormsg[80];
        sprintf(errormsg,"Matrix dimension mismatch. "
                         "Number of rows (%d) is not equal to number of columns (%d).",
                         mat_in_nrows, mat_in_ncols);
        SETERRQ(comm, 1, errormsg);
    }

    svd = nullptr;
    ierr = SVDCreate(comm, &svd); CHKERRQ(ierr);
    ierr = SVDSetOperator(svd, mat_in); CHKERRQ(ierr);
    ierr = SVDSetType(svd, SVDTRLANCZOS); CHKERRQ(ierr);
    // ierr = SVDSetDimensions(svd, mat_in_nrows, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);
    ierr = SVDSetDimensions(svd, mat_in_nrows, mat_in_nrows, PETSC_DEFAULT); CHKERRQ(ierr);
    ierr = SVDSetWhichSingularTriplets(svd, SVD_LARGEST); CHKERRQ(ierr);
    ierr = SVDSetFromOptions(svd); CHKERRQ(ierr);

    #define __SVD_SOLVE "        SVDSolve"
    LINALG_TOOLS_TIMINGS_START(__SVD_SOLVE)

    ierr = SVDSolve(svd);CHKERRQ(ierr);

    LINALG_TOOLS_TIMINGS_END(__SVD_SOLVE)
    #undef __SVD_SOLVE

    #define __SVD_LOAD  "        SVDLoad"
    LINALG_TOOLS_TIMINGS_START(__SVD_LOAD)


    ierr = SVDGetConverged(svd, &nconv); CHKERRQ(ierr);
    if (nconv < mat_in_nrows)
    {
        char errormsg[120];
        sprintf(errormsg,"Number of converged singular values (%d) is less than requested (%d).", nconv, mat_in_nrows);
        PetscPrintf(comm,"WARNING: %s\n", errormsg);
        // SETERRQ(comm, 1, errormsg);
    }

    #ifdef __PRINT_SVD_CONVERGENCE
        PetscPrintf(comm, "%12sSVD requested mstates: %d\n","",mstates);
        PetscPrintf(comm, "%12sSVD no of conv states: %d\n","",nconv);
    #endif

    #ifdef __PRINT_SVD_LARGEST
        SVDType        type;
        PetscReal      tol;
        PetscInt       maxit,its,nsv;
        PetscBool      terse;

        ierr = SVDGetIterationNumber(svd,&its);CHKERRQ(ierr);
        ierr = PetscPrintf(comm," Number of iterations of the method: %D\n",its);CHKERRQ(ierr);

        ierr = SVDGetType(svd,&type);CHKERRQ(ierr);
        ierr = PetscPrintf(comm," Solution method: %s\n\n",type);CHKERRQ(ierr);
        ierr = SVDGetDimensions(svd,&nsv,NULL,NULL);CHKERRQ(ierr);
        ierr = PetscPrintf(comm," Number of requested singular values: %D\n",nsv);CHKERRQ(ierr);
        ierr = SVDGetTolerances(svd,&tol,&maxit);CHKERRQ(ierr);
        ierr = PetscPrintf(comm," Stopping condition: tol=%.4g, maxit=%D\n",(double)tol,maxit);CHKERRQ(ierr);
        /*
            Show detailed info unless -terse option is given by user
         */
        ierr = PetscOptionsHasName(NULL,NULL,"-terse",&terse);CHKERRQ(ierr);
        if (terse) {
            ierr = SVDErrorView(svd,SVD_ERROR_RELATIVE,NULL);CHKERRQ(ierr);
        } else {
            ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL);CHKERRQ(ierr);
            ierr = SVDReasonView(svd,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
            ierr = SVDErrorView(svd,SVD_ERROR_RELATIVE,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
            ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        }
    #endif



    LINALG_TOOLS_TIMINGS_END(__SVD_LOAD)

    // ierr = SVDDestroy(&svd);
    return ierr;
}
