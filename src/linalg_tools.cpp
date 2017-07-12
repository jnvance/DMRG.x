
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
PetscErrorCode MatPeek(const Mat mat, const char* label)
{
    PetscErrorCode  ierr = 0;
    const MPI_Comm comm = PetscObjectComm((PetscObject) mat);
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
#define __FUNCT__ "MatMultSelfHC"
PetscErrorCode MatMultSelfHC(const Mat& mat_in, Mat& mat, const PetscBool hc_right)
{
    PetscErrorCode  ierr = 0;
    /*
        The resulting matrix will be created in PETSC_COMM_WORLD
    */
    MPI_Comm comm = PETSC_COMM_WORLD;
    #ifndef PETSC_USE_COMPLEX
        SETERRQ(comm, 1, "Not implemented for real scalars.");
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
    ierr = MatCreateDense(comm, PETSC_DECIDE, PETSC_DECIDE, mat_dim, mat_dim, NULL, &mat); CHKERRQ(ierr);
    /*
        Get ownership info
    */
    PetscInt    Istart, Iend, nrows;
    ierr = MatGetOwnershipRange(mat, &Istart, &Iend); CHKERRQ(ierr);
    nrows = Iend - Istart;

    Mat mat_in_loc = nullptr;
    Mat mat_out_loc = nullptr;
    Mat mat_in_hc  = nullptr;
    PetscScalar *vals_mat_in, *vals_mat_in_loc;
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
        ierr = MatDenseGetArray(mat_in_loc, &vals_mat_in_loc); CHKERRQ(ierr);

        if (hc_right == PETSC_TRUE){
            ierr = MatDenseGetArray(mat_in, &vals_mat_in); CHKERRQ(ierr);
            for (PetscInt Icol = 0; Icol < N; ++Icol)
            {
                ierr = PetscMemcpy(&vals_mat_in_loc[Icol*nrows],&vals_mat_in[Istart+Icol*M], nrows*sizeof(PetscScalar));
                CHKERRQ(ierr);
            }
            ierr = MatDenseRestoreArray(mat_in, &vals_mat_in); CHKERRQ(ierr);
            ierr = MatMatMult(mat_in_loc,mat_in_hc,MAT_REUSE_MATRIX,PETSC_DEFAULT,&mat_out_loc); CHKERRQ(ierr);
        } else
        {
            ierr = MatDenseGetArray(mat_in_hc, &vals_mat_in); CHKERRQ(ierr);
            for (PetscInt Icol = 0; Icol < M; ++Icol)
            {
                ierr = PetscMemcpy(&vals_mat_in_loc[Icol*nrows],&vals_mat_in[Istart+Icol*N], nrows*sizeof(PetscScalar));
                CHKERRQ(ierr);
            }
            ierr = MatDenseRestoreArray(mat_in, &vals_mat_in); CHKERRQ(ierr);
            ierr = MatMatMult(mat_in_loc,mat_in,MAT_REUSE_MATRIX,PETSC_DEFAULT,&mat_out_loc); CHKERRQ(ierr);
        }
        ierr = MatDenseRestoreArray(mat_in_loc, &vals_mat_in_loc); CHKERRQ(ierr);
    }

    ierr = MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    if(mat_in_loc) {ierr = MatDestroy(&mat_in_loc); CHKERRQ(ierr);}
    if(mat_in_hc ) {ierr = MatDestroy(&mat_in_hc ); CHKERRQ(ierr);}

    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "MatGetSVD"
PetscErrorCode MatGetSVD(const Mat& mat, SVD& svd)
{
    PetscErrorCode  ierr = 0;

    #ifndef PETSC_USE_COMPLEX
        SETERRQ(comm, 1, "Not implemented for real scalars.");
    #endif

    MPI_Comm comm = PetscObjectComm((PetscObject)mat);

    /*
        Determine the number of singular values to compute
        based on the size of the input matrix.
     */
    PetscInt nsv;
    MatGetSize(mat, &nsv, nullptr);

    /*
        Get the SVD of the reduced density matrix corresponding
        to the left block system
     */
    ierr = SVDCreate(comm, &svd); CHKERRQ(ierr);
    ierr = SVDSetOperator(svd, mat); CHKERRQ(ierr);
    ierr = SVDSetFromOptions(svd); CHKERRQ(ierr);
    ierr = SVDSetType(svd, SVDTRLANCZOS);
    ierr = SVDSetDimensions(svd, nsv, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);
    ierr = SVDSetWhichSingularTriplets(svd,SVD_LARGEST); CHKERRQ(ierr);
    ierr = SVDSolve(svd);CHKERRQ(ierr);
    /*
        Optional: Get some information from the solver and display it
     */
    #ifdef __TESTING
        SVDType        type;
        PetscReal      tol;
        PetscInt       maxit,its;
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
    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "SVDGetTruncatedSingularValues"
PetscErrorCode SVDGetTruncatedSingularValues(const Mat& mat_in, const SVD& svd, const PetscInt mstates, PetscScalar& error, Mat& mat)
{
    PetscErrorCode ierr = 0;

    MPI_Comm    comm = PetscObjectComm((PetscObject)mat_in);
    PetscInt    nconv, mat_nrows, mat_ncols;
    PetscInt    Istart_vec, Iend_vec, Istart_mat, Iend_mat;
    PetscReal   sigma;
    PetscScalar *vals;
    Vec         u = nullptr;
    Vec         v = nullptr;
    PetscInt    u_size = 0;

    ierr = SVDGetConverged(svd, &nconv); CHKERRQ(ierr);
    if (nconv < mstates)
    {
        char errormsg[80];
        sprintf(errormsg,"Number of converged singular values (%d) is less than mstates (%d).", nconv, mstates);
        SETERRQ(comm, 1, errormsg);
    }

    /**
        The output matrix is a dense matrix but stored as SPARSE.
     */
    ierr = MatCreate(comm, &mat); CHKERRQ(ierr);

    /*
        Setup the sizes
     */
    ierr = MatGetSize(mat_in, &mat_nrows, nullptr); CHKERRQ(ierr);
    mat_ncols = mstates;
    ierr = MatSetSizes(mat, PETSC_DECIDE, PETSC_DECIDE, mat_nrows, mat_ncols); CHKERRQ(ierr);

    /*
        Setup the matrix
     */
    ierr = MatSetFromOptions(mat); CHKERRQ(ierr);
    ierr = MatSetUp(mat); CHKERRQ(ierr);
    ierr = MatZeroEntries(mat); CHKERRQ(ierr);

    /*
        First m singular triplets contribute to the rotation matrix
     */
    ierr = MatCreateVecs(mat_in,nullptr,&u); CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(mat, &Istart_mat, &Iend_mat); CHKERRQ(ierr);
    ierr = VecGetSize(u, &u_size); CHKERRQ(ierr);
    ierr = VecGetOwnershipRange(u, &Istart_vec, &Iend_vec); CHKERRQ(ierr);

    if (!(Istart_vec == Istart_mat && Iend_vec == Iend_mat))
        SETERRQ(comm, 1, "Matrix and vector layout do not match.");

    for (PetscInt Icol = 0; Icol < mstates; ++Icol)
    {
        ierr = SVDGetSingularTriplet(svd, Icol, &sigma, u, nullptr); CHKERRQ(ierr);
        ierr = VecGetArray(u, &vals);

        /*
            Load vector as a column of mat
         */
        for (PetscInt Irow = Istart_vec; Irow < Iend_vec; ++Irow)
        {
            ierr = MatSetValue(mat, Irow, Icol, vals[Irow - Istart_vec], INSERT_VALUES); CHKERRQ(ierr);
            // printf("%f\n", PetscRealPart(vals[Irow - Istart_vec]));
        }

        ierr = VecRestoreArray(u, &vals);
    }

    ierr = MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    /*
        Remaining singular values add up to the error
     */
    error = 0;
    for (PetscInt Icol = mstates; Icol < nconv; ++Icol)
    {
        ierr = SVDGetSingularTriplet(svd, Icol, &sigma, nullptr, nullptr); CHKERRQ(ierr);
        ierr = PetscPrintf(comm, "xx %18e\n", sigma); CHKERRQ(ierr);
        error += sigma;
    }

    if(u) {ierr = VecDestroy(&u); CHKERRQ(ierr);}
    if(v) {ierr = VecDestroy(&v); CHKERRQ(ierr);}

    return ierr;
}
