#include "kron.hpp"

#undef __FUNCT__
#define __FUNCT__ "MatKron"
PetscErrorCode
MatKron(const Mat& A, const Mat& B, Mat& C, const MPI_Comm& comm)
{
    PetscErrorCode ierr = 0;

    PetscMPIInt     nprocs, rank;           // MPI comm variables
    PetscInt        M_A, N_A, M_B, N_B, M_C, N_C;
    /*
        Get information from MPI
    */
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    /*
        Put input matrices in correct state for submatrix extraction
    */
    // LINALG_TOOLS__MATASSEMBLY(A);
    // LINALG_TOOLS__MATASSEMBLY(B);
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    /*
        Determine dimensions of C and initialize
    */
    MatGetSize(A, &M_A, &N_A);
    MatGetSize(B, &M_B, &N_B);
    M_C = M_A * M_B;
    N_C = N_A * N_B;
    /*
        Setup matrix C
        TODO: maybe hardcode some setup options
    */
    ierr = MatCreate(comm, &C); CHKERRQ(ierr);
    ierr = MatSetSizes(C, PETSC_DECIDE, PETSC_DECIDE, M_C, N_C); CHKERRQ(ierr);
    ierr = MatSetFromOptions(C); CHKERRQ(ierr);
    ierr = MatSetUp(C); CHKERRQ(ierr);
    ierr = MatZeroEntries(C); CHKERRQ(ierr);

    /*
        Preallocaton Method 1: Naive, preallocate everything
     */
    /* Get the full size of the diagonal submatrix */

    // ierr = MatMPIAIJSetPreallocation(C, PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[]); CHKERRQ(ierr);

    MatKronAdd(A, B, C, comm);

    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "MatKronAdd"
PetscErrorCode
MatKronAdd(const Mat& A, const Mat& B, Mat& C, const MPI_Comm& comm)
{
    PetscErrorCode ierr = 0;

    ierr = MatKronScaleAdd(1., A, B, C, comm); CHKERRQ(ierr);

    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "MatKronScaleAdd"
PetscErrorCode
MatKronScaleAdd(const PetscScalar a, const Mat& A, const Mat& B, Mat& C, const MPI_Comm& comm)
{
    PetscErrorCode ierr = 0;

    KRON_TIMINGS_INIT(__FUNCT__);
    KRON_TIMINGS_START(__FUNCT__);

    PetscMPIInt     nprocs, rank;
    PetscInt        M_A, N_A, M_B, N_B, M_C, N_C;
    /*
        Get information from MPI
    */
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    /*
        Put input matrices in correct state for submatrix extraction
    */
    ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    /*
        Determine dimensions of C and initialize
    */
    MatGetSize(A, &M_A, &N_A);
    MatGetSize(B, &M_B, &N_B);
    M_C = M_A * M_B;
    N_C = N_A * N_B;
    /*
        Determine whether C has the correct size
    */
    PetscInt M_C_input, N_C_input;
    MatGetSize(C, &M_C_input, &N_C_input);
    if( (M_C_input != M_C) || (N_C_input != N_C) ){
        char errormsg[200];
        sprintf(errormsg, "Incorrect Matrix size: Input(%d, %d) != Expected(%d, %d)\n", M_C_input, N_C_input, M_C, N_C);
        SETERRQ(comm,1,errormsg);
    }
    /*
        Create the submatrix for A
    */
    PetscInt Istart, Iend;
    Mat submat_A;
    /*
        Determine required rows from A
    */
    MatGetOwnershipRange(C, &Istart, &Iend);
    PetscInt Astart, Aend;
    Astart = Istart/M_B;
    Aend = 1+(Iend-1)/M_B;
    PetscInt M_req_A = Aend - Astart;
    /*
        List down indices needed for submatrix A
    */
    PetscInt id_rows_A[M_req_A], id_cols_A[N_A];
    for (PetscInt Irow = Astart; Irow < Aend; ++Irow)
        id_rows_A[Irow-Astart] = Irow;
    for (PetscInt Icol = 0; Icol < N_A; ++Icol)
        id_cols_A[Icol] = Icol;
    /*
        Construct index set
    */
    IS isrow_A, iscol_A;
    ISCreateGeneral(comm, M_req_A, id_rows_A, PETSC_COPY_VALUES, &isrow_A);
    ISCreateGeneral(comm, N_A,     id_cols_A, PETSC_COPY_VALUES, &iscol_A);
    /*
        Construct submatrix_A and get local indices
    */
    MatGetSubMatrix(A, isrow_A, iscol_A, MAT_INITIAL_MATRIX, &submat_A);
    PetscInt A_sub_start, A_sub_end;
    MatGetOwnershipRange(submat_A, &A_sub_start, &A_sub_end);
    /*
        Test: Confirm correct ownership
    */
    #ifdef __PRINT_TESTS__
        for(PetscInt Irank = 0; Irank < nprocs; ++Irank){
            if(Irank==rank)
                printf("[%2d] A_submat start: %-5d end: %-5d\n", rank, A_sub_start, A_sub_end);
            MPI_Barrier(comm);
        }
        PetscPrintf(comm, "\nRow index set for A\n");
        ISView(isrow_A, PETSC_VIEWER_STDOUT_WORLD);
    #endif
    /*
        Create submatrix for B

        Get indices for submatrix B, with each process taking all elements
        TODO: Reallocate aligned
        TODO: Collect elements needed only on processor
              If processor takes no non-zero value in A, do not take values for B
    */
    Mat submat_B;
    PetscInt id_rows_B[M_B], id_cols_B[N_B];
    for (PetscInt Irow = 0; Irow < M_B; ++Irow)
        id_rows_B[Irow] = Irow;
    for (PetscInt Icol = 0; Icol < N_B; ++Icol)
        id_cols_B[Icol] = Icol;
    /*
        Construct index set
    */
    IS isrow_B, iscol_B;
    ISCreateGeneral(comm, M_B, id_rows_B, PETSC_COPY_VALUES, &isrow_B);
    ISCreateGeneral(comm, N_B, id_cols_B, PETSC_COPY_VALUES, &iscol_B);
    /*
        Construct submatrix_B and get local indices
    */
    MatGetSubMatrix(B, isrow_B, iscol_B, MAT_INITIAL_MATRIX, &submat_B);
    PetscInt B_sub_start, B_sub_end;
    MatGetOwnershipRange(submat_B, &B_sub_start, &B_sub_end);
    /*
        Map ownership
        Input: the row INDEX in the global matrix A/B
        Output: the corresponding row index in the locally-owned rows of submatrix A/B
    */
    #define ROW_MAP_A(INDEX) ((INDEX) - Astart + A_sub_start)
    #define ROW_MAP_B(INDEX) ((INDEX) + B_sub_start)
    /*
        Submatrix constructions offsets the starting column
        Input: the corresponding column index in the locally-owned submatrix A/B
        Output: the column INDEX in the global matrix A/B
    */
    #define COL_MAP_A(INDEX) ((INDEX) - N_A * (nprocs - 1) )
    #define COL_MAP_B(INDEX) ((INDEX) - N_B * (nprocs - 1) )
    /*
        Determine non-zero indices of submatrices
    */
    const PetscInt*     cols_A;
    const PetscScalar*  vals_A;
    const PetscInt*     cols_B;
    const PetscScalar*  vals_B;
    PetscInt            ncols_A, ncols_B;
    PetscInt            Arow, Brow;
    PetscInt            ncols_C;
    /*
        IMPLEMENTATION OPTIONS:
        Load A and B one row at a time (check)
        Pre-allocate an array for calculating a row of C using
            the maximum possible number of columns scanned through all A and B
        Load B all at once and destroy submatrix
            Find a way to get and set values using the entire submatrix B
    */
    // #define __SEQ_ORDER__
    #ifdef __SEQ_ORDER__

    for(PetscInt Irank = 0; Irank < nprocs; ++Irank){
        if(Irank==rank)

    #endif
    // #define __NO_KRON__
    #ifndef __NO_KRON__



    #define __KRONLOOP "  KronLoop"
    KRON_TIMINGS_INIT(__KRONLOOP);

    #define __MATSETVALUES "    MatSetValues"
    KRON_TIMINGS_ACCUM_INIT(__MATSETVALUES);

    #define __CALC_VALUES "    CalculateKronValues"
    KRON_TIMINGS_ACCUM_INIT(__CALC_VALUES);

    KRON_TIMINGS_START(__KRONLOOP);

    #define KRON_OPTIMIZATION

    #ifdef KRON_OPTIMIZATION
    PetscInt        max_ncols_C = N_A * N_B;
    PetscInt*       cols_C = new PetscInt[max_ncols_C];
    PetscScalar*    vals_C = new PetscScalar[max_ncols_C];
    PetscInt        j_A, j_B, last;
    for (PetscInt Irow = Istart; Irow < Iend; ++Irow)
    {
        Arow = Irow / M_B;
        Brow = Irow % M_B;

        MatGetRow(submat_A, ROW_MAP_A(Arow), &ncols_A, &cols_A, &vals_A);
        MatGetRow(submat_B, ROW_MAP_B(Brow), &ncols_B, &cols_B, &vals_B);

        ncols_C = ncols_A * ncols_B;

        KRON_TIMINGS_ACCUM_START(__CALC_VALUES);
        for (j_A = 0; j_A < ncols_A; ++j_A)
        {
            for (j_B = 0; j_B < ncols_B; ++j_B)
            {
                cols_C [ j_A * ncols_B + j_B ] = COL_MAP_A(cols_A[j_A]) * N_B + COL_MAP_B(cols_B[j_B]);
            }
        }
        for (j_A = 0; j_A < ncols_A; ++j_A)
        {
            for (j_B = 0; j_B < ncols_B; ++j_B)
            {
                vals_C [ j_A * ncols_B + j_B ] = a * vals_A[j_A] * vals_B[j_B];
            }
        }
        KRON_TIMINGS_ACCUM_END(__CALC_VALUES);

        KRON_TIMINGS_ACCUM_START(__MATSETVALUES);
        MatSetValues(C, 1, &Irow, ncols_C, cols_C, vals_C, ADD_VALUES );
        KRON_TIMINGS_ACCUM_END(__MATSETVALUES);

        MatRestoreRow(submat_B, ROW_MAP_B(Brow), &ncols_B, &cols_B, &vals_B);
        MatRestoreRow(submat_A, ROW_MAP_A(Arow), &ncols_A, &cols_A, &vals_A);
    };

    #else

    PetscInt        max_ncols_C = N_A * N_B;
    PetscInt*       cols_C = new PetscInt[max_ncols_C];
    PetscScalar*    vals_C = new PetscScalar[max_ncols_C];
    if (a == 1.)
    {
        for (PetscInt Irow = Istart; Irow < Iend; ++Irow)
        {
            Arow = Irow/M_B;
            Brow = Irow % M_B;

            MatGetRow(submat_A, ROW_MAP_A(Arow), &ncols_A, &cols_A, &vals_A);
            MatGetRow(submat_B, ROW_MAP_B(Brow), &ncols_B, &cols_B, &vals_B);

            ncols_C = ncols_A * ncols_B;

            KRON_TIMINGS_ACCUM_START(__CALC_VALUES);
            for (int j_A = 0; j_A < ncols_A; ++j_A)
            {
                for (int j_B = 0; j_B < ncols_B; ++j_B)
                {
                    cols_C [ j_A * ncols_B + j_B ] = COL_MAP_A(cols_A[j_A]) * N_B + COL_MAP_B(cols_B[j_B]);
                    vals_C [ j_A * ncols_B + j_B ] = vals_A[j_A] * vals_B[j_B];
                }
            }
            KRON_TIMINGS_ACCUM_END(__CALC_VALUES);

            KRON_TIMINGS_ACCUM_START(__MATSETVALUES);
            MatSetValues(C, 1, &Irow, ncols_C, cols_C, vals_C, ADD_VALUES );
            KRON_TIMINGS_ACCUM_END(__MATSETVALUES);

            MatRestoreRow(submat_B, ROW_MAP_B(Brow), &ncols_B, &cols_B, &vals_B);
            MatRestoreRow(submat_A, ROW_MAP_A(Arow), &ncols_A, &cols_A, &vals_A);
        };
    }
    else
    {
        for (PetscInt Irow = Istart; Irow < Iend; ++Irow)
        {
            Arow = Irow/M_B;
            Brow = Irow % M_B;

            MatGetRow(submat_A, ROW_MAP_A(Arow), &ncols_A, &cols_A, &vals_A);
            MatGetRow(submat_B, ROW_MAP_B(Brow), &ncols_B, &cols_B, &vals_B);

            ncols_C = ncols_A * ncols_B;

            KRON_TIMINGS_ACCUM_START(__CALC_VALUES);
            for (int j_A = 0; j_A < ncols_A; ++j_A)
            {
                for (int j_B = 0; j_B < ncols_B; ++j_B)
                {
                    cols_C [ j_A * ncols_B + j_B ] = COL_MAP_A(cols_A[j_A]) * N_B + COL_MAP_B(cols_B[j_B]);
                    vals_C [ j_A * ncols_B + j_B ] = a * vals_A[j_A] * vals_B[j_B];
                }
            }
            // PetscPrintf(PETSC_COMM_WORLD,"\n");
            KRON_TIMINGS_ACCUM_END(__CALC_VALUES);

            KRON_TIMINGS_ACCUM_START(__MATSETVALUES);
            MatSetValues(C, 1, &Irow, ncols_C, cols_C, vals_C, ADD_VALUES );
            KRON_TIMINGS_ACCUM_END(__MATSETVALUES);

            MatRestoreRow(submat_B, ROW_MAP_B(Brow), &ncols_B, &cols_B, &vals_B);
            MatRestoreRow(submat_A, ROW_MAP_A(Arow), &ncols_A, &cols_A, &vals_A);
        };
    }
    #endif

    delete [] cols_C;
    delete [] vals_C;

    KRON_TIMINGS_END(__KRONLOOP);
    #undef __KRONLOOP

    KRON_TIMINGS_ACCUM_PRINT(__CALC_VALUES);
    #undef __CALC_VALUES

    KRON_TIMINGS_ACCUM_PRINT(__MATSETVALUES);
    #undef __MATSETVALUES

    #endif // __NO_KRON__

    #ifdef __SEQ_ORDER__
        MPI_Barrier(comm);
    }
    #endif
    #undef __SEQ_ORDER__

    #undef ROW_MAP_A
    #undef ROW_MAP_B


    /*
        Write submatrices to file
    */
    // #define __KRON_WRITE_SUBMAT__
    #ifdef __KRON_WRITE_SUBMAT__
        PetscViewer writer = nullptr;
        #define WRITE(MAT,FILE) \
            MatAssemblyBegin(MAT, MAT_FLUSH_ASSEMBLY);\
            MatAssemblyEnd(MAT, MAT_FINAL_ASSEMBLY);\
            PetscViewerBinaryOpen(PETSC_COMM_WORLD,FILE,FILE_MODE_WRITE,&writer);\
            MatView(MAT, writer);\
            PetscViewerDestroy(&writer);

        WRITE(submat_A,"test_kron/submat_A.dat")
        WRITE(submat_B,"test_kron/submat_B.dat")
        #undef WRITE
        PetscViewerDestroy(&writer);
    #endif
    #undef __KRON_WRITE_SUBMAT__

    KRON_TIMINGS_END(__FUNCT__);
    KRON_TIMINGS_PRINT(" ");

    /*
        This MatAssembly may be clearing up the unused
        preallocation.
        Solution: Explicitly fill zeros.
    */
    MatAssemblyBegin(C, MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(C, MAT_FLUSH_ASSEMBLY);

    MatDestroy(&submat_A);
    MatDestroy(&submat_B);
    return ierr;
}
