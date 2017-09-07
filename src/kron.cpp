#include "kron.hpp"


#undef __FUNCT__
#define __FUNCT__ "InitialChecks"
PetscErrorCode InitialChecks(
    const std::vector<PetscScalar>& a,
    const std::vector<Mat>& A,
    const std::vector<Mat>& B,
    Mat& C,
    MPI_Comm& comm,
    PetscInt& nterms,
    std::vector<PetscInt>& M_A,
    std::vector<PetscInt>& N_A,
    std::vector<PetscInt>& M_B,
    std::vector<PetscInt>& N_B,
    PetscInt& M_C,
    PetscInt& N_C)
{
    PetscErrorCode ierr;

    /*
        INITIAL CHECKPOINTS

        Check that a vectors a, A and B all have the same non-zero lengths
    */
    nterms = a.size();

    if ((size_t)nterms != A.size()) SETERRQ2(comm, 1,
        "Incompatible length of a and A: %d != %d\n", a.size(), A.size());

    if (A.size() != B.size()) SETERRQ2(comm, 1,
        "Incompatible length of A and B: %d != %d\n", A.size(), B.size());

    if (nterms < 1) SETERRQ(comm, 1,
        "A and B must each contain at least one matrix.\n");
    /*
        Collect matrix sizes
    */
    // std::vector<PetscInt> M_A(nterms), N_A(nterms), M_B(nterms), N_B(nterms);
    M_A.resize(nterms);
    N_A.resize(nterms);
    M_B.resize(nterms);
    N_B.resize(nterms);
    PetscInt M_C_temp, N_C_temp;

    for (PetscInt i = 0; i < nterms; ++i)
    {
        ierr = MatGetSize(A[i], M_A.data()+i, N_A.data()+i);
        ierr = MatGetSize(B[i], M_B.data()+i, N_B.data()+i);
    }
    /*
        Check whether the resulting sizes of the tensor products are equal
    */
    M_C = M_A[0] * M_B[0];
    N_C = N_A[0] * N_B[0];

    for (PetscInt i = 1; i < nterms; ++i)
    {
        M_C_temp = M_A[i] * M_B[i];
        N_C_temp = N_A[i] * N_B[i];
        if (M_C_temp != M_C) SETERRQ3(comm, 1,
            "Incompatible resultant matrix dimension M_C "
            "between entries %d and 0: "
            "%d != %d", i, M_C_temp, M_C);
        if (N_C_temp != N_C) SETERRQ3(comm, 1,
            "Incompatible resultant matrix dimension N_C "
            "between entries %d and 0: "
            "%d != %d", i, N_C_temp, N_C);
    }
    /*
        NOTE: One simplifying assumption is that the sizes of all A matrices
                and all B matrices are already equal.
        TODO: Relax this assumption later
    */
    for (PetscInt i = 1; i < nterms; ++i)
    {
        if (M_A[i] != M_A[0]) SETERRQ3(comm, 1,
            "Incompatible matrix shapes for M between A[%d] and A[0]: %d != %d",
            i, M_A[i], M_A[0]);
        if (N_A[i] != N_A[0]) SETERRQ3(comm, 1,
            "Incompatible matrix shapes for N between A[%d] and A[0]: %d != %d",
            i, N_A[i], N_A[0]);
        if (M_B[i] != M_B[0]) SETERRQ3(comm, 1,
            "Incompatible matrix shapes for M between B[%d] and B[0]: %d != %d",
            i, M_B[i], M_B[0]);
        if (N_B[i] != N_B[0]) SETERRQ3(comm, 1,
            "Incompatible matrix shapes for N between B[%d] and B[0]: %d != %d",
            i, N_B[i], N_B[0]);
    }

    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "GetSubmatrix"
PetscErrorCode GetSubmatrix(
    const std::vector<Mat>&       A,
    const std::vector<PetscInt>&  N_A,
    const PetscInt&               nterms,
    const PetscInt&               M_req_A,
    const PetscInt                *id_rows_A,
    std::vector<Mat>&       submat_A,
    PetscInt&               A_sub_start,
    PetscInt&               A_sub_end)
{
    PetscErrorCode ierr = 0;
    PetscBool      assembled;
    MPI_Comm comm = PETSC_COMM_WORLD;

    PetscInt *id_cols_A;
    ierr = PetscMalloc1(N_A[0], &id_cols_A); CHKERRQ(ierr);
    for (PetscInt Icol = 0; Icol < N_A[0]; ++Icol)
        id_cols_A[Icol] = Icol;

    IS isrow_A = nullptr, iscol_A = nullptr;
    PetscInt A_sub_start_temp, A_sub_end_temp;

    for (PetscInt i = 0; i < nterms; ++i)
    {
        LINALG_TOOLS__MATASSEMBLY_FINAL(A[i]); /* Removes segfault issue*/
        /*
            Checkpoint column sizes
         */
        if( N_A[i]!=N_A[0])
            SETERRQ(comm, 1, "Shapes of A matrices are not equal.");

        ierr = ISCreateGeneral(comm, M_req_A, id_rows_A, PETSC_USE_POINTER, &isrow_A); CHKERRQ(ierr);
        ierr = ISCreateGeneral(comm, N_A[i],  id_cols_A, PETSC_USE_POINTER, &iscol_A); CHKERRQ(ierr);
        /*
            Construct submatrix_A and get local indices
        */
        submat_A[i] = nullptr;
        ierr = MatGetSubMatrix(A[i], isrow_A, iscol_A, MAT_INITIAL_MATRIX, submat_A.data()+i); CHKERRQ(ierr);
        ierr = MatGetOwnershipRange(submat_A[i], &A_sub_start, &A_sub_end); CHKERRQ(ierr);
        /*
            Checkpoint row ownership ranges
        */
        if (i && (A_sub_start_temp != A_sub_start) && (A_sub_end_temp != A_sub_end)){
            SETERRQ(comm, 1, "Shapes of A matrices are not equal.");
        } else {
            A_sub_start_temp = A_sub_start;
            A_sub_end_temp = A_sub_end;
        }
        /*
            Destroy index set
        */
        if(isrow_A) ierr = ISDestroy(&isrow_A); CHKERRQ(ierr); isrow_A = nullptr;
        if(iscol_A) ierr = ISDestroy(&iscol_A); CHKERRQ(ierr); iscol_A = nullptr;
    }

    ierr = PetscFree(id_cols_A); CHKERRQ(ierr);

    return ierr;
}



#undef __FUNCT__
#define __FUNCT__ "MatKronProdSum"
PetscErrorCode MatKronProdSum(
    const std::vector<PetscScalar>& a,
    const std::vector<Mat>& A,
    const std::vector<Mat>& B,
    Mat& C,
    const PetscBool prealloc)
{
    PetscErrorCode ierr = 0;
    KRON_TIMINGS_INIT(__FUNCT__);
    KRON_TIMINGS_START(__FUNCT__);
    /*
        Get information from MPI
    */
    PetscMPIInt     nprocs, rank;
    MPI_Comm comm = PETSC_COMM_WORLD;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    /*
        Perform initial checks and get sizes
    */
    std::vector<PetscInt>   M_A, N_A, M_B, N_B;
    PetscInt                nterms, M_C, N_C;
    ierr = InitialChecks(a,A,B,C,comm,nterms,M_A,N_A,M_B,N_B,M_C,N_C); CHKERRQ(ierr);
    /*
        Guess the local ownership of resultant matrix C
    */
    PetscInt remrows = M_C % nprocs;
    PetscInt locrows = M_C / nprocs;
    PetscInt Istart = locrows * rank;

    if (rank < remrows){
        locrows += 1;
        Istart += rank;
    } else {
        Istart += remrows;
    }

    PetscInt Iend = Istart + locrows;

    #define KRON_SUBMATRIX "        Kron: Submatrix collection"
    KRON_PS_TIMINGS_INIT(KRON_SUBMATRIX)
    KRON_PS_TIMINGS_START(KRON_SUBMATRIX)
    /*

        SUBMATRIX A

        Acquire the submatrices of local and nonlocal rows needed to build
        the local rows of C

        Determine the required rows from A_i
    */
    PetscInt Astart = Istart/M_B[0];
    PetscInt Aend = 1+(Iend-1)/M_B[0];
    PetscInt M_req_A = Aend - Astart;
    /*
        Build the submatrices for each term
    */
    std::vector<Mat>    submat_A(nterms);
    PetscInt            A_sub_start, A_sub_end;
    PetscInt            *id_rows_A;
    /*
        Assumes equal shapes for A and B matrices
    */
    ierr = PetscMalloc1(M_req_A, &id_rows_A); CHKERRQ(ierr);
    /*
        Construct index set
    */
    for (PetscInt Irow = Astart; Irow < Aend; ++Irow)
        id_rows_A[Irow-Astart] = Irow;
    /*
        Obtain submatrices
    */
    ierr = GetSubmatrix(A,N_A,nterms,M_req_A,id_rows_A,submat_A,A_sub_start,A_sub_end); CHKERRQ(ierr);
    /*
        Destroy row indices
    */
    ierr = PetscFree(id_rows_A); CHKERRQ(ierr);
    /*

        SUBMATRIX B

        Acquire the submatrices of local and nonlocal rows needed to build
        the local rows of C

        Build the submatrices for each term

    */
    std::vector<Mat>      submat_B(nterms);
    PetscInt B_sub_start, B_sub_end;
    /*
        NOTE: Assumes equal shapes for A and B matrices
    */
    PetscInt *id_rows_B;

    ierr = PetscMalloc1(M_B[0], &id_rows_B); CHKERRQ(ierr);

    for (PetscInt Irow = 0; Irow < M_B[0]; ++Irow)
        id_rows_B[Irow] = Irow;

    ierr = GetSubmatrix(B,N_B,nterms,M_B[0],id_rows_B,submat_B,B_sub_start,B_sub_end); CHKERRQ(ierr);

    ierr = PetscFree(id_rows_B); CHKERRQ(ierr);

    KRON_PS_TIMINGS_END(KRON_SUBMATRIX)
    #undef KRON_SUBMATRIX
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
    #define COL_MAP_A(INDEX) ((INDEX) - N_A[i] * (nprocs - 1) )
    #define COL_MAP_B(INDEX) ((INDEX) - N_B[i] * (nprocs - 1) )
    /*

        PREALLOCATION

        Run through all terms and calculate an overestimated preallocation
        by adding all the non-zeros needed for each row.
    */
    if(prealloc)
    {
        #define KRON_SUBMATRIX "        Kron: Preallocation"
        KRON_PS_TIMINGS_INIT(KRON_SUBMATRIX)
        KRON_PS_TIMINGS_START(KRON_SUBMATRIX)

        ierr = MatCreate(comm, &C); CHKERRQ(ierr);
        ierr = MatSetSizes(C, PETSC_DECIDE, PETSC_DECIDE, M_C, N_C); CHKERRQ(ierr);
        ierr = MatSetFromOptions(C); CHKERRQ(ierr);

        #ifdef __KRON_DENSE_PREALLOCATION
        /*
            Naive / dense preallocation
        */
        ierr = MatMPIAIJSetPreallocation(C, locrows, NULL, M_C - locrows, NULL); CHKERRQ(ierr);
        ierr = MatSeqAIJSetPreallocation(C, M_C, NULL); CHKERRQ(ierr);
        #else
        /*
            More accurate preallocation (slightly overestimated)
        */
        PetscInt *d_nnz, *o_nnz, ncols_A, ncols_B, ncols_C_max;
        PetscInt Arow, Brow, Ccol, diag;
        const PetscInt *cols_A, *cols_B;
        ierr = PetscMalloc1(locrows,&d_nnz); CHKERRQ(ierr);
        ierr = PetscMalloc1(locrows,&o_nnz); CHKERRQ(ierr);

        PetscInt tot_entries = 0;
        for (PetscInt Irow = Istart; Irow < Iend; ++Irow)
        {
            Arow = Irow / M_B[0];
            Brow = Irow % M_B[0];

            diag        = 0;
            ncols_C_max = 0;
            for (PetscInt i = 0; i < nterms; ++i)
            {
                ierr = MatGetRow(submat_A[i], ROW_MAP_A(Arow), &ncols_A, &cols_A, nullptr); CHKERRQ(ierr);
                ierr = MatGetRow(submat_B[i], ROW_MAP_B(Brow), &ncols_B, &cols_B, nullptr); CHKERRQ(ierr);

                ncols_C_max += ncols_A * ncols_B;
                for (PetscInt j_A = 0; j_A < ncols_A; ++j_A)
                {
                    for (PetscInt j_B = 0; j_B < ncols_B; ++j_B)
                    {
                        Ccol = COL_MAP_A(cols_A[j_A]) * N_B[i] + COL_MAP_B(cols_B[j_B]);
                        if ( Istart <= Ccol && Ccol < Iend ) diag += 1;
                    }
                }

                ierr = MatRestoreRow(submat_A[i], ROW_MAP_A(Arow), &ncols_A, &cols_A, nullptr); CHKERRQ(ierr);
                ierr = MatRestoreRow(submat_B[i], ROW_MAP_B(Brow), &ncols_B, &cols_B, nullptr); CHKERRQ(ierr);
            }
            d_nnz[Irow-Istart] = std::min(diag, locrows);
            o_nnz[Irow-Istart] = std::min(ncols_C_max - diag, M_C - locrows);
            tot_entries += ncols_C_max;
        }

        ierr = MatMPIAIJSetPreallocation(C, -1, d_nnz, -1, o_nnz); CHKERRQ(ierr);
        ierr = MatSeqAIJSetPreallocation(C, -1, d_nnz); CHKERRQ(ierr);

        ierr = PetscFree(d_nnz); CHKERRQ(ierr);
        ierr = PetscFree(o_nnz); CHKERRQ(ierr);

        #ifdef __KRON_PS_TIMINGS // print info on expected sparsity
            // printf("[%d] %d \n", rank, tot_entries);
            PetscInt tot_entries_reduced;
            MPI_Reduce( &tot_entries, &tot_entries_reduced, 1, MPI_INT, MPI_SUM, 0, comm);
            PetscPrintf(comm, "%20s Nonzeros: %d/(%-d)^2 = %f%%\n", " ",tot_entries_reduced, M_C,
                100.0*(double)tot_entries_reduced/((double)(M_C) * (double)(M_C)));
        #endif
        #endif
        /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

            MATRIX OPTIONS

            You know each process will only set values for its own rows,
            will generate an error if any process sets values for another process.
            This avoids all reductions in the MatAssembly routines and thus
            improves performance for very large process counts.
        */
        ierr = MatSetOption(C, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE);
        /*

            You know each process will only zero its own rows.
            This avoids all reductions in the zero row routines and thus
            improves performance for very large process counts.
        */
        ierr = MatSetOption(C, MAT_NO_OFF_PROC_ZERO_ROWS, PETSC_TRUE);
        /*

            Set to PETSC_TRUE indicates entries destined for other processors should be dropped,
            rather than stashed. This is useful if you know that the "owning" processor is also
            always generating the correct matrix entries, so that PETSc need not transfer
            duplicate entries generated on another processor.
        */
        ierr = MatSetOption(C, MAT_IGNORE_OFF_PROC_ENTRIES, PETSC_TRUE);
        /*

            indicates when MatZeroRows() is called the zeroed entries are kept in the nonzero structure
            NOTE: significant improvement not yet observed
        */
        ierr = MatSetOption(C, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);
        /*

            set to PETSC_TRUE indicates that any add or insertion that would generate a new entry
            in the nonzero structure instead produces an error. (Currently supported for
            AIJ and BAIJ formats only.) If this option is set then the MatAssemblyBegin/End()
            processes has one less global reduction
         */
        ierr = MatSetOption(C, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE);
        /*

            set to PETSC_TRUE indicates that any add or insertion that would generate a new entry
            that has not been preallocated will instead produce an error. (Currently supported
            for AIJ and BAIJ formats only.) This is a useful flag when debugging matrix memory
            preallocation. If this option is set then the MatAssemblyBegin/End() processes has one
            less global reduction
         */
        ierr = MatSetOption(C, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE);
        /*

            for AIJ/IS matrices this will stop zero values from creating a zero location in the matrix
        */
        ierr = MatSetOption(C, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);

        /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

        /*
            Check ownership guess
        */
        if(nprocs > 1){
            ierr = MatGetOwnershipRange(C, &Istart, &Iend);
            PetscInt Irows = Iend - Istart;
            if(Irows != locrows) {
                char errormsg[200];
                sprintf(errormsg,"WRONG GUESS: Irows=%d  locrows=%d\n", Irows,locrows);
                SETERRQ(comm, 1, errormsg);
            }
        }

        KRON_PS_TIMINGS_END(KRON_SUBMATRIX)
        #undef KRON_SUBMATRIX
    }
    else
    {
        ierr = MatZeroEntries(C); CHKERRQ(ierr);
    }
    /*
        CALCULATE ENTRIES
    */
    #define __KRONLOOP     "        KronLoop"
    KRON_PS_TIMINGS_INIT(__KRONLOOP);

    #define __MATSETVALUES "            MatSetValues"
    KRON_PS_TIMINGS_ACCUM_INIT(__MATSETVALUES);

    #define __CALC_VALUES  "            CalculateKronValues"
    KRON_PS_TIMINGS_ACCUM_INIT(__CALC_VALUES);

    KRON_PS_TIMINGS_START(__KRONLOOP);

    const PetscInt*     cols_A[nterms];
    const PetscScalar*  vals_A[nterms];
    const PetscInt*     cols_B[nterms];
    const PetscScalar*  vals_B[nterms];
    PetscInt            ncols_A[nterms], ncols_B[nterms];
    PetscInt            Arow, Brow;
    PetscInt            ncols_C[nterms];

    PetscInt        max_ncols_C = N_A[0] * N_B[0]; /* Assumes same size of matrices in A and B */
    PetscInt*       cols_C;
    PetscScalar*    vals_C;
    ierr = PetscMalloc1(max_ncols_C,&cols_C); CHKERRQ(ierr);
    ierr = PetscMalloc1(max_ncols_C,&vals_C); CHKERRQ(ierr);

    // #define __KRON_SWAP_LOOP
    #ifdef __KRON_SWAP_LOOP
    /*
     *  Outer loop through operators, inner loop through rows of each operator
     */
    for (PetscInt i = 0; i < nterms; ++i)
    {
        for (PetscInt Irow = Istart; Irow < Iend; ++Irow)
        {
            Arow = Irow / M_B[0];
            Brow = Irow % M_B[0];

            ierr = MatGetRow(submat_A[i], ROW_MAP_A(Arow), &ncols_A[i], &cols_A[i], &vals_A[i]); CHKERRQ(ierr);
            ierr = MatGetRow(submat_B[i], ROW_MAP_B(Brow), &ncols_B[i], &cols_B[i], &vals_B[i]); CHKERRQ(ierr);
            ncols_C[i] = ncols_A[i] * ncols_B[i];
            /*
                Assumes that matrices in A and in B have the same shapes
            */
            KRON_PS_TIMINGS_ACCUM_START(__CALC_VALUES);
            for (PetscInt j_A = 0; j_A < ncols_A[i]; ++j_A)
            {
                for (PetscInt j_B = 0; j_B < ncols_B[i]; ++j_B)
                {
                    cols_C [ j_A * ncols_B[i] + j_B ] = COL_MAP_A(cols_A[i][j_A]) * N_B[i] + COL_MAP_B(cols_B[i][j_B]);
                    vals_C [ j_A * ncols_B[i] + j_B ] = a[i] * vals_A[i][j_A] * vals_B[i][j_B];
                }
            }
            KRON_PS_TIMINGS_ACCUM_END(__CALC_VALUES);

            KRON_PS_TIMINGS_ACCUM_START(__MATSETVALUES);
            ierr = MatSetValues(C, 1, &Irow, ncols_C[i], cols_C, vals_C, ADD_VALUES ); CHKERRQ(ierr);
            KRON_PS_TIMINGS_ACCUM_END(__MATSETVALUES);

            ierr = MatRestoreRow(submat_B[i], ROW_MAP_B(Brow), &ncols_B[i], &cols_B[i], &vals_B[i]); CHKERRQ(ierr);
            ierr = MatRestoreRow(submat_A[i], ROW_MAP_A(Arow), &ncols_A[i], &cols_A[i], &vals_A[i]); CHKERRQ(ierr);
        }
    }
    /*
     *
     */
    #else
    /*
     *  Default behavior:
     *  Outer loop through rows, inner loop through operators.
     */
    for (PetscInt Irow = Istart; Irow < Iend; ++Irow)
    {
        Arow = Irow / M_B[0];
        Brow = Irow % M_B[0];
        for (PetscInt i = 0; i < nterms; ++i)
        {
            ierr = MatGetRow(submat_A[i], ROW_MAP_A(Arow), &ncols_A[i], &cols_A[i], &vals_A[i]); CHKERRQ(ierr);
            ierr = MatGetRow(submat_B[i], ROW_MAP_B(Brow), &ncols_B[i], &cols_B[i], &vals_B[i]); CHKERRQ(ierr);
            ncols_C[i] = ncols_A[i] * ncols_B[i];
        }
        /*
            Assume that matrices in A and in B have the same shapes
        */
        for (PetscInt i = 0; i < nterms; ++i)
        {
            KRON_PS_TIMINGS_ACCUM_START(__CALC_VALUES);
            for (PetscInt j_A = 0; j_A < ncols_A[i]; ++j_A)
            {
                for (PetscInt j_B = 0; j_B < ncols_B[i]; ++j_B)
                {
                    cols_C [ j_A * ncols_B[i] + j_B ] = COL_MAP_A(cols_A[i][j_A]) * N_B[i] + COL_MAP_B(cols_B[i][j_B]);
                    vals_C [ j_A * ncols_B[i] + j_B ] = a[i] * vals_A[i][j_A] * vals_B[i][j_B];
                }
            }
            KRON_PS_TIMINGS_ACCUM_END(__CALC_VALUES);

            KRON_PS_TIMINGS_ACCUM_START(__MATSETVALUES);
            ierr = MatSetValues(C, 1, &Irow, ncols_C[i], cols_C, vals_C, ADD_VALUES ); CHKERRQ(ierr);
            KRON_PS_TIMINGS_ACCUM_END(__MATSETVALUES);
        }

        for (PetscInt i = 0; i < nterms; ++i)
        {
            ierr = MatRestoreRow(submat_B[i], ROW_MAP_B(Brow), &ncols_B[i], &cols_B[i], &vals_B[i]); CHKERRQ(ierr);
            ierr = MatRestoreRow(submat_A[i], ROW_MAP_A(Arow), &ncols_A[i], &cols_A[i], &vals_A[i]); CHKERRQ(ierr);
        };
    }

    #endif


    KRON_PS_TIMINGS_ACCUM_PRINT(__CALC_VALUES);
    #undef __CALC_VALUES

    KRON_PS_TIMINGS_ACCUM_PRINT(__MATSETVALUES);
    #undef __MATSETVALUES

    KRON_PS_TIMINGS_END(__KRONLOOP);
    #undef __KRONLOOP

    ierr = PetscFree(cols_C); CHKERRQ(ierr);
    ierr = PetscFree(vals_C); CHKERRQ(ierr);
    /*
        Destroy submatrices
    */
    for (PetscInt i = 0; i < nterms; ++i){
        if(submat_A.data()+i) ierr = MatDestroy(submat_A.data()+i); CHKERRQ(ierr);
    }
    for (PetscInt i = 0; i < nterms; ++i){
        if(submat_B.data()+i) ierr = MatDestroy(submat_B.data()+i); CHKERRQ(ierr);
    }

    #undef ROW_MAP_A
    #undef ROW_MAP_B
    #undef COL_MAP_A
    #undef COL_MAP_B

    KRON_TIMINGS_END(__FUNCT__);
    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "MatKronProd"
PetscErrorCode MatKronProd(const PetscScalar& a, const Mat& A, const Mat& B, Mat& C)
{
    PetscErrorCode ierr = 0;

    std::vector<PetscScalar>    a_vec = {a};
    std::vector<Mat>            A_vec = {A};
    std::vector<Mat>            B_vec = {B};

    ierr =  MatKronProdSum(a_vec, A_vec, B_vec, C,PETSC_TRUE); CHKERRQ(ierr);

    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "MatKronProdSumIdx_copy"
PetscErrorCode MatKronProdSumIdx_copy(
    const std::vector<PetscScalar>& a,
    const std::vector<Mat>& A,
    const std::vector<Mat>& B,
    Mat& C,
    const std::vector<PetscInt> idx)
{
    PetscErrorCode ierr = 0;

    PetscMPIInt     nprocs, rank;
    MPI_Comm comm = PETSC_COMM_WORLD;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    /* Verify that idx are all valid
     * Assumes A and B matrices have the same sizes
     */
    PetscInt M_A, M_B, M_C;
    ierr = MatGetSize(A[0], &M_A, nullptr);
    ierr = MatGetSize(B[0], &M_B, nullptr);

    M_C = M_A * M_B;
    for (auto id: idx)
        if (id >= M_C)
            SETERRQ1(comm,1,"Invalid key: %d", id);

    /* Calculate full matrix */

    Mat C_temp = nullptr;
    ierr = MatKronProdSum(a, A, B, C_temp, PETSC_TRUE); CHKERRQ(ierr);

    /* Guess final row ownership ranges */

    PetscInt M_C_final = idx.size();
    PetscInt N_C_final = idx.size();
    PetscInt remrows = M_C_final % nprocs;
    PetscInt locrows = M_C_final / nprocs;
    PetscInt Istart = locrows * rank;

    if (rank < remrows){
        locrows += 1;
        Istart += rank;
    } else {
        Istart += remrows;
    }

    PetscInt Iend = Istart + locrows;

    /* Construct row indices */

    PetscInt *id_rows;
    ierr = PetscMalloc1(locrows,    &id_rows); CHKERRQ(ierr);
    for (PetscInt Irow = Istart; Irow < Iend; ++Irow)
        id_rows[Irow-Istart] = idx[Irow];

    IS is_rows = nullptr;
    ierr = ISCreateGeneral(comm, locrows, id_rows, PETSC_USE_POINTER, &is_rows); CHKERRQ(ierr);

    /* Construct column indices */

    PetscInt *id_cols;
    ierr = PetscMalloc1(idx.size(), &id_cols); CHKERRQ(ierr);
    for (PetscInt Icol = 0; Icol < idx.size(); ++Icol)
        id_cols[Icol] = idx[Icol];

    IS is_cols = nullptr;
    ierr = ISCreateGeneral(comm, idx.size(), id_cols, PETSC_USE_POINTER, &is_cols); CHKERRQ(ierr);

    /* Get submatrix based on desired indices */

    PetscBool assembled;
    LINALG_TOOLS__MATASSEMBLY_FINAL(C_temp);

    Mat C_sub;
    ierr = MatGetSubMatrix(C_temp, is_rows, is_cols, MAT_INITIAL_MATRIX, &C_sub); CHKERRQ(ierr);

    /* Local to global mapping */

    PetscInt col_map_shift = - N_C_final * (nprocs - 1);
    #define COL_MAP(INDEX) ((INDEX) + col_map_shift )

    /* Create a new square matrix and populate with elements shifted with COL_MAP */

    ierr = MatCreate(comm, &C); CHKERRQ(ierr);
    ierr = MatSetSizes(C, PETSC_DECIDE, PETSC_DECIDE, M_C_final, N_C_final); CHKERRQ(ierr);
    ierr = MatSetFromOptions(C); CHKERRQ(ierr);

    /* Preallocation */

    const PetscInt    *cols;
    const PetscScalar *vals;
    PetscInt *cols_shifted, ncols;

    PetscInt *d_nnz, *o_nnz;

    ierr = PetscMalloc1(locrows, &d_nnz); CHKERRQ(ierr);
    ierr = PetscMalloc1(locrows, &o_nnz); CHKERRQ(ierr);

    for (PetscInt Irow = Istart; Irow < Iend; ++Irow)
    {
        ierr = MatGetRow(C_sub, Irow, &ncols, &cols, nullptr);

        d_nnz[Irow-Istart] = 0;
        o_nnz[Irow-Istart] = 0;

        for (PetscInt Icol = 0; Icol < ncols; ++Icol){
            if ( Istart <= COL_MAP(cols[Icol]) && COL_MAP(cols[Icol]) < Iend ){
                d_nnz[Irow-Istart] += 1;
            } else {
                o_nnz[Irow-Istart] += 1;
            }
        }

        ierr = MatRestoreRow(C_sub, Irow, &ncols, &cols, nullptr);
    }

    ierr = MatMPIAIJSetPreallocation(C, -1, d_nnz, -1, o_nnz); CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(C, -1, d_nnz); CHKERRQ(ierr);

    ierr = PetscFree(d_nnz); CHKERRQ(ierr);
    ierr = PetscFree(o_nnz); CHKERRQ(ierr);

    /* Check correct ownership ranges */

    PetscInt Istart_C, Iend_C;

    ierr = MatGetOwnershipRange(C, &Istart_C, &Iend_C);

    if(Istart_C != Istart)
        SETERRQ2(comm, 1, "Incorrect ownership range for Istart. Expected %d. Got %d.", Istart, Istart_C);

    if(Iend_C != Iend)
        SETERRQ2(comm, 1, "Incorrect ownership range for Iend. Expected %d. Got %d.", Iend, Iend_C);

    /* Set some optimization options */

    ierr = MatSetOption(C, MAT_NO_OFF_PROC_ENTRIES,         PETSC_TRUE); CHKERRQ(ierr);
    ierr = MatSetOption(C, MAT_NO_OFF_PROC_ZERO_ROWS,       PETSC_TRUE); CHKERRQ(ierr);
    ierr = MatSetOption(C, MAT_IGNORE_OFF_PROC_ENTRIES,     PETSC_TRUE); CHKERRQ(ierr);
    ierr = MatSetOption(C, MAT_KEEP_NONZERO_PATTERN,        PETSC_TRUE); CHKERRQ(ierr);
    ierr = MatSetOption(C, MAT_NEW_NONZERO_LOCATION_ERR,    PETSC_TRUE); CHKERRQ(ierr);
    ierr = MatSetOption(C, MAT_NEW_NONZERO_ALLOCATION_ERR,  PETSC_TRUE); CHKERRQ(ierr);
    ierr = MatSetOption(C, MAT_IGNORE_ZERO_ENTRIES,         PETSC_TRUE); CHKERRQ(ierr);

    /* Dump values from submatrix to final matrix in correct location */

    ierr = PetscMalloc1(N_C_final, &cols_shifted);
    for (PetscInt Irow = Istart; Irow < Iend; ++Irow)
    {
        ierr = MatGetRow(C_sub, Irow, &ncols, &cols, &vals);

        for (PetscInt Icol = 0; Icol < ncols; ++Icol)
            cols_shifted[Icol] = COL_MAP(cols[Icol]);

        ierr = MatSetValues(C, 1, &Irow, ncols, cols_shifted, vals, INSERT_VALUES); CHKERRQ(ierr);

        ierr = MatRestoreRow(C_sub, Irow, &ncols, &cols, &vals);
    }
    ierr = PetscFree(cols_shifted);

    #undef COL_MAP

    /* Free/destroy temporary data structures */

    if(C_temp)  ierr = MatDestroy(&C_temp); CHKERRQ(ierr);
    if(C_sub)   ierr = MatDestroy(&C_sub); CHKERRQ(ierr);
    if(is_rows) ierr = ISDestroy(&is_rows); CHKERRQ(ierr);
    if(is_cols) ierr = ISDestroy(&is_cols); CHKERRQ(ierr);

    ierr = PetscFree(id_cols); CHKERRQ(ierr);
    ierr = PetscFree(id_rows); CHKERRQ(ierr);

    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "MatKronProdSumIdx"
PetscErrorCode MatKronProdSumIdx(
    const std::vector<PetscScalar>& a,
    const std::vector<Mat>& A,
    const std::vector<Mat>& B,
    Mat& C,
    const std::vector<PetscInt> idx)
{
    PetscErrorCode ierr = 0;

    KRON_TIMINGS_INIT(__FUNCT__);
    KRON_TIMINGS_START(__FUNCT__);
    /*
        Get information from MPI
    */
    PetscMPIInt     nprocs, rank;
    MPI_Comm comm = PETSC_COMM_WORLD;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);
    /*
        Perform initial checks and get sizes
    */
    std::vector<PetscInt>   M_A, N_A, M_B, N_B;
    PetscInt                nterms, M_C, N_C;
    ierr = InitialChecks(a,A,B,C,comm,nterms,M_A,N_A,M_B,N_B,M_C,N_C); CHKERRQ(ierr);
    /*
        Determine final sizes based on desired indices
    */
    PetscInt M_C_final = idx.size();
    PetscInt N_C_final = idx.size();
    /*
        Guess the local ownership of resultant matrix C
    */
    PetscInt remrows = M_C_final % nprocs;
    PetscInt locrows = M_C_final / nprocs;
    PetscInt Istart = locrows * rank;

    if (rank < remrows){
        locrows += 1;
        Istart += rank;
    } else {
        Istart += remrows;
    }

    PetscInt Iend = Istart + locrows;
    /*
        Determine which rows of A and B to take and populate corresponding sets
        Then dump (ordered) set into array
    */
    std::set<PetscInt> set_Arows, set_Brows;

    for (PetscInt i = Istart; i < Iend; ++i)
        set_Arows.insert(idx[i] / M_B[0]);

    for (PetscInt i = Istart; i < Iend; ++i)
        set_Brows.insert(idx[i] % M_B[0]);

    PetscInt M_req_A = set_Arows.size();
    PetscInt M_req_B = set_Brows.size();

    PetscInt *id_rows_A;
    ierr = PetscMalloc1(M_req_A, &id_rows_A); CHKERRQ(ierr);
    {
        PetscInt i = 0;
        for (auto elem: set_Arows){
            id_rows_A[i] = elem;
            ++i;
        }
    }

    PetscInt *id_rows_B;
    ierr = PetscMalloc1(M_req_B, &id_rows_B); CHKERRQ(ierr);
    {
        PetscInt i = 0;
        for (auto elem: set_Brows){
            id_rows_B[i] = elem;
            ++i;
        }
    }

    std::vector<Mat>    submat_A(nterms), submat_B(nterms);
    PetscInt            A_sub_start, A_sub_end, B_sub_start, B_sub_end;

    ierr = GetSubmatrix(A,N_A,nterms,M_req_A,id_rows_A,submat_A,A_sub_start,A_sub_end); CHKERRQ(ierr);

    ierr = GetSubmatrix(B,N_B,nterms,M_req_B,id_rows_B,submat_B,B_sub_start,B_sub_end); CHKERRQ(ierr);

    /*
        Create map from global matrix row index to local submatrix index
        TODO: integrate ROW_MAP_A function here
    */
    std::map<PetscInt,PetscInt> map_A;
    for (PetscInt i = 0; i < set_Arows.size(); ++i)
        map_A[ id_rows_A[i] ] = i;

    std::map<PetscInt,PetscInt> map_B;
    for (PetscInt i = 0; i < set_Brows.size(); ++i)
        map_B[ id_rows_B[i] ] = i;

    ierr = PetscFree(id_rows_A); CHKERRQ(ierr);
    ierr = PetscFree(id_rows_B); CHKERRQ(ierr);


    // for (auto elem: set_Arows) printf("[%d] A: %d\n", rank, elem);
    // for (auto elem: set_Brows) printf("[%d] B: %d\n", rank, elem);
    /*
        Put these rows to a local submatrix
    */


    #define ROW_MAP_A(INDEX) (map_A[INDEX] + A_sub_start)
    #define ROW_MAP_B(INDEX) (map_B[INDEX] + B_sub_start)

    #define COL_MAP_A(INDEX) ((INDEX) - N_A[i] * (nprocs - 1) )
    #define COL_MAP_B(INDEX) ((INDEX) - N_B[i] * (nprocs - 1) )


    /*

        PREALLOCATION

        Run through all terms and calculate an overestimated preallocation
        by adding all the non-zeros needed for each row.

        Note: Always preallocate
    */
    if(C) MatDestroy(&C);
    if(true)
    {
        #define KRON_SUBMATRIX "        Kron: Preallocation"
        KRON_PS_TIMINGS_INIT(KRON_SUBMATRIX)
        KRON_PS_TIMINGS_START(KRON_SUBMATRIX)

        ierr = MatCreate(comm, &C); CHKERRQ(ierr);
        ierr = MatSetSizes(C, PETSC_DECIDE, PETSC_DECIDE, M_C_final, N_C_final); CHKERRQ(ierr);
        ierr = MatSetFromOptions(C); CHKERRQ(ierr);

        #if 0
        /*
            Naive / dense preallocation
        */
        ierr = MatMPIAIJSetPreallocation(C, locrows, NULL, M_C - locrows, NULL); CHKERRQ(ierr);
        ierr = MatSeqAIJSetPreallocation(C, M_C, NULL); CHKERRQ(ierr);
        #else
        /*
            More accurate preallocation (slightly overestimated)
        */
        PetscInt *d_nnz, *o_nnz, ncols_A, ncols_B;
        PetscInt Arow, Brow, Irow, Ccol;
        const PetscInt *cols_A, *cols_B;
        ierr = PetscMalloc1(locrows,&d_nnz); CHKERRQ(ierr);
        ierr = PetscMalloc1(locrows,&o_nnz); CHKERRQ(ierr);

        /* Map idx to its column */
        std::map<PetscInt,PetscInt> idx_map;
        std::map<PetscInt,PetscInt>::iterator it;
        {
            PetscInt i = 0;
            for (auto elem: idx){
                idx_map[elem] = i;
                ++i;
            }
        }

        for (PetscInt Crow = Istart; Crow < Iend; ++Crow)
        {
            Irow = idx[Crow];

            Arow = Irow / M_B[0];
            Brow = Irow % M_B[0];

            /*  Assume that matrices in A and in B have the same shapes  */

            d_nnz[Crow-Istart] = 0;
            o_nnz[Crow-Istart] = 0;

            for (PetscInt i = 0; i < nterms; ++i)
            {
                ierr = MatGetRow(submat_A[i], ROW_MAP_A(Arow), &ncols_A, &cols_A, nullptr); CHKERRQ(ierr);
                ierr = MatGetRow(submat_B[i], ROW_MAP_B(Brow), &ncols_B, &cols_B, nullptr); CHKERRQ(ierr);

                for (PetscInt j_A = 0; j_A < ncols_A; ++j_A)
                {
                    for (PetscInt j_B = 0; j_B < ncols_B; ++j_B)
                    {
                        Ccol = COL_MAP_A(cols_A[j_A]) * N_B[i] + COL_MAP_B(cols_B[j_B]);

                        /* Determine where Ccol enters into the final matrix */
                        it = idx_map.find(Ccol);
                        if (it != idx_map.end()){
                            Ccol = it->second;
                            if ( Istart <= Ccol && Ccol < Iend ){
                                d_nnz[Crow-Istart] += 1;
                            } else {
                                o_nnz[Crow-Istart] += 1;
                            }
                        }
                    }
                }

                ierr = MatRestoreRow(submat_A[i], ROW_MAP_A(Arow), &ncols_A, &cols_A, nullptr); CHKERRQ(ierr);
                ierr = MatRestoreRow(submat_B[i], ROW_MAP_B(Brow), &ncols_B, &cols_B, nullptr); CHKERRQ(ierr);
            }

            d_nnz[Crow-Istart] = std::min(locrows,             d_nnz[Crow-Istart]);
            o_nnz[Crow-Istart] = std::min(N_C_final - locrows, o_nnz[Crow-Istart]);

        }

        ierr = MatMPIAIJSetPreallocation(C, -1, d_nnz, -1, o_nnz); CHKERRQ(ierr);
        ierr = MatSeqAIJSetPreallocation(C, -1, d_nnz); CHKERRQ(ierr);

        ierr = PetscFree(d_nnz); CHKERRQ(ierr);
        ierr = PetscFree(o_nnz); CHKERRQ(ierr);

        #ifdef __KRON_PS_TIMINGS // print info on expected sparsity
            // printf("[%d] %d \n", rank, tot_entries);
            PetscInt tot_entries_reduced;
            MPI_Reduce( &tot_entries, &tot_entries_reduced, 1, MPI_INT, MPI_SUM, 0, comm);
            PetscPrintf(comm, "%20s Nonzeros: %d/(%-d)^2 = %f%%\n", " ",tot_entries_reduced, M_C,
                100.0*(double)tot_entries_reduced/((double)(M_C) * (double)(M_C)));
        #endif
        #endif
        /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

            MATRIX OPTIONS

            You know each process will only set values for its own rows,
            will generate an error if any process sets values for another process.
            This avoids all reductions in the MatAssembly routines and thus
            improves performance for very large process counts.
        */
        ierr = MatSetOption(C, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE);
        /*

            You know each process will only zero its own rows.
            This avoids all reductions in the zero row routines and thus
            improves performance for very large process counts.
        */
        ierr = MatSetOption(C, MAT_NO_OFF_PROC_ZERO_ROWS, PETSC_TRUE);
        /*

            Set to PETSC_TRUE indicates entries destined for other processors should be dropped,
            rather than stashed. This is useful if you know that the "owning" processor is also
            always generating the correct matrix entries, so that PETSc need not transfer
            duplicate entries generated on another processor.
        */
        ierr = MatSetOption(C, MAT_IGNORE_OFF_PROC_ENTRIES, PETSC_TRUE);
        /*

            indicates when MatZeroRows() is called the zeroed entries are kept in the nonzero structure
            NOTE: significant improvement not yet observed
        */
        ierr = MatSetOption(C, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);
        /*

            set to PETSC_TRUE indicates that any add or insertion that would generate a new entry
            in the nonzero structure instead produces an error. (Currently supported for
            AIJ and BAIJ formats only.) If this option is set then the MatAssemblyBegin/End()
            processes has one less global reduction
         */
        ierr = MatSetOption(C, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE);
        /*

            set to PETSC_TRUE indicates that any add or insertion that would generate a new entry
            that has not been preallocated will instead produce an error. (Currently supported
            for AIJ and BAIJ formats only.) This is a useful flag when debugging matrix memory
            preallocation. If this option is set then the MatAssemblyBegin/End() processes has one
            less global reduction
         */
        ierr = MatSetOption(C, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE);
        // ierr = MatSetOption(C, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE);
        /*

            for AIJ/IS matrices this will stop zero values from creating a zero location in the matrix
        */
        ierr = MatSetOption(C, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE);

        /* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

        /*
            Check ownership guess
        */
        if(nprocs > 1){
            PetscInt Istart_final, Iend_final;
            ierr = MatGetOwnershipRange(C, &Istart_final, &Iend_final);
            PetscInt Irows_final = Iend_final - Istart_final;
            if(Irows_final != locrows) {
                char errormsg[200];
                sprintf(errormsg,"WRONG GUESS: Irows=%d  locrows=%d\n", Irows_final,locrows);
                SETERRQ(comm, 1, errormsg);
            }
        }

        KRON_PS_TIMINGS_END(KRON_SUBMATRIX)
        #undef KRON_SUBMATRIX
    }
    else
    {
        ierr = MatZeroEntries(C); CHKERRQ(ierr);
    }
    /*
        CALCULATE ENTRIES
    */
    #define __KRONLOOP     "        KronLoop"
    KRON_PS_TIMINGS_INIT(__KRONLOOP);

    #define __MATSETVALUES "            MatSetValues"
    KRON_PS_TIMINGS_ACCUM_INIT(__MATSETVALUES);

    #define __CALC_VALUES  "            CalculateKronValues"
    KRON_PS_TIMINGS_ACCUM_INIT(__CALC_VALUES);

    KRON_PS_TIMINGS_START(__KRONLOOP);

    const PetscInt*     cols_A[nterms];
    const PetscScalar*  vals_A[nterms];
    const PetscInt*     cols_B[nterms];
    const PetscScalar*  vals_B[nterms];
    PetscInt            ncols_A[nterms], ncols_B[nterms];
    PetscInt            Arow, Brow, Irow;

    PetscInt        max_ncols_C = N_A[0] * N_B[0]; /* Assumes same size of matrices in A and B */
    PetscInt        *cols_C;
    PetscScalar     *vals_C;
    ierr = PetscMalloc1(max_ncols_C,&cols_C); CHKERRQ(ierr);
    ierr = PetscMalloc1(max_ncols_C,&vals_C); CHKERRQ(ierr);

    std::map<PetscInt,PetscScalar> C_map;
    std::map<PetscInt,PetscScalar>::iterator C_it;

    for (PetscInt Crow = Istart; Crow < Iend; ++Crow)
    {
        Irow = idx[Crow];

        Arow = Irow / M_B[0];
        Brow = Irow % M_B[0];
        for (PetscInt i = 0; i < nterms; ++i)
        {
            ierr = MatGetRow(submat_A[i], ROW_MAP_A(Arow), &ncols_A[i], &cols_A[i], &vals_A[i]); CHKERRQ(ierr);
            ierr = MatGetRow(submat_B[i], ROW_MAP_B(Brow), &ncols_B[i], &cols_B[i], &vals_B[i]); CHKERRQ(ierr);
        }

        /*  Assume that matrices in A and in B have the same shapes  */

        for (PetscInt i = 0; i < nterms; ++i)
        {
            C_map.clear();
            KRON_PS_TIMINGS_ACCUM_START(__CALC_VALUES);
            for (PetscInt j_A = 0; j_A < ncols_A[i]; ++j_A)
            {
                for (PetscInt j_B = 0; j_B < ncols_B[i]; ++j_B)
                {
                    /* Transform into map: key-cols, value-vals */
                    #define KEY COL_MAP_A(cols_A[i][j_A]) * N_B[i] + COL_MAP_B(cols_B[i][j_B])
                    #define VAL a[i] * vals_A[i][j_A] * vals_B[i][j_B]
                    C_map[KEY] += VAL;
                    #undef KEY
                    #undef VAL
                }
            }

            /* Select only cols that match with idx */
            PetscInt id_C = 0, idx_col = 0;
            for (auto col: idx)
            {
                C_it = C_map.find(col);
                if(C_it != C_map.end()){
                    if (C_it->second != 0.0)
                    {
                        cols_C[id_C] = idx_col;
                        vals_C[id_C] = C_it->second;
                        ++id_C;
                    }
                }
                ++idx_col;
            }
            KRON_PS_TIMINGS_ACCUM_END(__CALC_VALUES);

            KRON_PS_TIMINGS_ACCUM_START(__MATSETVALUES);
            ierr = MatSetValues(C, 1, &Crow, id_C, cols_C, vals_C, ADD_VALUES ); CHKERRQ(ierr);
            KRON_PS_TIMINGS_ACCUM_END(__MATSETVALUES);

        }

        for (PetscInt i = 0; i < nterms; ++i)
        {
            ierr = MatRestoreRow(submat_B[i], ROW_MAP_B(Brow), &ncols_B[i], &cols_B[i], &vals_B[i]); CHKERRQ(ierr);
            ierr = MatRestoreRow(submat_A[i], ROW_MAP_A(Arow), &ncols_A[i], &cols_A[i], &vals_A[i]); CHKERRQ(ierr);
        };
    }


    KRON_PS_TIMINGS_ACCUM_PRINT(__CALC_VALUES);
    #undef __CALC_VALUES

    KRON_PS_TIMINGS_ACCUM_PRINT(__MATSETVALUES);
    #undef __MATSETVALUES

    KRON_PS_TIMINGS_END(__KRONLOOP);
    #undef __KRONLOOP

    /*
        Destroy temporary objects and submatrices
    */

    C_map.clear();

    ierr = PetscFree(cols_C); CHKERRQ(ierr);
    ierr = PetscFree(vals_C); CHKERRQ(ierr);

    for (PetscInt i = 0; i < nterms; ++i){
        if(submat_A.data()+i) ierr = MatDestroy(submat_A.data()+i); CHKERRQ(ierr);
    }
    for (PetscInt i = 0; i < nterms; ++i){
        if(submat_B.data()+i) ierr = MatDestroy(submat_B.data()+i); CHKERRQ(ierr);
    }

    #undef ROW_MAP_A
    #undef ROW_MAP_B
    #undef COL_MAP_A
    #undef COL_MAP_B

    KRON_TIMINGS_END(__FUNCT__);

    PetscBool assembled;
    LINALG_TOOLS__MATASSEMBLY_FINAL(C);

    return ierr;
}
