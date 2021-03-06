#include "kron.hpp"
#include <../src/mat/impls/aij/seq/aij.h> /* Mat_SeqAIJ */

#undef __FUNCT__
#define __FUNCT__ "InitialChecks"
PETSC_EXTERN PetscErrorCode InitialChecks(
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
    PetscErrorCode ierr = 0;

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
        ierr = MatGetSize(A[i], M_A.data()+i, N_A.data()+i); CHKERRQ(ierr);
        ierr = MatGetSize(B[i], M_B.data()+i, N_B.data()+i); CHKERRQ(ierr);
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
PETSC_EXTERN PetscErrorCode GetSubmatrix(
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
    MPI_Comm comm = PETSC_COMM_WORLD;

    PetscInt *id_cols_A;
    ierr = PetscMalloc1(N_A[0], &id_cols_A); CHKERRQ(ierr);
    for (PetscInt Icol = 0; Icol < N_A[0]; ++Icol)
        id_cols_A[Icol] = Icol;

    IS isrow_A = nullptr, iscol_A = nullptr;
    PetscInt A_sub_start_temp = 0, A_sub_end_temp = 0;

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

        #if 1
        /* New submatrix layout using GetSubmatrices */
        Mat *p_submat_A; /* reference to submatrix array to be filled by 1 submatrix */
        ierr = MatGetSubMatrices(A[i], 1, &isrow_A, &iscol_A, MAT_INITIAL_MATRIX, &p_submat_A); CHKERRQ(ierr);
        submat_A[i] = *p_submat_A;

        #else
        /* Old submatrix layout */
        ierr = MatGetSubMatrix(A[i], isrow_A, iscol_A, MAT_INITIAL_MATRIX, submat_A.data()+i); CHKERRQ(ierr);

        #endif

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
#define __FUNCT__ "MatKronProdSum_2"
PetscErrorCode MatKronProdSum_2(
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

        Get only the rows of B needed for this operation

        Build the submatrices for each term

        NOTE: Assumes equal shapes for all A and all B matrices

    */
    std::vector<Mat>      submat_B(nterms);
    PetscInt B_sub_start, B_sub_end;
    /*
        Determine which rows of B to take and populate corresponding sets
        to remove duplicates. Then dump (ordered) set into array

        Map ownership
        Input: the row INDEX in the global matrix B
        Output: the corresponding row index in the locally-owned rows of submatrix B
    */

    PetscInt *id_rows_B;

    #if 1

        std::set<PetscInt> set_Brows;
        for (PetscInt Irow = Istart; Irow < Iend; ++Irow)
            set_Brows.insert(Irow % M_B[0]);

        PetscInt M_req_B = set_Brows.size();
        ierr = PetscMalloc1(M_req_B, &id_rows_B); CHKERRQ(ierr);

        PetscInt counter = 0;
        for (auto elem: set_Brows){
            id_rows_B[counter] = elem;
            ++counter;
        }
        ierr = GetSubmatrix(B,N_B,nterms,M_req_B,id_rows_B,submat_B,B_sub_start,B_sub_end); CHKERRQ(ierr);

        std::vector<PetscInt> map_B(M_B[0]);
        for (size_t i = 0; i < set_Brows.size(); ++i){
            map_B[ id_rows_B[i] ] = i + B_sub_start;
        }

        #define ROW_MAP_B(INDEX) (map_B[INDEX])

    #else

        ierr = PetscMalloc1(M_B[0], &id_rows_B); CHKERRQ(ierr);
        for (PetscInt Irow = 0; Irow < M_B[0]; ++Irow)
            id_rows_B[Irow] = Irow;
        ierr = GetSubmatrix(B,N_B,nterms,M_B[0],id_rows_B,submat_B,B_sub_start,B_sub_end); CHKERRQ(ierr);

        const PetscInt ROW_SHIFT_B = + B_sub_start;
        #define ROW_MAP_B(INDEX) ((INDEX) + ROW_SHIFT_B)

    #endif

    ierr = PetscFree(id_rows_B); CHKERRQ(ierr);

    /*
        Map ownership
        Input: the row INDEX in the global matrix A
        Output: the corresponding row index in the locally-owned rows of submatrix A
    */
    const PetscInt ROW_SHIFT_A = - Astart + A_sub_start;
    #define ROW_MAP_A(INDEX) ((INDEX) + ROW_SHIFT_A)
    /*
        Submatrix constructions offsets the starting column
        Input: the corresponding column index in the locally-owned submatrix A/B
        Output: the column INDEX in the global matrix A/B
    */
    #if 1
    /* New submatrix layout */
    #define COL_MAP_A(INDEX) ((INDEX))
    #define COL_MAP_B(INDEX) ((INDEX))

    #else
    /* Old submatrix layout */
    const PetscInt COL_SHIFT_A = - N_A[0] * (nprocs - 1);
    const PetscInt COL_SHIFT_B = - N_B[0] * (nprocs - 1);

    #define COL_MAP_A(INDEX) ((INDEX) + COL_SHIFT_A )
    #define COL_MAP_B(INDEX) ((INDEX) + COL_SHIFT_B )
    #endif

    KRON_PS_TIMINGS_END(KRON_SUBMATRIX)
    #undef KRON_SUBMATRIX
    /*

        PREALLOCATION

        Run through all terms and calculate an overestimated preallocation
        by adding all the non-zeros needed for each row.
    */
    PetscInt max_ncols_C = 0;
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

        /* Also determine the maximum number of nonzeros among all rows */

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
            if (ncols_C_max > max_ncols_C) max_ncols_C = ncols_C_max;
        }

        ierr = MatMPIAIJSetPreallocation(C, -1, d_nnz, -1, o_nnz); CHKERRQ(ierr);
        ierr = MatSeqAIJSetPreallocation(C, -1, d_nnz); CHKERRQ(ierr);

        #ifdef __KRON_PS_TIMINGS // print info on expected sparsity
            unsigned long int tot_entries=0, tot_entries_reduced=0, M_C_final=M_C;
            for (PetscInt i = 0; i < locrows; ++i) tot_entries += d_nnz[i] + o_nnz[i];
            MPI_Reduce( &tot_entries, &tot_entries_reduced, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, comm);
            PetscPrintf(comm, "%24s Nonzeros: %lu/(%-d)^2 = %f%%\n", " ",tot_entries_reduced, M_C_final,
                100.0*(double)tot_entries_reduced/((double)(M_C_final) * (double)(M_C_final)));
        #endif

        ierr = PetscFree(d_nnz); CHKERRQ(ierr);
        ierr = PetscFree(o_nnz); CHKERRQ(ierr);

        #endif // #ifdef __KRON_DENSE_PREALLOCATION
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

        PetscInt ncols_A, ncols_B;
        for (PetscInt Irow = Istart; Irow < Iend; ++Irow)
        {
            PetscInt Arow = Irow / M_B[0];
            PetscInt Brow = Irow % M_B[0];

            PetscInt ncols_C_max = 0;
            for (PetscInt i = 0; i < nterms; ++i)
            {
                ierr = MatGetRow(submat_A[i], ROW_MAP_A(Arow), &ncols_A, nullptr, nullptr); CHKERRQ(ierr);
                ierr = MatGetRow(submat_B[i], ROW_MAP_B(Brow), &ncols_B, nullptr, nullptr); CHKERRQ(ierr);

                ncols_C_max += ncols_A * ncols_B;

                ierr = MatRestoreRow(submat_A[i], ROW_MAP_A(Arow), &ncols_A, nullptr, nullptr); CHKERRQ(ierr);
                ierr = MatRestoreRow(submat_B[i], ROW_MAP_B(Brow), &ncols_B, nullptr, nullptr); CHKERRQ(ierr);
            }
            if (ncols_C_max > max_ncols_C) max_ncols_C = ncols_C_max;
        }

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

    PetscInt*       cols_C;
    PetscScalar*    vals_C;
    ierr = PetscMalloc1(max_ncols_C+1,&cols_C); CHKERRQ(ierr);
    ierr = PetscMalloc1(max_ncols_C+1,&vals_C); CHKERRQ(ierr);

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


/* Dump values to sequential matrices */
#undef __FUNCT__
#define __FUNCT__ "MatKronProdSum_selectiverows_3"
PetscErrorCode MatKronProdSum_selectiverows_3(
    const std::vector<PetscScalar>& a,
    const std::vector<Mat>& A,
    const std::vector<Mat>& B,
    Mat& C,
    const std::vector<PetscInt> idx)
{
    PetscErrorCode ierr = 0;

    /**************************************************/
    KRON_TIMINGS_INIT(__FUNCT__);
    KRON_TIMINGS_START(__FUNCT__);

    #define KRON_SUBMATRIX "    Kron: Init and Submatrix collection"
    KRON_PS_TIMINGS_INIT(KRON_SUBMATRIX)
    KRON_PS_TIMINGS_START(KRON_SUBMATRIX)
    /**************************************************/
    /*
        Get information from MPI
    */
    PetscMPIInt     nprocs, rank;
    MPI_Comm comm = PETSC_COMM_WORLD;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    {
        /* Verify that idx are all valid
         * Assumes A and B matrices have the same sizes
         */
        PetscInt M_A, M_B, M_C;
        ierr = MatGetSize(A[0], &M_A, nullptr); CHKERRQ(ierr);
        ierr = MatGetSize(B[0], &M_B, nullptr); CHKERRQ(ierr);

        M_C = M_A * M_B;
        for (auto id: idx)
            if (id >= M_C)
                SETERRQ1(comm,1,"Invalid key: %d", id);
    }
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
    PetscInt N_C_final = N_C;
    /*
        Guess the local ownership of resultant matrix C
    */
    PetscInt remrows = M_C_final % nprocs;
    PetscInt locrows = M_C_final / nprocs;
    PetscInt Istart  = locrows * rank;

    if (rank < remrows){
        locrows += 1;
        Istart += rank;
    } else {
        Istart += remrows;
    }

    PetscInt Iend = Istart + locrows;
    /*
        Guess column layout / submatrices
     */
    PetscInt remcols = N_C_final % nprocs;
    PetscInt locdiag = N_C_final / nprocs;
    PetscInt Cstart  = locdiag * rank;

    if (rank < remcols){
        locdiag += 1;
        Cstart += rank;
    } else {
        Cstart += remcols;
    }

    // PetscInt Cend = Cstart + locdiag;
    /*
        Determine which rows of A and B to take and populate corresponding sets
        to remove duplicates. Then dump (ordered) set into array
    */
    std::set<PetscInt> set_Arows, set_Brows;

    for (PetscInt i = Istart; i < Iend; ++i)
        set_Arows.insert(idx[i] / M_B[0]);

    for (PetscInt i = Istart; i < Iend; ++i)
        set_Brows.insert(idx[i] % M_B[0]);

    PetscInt M_req_A = set_Arows.size();
    PetscInt M_req_B = set_Brows.size();

    PetscInt *id_rows_A, *id_rows_B;
    ierr = PetscMalloc1(M_req_A, &id_rows_A); CHKERRQ(ierr);
    ierr = PetscMalloc1(M_req_B, &id_rows_B); CHKERRQ(ierr);

    PetscInt counter = 0;
    for (auto elem: set_Arows){
        id_rows_A[counter] = elem;
        ++counter;
    }

    counter = 0;
    for (auto elem: set_Brows){
        id_rows_B[counter] = elem;
        ++counter;
    }

    /*
        Get required rows and dump into respective submatrix
     */

    std::vector<Mat>    submat_A(nterms), submat_B(nterms);
    PetscInt            A_sub_start, A_sub_end, B_sub_start, B_sub_end;

    ierr = GetSubmatrix(A,N_A,nterms,M_req_A,id_rows_A,submat_A,A_sub_start,A_sub_end); CHKERRQ(ierr);
    ierr = GetSubmatrix(B,N_B,nterms,M_req_B,id_rows_B,submat_B,B_sub_start,B_sub_end); CHKERRQ(ierr);

    /*
        Create map from global matrix row index to local submatrix index
        using a map (slower, less memory) or a vector (faster, more unused memory)
    */

    #if 0

        std::map<PetscInt,PetscInt> map_A;
        for (PetscInt i = 0; i < set_Arows.size(); ++i)
            map_A[ id_rows_A[i] ] = i + A_sub_start;

        std::map<PetscInt,PetscInt> map_B;
        for (PetscInt i = 0; i < set_Brows.size(); ++i)
            map_B[ id_rows_B[i] ] = i + B_sub_start;

    #else

        std::vector<PetscInt> map_A(M_A[0]);
        std::vector<PetscInt> map_B(M_B[0]);
        for (size_t i = 0; i < set_Arows.size(); ++i){
            map_A[ id_rows_A[i] ] = i + A_sub_start;
        }
        for (size_t i = 0; i < set_Brows.size(); ++i){
            map_B[ id_rows_B[i] ] = i + B_sub_start;
        }

    #endif

    ierr = PetscFree(id_rows_A); CHKERRQ(ierr);
    ierr = PetscFree(id_rows_B); CHKERRQ(ierr);

    /*
        Map ownership
        Input: the row INDEX in the global matrix A/B
        Output: the corresponding row index in the locally-owned rows of submatrix A/B
    */
    #define ROW_MAP_A(INDEX) (map_A[INDEX])
    #define ROW_MAP_B(INDEX) (map_B[INDEX])
    /*
        Submatrix constructions offsets the starting column
        Input: the corresponding column index in the locally-owned submatrix A/B
        Output: the column INDEX in the global matrix A/B
    */
    #if 1
    /* New submatrix layout */
    #define COL_MAP_A(INDEX) ((INDEX))
    #define COL_MAP_B(INDEX) ((INDEX))

    #else
    /* Old submatrix layout */
    PetscInt A_shift = N_A[0] * (nprocs - 1);
    PetscInt B_shift = N_B[0] * (nprocs - 1);
    #define COL_MAP_A(INDEX) ((INDEX) - A_shift)
    #define COL_MAP_B(INDEX) ((INDEX) - B_shift)
    #endif
    /*
        Input: the column INDEX in the global matrix A/B
        Output: the corresponding column index in the locally-owned submatrix A/B
    */
    #if 1
    /* New submatrix layout */
    #define COL_INV_A(INDEX) ((INDEX))
    #define COL_INV_B(INDEX) ((INDEX))

    #else
    /* Old submatrix layout */
    #define COL_INV_A(INDEX) ((INDEX) + A_shift)
    #define COL_INV_B(INDEX) ((INDEX) + B_shift)
    #endif

    /**************************************************/
    KRON_PS_TIMINGS_END(KRON_SUBMATRIX)
    #undef KRON_SUBMATRIX

    #define KRON_PREALLOC "    Kron:Preallocation"
    KRON_PS_TIMINGS_INIT(KRON_PREALLOC)
    KRON_PS_TIMINGS_START(KRON_PREALLOC)
    /**************************************************/

    /*

        PREALLOCATION

        Run through all terms and calculate an overestimated preallocation
        by adding all the non-zeros needed for each row.

        The matrix created is seqaij

        Note: Always preallocate
    */
    if(C) MatDestroy(&C);

    ierr = MatCreate(PETSC_COMM_SELF, &C); CHKERRQ(ierr);
    ierr = MatSetType(C, MATSEQAIJ);
    ierr = MatSetSizes(C, PETSC_DECIDE, PETSC_DECIDE, locrows, N_C_final); CHKERRQ(ierr);
    ierr = MatSetFromOptions(C); CHKERRQ(ierr);

    PetscInt *nnz;
    ierr = PetscMalloc1(locrows,&nnz); CHKERRQ(ierr);

    PetscInt ncols_A, ncols_B, Arow, Brow, Irow;
    const PetscInt *cols_A, *cols_B;

    PetscInt        max_ncols_C = 0;
    for (PetscInt Crow = Istart; Crow < Iend; ++Crow)
    {
        Irow = idx[Crow];

        Arow = Irow / M_B[0];
        Brow = Irow % M_B[0];

        nnz[Crow-Istart] = 0;

        for (PetscInt i = 0; i < nterms; ++i)
        {
            ierr = MatGetRow(submat_A[i], ROW_MAP_A(Arow), &ncols_A, &cols_A, nullptr); CHKERRQ(ierr);
            ierr = MatGetRow(submat_B[i], ROW_MAP_B(Brow), &ncols_B, &cols_B, nullptr); CHKERRQ(ierr);

            nnz[Crow-Istart] += ncols_A*ncols_B;

            ierr = MatRestoreRow(submat_A[i], ROW_MAP_A(Arow), &ncols_A, &cols_A, nullptr); CHKERRQ(ierr);
            ierr = MatRestoreRow(submat_B[i], ROW_MAP_B(Brow), &ncols_B, &cols_B, nullptr); CHKERRQ(ierr);
        }

        nnz[Crow-Istart] = std::min(N_C_final, nnz[Crow-Istart]);
        if(max_ncols_C < nnz[Crow-Istart]) max_ncols_C = nnz[Crow-Istart];
    }

    ierr = MatSeqAIJSetPreallocation(C, -1, nnz); CHKERRQ(ierr);

    #ifdef __KRON_PS_TIMINGS // print info on expected sparsity
        unsigned long int tot_entries=0, tot_entries_reduced=0;
        for (PetscInt i = 0; i < locrows; ++i) tot_entries += nnz[i];
        MPI_Reduce( &tot_entries, &tot_entries_reduced, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, comm);
        PetscPrintf(comm, "%24s Nonzeros: %lu/(%-d x %-d)^2 = %f%%\n", " ",tot_entries_reduced, M_C_final, N_C_final,
            100.0*(double)tot_entries_reduced/( (double)(M_C_final) * (double)(N_C_final)) );
    #endif

    ierr = PetscFree(nnz); CHKERRQ(ierr);

    /* Set some matrix options for optimization */
    ierr = MatSetOption(C, MAT_NO_OFF_PROC_ENTRIES          , PETSC_TRUE);
    ierr = MatSetOption(C, MAT_NO_OFF_PROC_ZERO_ROWS        , PETSC_TRUE);
    ierr = MatSetOption(C, MAT_IGNORE_OFF_PROC_ENTRIES      , PETSC_TRUE);
    // ierr = MatSetOption(C, MAT_KEEP_NONZERO_PATTERN         , PETSC_TRUE);
    ierr = MatSetOption(C, MAT_NEW_NONZERO_LOCATION_ERR     , PETSC_TRUE);
    ierr = MatSetOption(C, MAT_NEW_NONZERO_ALLOCATION_ERR   , PETSC_TRUE);
    ierr = MatSetOption(C, MAT_IGNORE_ZERO_ENTRIES          , PETSC_TRUE);

    /**************************************************/
    KRON_PS_TIMINGS_END(KRON_PREALLOC)
    #undef KRON_PREALLOC
    /**************************************************/

    /*
        CALCULATE ENTRIES
    */
    const PetscScalar*  vals_A;
    const PetscScalar*  vals_B;

    PetscInt        *cols_C;
    PetscScalar     *vals_C;
    ierr = PetscMalloc1(max_ncols_C+1,&cols_C); CHKERRQ(ierr);
    ierr = PetscMalloc1(max_ncols_C+1,&vals_C); CHKERRQ(ierr);

    /**************************************************/
    #define __KRONLOOP     "    Kron:Loop"
    KRON_PS_TIMINGS_INIT(__KRONLOOP);

    #define __MATSETVALUES "        Kron:MatSetValues"
    KRON_PS_TIMINGS_ACCUM_INIT(__MATSETVALUES);

    #define __CALC_VALUES  "        Kron:CalculateKronValues"
    KRON_PS_TIMINGS_ACCUM_INIT(__CALC_VALUES);

    KRON_PS_TIMINGS_START(__KRONLOOP);
    /**************************************************/

    for (PetscInt Crow = 0; Crow < locrows; ++Crow)
    {
        Irow = idx[Crow+Istart];

        Arow = Irow / M_B[0];
        Brow = Irow % M_B[0];

        for (PetscInt i = 0; i < nterms; ++i)
        {
            /**************************************************/
            KRON_PS_TIMINGS_ACCUM_START(__CALC_VALUES);
            /**************************************************/

            ierr = MatGetRow(submat_A[i], ROW_MAP_A(Arow), &ncols_A, &cols_A, &vals_A); CHKERRQ(ierr);
            ierr = MatGetRow(submat_B[i], ROW_MAP_B(Brow), &ncols_B, &cols_B, &vals_B); CHKERRQ(ierr);

            for (PetscInt j_A = 0; j_A < ncols_A; ++j_A)
            {
                for (PetscInt j_B = 0; j_B < ncols_B; ++j_B)
                {
                    cols_C [ j_A * ncols_B + j_B ] = COL_MAP_A(cols_A[j_A]) * N_B[i] + COL_MAP_B(cols_B[j_B]);
                    vals_C [ j_A * ncols_B + j_B ] = a[i] * vals_A[j_A] * vals_B[j_B];
                }
            }
            PetscInt ncols_C = ncols_A*ncols_B;

            /**************************************************/
            KRON_PS_TIMINGS_ACCUM_END(__CALC_VALUES);
            KRON_PS_TIMINGS_ACCUM_START(__MATSETVALUES);
            /**************************************************/

            ierr = MatSetValues(C, 1, &Crow, ncols_C, cols_C, vals_C, ADD_VALUES ); CHKERRQ(ierr);

            ierr = MatRestoreRow(submat_B[i], ROW_MAP_B(Brow), &ncols_B, &cols_B, &vals_B); CHKERRQ(ierr);
            ierr = MatRestoreRow(submat_A[i], ROW_MAP_A(Arow), &ncols_A, &cols_A, &vals_A); CHKERRQ(ierr);

            /**************************************************/
            KRON_PS_TIMINGS_ACCUM_END(__MATSETVALUES);
            /**************************************************/
        }
    }

    /**************************************************/
    KRON_PS_TIMINGS_ACCUM_PRINT(__CALC_VALUES);
    #undef __CALC_VALUES

    KRON_PS_TIMINGS_ACCUM_PRINT(__MATSETVALUES);
    #undef __MATSETVALUES

    KRON_PS_TIMINGS_END(__KRONLOOP);
    #undef __KRONLOOP
    /**************************************************/

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
    #undef COL_INV_A
    #undef COL_INV_B

    /**************************************************/
    #define __FINALASSEMBLY     "    Kron:FinalAssemly"
    KRON_PS_TIMINGS_INIT(__FINALASSEMBLY);
    KRON_PS_TIMINGS_START(__FINALASSEMBLY);
    /**************************************************/

    LINALG_TOOLS__MATASSEMBLY_FINAL(C);

    /**************************************************/
    KRON_PS_TIMINGS_END(__FINALASSEMBLY);
    #undef __FINALASSEMBLY

    KRON_TIMINGS_END(__FUNCT__);
    /**************************************************/
    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "MatKronProdSumIdx_copy_3"
PetscErrorCode MatKronProdSumIdx_copy_3(
    const std::vector<PetscScalar>& a,
    const std::vector<Mat>& A,
    const std::vector<Mat>& B,
    Mat& C,
    const std::vector<PetscInt> idx)
{
    PetscErrorCode ierr = 0;

    PetscMPIInt     nprocs, rank;
    MPI_Comm comm = PetscObjectComm((PetscObject) A[0]);
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    // /* Verify that idx are all valid
    //  * Assumes A and B matrices have the same sizes
    //  */
    PetscInt M_A, M_B, M_C;
    ierr = MatGetSize(A[0], &M_A, nullptr); CHKERRQ(ierr);
    ierr = MatGetSize(B[0], &M_B, nullptr); CHKERRQ(ierr);

    M_C = M_A * M_B;
    // for (auto id: idx)
    //     if (id >= M_C)
    //         SETERRQ1(comm,1,"Invalid key: %d", id);

    /*
        Calculate partial elements of full matrix.
        Run separate routine to calculate only selected rows of
        C_temp and rewrite indexing here accordingly
    */

    Mat C_temp = nullptr;
    ierr = MatKronProdSum_selectiverows_3(a, A, B, C_temp, idx); CHKERRQ(ierr);

    LINALG_TOOLS__MATASSEMBLY_FINAL(C_temp);

    /* Verify that matrix is of sequential type */
    PetscBool flg;
    ierr = PetscObjectTypeCompare((PetscObject)C_temp,MATSEQAIJ,&flg); CHKERRQ(ierr);
    if(!flg){
        MatType type;
        ierr = MatGetType(C_temp,&type); CHKERRQ(ierr);
        SETERRQ2(comm,1,"Wrong matrix type. Expected %s. Got %s.",MATSEQAIJ,type);
    }

    MPI_Comm comm_temp = PetscObjectComm((PetscObject) C_temp);

    /**************************************************/
    KRON_TIMINGS_INIT(__FUNCT__);
    KRON_TIMINGS_START(__FUNCT__);

    #define __PREP     "    Kron:Prep"
    KRON_PS_TIMINGS_INIT(__PREP);
    KRON_PS_TIMINGS_START(__PREP);
    /**************************************************/

    /* Guess final row ownership ranges */

    PetscInt M_C_final = idx.size();
    PetscInt remrows = M_C_final % nprocs;
    PetscInt locrows = M_C_final / nprocs;
    PetscInt Istart = locrows * rank;

    if (rank < remrows){
        locrows += 1;
        Istart += rank;
    } else {
        Istart += remrows;
    }

    /* Check the size of C_temp */

    PetscInt M_C_temp, N_C_temp;
    ierr = MatGetSize(C_temp, &M_C_temp, &N_C_temp); CHKERRQ(ierr);
    if(M_C_temp != locrows)
        SETERRQ2(comm_temp,1,"Incorrect number of rows in C_temp. Expected %d. Got %d.",locrows,M_C_temp);
    if(N_C_temp != M_C)
        SETERRQ2(comm_temp,1,"Incorrect number of columns in C_temp. Expected %d. Got %d.",M_C,N_C_temp);

    /* Construct row indices */

    PetscInt *id_rows;
    ierr = PetscMalloc1(locrows,    &id_rows); CHKERRQ(ierr);
    for (PetscInt Irow = 0; Irow < locrows; ++Irow)
        id_rows[Irow] = Irow;

    IS is_rows = nullptr;
    ierr = ISCreateGeneral(comm_temp, locrows, id_rows, PETSC_USE_POINTER, &is_rows); CHKERRQ(ierr);

    /* Construct column indices */

    PetscInt *id_cols;
    ierr = PetscMalloc1(idx.size(), &id_cols); CHKERRQ(ierr);
    for (size_t Icol = 0; Icol < idx.size(); ++Icol)
        id_cols[Icol] = idx[Icol];

    IS is_cols = nullptr;
    ierr = ISCreateGeneral(comm_temp, idx.size(), id_cols, PETSC_USE_POINTER, &is_cols); CHKERRQ(ierr);

    /* Get submatrix based on desired indices */

    /**************************************************/
    KRON_PS_TIMINGS_END(__PREP);
    #undef __PREP

    #define __GETSUBMAT     "    MatGetSubMatrix"
    KRON_PS_TIMINGS_INIT(__GETSUBMAT);
    KRON_PS_TIMINGS_START(__GETSUBMAT);
    /**************************************************/

    Mat C_sub;
    ierr = MatGetSubMatrix(C_temp, is_rows, is_cols, MAT_INITIAL_MATRIX, &C_sub); CHKERRQ(ierr);

    /* Destroy C_temp earlier */
    if(C_temp) ierr = MatDestroy(&C_temp); CHKERRQ(ierr);


    /**************************************************/
    KRON_PS_TIMINGS_END(__GETSUBMAT);
    #undef __GETSUBMAT

/* Test this feature */
#if 1

    #define __CONCATENATE      "    Concatenate"
    KRON_PS_TIMINGS_INIT(__CONCATENATE);
    KRON_PS_TIMINGS_START(__CONCATENATE);
    /**************************************************/

    MatCreateMPIMatConcatenateSeqMat(comm, C_sub, PETSC_DECIDE, MAT_INITIAL_MATRIX, &C);

    /**************************************************/
    KRON_PS_TIMINGS_END(__CONCATENATE);
    #undef __CONCATENATE
    /**************************************************/

#else

    PetscInt N_C_final = idx.size();
    PetscInt Iend = Istart + locrows;

    #define __PREALLOC      "    Preallocation"
    KRON_PS_TIMINGS_INIT(__PREALLOC);
    KRON_PS_TIMINGS_START(__PREALLOC);
    /**************************************************/

    /* Local to global mapping (Direct) */

    #define COL_MAP(INDEX) ((INDEX))

    /* Create a new square matrix and populate with elements */

    ierr = MatCreate(comm, &C); CHKERRQ(ierr);
    ierr = MatSetSizes(C, PETSC_DECIDE, PETSC_DECIDE, M_C_final, N_C_final); CHKERRQ(ierr);
    ierr = MatSetFromOptions(C); CHKERRQ(ierr);

    /* Preallocation */

    const PetscInt    *cols;
    const PetscScalar *vals;
    PetscInt ncols;

    PetscInt *d_nnz, *o_nnz;
    ierr = PetscCalloc1(locrows, &d_nnz); CHKERRQ(ierr);
    ierr = PetscCalloc1(locrows, &o_nnz); CHKERRQ(ierr);

    for (PetscInt Irow = 0; Irow < locrows; ++Irow)
    {
        ierr = MatGetRow(C_sub, Irow, &ncols, &cols, nullptr);

        for (PetscInt Icol = 0; Icol < ncols; ++Icol){
            // if ( Istart <= COL_MAP(cols[Icol]) && COL_MAP(cols[Icol]) < Iend )
                // ++d_nnz[Irow];
            d_nnz[Irow] += ( Istart <= COL_MAP(cols[Icol]) && COL_MAP(cols[Icol]) < Iend ) ? 1 : 0 ;
        }

        o_nnz[Irow] = ncols - d_nnz[Irow];

        ierr = MatRestoreRow(C_sub, Irow, &ncols, &cols, nullptr);
    }

    ierr = MatMPIAIJSetPreallocation(C, -1, d_nnz, -1, o_nnz); CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(C, -1, d_nnz); CHKERRQ(ierr);

    #ifdef __KRON_PS_TIMINGS // print info on expected sparsity
        unsigned long int tot_entries=0, tot_entries_reduced=0;
        for (size_t i = 0; i < locrows; ++i) tot_entries += d_nnz[i] + o_nnz[i];
        MPI_Reduce( &tot_entries, &tot_entries_reduced, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, comm);
        PetscPrintf(comm, "%24s Nonzeros: %lu/(%-d)^2 = %f%%\n", " ", tot_entries_reduced, M_C_final,
            100.0*(double)tot_entries_reduced/((double)(M_C_final) * (double)(M_C_final)));
        PetscPrintf(comm, "%24s TotalRows: %-10d LocalRows: %d\n", " ", M_C_final, locrows);
    #endif

    ierr = PetscFree(d_nnz); CHKERRQ(ierr);
    ierr = PetscFree(o_nnz); CHKERRQ(ierr);

    /* Set some optimization options */

    ierr = MatSetOption(C, MAT_NO_OFF_PROC_ENTRIES,         PETSC_TRUE); CHKERRQ(ierr);
    ierr = MatSetOption(C, MAT_NO_OFF_PROC_ZERO_ROWS,       PETSC_TRUE); CHKERRQ(ierr);
    ierr = MatSetOption(C, MAT_IGNORE_OFF_PROC_ENTRIES,     PETSC_TRUE); CHKERRQ(ierr);
    // ierr = MatSetOption(C, MAT_KEEP_NONZERO_PATTERN,        PETSC_TRUE); CHKERRQ(ierr);
    ierr = MatSetOption(C, MAT_IGNORE_ZERO_ENTRIES,         PETSC_TRUE); CHKERRQ(ierr);
    ierr = MatSetOption(C, MAT_NEW_NONZERO_LOCATION_ERR,    PETSC_TRUE); CHKERRQ(ierr);
    ierr = MatSetOption(C, MAT_NEW_NONZERO_ALLOCATION_ERR,  PETSC_TRUE); CHKERRQ(ierr);

    /**************************************************/
    KRON_PS_TIMINGS_END(__PREALLOC);
    #undef __PREALLOC

    #define __SETVALS     "    SetValues"
    KRON_PS_TIMINGS_INIT(__SETVALS);
    KRON_PS_TIMINGS_START(__SETVALS);
    /**************************************************/

    /* Check correct ownership ranges */

    PetscInt Istart_C, Iend_C;

    ierr = MatGetOwnershipRange(C, &Istart_C, &Iend_C);

    if(Istart_C != Istart)
        SETERRQ2(comm, 1, "Incorrect ownership range for Istart. Expected %d. Got %d.", Istart, Istart_C);

    if(Iend_C != Iend)
        SETERRQ2(comm, 1, "Incorrect ownership range for Iend. Expected %d. Got %d.", Iend, Iend_C);

    /* Dump values from submatrix to final matrix */

    for (PetscInt Irow = Istart; Irow < Iend; ++Irow)
    {
        ierr = MatGetRow(C_sub, Irow-Istart, &ncols, &cols, &vals);
        ierr = MatSetValues(C, 1, &Irow, ncols, cols, vals, INSERT_VALUES); CHKERRQ(ierr);
        ierr = MatRestoreRow(C_sub, Irow-Istart, &ncols, &cols, &vals);
    }

    /**************************************************/
    #undef COL_MAP
    KRON_PS_TIMINGS_END(__SETVALS);
    #undef __SETVALS
    /**************************************************/

#endif

    /* Free/destroy temporary data structures */

    if(C_sub)   ierr = MatDestroy(&C_sub); CHKERRQ(ierr);
    if(is_rows) ierr = ISDestroy(&is_rows); CHKERRQ(ierr);
    if(is_cols) ierr = ISDestroy(&is_cols); CHKERRQ(ierr);

    ierr = PetscFree(id_cols); CHKERRQ(ierr);
    ierr = PetscFree(id_rows); CHKERRQ(ierr);


    /**************************************************/
    KRON_TIMINGS_END(__FUNCT__);
    /**************************************************/

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

    #ifdef __DMRG_MPI_BARRIERS
        ierr = MPI_Barrier(PETSC_COMM_WORLD); CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "\n======== Start of %s ========\n\n",__FUNCT__); CHKERRQ(ierr);
    #endif

    ierr = MatKronProdSum_2(a,A,B,C,prealloc); CHKERRQ(ierr);

    #ifdef __DMRG_MPI_BARRIERS
        ierr = MPI_Barrier(PETSC_COMM_WORLD); CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "\n======== End of %s ========\n\n",__FUNCT__); CHKERRQ(ierr);
    #endif

    return ierr;
}


PetscErrorCode MatKronProdSumIdx(
    const std::vector<PetscScalar>& a,
    const std::vector<Mat>& A,
    const std::vector<Mat>& B,
    Mat& C,
    const std::vector<PetscInt> idx)
{
    PetscErrorCode ierr = 0;

    C = nullptr;

    #ifdef __DMRG_MPI_BARRIERS
        ierr = MPI_Barrier(PETSC_COMM_WORLD); CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "\n======== Start of %s ========\n\n",__FUNCT__); CHKERRQ(ierr);
    #endif

    ierr = MatKronProdSumIdx_copy_3(a, A, B, C, idx); CHKERRQ(ierr);

    if (!C) SETERRQ(PETSC_COMM_WORLD, 1, "Matrix was not generated.");

    #ifdef __DMRG_MPI_BARRIERS
        ierr = MPI_Barrier(PETSC_COMM_WORLD); CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "\n======== End of %s ========\n\n",__FUNCT__); CHKERRQ(ierr);
    #endif

    return ierr;
}


/*
    Matrix-free construction of the Hamiltonian using L-R approach
 */


typedef struct {

    /* Inputs */
    PetscInt                    nterms;
    std::vector<PetscScalar>    a;
    std::vector<PetscInt>       M_A;
    std::vector<PetscInt>       N_A;
    std::vector<PetscInt>       M_B;
    std::vector<PetscInt>       N_B;

    /* Submatrices */
    std::vector<Mat>            submat_A;
    std::vector<Mat>            submat_B;
    PetscBool                   term_row_loop;

    /* Matrix Layout */
    PetscInt                    Istart, Iend;
    PetscInt                    locrows;
    std::vector<PetscInt>       map_B;
    PetscInt                    row_shift_A;

    /* MPI Info */
    PetscMPIInt                 rank;

    /* Mat-Vec multiplication */
    VecScatter                  vsctx;
    Vec                         x_seq;
    PetscInt                    y_loclength;

} CTX_KRON;


#define ROW_MAP_A(INDEX) ((INDEX) + ctx->row_shift_A)
#define ROW_MAP_B(INDEX) (ctx->map_B[(INDEX)])
#define COL_MAP_A(INDEX) ((INDEX))
#define COL_MAP_B(INDEX) ((INDEX))


PetscErrorCode MatMult_KronProdSum_MATSHELL(Mat A, Vec x, Vec y)
{
    PetscErrorCode ierr = 0;

    CTX_KRON *ctx;
    ierr = MatShellGetContext(A,(void**)&ctx); CHKERRQ(ierr);
    ierr = VecScatterBegin(ctx->vsctx, x, ctx->x_seq, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
    ierr = VecScatterEnd(ctx->vsctx, x, ctx->x_seq, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

    const PetscScalar *xvals;
    PetscScalar *yvals;
    ierr = VecGetArrayRead(ctx->x_seq, &xvals); CHKERRQ(ierr);
    ierr = VecGetArray(y, &yvals); CHKERRQ(ierr);
    ierr = PetscMemzero(yvals, (ctx->locrows)*sizeof(PetscScalar)); CHKERRQ(ierr);

    const PetscInt      *cols_A;
    const PetscScalar   *vals_A;
    const PetscInt      *cols_B;
    const PetscScalar   *vals_B;
    PetscInt            ncols_A, ncols_B;
    PetscInt            Arow, Brow;
    const Mat_SeqAIJ    *submat_A;
    const Mat_SeqAIJ    *submat_B;
    PetscScalar         yval;

    if(ctx->term_row_loop)
    {
        for(PetscInt iterm = 0; iterm < ctx->nterms; ++iterm)
        {
            PetscScalar   a = ctx->a[iterm];
            PetscInt    M_B = ctx->M_B[iterm];
            PetscInt    N_B = ctx->N_B[iterm];
            submat_A = (Mat_SeqAIJ*)(ctx->submat_A[iterm])->data;
            submat_B = (Mat_SeqAIJ*)(ctx->submat_B[iterm])->data;

            for(PetscInt Irow = ctx->Istart; Irow < ctx->Iend; ++Irow)
            {
                Arow = ROW_MAP_A(Irow / M_B);
                ncols_A  = submat_A->i[Arow+1] - submat_A->i[Arow];
                vals_A   = submat_A->a + submat_A->i[Arow];
                cols_A   = ncols_A ? submat_A->j + submat_A->i[Arow] : 0;

                Brow = ROW_MAP_B(Irow % M_B);
                ncols_B  = submat_B->i[Brow+1] - submat_B->i[Brow];
                vals_B   = submat_B->a + submat_B->i[Brow];
                cols_B   = ncols_B ? (submat_B->j + submat_B->i[Brow]) : 0;

                yval = 0;
                if(a==1.0)
                {
                    for(PetscInt Acol = 0; Acol < ncols_A; ++Acol)
                    {
                        PetscInt    cA = cols_A[Acol];
                        PetscScalar vA = vals_A[Acol];
                        for(PetscInt Bcol = 0; Bcol < ncols_B; ++Bcol)
                        {
                            yval += vA * vals_B[Bcol] * xvals[cA * N_B + cols_B[Bcol]];
                        }
                    }
                }
                else
                {
                    for(PetscInt Acol = 0; Acol < ncols_A; ++Acol)
                    {
                        PetscInt    cA = cols_A[Acol];
                        PetscScalar vA = vals_A[Acol];
                        for(PetscInt Bcol = 0; Bcol < ncols_B; ++Bcol)
                        {
                            yval += a * vA * vals_B[Bcol] * xvals[cA * N_B + cols_B[Bcol]];
                        }
                    }
                }
                yvals[Irow-ctx->Istart] += yval;
            }
        }
    }
    else
    {
        for(PetscInt Irow = ctx->Istart; Irow < ctx->Iend; ++Irow)
        {
            yval = 0;
            for(PetscInt iterm = 0; iterm < ctx->nterms; ++iterm)
            {
                PetscScalar   a = ctx->a[iterm];
                PetscInt    M_B = ctx->M_B[iterm];
                PetscInt    N_B = ctx->N_B[iterm];
                Arow = ROW_MAP_A(Irow / M_B);
                Brow = ROW_MAP_B(Irow % M_B);

                /* Manual inlining of MatGetRow_SeqAIJ */
                submat_A = (Mat_SeqAIJ*)(ctx->submat_A[iterm])->data;
                ncols_A  = submat_A->i[Arow+1] - submat_A->i[Arow];
                vals_A   = submat_A->a + submat_A->i[Arow];
                cols_A   = ncols_A ? submat_A->j + submat_A->i[Arow] : 0;

                submat_B = (Mat_SeqAIJ*)(ctx->submat_B[iterm])->data;
                ncols_B  = submat_B->i[Brow+1] - submat_B->i[Brow];
                vals_B   = submat_B->a + submat_B->i[Brow];
                cols_B   = ncols_B ? (submat_B->j + submat_B->i[Brow]) : 0;

                if(a==1.0)
                {
                    for(PetscInt Acol = 0; Acol < ncols_A; ++Acol)
                    {
                        PetscInt    cA = cols_A[Acol];
                        PetscScalar vA = vals_A[Acol];
                        for(PetscInt Bcol = 0; Bcol < ncols_B; ++Bcol)
                        {
                            yval += vA * vals_B[Bcol] * xvals[cA * N_B + cols_B[Bcol]];
                        }
                    }
                }
                else
                {
                    for(PetscInt Acol = 0; Acol < ncols_A; ++Acol)
                    {
                        PetscInt    cA = cols_A[Acol];
                        PetscScalar vA = vals_A[Acol];
                        for(PetscInt Bcol = 0; Bcol < ncols_B; ++Bcol)
                        {
                            yval += a * vA * vals_B[Bcol] * xvals[cA * N_B + cols_B[Bcol]];
                        }
                    }
                }

            }
            yvals[Irow-ctx->Istart] = yval;
        }
    }

    ierr = VecRestoreArrayRead(ctx->x_seq, &xvals);
    ierr = VecRestoreArray(y, &yvals);
    return ierr;
}


PetscErrorCode MatDestroy_KronProdSum_MATSHELL(Mat mat)
{
    PetscErrorCode ierr = 0;

    CTX_KRON *ctx;
    ierr = MatShellGetContext(mat,(void**)&ctx); CHKERRQ(ierr);

    /* Destroy objects in context */
    for (PetscInt i = 0; i < ctx->nterms; ++i){
        if(ctx->submat_A.data()+i) ierr = MatDestroy(ctx->submat_A.data()+i); CHKERRQ(ierr);
    }
    for (PetscInt i = 0; i < ctx->nterms; ++i){
        if(ctx->submat_B.data()+i) ierr = MatDestroy(ctx->submat_B.data()+i); CHKERRQ(ierr);
    }
    ierr = VecDestroy(&ctx->x_seq); CHKERRQ(ierr);
    ierr = VecScatterDestroy(&ctx->vsctx); CHKERRQ(ierr);

    /* Destroy context */
    ierr = PetscFree(ctx); CHKERRQ(ierr);

    /* Destroy matrix */
    // ierr = MatDestroymat); CHKERRQ(ierr);

    return ierr;
}


PetscErrorCode MatKronProdSum_MATSHELL(
    const std::vector<PetscScalar>& a,
    const std::vector<Mat>& A,
    const std::vector<Mat>& B,
    Mat& C)
{
    PetscErrorCode ierr = 0;
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
    CTX_KRON *ctx;
    ierr = PetscNew(&ctx); CHKERRQ(ierr);
    ctx->a = a;
    ctx->rank = rank;

    PetscInt M_C, N_C;
    ierr = InitialChecks(ctx->a,A,B,C,comm,ctx->nterms,
        ctx->M_A,ctx->N_A,ctx->M_B,ctx->N_B,M_C,N_C); CHKERRQ(ierr);
    ctx->term_row_loop = PETSC_TRUE;
    /*
        Guess the local ownership of resultant matrix C
    */
    PetscInt remrows = M_C % nprocs;
    ctx->locrows     = M_C / nprocs;
    ctx->Istart      = ctx->locrows * rank;

    if (rank < remrows){
        ctx->locrows += 1;
        ctx->Istart += rank;
    } else {
        ctx->Istart += remrows;
    }
    ctx->Iend = ctx->Istart + ctx->locrows;

    /******************* TIMINGS *******************/
    #define KRON_SUBMATRIX "KronMatFree: Submatrix collection"
    KRON_PS_TIMINGS_INIT(KRON_SUBMATRIX)
    KRON_PS_TIMINGS_START(KRON_SUBMATRIX)
    /***********************************************/

    /*

        SUBMATRIX A

        Acquire the submatrices of local and nonlocal rows needed to build
        the local rows of C

        Determine the required rows from A_i
    */
    PetscInt Astart = ctx->Istart/ctx->M_B[0];
    PetscInt Aend = 1+(ctx->Iend-1)/ctx->M_B[0];
    PetscInt M_req_A = Aend - Astart;
    /*
        Build the submatrices for each term
    */
    ctx->submat_A.resize(ctx->nterms);
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
    PetscInt A_sub_start, A_sub_end;
    ierr = GetSubmatrix(A, ctx->N_A, ctx->nterms, M_req_A, id_rows_A, ctx->submat_A, A_sub_start, A_sub_end); CHKERRQ(ierr);
    /*
        Destroy row indices
    */
    ierr = PetscFree(id_rows_A); CHKERRQ(ierr);
    /*

        SUBMATRIX B

        Acquire the submatrices of local and nonlocal rows needed to build
        the local rows of C

        Get only the rows of B needed for this operation

        Build the submatrices for each term

        NOTE: Assumes equal shapes for all A and all B matrices

    */
    ctx->submat_B.resize(ctx->nterms);
    PetscInt *id_rows_B;

    std::set<PetscInt> set_Brows;
    for (PetscInt Irow = ctx->Istart; Irow < ctx->Iend; ++Irow)
        set_Brows.insert(Irow % ctx->M_B[0]);

    PetscInt M_req_B = set_Brows.size();
    ierr = PetscMalloc1(M_req_B, &id_rows_B); CHKERRQ(ierr);

    PetscInt counter = 0;
    for (auto elem: set_Brows){
        id_rows_B[counter] = elem;
        ++counter;
    }
    PetscInt B_sub_start, B_sub_end;
    ierr = GetSubmatrix(B, ctx->N_B, ctx->nterms, M_req_B, id_rows_B, ctx->submat_B, B_sub_start, B_sub_end); CHKERRQ(ierr);

    ctx->map_B.resize(ctx->M_B[0]);
    for (size_t i = 0; i < set_Brows.size(); ++i){
        ctx->map_B[ id_rows_B[i] ] = i + B_sub_start;
    }
    ierr = PetscFree(id_rows_B); CHKERRQ(ierr);
    /*
        Map ownership
        Input: the row INDEX in the global matrix A
        Output: the corresponding row index in the locally-owned rows of submatrix A
    */
    ctx->row_shift_A = - Astart + A_sub_start;

    /******************* TIMINGS *******************/
    KRON_PS_TIMINGS_END(KRON_SUBMATRIX)
    #undef KRON_SUBMATRIX

    #define KRON_MAT "KronMatFree: Matrix Creation"
    KRON_PS_TIMINGS_INIT(KRON_MAT)
    KRON_PS_TIMINGS_START(KRON_MAT)
    /***********************************************/

    /* Create vector scatter object */
    Vec x_mpi;
    ierr = VecCreateMPI(comm, ctx->locrows, N_C, &x_mpi); CHKERRQ(ierr);
    ierr = VecScatterCreateToAll(x_mpi, &ctx->vsctx, &ctx->x_seq); CHKERRQ(ierr);
    ierr = VecDestroy(&x_mpi); CHKERRQ(ierr);

    /* Create the matrix */
    ierr = MatCreateShell(comm, ctx->locrows, ctx->locrows, M_C, N_C, (void*)ctx, &C); CHKERRQ(ierr);
    ierr = MatShellSetOperation(C, MATOP_MULT, (void(*)())MatMult_KronProdSum_MATSHELL); CHKERRQ(ierr);
    ierr = MatShellSetOperation(C, MATOP_DESTROY, (void(*)())MatDestroy_KronProdSum_MATSHELL); CHKERRQ(ierr);

    /******************* TIMINGS *******************/
    KRON_PS_TIMINGS_END(KRON_MAT)
    #undef KRON_MAT
    /***********************************************/

    return ierr;
}

#undef ROW_MAP_A
#undef ROW_MAP_B
#undef COL_MAP_A
#undef COL_MAP_B


