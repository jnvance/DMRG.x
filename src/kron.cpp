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
    MatCreate(comm, &C);
    MatSetSizes(C, PETSC_DECIDE, PETSC_DECIDE, M_C, N_C);
    MatSetFromOptions(C);
    MatSetUp(C);
    MatZeroEntries(C);

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
        SETERRQ(comm,1,"Incorrect Matrix size");
        PetscPrintf(comm, "Input(%d, %d) != Expected(%d, %d)\n", M_C_input, N_C_input, M_C, N_C);
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

            for (PetscInt Irow = Istart; Irow < Iend; ++Irow)
            {
                Arow = Irow/M_B;
                Brow = Irow % M_B;

                MatGetRow(submat_A, ROW_MAP_A(Arow), &ncols_A, &cols_A, &vals_A);
                MatGetRow(submat_B, ROW_MAP_B(Brow), &ncols_B, &cols_B, &vals_B);

                #ifdef __PRINT_TESTS__
                printf("[%2d] Irow: %-5d Arow: %-5d Brow: %-5d ncols_A: %-5d ncols_B: %-5d\n",
                    rank, Irow, Arow, Brow, ncols_A, ncols_B);
                #endif

                ncols_C = ncols_A * ncols_B;
                /*
                    TODO: This malloc might be costly, try to estimate the max number of nonzeros
                    of the product matrix and pre-allocate outside the loop
                */
                PetscInt*       cols_C = new PetscInt[ncols_C];
                PetscScalar*    vals_C = new PetscScalar[ncols_C];


                if (a == 1.)
                {
                    for (int j_A = 0; j_A < ncols_A; ++j_A)
                    {
                        for (int j_B = 0; j_B < ncols_B; ++j_B)
                        {
                            cols_C [ j_A * ncols_B + j_B ] = COL_MAP_A(cols_A[j_A]) * N_B + COL_MAP_B(cols_B[j_B]);
                            vals_C [ j_A * ncols_B + j_B ] = vals_A[j_A] * vals_B[j_B];
                        }
                    }
                }
                else
                {
                    for (int j_A = 0; j_A < ncols_A; ++j_A)
                    {
                        for (int j_B = 0; j_B < ncols_B; ++j_B)
                        {
                            cols_C [ j_A * ncols_B + j_B ] = COL_MAP_A(cols_A[j_A]) * N_B + COL_MAP_B(cols_B[j_B]);
                            vals_C [ j_A * ncols_B + j_B ] = a * vals_A[j_A] * vals_B[j_B];
                        }
                    }
                }

                MatSetValues(C, 1, &Irow, ncols_C, cols_C, vals_C, ADD_VALUES );

                delete [] cols_C;
                delete [] vals_C;

                MatRestoreRow(submat_B, ROW_MAP_B(Brow), &ncols_B, &cols_B, &vals_B);
                MatRestoreRow(submat_A, ROW_MAP_A(Arow), &ncols_A, &cols_A, &vals_A);
            };
    #endif

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


    MatDestroy(&submat_A);
    MatDestroy(&submat_B);
    return ierr;
}
