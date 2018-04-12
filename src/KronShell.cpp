#include "kron.hpp"
#include <../src/mat/impls/aij/seq/aij.h>

/*
    Contains submatrices
    and row mapping
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
    PetscInt& N_C);


PETSC_EXTERN PetscErrorCode GetSubmatrix(
    const std::vector<Mat>&       A,
    const std::vector<PetscInt>&  N_A,
    const PetscInt&               nterms,
    const PetscInt&               M_req_A,
    const PetscInt                *id_rows_A,
    std::vector<Mat>&       submat_A,
    PetscInt&               A_sub_start,
    PetscInt&               A_sub_end);


#define ROW_MAP_A(INDEX) ((INDEX) + ctx->row_shift_A)
#define ROW_MAP_B(INDEX) (ctx->map_B[(INDEX)])
#define COL_MAP_A(INDEX) ((INDEX))
#define COL_MAP_B(INDEX) ((INDEX))


PETSC_EXTERN PetscErrorCode MatMult_KronProdSum_MATSHELL(Mat A, Vec x, Vec y)
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



PETSC_EXTERN PetscErrorCode MatDestroy_KronProdSum_MATSHELL(Mat *p_mat)
{
    PetscErrorCode ierr = 0;

    CTX_KRON *ctx;
    ierr = MatShellGetContext(*p_mat,(void**)&ctx); CHKERRQ(ierr);

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
    ierr = MatDestroy(p_mat); CHKERRQ(ierr);

    return ierr;
}


PETSC_EXTERN PetscErrorCode MatKronProdSum_MATSHELL(
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
