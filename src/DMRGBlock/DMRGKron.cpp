#include <vector>
#include <iostream>
#include <set>
#include <map>
#include <unordered_map>

#include "DMRGKron.hpp"
#include <../src/mat/impls/aij/seq/aij.h>    /* Mat_SeqAIJ */
#include <../src/mat/impls/aij/mpi/mpiaij.h> /* Mat_MPIAIJ */

#define DMRG_KRON_TESTING 0

#if DMRG_KRON_TESTING
    #ifndef PRINT_RANK_BEGIN
    #define PRINT_RANK_BEGIN() \
        for(PetscMPIInt irank = 0; irank < mpi_size; ++irank){\
            if(irank==mpi_rank){std::cout << "[" << mpi_rank << "]<<" << std::endl;
    #endif

    #ifndef PRINT_RANK_END
    #define PRINT_RANK_END() \
            std::cout << ">>[" << mpi_rank << "]" << std::endl;}\
        ierr = MPI_Barrier(mpi_comm); CHKERRQ(ierr);}
    #endif
#else
    #ifndef PRINT_RANK_BEGIN
    #define PRINT_RANK_BEGIN()
    #endif

    #ifndef PRINT_RANK_END
    #define PRINT_RANK_END()
    #endif
#endif

PETSC_EXTERN PetscErrorCode PreSplitOwnership(const MPI_Comm comm, const PetscInt N, PetscInt& locrows, PetscInt& Istart);
PETSC_EXTERN PetscErrorCode MatEnsureAssembled(const Mat& matin);
PETSC_EXTERN PetscErrorCode MatSetOption_MultipleMats(
    const std::vector<Mat>& matrices,
    const std::vector<MatOption>& options,
    const std::vector<PetscBool>& flgs);
PETSC_EXTERN PetscErrorCode MatSetOption_MultipleMatGroups(
    const std::vector<std::vector<Mat>>& matgroups,
    const std::vector<MatOption>& options,
    const std::vector<PetscBool>& flgs);
PETSC_EXTERN PetscErrorCode MatEnsureAssembled_MultipleMatGroups(const std::vector<std::vector<Mat>>& matgroups);
PETSC_EXTERN PetscErrorCode InitSingleSiteOperator(const MPI_Comm& comm, const PetscInt dim, Mat* mat);

static const PetscScalar one = 1.0;

/** Constructs the Kronecker product matrices explicitly. */
PetscErrorCode MatKronEyeConstruct(
    Block::SpinOneHalf& LeftBlock,
    Block::SpinOneHalf& RightBlock,
    const KronBlocks_t& KronBlocks,
    Block::SpinOneHalf& BlockOut
    )
{
    PetscErrorCode ierr = 0;

    MPI_Comm mpi_comm = LeftBlock.MPIComm();
    PetscMPIInt mpi_rank, mpi_size;
    ierr = MPI_Comm_rank(mpi_comm, &mpi_rank); CHKERRQ(ierr);
    ierr = MPI_Comm_size(mpi_comm, &mpi_size); CHKERRQ(ierr);

    /*  Determine the ownership range from the Sz matrix of the 0th site */
    if(!BlockOut.NumSites()) SETERRQ(PETSC_COMM_SELF,1,"BlockOut must have at least one site.");
    Mat matin = BlockOut.Sz(0);

    const PetscInt rstart = matin->rmap->rstart;
    const PetscInt lrows  = matin->rmap->n;
    const PetscInt cstart = matin->cmap->rstart;
    const PetscInt cend   = matin->cmap->rend;

    /*  Verify that the row and column mapping match what is expected */
    {
        PetscInt M,N, locrows, loccols, Istart, Cstart, lcols=cend-cstart;
        ierr = MatGetSize(matin, &M, &N); CHKERRQ(ierr);
        ierr = PreSplitOwnership(mpi_comm, M, locrows, Istart); CHKERRQ(ierr);
        if(locrows!=lrows) SETERRQ4(PETSC_COMM_SELF, 1,
            "Incorrect guess for locrows. Expected %d. Got %d. Size: %d x %d.", locrows, lrows, M, N);
        if(Istart!=rstart) SETERRQ4(PETSC_COMM_SELF, 1,
            "Incorrect guess for Istart. Expected %d. Got %d. Size: %d x %d.",  Istart, rstart, M, N);
        ierr = PreSplitOwnership(mpi_comm, N, loccols, Cstart); CHKERRQ(ierr);
        if(loccols!=lcols) SETERRQ4(PETSC_COMM_SELF, 1,
            "Incorrect guess for loccols. Expected %d. Got %d. Size: %d x %d.", loccols, lcols, M, N);
        if(Cstart!=cstart) SETERRQ4(PETSC_COMM_SELF, 1,
            "Incorrect guess for Cstart. Expected %d. Got %d. Size: %d x %d.",  Cstart, cstart, M, N);
    }

    const PetscInt TotSites = BlockOut.NumSites();
    const std::vector<PetscInt> NumSites_LR = {LeftBlock.NumSites(), RightBlock.NumSites()};

    /*  Maps the global indices of the rows of L and R to their local indices in the corresponding submatrices */
    std::unordered_map<PetscInt,PetscInt> MapRowsL, MapRowsR;

    /*  PETSc-compatible arrays which store the values of the global indices of the rows which will be
        stored into local submatrices */
    PetscInt *ReqRowsL, *ReqRowsR;
    size_t NReqRowsL, NReqRowsR;

    /**************************
        SUBMATRIX COLLECTION
     **************************/
    {
        KronBlocksIterator     KIter(KronBlocks,    rstart, rstart+lrows);

        /* Temporarily stores the global rows of L and R needed for the local rows of O */
        std::set<PetscInt> SetRowsL, SetRowsR;

        for( ; KIter.Loop(); ++KIter)
        {
            const PetscInt Row_BlockIdx_L = KIter.BlockIdxLeft();
            const PetscInt Row_BlockIdx_R = KIter.BlockIdxRight();
            const PetscInt Row_NumStates_R = RightBlock.Magnetization.Sizes()[Row_BlockIdx_R];
            const PetscInt Row_LocIdx_L = KIter.LocIdx() / Row_NumStates_R;
            const PetscInt Row_LocIdx_R = KIter.LocIdx() % Row_NumStates_R;
            const PetscInt Row_L = LeftBlock.Magnetization.BlockIdxToGlobalIdx(Row_BlockIdx_L, Row_LocIdx_L);
            const PetscInt Row_R = RightBlock.Magnetization.BlockIdxToGlobalIdx(Row_BlockIdx_R, Row_LocIdx_R);

            SetRowsL.insert(Row_L);
            SetRowsR.insert(Row_R);
        }

        /*  Store the results from the sets into the corresponding map where the key represents the global row while
            the value represents the sequential index where that global row will be stored */
        NReqRowsL = SetRowsL.size();
        NReqRowsR = SetRowsR.size();

        /*  Allocate enough space and store the required rows into a standard array for generating an index set */
        ierr = PetscCalloc1(NReqRowsL, &ReqRowsL); CHKERRQ(ierr);
        ierr = PetscCalloc1(NReqRowsR, &ReqRowsR); CHKERRQ(ierr);
        /*  Dump the set values into the array for local->global lookup and into the map for global->local lookup */
        {
            size_t idx = 0;
            for(PetscInt row: SetRowsL){
                ReqRowsL[idx] = row;
                MapRowsL[row] = idx++;
            }
            idx = 0;
            for(PetscInt row: SetRowsR){
                ReqRowsR[idx] = row;
                MapRowsR[row] = idx++;
            }
        }
    }

    /*  Submatrix array containing local rows of all operators of both the left and right sides */
    Mat **SubMatArray;
    ierr = PetscCalloc1(2*TotSites, &SubMatArray); CHKERRQ(ierr);
    const std::vector<PetscInt> SiteShifts_LR = {0, LeftBlock.NumSites()};
    #define p_SubMat(OPTYPE, SIDETYPE, ISITE) (SubMatArray[ (ISITE + (SiteShifts_LR [SIDETYPE]) )*2+(OPTYPE) ])
    #define SubMat(OPTYPE, SIDETYPE, ISITE) (*p_SubMat((OPTYPE), (SIDETYPE), (ISITE)))

    /*  Generate the index sets needed to get the rows and columns */
    IS isrow_L, isrow_R, iscol_L, iscol_R;
    /*  Get only some required rows */
    ierr = ISCreateGeneral(mpi_comm, NReqRowsL, ReqRowsL, PETSC_USE_POINTER, &isrow_L); CHKERRQ(ierr);
    ierr = ISCreateGeneral(mpi_comm, NReqRowsR, ReqRowsR, PETSC_USE_POINTER, &isrow_R); CHKERRQ(ierr);
    /*  Get all columns in each required row */
    ierr = ISCreateStride(mpi_comm, LeftBlock.NumStates(), 0, 1, &iscol_L); CHKERRQ(ierr);
    ierr = ISCreateStride(mpi_comm, RightBlock.NumStates(), 0, 1, &iscol_R); CHKERRQ(ierr);

    /*  Looping through all sites and matrices, get the submatrices containing the required rows */
    for(PetscInt isite=0; isite<LeftBlock.NumSites(); ++isite){
        ierr = MatCreateSubMatrices(LeftBlock.Sz(isite), 1, &isrow_L, &iscol_L,
                MAT_INITIAL_MATRIX, &p_SubMat(OpSz, SideLeft, isite)); CHKERRQ(ierr);
        ierr = MatCreateSubMatrices(LeftBlock.Sp(isite), 1, &isrow_L, &iscol_L,
                MAT_INITIAL_MATRIX, &p_SubMat(OpSp, SideLeft, isite)); CHKERRQ(ierr);
    }
    for(PetscInt isite=0; isite<RightBlock.NumSites(); ++isite){
        ierr = MatCreateSubMatrices(RightBlock.Sz(isite), 1, &isrow_R, &iscol_R,
                MAT_INITIAL_MATRIX, &p_SubMat(OpSz, SideRight, isite)); CHKERRQ(ierr);
        ierr = MatCreateSubMatrices(RightBlock.Sp(isite), 1, &isrow_R, &iscol_R,
                MAT_INITIAL_MATRIX, &p_SubMat(OpSp, SideRight, isite)); CHKERRQ(ierr);
    }

    /*******************
        PREALLOCATION
     *******************/
    /*  Array of vectors containing the number of elements in the diagonal and off-diagonal
        blocks of Sz and Sp matrices on each site */
    PetscInt *D_NNZ_all, *O_NNZ_all;
    ierr = PetscCalloc1(2*TotSites*lrows, &D_NNZ_all); CHKERRQ(ierr);
    ierr = PetscCalloc1(2*TotSites*lrows, &O_NNZ_all); CHKERRQ(ierr);
    #define Dnnz(OPTYPE,SIDETYPE,ISITE,LROW) (D_NNZ_all[ ((ISITE + (SiteShifts_LR [SIDETYPE]) )*2+(OPTYPE))*lrows + LROW ])
    #define Onnz(OPTYPE,SIDETYPE,ISITE,LROW) (O_NNZ_all[ ((ISITE + (SiteShifts_LR [SIDETYPE]) )*2+(OPTYPE))*lrows + LROW ])

    /*  Require all output block matrices to be preallocated */
    ierr = MatSetOption_MultipleMatGroups({ BlockOut.Sz(), BlockOut.Sp() },
        { MAT_NO_OFF_PROC_ENTRIES, MAT_NEW_NONZERO_LOCATION_ERR }, { PETSC_TRUE, PETSC_TRUE }); CHKERRQ(ierr);

    const std::vector<std::vector<Mat>>& MatOut_ZP = {BlockOut.Sz(), BlockOut.Sp()};
    PetscInt MaxElementsPerRow = 0;
    {
        std::vector<PetscInt> fws_O_Sp_LR, col_NStatesR_LR;

        KronBlocksIterator     KIter(KronBlocks,    rstart, rstart+lrows);
        for( ; KIter.Loop(); ++KIter)
        {
            const PetscInt lrow = KIter.Steps();
            const PetscInt Row_BlockIdx_L = KIter.BlockIdxLeft();
            const PetscInt Row_BlockIdx_R = KIter.BlockIdxRight();
            const PetscInt Row_NumStates_R = RightBlock.Magnetization.Sizes()[Row_BlockIdx_R];
            const PetscInt Row_LocIdx_L = KIter.LocIdx() / Row_NumStates_R;
            const PetscInt Row_LocIdx_R = KIter.LocIdx() % Row_NumStates_R;
            const PetscInt Row_L = LeftBlock.Magnetization.BlockIdxToGlobalIdx(Row_BlockIdx_L, Row_LocIdx_L);
            const PetscInt Row_R = RightBlock.Magnetization.BlockIdxToGlobalIdx(Row_BlockIdx_R, Row_LocIdx_R);
            const PetscInt LocRow_L = MapRowsL[Row_L];
            const PetscInt LocRow_R = MapRowsR[Row_R];

            PetscBool flg[2];
            PetscInt nz_L, nz_R, col_NStatesR;
            const PetscInt *idx_L, *idx_R;
            const PetscScalar *v_L, *v_R, *v_O;

            /* Precalculate the post-shift for Sz operators */
            const PetscInt fws_O_Sz = KIter.BlockStartIdx(OpSz);
            /*  Reduced redundant map lookup by pre-calculating all possible post-shifts and rblock numstates
                for S+ operators in the left and right blocks and updating them only when the Row_BlockIdx is updated */
            if(KIter.UpdatedBlock()){
                fws_O_Sp_LR = {
                    KronBlocks.Offsets(Row_BlockIdx_L + 1, Row_BlockIdx_R),
                    KronBlocks.Offsets(Row_BlockIdx_L, Row_BlockIdx_R + 1)
                };
                col_NStatesR_LR = {
                    RightBlock.Magnetization.Sizes(Row_BlockIdx_R),
                    RightBlock.Magnetization.Sizes(Row_BlockIdx_R + 1)
                };
            }

            /* Operator-dependent scope */
            for(Op_t OpType: BasicOpTypes)
            {
                /*  Calculate the backward pre-shift associated to taking only the non-zero quantum number block */
                const std::vector<PetscInt> shift_L = {
                    LeftBlock.Magnetization.OpBlockToGlobalRangeStart(Row_BlockIdx_L, OpType, flg[SideLeft]), 0 };
                const std::vector<PetscInt> shift_R = {
                    0, RightBlock.Magnetization.OpBlockToGlobalRangeStart(Row_BlockIdx_R, OpType, flg[SideRight])};

                for (Side_t SideType: SideTypes)
                {
                    /*  Calculate the forward shift for the final elements
                        corresponding to the first element of the non-zero block */
                    PetscInt fws_O;
                    if(OpType == OpSz)
                    {
                        col_NStatesR = Row_NumStates_R;
                        fws_O = fws_O_Sz; /* The row and column indices of the block are the same */
                        if(fws_O == -1) continue;
                    }
                    else if(OpType == OpSp)
                    {
                        /*  +1 on block index that corresponds to the side */
                        col_NStatesR = col_NStatesR_LR[SideType];
                        fws_O = fws_O_Sp_LR[SideType];
                        if(fws_O == -1) continue;
                    }
                    else
                    {
                        SETERRQ(mpi_comm, 1, "Invalid operator type.");
                    }

                    /*  Site-dependent scope */
                    for(PetscInt isite=0; isite < NumSites_LR[SideType]; ++isite)
                    {
                        if(!flg[SideType]) continue;

                        /* Get the pre-shift value of the operator based on whether [L,R] and [Sz,Sp] and set the other as 0 */
                        const PetscInt bks_L = shift_L[SideType];
                        const PetscInt bks_R = shift_R[SideType];
                        const Mat mat = SubMat(OpType, SideType, isite);

                        /* Fill one side (L/R) with operator values and fill the other (R/L) with the indentity */
                        if(SideType) /* Right */
                        {
                            nz_L = 1;
                            idx_L = &Row_LocIdx_L;
                            v_L = &one;
                            ierr = (*mat->ops->getrow)(mat, LocRow_R, &nz_R, (PetscInt**)&idx_R, (PetscScalar**)&v_R); CHKERRQ(ierr);
                            v_O = v_R;
                        }
                        else /* Left */
                        {
                            ierr = (*mat->ops->getrow)(mat, LocRow_L, &nz_L, (PetscInt**)&idx_L, (PetscScalar**)&v_L); CHKERRQ(ierr);
                            nz_R = 1;
                            idx_R = &Row_LocIdx_R;
                            v_R = &one;
                            v_O = v_L;
                        }

                        /* Calculate the resulting indices */
                        PetscInt idx;
                        PetscInt& diag  = Dnnz(OpType, SideType, isite,lrow);
                        PetscInt& odiag = Onnz(OpType, SideType, isite,lrow);
                        for(size_t l=0; l<nz_L; ++l){
                            for(size_t r=0; r<nz_R; ++r)
                            {
                                idx = (idx_L[l] - bks_L) * col_NStatesR + (idx_R[r] - bks_R) + fws_O;
                                if ( cstart <= idx && idx < cend ) ++diag;
                                else ++odiag;
                            }
                        }
                        PetscInt nelts = nz_L * nz_R;
                        if (nelts > MaxElementsPerRow) MaxElementsPerRow = nelts;
                    }
                }
            }
        }
    }

    /*  Call the preallocation for all matrices */
    for(Side_t SideType: SideTypes){
        for(Op_t OpType: BasicOpTypes){
            for(PetscInt isite = 0; isite < NumSites_LR[SideType]; ++isite){
                ierr = MatMPIAIJSetPreallocation(
                        MatOut_ZP[OpType][isite+SiteShifts_LR[SideType]],
                        0, &Dnnz(OpType,SideType,isite,0),
                        0, &Onnz(OpType,SideType,isite,0)); CHKERRQ(ierr);
                /* Note: Preallocation for seq not required as long as mpiaij(mkl) matrices are specified */
            }
        }
    }

    /*************************
        MATRIX CONSTRUCTION
     *************************/

    /* Allocate static workspace for idx */
    PetscInt *idx;
    ierr = PetscCalloc1(MaxElementsPerRow+1, &idx); CHKERRQ(ierr);

    {
        KronBlocksIterator     KIter(KronBlocks,    rstart, rstart+lrows); /* Iterates through component subspaces and final block */
        std::vector<PetscInt> fws_O_Sp_LR, col_NStatesR_LR;
         /* Iterate through all basis states belonging to local rows */
        for( ; KIter.Loop(); ++KIter)
        {
            const PetscInt lrow = KIter.Steps(); /* output local row index */
            const PetscInt Irow = lrow + rstart; /* output global row index */
            /* Index of the block in the Kronecker-product block */
            const PetscInt Row_BlockIdx_L = KIter.BlockIdxLeft();
            const PetscInt Row_BlockIdx_R = KIter.BlockIdxRight();
            const PetscInt Row_NumStates_R = RightBlock.Magnetization.Sizes()[Row_BlockIdx_R];
            /* Local index of the row in the Kronecker-product block */
            const PetscInt Row_LocIdx_L = KIter.LocIdx() / Row_NumStates_R;
            const PetscInt Row_LocIdx_R = KIter.LocIdx() % Row_NumStates_R;
            /* MPI row indices of the left and right blocks */
            const PetscInt Row_L = LeftBlock.Magnetization.BlockIdxToGlobalIdx(Row_BlockIdx_L, Row_LocIdx_L);
            const PetscInt Row_R = RightBlock.Magnetization.BlockIdxToGlobalIdx(Row_BlockIdx_R, Row_LocIdx_R);
            /* Corresponding indices in the sequential submatrices */
            const PetscInt LocRow_L = MapRowsL[Row_L];
            const PetscInt LocRow_R = MapRowsR[Row_R];

            PetscBool flg[2];
            PetscInt nz_L, nz_R, col_NStatesR;
            const PetscInt *idx_L, *idx_R;
            const PetscScalar *v_L, *v_R, *v_O;

            /* Precalculate the post-shift for Sz operators */
            const PetscInt fws_O_Sz = KIter.BlockStartIdx(OpSz);
            /*  Reduced redundant map lookup by pre-calculating all possible post-shifts and rblock numstates
                for S+ operators in the left and right blocks and updating them only when the Row_BlockIdx is updated */
            if(KIter.UpdatedBlock()){
                fws_O_Sp_LR = {
                    KronBlocks.Offsets(Row_BlockIdx_L + 1, Row_BlockIdx_R),
                    KronBlocks.Offsets(Row_BlockIdx_L, Row_BlockIdx_R + 1)
                };
                col_NStatesR_LR = {
                    RightBlock.Magnetization.Sizes(Row_BlockIdx_R),
                    RightBlock.Magnetization.Sizes(Row_BlockIdx_R + 1)
                };
            }

            /* Operator-dependent scope */
            const std::vector<std::vector<Mat>>& MatOut = {BlockOut.Sz(), BlockOut.Sp()};
            for(Op_t OpType: BasicOpTypes)
            {
                /*  Calculate the backward pre-shift associated to taking only the non-zero quantum number block */
                const std::vector<PetscInt> shift_L = {
                    LeftBlock.Magnetization.OpBlockToGlobalRangeStart(Row_BlockIdx_L, OpType, flg[SideLeft]), 0 };
                const std::vector<PetscInt> shift_R = {
                    0, RightBlock.Magnetization.OpBlockToGlobalRangeStart(Row_BlockIdx_R, OpType, flg[SideRight])};

                for (Side_t SideType: SideTypes)
                {
                    /*  Calculate the forward shift for the final elements
                        corresponding to the first element of the non-zero block */
                    PetscInt fws_O;
                    if(OpType == OpSz)
                    {
                        col_NStatesR = Row_NumStates_R;
                        fws_O = fws_O_Sz; /* The row and column indices of the block are the same */
                        if(fws_O == -1) continue;
                    }
                    else if(OpType == OpSp)
                    {
                        /*  +1 on block index that corresponds to the side */
                        col_NStatesR = col_NStatesR_LR[SideType];
                        fws_O = fws_O_Sp_LR[SideType];
                        if(fws_O == -1) continue;
                    }
                    else
                    {
                        SETERRQ(mpi_comm, 1, "Invalid operator type.");
                    }

                    /*  Corresponding shift in position of the final site */
                    const PetscInt ishift = SiteShifts_LR[SideType];

                    /*  Site-dependent scope */
                    for(PetscInt isite=0; isite < NumSites_LR[SideType]; ++isite)
                    {
                        if(!flg[SideType]) continue;

                        /* Get the pre-shift value of the operator based on whether [L,R] and [Sz,Sp] and set the other as 0 */
                        const PetscInt bks_L = shift_L[SideType];
                        const PetscInt bks_R = shift_R[SideType];
                        const Mat mat = SubMat(OpType, SideType, isite);

                        /* Fill one side (L/R) with operator values and fill the other (R/L) with the indentity */
                        if(SideType) /* Right */
                        {
                            nz_L = 1;
                            idx_L = &Row_LocIdx_L;
                            v_L = &one;
                            ierr = (*mat->ops->getrow)(mat, LocRow_R, &nz_R, (PetscInt**)&idx_R, (PetscScalar**)&v_R); CHKERRQ(ierr);
                            v_O = v_R;
                        }
                        else /* Left */
                        {
                            ierr = (*mat->ops->getrow)(mat, LocRow_L, &nz_L, (PetscInt**)&idx_L, (PetscScalar**)&v_L); CHKERRQ(ierr);
                            nz_R = 1;
                            idx_R = &Row_LocIdx_R;
                            v_R = &one;
                            v_O = v_L;
                        }

                        /* Calculate the resulting indices */
                        for(size_t l=0; l<nz_L; ++l)
                            for(size_t r=0; r<nz_R; ++r)
                                idx[l*nz_R+r] = (idx_L[l] - bks_L) * col_NStatesR + (idx_R[r] - bks_R) + fws_O;

                        /* Set the matrix elements for this row in the output matrix */
                        ierr = MatSetValues(MatOut[OpType][isite+ishift], 1, &Irow, nz_L*nz_R, &idx[0], v_O, INSERT_VALUES); CHKERRQ(ierr);
                    }
                }
            }
        }
    }

    /*  Assemble all output block matrices */
    ierr = MatEnsureAssembled_MultipleMatGroups({BlockOut.Sz(), BlockOut.Sp()}); CHKERRQ(ierr);

    for(PetscInt i=0; i<2*TotSites; ++i){
        ierr = MatDestroy(SubMatArray[i]); CHKERRQ(ierr);
    }
    ierr = PetscFree(idx); CHKERRQ(ierr);
    ierr = PetscFree(SubMatArray); CHKERRQ(ierr);
    ierr = ISDestroy(&isrow_L); CHKERRQ(ierr);
    ierr = ISDestroy(&isrow_R); CHKERRQ(ierr);
    ierr = ISDestroy(&iscol_L); CHKERRQ(ierr);
    ierr = ISDestroy(&iscol_R); CHKERRQ(ierr);
    ierr = PetscFree(ReqRowsL); CHKERRQ(ierr);
    ierr = PetscFree(ReqRowsR); CHKERRQ(ierr);
    ierr = PetscFree(D_NNZ_all); CHKERRQ(ierr);
    ierr = PetscFree(O_NNZ_all); CHKERRQ(ierr);
    #undef p_SubMat
    #undef SubMat
    #undef Dnnz
    #undef Onnz
    return ierr;
}


PetscErrorCode KronEye_Explicit(
    Block::SpinOneHalf& LeftBlock,
    Block::SpinOneHalf& RightBlock,
    Block::SpinOneHalf& BlockOut
    )
{
    PetscErrorCode ierr = 0;

    MPI_Comm mpi_comm = LeftBlock.MPIComm();
    if(mpi_comm != RightBlock.MPIComm()) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Input blocks must have the same communicator.");
    PetscMPIInt mpi_rank, mpi_size;
    ierr = MPI_Comm_rank(mpi_comm, &mpi_rank); CHKERRQ(ierr);
    ierr = MPI_Comm_size(mpi_comm, &mpi_size); CHKERRQ(ierr);

    /*  For checking the accuracy of the routine. TODO: Remove later */
    #if DMRG_KRON_TESTING
        PRINT_RANK_BEGIN()
        std::cout << "***** Kron_Explicit *****" << std::endl;
        std::cout << "LeftBlock  qn_list:   ";
        for(auto i: LeftBlock.Magnetization.List()) std::cout << i << "   ";
        std::cout << std::endl;

        std::cout << "LeftBlock  qn_size:   ";
        for(auto i: LeftBlock.Magnetization.Sizes()) std::cout << i << "   ";
        std::cout << std::endl;

        std::cout << "LeftBlock  qn_offset: ";
        for(auto i: LeftBlock.Magnetization.Offsets()) std::cout << i << "   ";
        std::cout << std::endl;

        std::cout << std::endl;

        std::cout << "RightBlock qn_list:   ";
        for(auto i: RightBlock.Magnetization.List()) std::cout << i << "   ";
        std::cout << std::endl;

        std::cout << "RightBlock qn_size:   ";
        for(auto i: RightBlock.Magnetization.Sizes()) std::cout << i << "   ";
        std::cout << std::endl;

        std::cout << "RightBlock qn_offset: ";
        for(auto i: RightBlock.Magnetization.Offsets()) std::cout << i << "   ";
        std::cout << std::endl;
        PRINT_RANK_END()
    #endif

    /* Check the validity of input blocks */
    ierr = LeftBlock.CheckOperators(); CHKERRQ(ierr);
    ierr = LeftBlock.CheckSectors(); CHKERRQ(ierr);
    ierr = LeftBlock.CheckOperatorBlocks(); CHKERRQ(ierr); /* NOTE: Possibly costly operation */
    ierr = RightBlock.CheckOperators(); CHKERRQ(ierr);
    ierr = RightBlock.CheckSectors(); CHKERRQ(ierr);
    ierr = RightBlock.CheckOperatorBlocks(); CHKERRQ(ierr); /* NOTE: Possibly costly operation */

    /*  Create a list of tuples of quantum numbers following the kronecker product structure */
    KronBlocks_t KronBlocks(LeftBlock, RightBlock);

    #if DMRG_KRON_TESTING
        PRINT_RANK_BEGIN()
        {
            PetscInt i = 0;
            std::cout << "KronBlocks: \n";
            for(KronBlock_t kb: KronBlocks.data())
            {
                std::cout << "( "
                    << std::get<0>(kb) << ", "
                    << std::get<1>(kb) << ", "
                    << std::get<2>(kb) << ", "
                    << std::get<3>(kb) << ", "
                    << KronBlocks.Offsets()[i++] <<" )\n";
            }
            std::cout << "*************************" << std::endl;
        }
        PRINT_RANK_END()
    #endif

    /*  Count the input and output number of sites */
    PetscInt nsites_left  = LeftBlock.NumSites();
    PetscInt nsites_right = RightBlock.NumSites();
    PetscInt nsites_out   = nsites_left + nsites_right;

    /*  Count the input and output number of sectors */
    PetscInt nsectors_left  = LeftBlock.Magnetization.NumSectors();
    PetscInt nsectors_right = RightBlock.Magnetization.NumSectors();
    PetscInt nsectors_out   = nsectors_left * nsectors_right;
    if(PetscUnlikely((size_t) KronBlocks.size() != nsectors_out ))
        SETERRQ2(mpi_comm, 1, "Mismatch in number of sectors. Expected %lu. Got %d.", KronBlocks.size(), nsectors_out);

    /*  Count the input and output number of states */
    PetscInt nstates_left  = LeftBlock.Magnetization.NumStates();
    PetscInt nstates_right = RightBlock.Magnetization.NumStates();
    PetscInt nstates_out   = nstates_left * nstates_right;
    PetscInt KronBlocks_nstates = KronBlocks.NumStates();
    if(PetscUnlikely(KronBlocks_nstates != nstates_out))
        SETERRQ2(mpi_comm, 1, "Mismatch in number of states. Expected %lu. Got %d.", KronBlocks_nstates, nstates_out);

    /*  Some quantum numbers that appear multiple times need to be grouped into a single quantum number block */
    std::vector<PetscReal>  QN_List;
    std::vector<PetscInt>   QN_Size;
    PetscReal QN_last = 0;
    for (auto tup: KronBlocks.data()){
        const PetscReal& qn   = std::get<0>(tup);
        const PetscInt&  size = std::get<3>(tup);
        if(qn < QN_last || QN_List.size()==0){
            QN_List.push_back(qn);
            QN_Size.push_back(size);
        } else {
            QN_Size.back() += size;
        }
        QN_last = qn;
    }

    PetscInt QN_Size_total = 0;
    for(PetscInt size: QN_Size) QN_Size_total += size;
    if(PetscUnlikely(nstates_out != QN_Size_total))
        SETERRQ2(mpi_comm, 1, "Mismatch in number of states. Expected %d. Got %d.", nstates_out, QN_Size_total);

    #if DMRG_KRON_TESTING
        PRINT_RANK_BEGIN()
        std::cout << "QN_List: "; for(auto q: QN_List) std::cout << q << " "; std::cout << std::endl;
        std::cout << "Total Sites: " << nsites_out << std::endl;
        std::cout << "Total Sectors: " << nsectors_out << std::endl;
        std::cout << "Total States: " << nstates_out << std::endl;
        PRINT_RANK_END()
    #endif

    /*  Initialize the new block using the quantum number blocks */
    ierr = BlockOut.Initialize(mpi_comm, nsites_out, QN_List, QN_Size); CHKERRQ(ierr);

    #if DMRG_KRON_TESTING
        PRINT_RANK_BEGIN()
        std::cout << "Mag: QN_List: "; for(auto q: BlockOut.Magnetization.List()) std::cout << q << " "; std::cout << std::endl;
        std::cout << "Mag: QN_Size: "; for(auto q: BlockOut.Magnetization.Sizes()) std::cout << q << " "; std::cout << std::endl;
        PRINT_RANK_END()
    #endif

    /*  Combine sites from the old blocks to form the new block */
    /*  Expand the left-block states explicitly by padding identities to the right */
    /*  Expand the right-block states explicitly by padding identities to the left */
    ierr = MatKronEyeConstruct(LeftBlock, RightBlock, KronBlocks, BlockOut);  CHKERRQ(ierr);

    return ierr;
}


typedef struct {
    PetscInt rstart;
    PetscInt lrows;
    PetscInt cstart;
    PetscInt cend;

    /* Temporarily stores the global rows of L and R needed for the local rows of O */
    std::set<PetscInt> SetRowsL, SetRowsR;

    /*  Maps the global indices of the rows of L and R to their local indices in the corresponding submatrices */
    std::unordered_map<PetscInt,PetscInt> MapRowsL, MapRowsR;

} KronSumCtx;


PetscErrorCode KronBlocks_t::VerifySzAssumption(
    const std::vector< Mat >& Matrices,
    const Side_t& SideType
    )
{
    PetscErrorCode ierr = 0;
    for(const Mat& mat: Matrices){
        if (SideType == SideLeft){
            ierr = LeftBlock.MatCheckOperatorBlocks(OpSz, mat); CHKERRQ(ierr);
        } else if (SideType == SideRight){
            ierr = RightBlock.MatCheckOperatorBlocks(OpSz, mat); CHKERRQ(ierr);
        }
    }
    return(0);
}


PetscErrorCode KronBlocks_t::MatKronEyeAdd(
    const std::vector< Mat >& Matrices,
    const Side_t& SideType,
    Mat& MatOut
    )
{
    PetscErrorCode ierr = 0;

    return ierr;
}


PetscErrorCode KronBlocks_t::KronSumConstruct(
    const std::vector< Hamiltonians::Term >& Terms,
    Mat& MatOut
    )
{
    PetscErrorCode ierr = 0;

    /*  Count the input and total number of sites */
    PetscInt nsites_left  = LeftBlock.NumSites();
    PetscInt nsites_right = RightBlock.NumSites();
    PetscInt nsites_out   = nsites_left + nsites_right;

    /*  Check that the maximum site index in Terms is less than the number of sites in the blocks */
    PetscInt Max_Isite = 0;
    for(const Hamiltonians::Term& term: Terms){
        Max_Isite = ( term.Isite > Max_Isite ) ? term.Isite : Max_Isite;
        Max_Isite = ( term.Jsite > Max_Isite ) ? term.Jsite : Max_Isite;
    }
    if(Max_Isite >= nsites_out) SETERRQ2(mpi_comm, 1, "Maximum site index from Terms (%d) has to be "
        "less than the total number of sites in the blocks (%d).", Max_Isite, nsites_out);

    /*  Check the validity of input blocks */
    ierr = LeftBlock.CheckOperators(); CHKERRQ(ierr);
    ierr = LeftBlock.CheckSectors(); CHKERRQ(ierr);
    ierr = LeftBlock.CheckOperatorBlocks(); CHKERRQ(ierr); /* NOTE: Possibly costly operation */
    ierr = RightBlock.CheckOperators(); CHKERRQ(ierr);
    ierr = RightBlock.CheckSectors(); CHKERRQ(ierr);
    ierr = RightBlock.CheckOperatorBlocks(); CHKERRQ(ierr); /* NOTE: Possibly costly operation */

    /*  Classify the terms according to whether an intra-block or inter-block product will be performed */
    std::vector< Hamiltonians::Term > TermsLL, TermsRR; /* Intra-block */
    std::vector< Hamiltonians::Term > TermsLR; /* Inter-block */
    for( const Hamiltonians::Term& term: Terms ){
        if      ((0 <= term.Isite && term.Isite < nsites_left) && (0 <= term.Jsite && term.Jsite < nsites_left)){
            TermsLL.push_back(term);
        }
        else if ((0 <= term.Isite && term.Isite < nsites_left) && (nsites_left <= term.Jsite && term.Jsite < nsites_out)){
            TermsLR.push_back(term);
        }
        else if ((nsites_left <= term.Isite && term.Isite < nsites_out) && (nsites_left <= term.Jsite && term.Jsite < nsites_out)){
            TermsRR.push_back(term);
        }
        else {
            SETERRQ4(mpi_comm, 1, "Invalid term: Isite=%d Jsite=%d for nsites_left=%d and nsites_right=%d.",
                term.Isite, term.Jsite, nsites_left, nsites_right);
        }
    }

    /*  REFLECTION SYMMETRY: Reverse the sequence of sites in the right block so that
        new sites are always located at the interface */
    for (Hamiltonians::Term& term: TermsLR){
        term.Jsite = nsites_out - 1 - term.Jsite;
    }
    for (Hamiltonians::Term& term: TermsRR){
        term.Isite = nsites_out - 1 - term.Isite;
        term.Jsite = nsites_out - 1 - term.Jsite;
    }

    /*  Check whether there is any need to create the Sm matrices */
    PetscBool CreateSmL = PETSC_FALSE, CreateSmR = PETSC_FALSE;
    for( const Hamiltonians::Term& term: TermsLL){
        if(term.Iop == OpSm || term.Jop == OpSm) { CreateSmL = PETSC_TRUE; break; }
    }
    for( const Hamiltonians::Term& term: TermsRR){
        if(term.Iop == OpSm || term.Jop == OpSm) { CreateSmR = PETSC_TRUE; break; }
    }
    if(!CreateSmL) for( const Hamiltonians::Term& term: TermsLR){
        if(term.Iop == OpSm) { CreateSmL = PETSC_TRUE; break; }
    }
    if(!CreateSmR) for( const Hamiltonians::Term& term: TermsLR){
        if(term.Jop == OpSm) { CreateSmR = PETSC_TRUE; break; }
    }
    if(CreateSmL){
        ierr = LeftBlock.CreateSm(); CHKERRQ(ierr);
    }
    if(CreateSmR){
        ierr = RightBlock.CreateSm(); CHKERRQ(ierr);
    }

    #if DMRG_KRON_TESTING
        if(!mpi_rank) {
            printf(" nsites_left=%d nsites_right=%d nsites_out=%d\n", nsites_left, nsites_right, nsites_out);
            printf(" TermsLL\n");
            for(const Hamiltonians::Term& term: TermsLL)
            {
                printf("%.2f %2s(%2d) %2s(%2d)\n", term.a, (OpString.find(term.Iop)->second).c_str(), term.Isite,
                    (OpString.find(term.Jop)->second).c_str(), term.Jsite );
            }
            printf(" TermsLR\n");
            for(const Hamiltonians::Term& term: TermsLR)
            {
                printf("%.2f %2s(%2d) %2s(%2d)\n", term.a, (OpString.find(term.Iop)->second).c_str(), term.Isite,
                    (OpString.find(term.Jop)->second).c_str(), term.Jsite );
            }
            printf(" TermsRR\n");
            for(const Hamiltonians::Term& term: TermsRR)
            {
                printf("%.2f %2s(%2d) %2s(%2d)\n", term.a, (OpString.find(term.Iop)->second).c_str(), term.Isite,
                    (OpString.find(term.Jop)->second).c_str(), term.Jsite );
            }
        }
    #endif

    /*  Initialize the output matrix with the correct dimensions */
    ierr = MatDestroy(&MatOut); CHKERRQ(ierr);
    ierr = InitSingleSiteOperator(mpi_comm, num_states, &MatOut); CHKERRQ(ierr);
    /*  Include other parameters or setup options here */

    KronSumCtx ctx;
    const PetscInt rstart = ctx.rstart = MatOut->rmap->rstart;
    const PetscInt lrows = ctx.lrows  = MatOut->rmap->n;
    const PetscInt cstart = ctx.cstart = MatOut->cmap->rstart;
    const PetscInt cend = ctx.cend   = MatOut->cmap->rend;

    /*  Verify that the row and column mapping match what is expected */
    {
        PetscInt M,N, locrows, loccols, Istart, Cstart, lcols=cend-cstart;
        ierr = MatGetSize(MatOut, &M, &N); CHKERRQ(ierr);
        ierr = PreSplitOwnership(mpi_comm, M, locrows, Istart); CHKERRQ(ierr);
        if(locrows!=lrows) SETERRQ4(PETSC_COMM_SELF, 1,
            "Incorrect guess for locrows. Expected %d. Got %d. Size: %d x %d.", locrows, lrows, M, N);
        if(Istart!=rstart) SETERRQ4(PETSC_COMM_SELF, 1,
            "Incorrect guess for Istart. Expected %d. Got %d. Size: %d x %d.",  Istart, rstart, M, N);
        ierr = PreSplitOwnership(mpi_comm, N, loccols, Cstart); CHKERRQ(ierr);
        if(loccols!=lcols) SETERRQ4(PETSC_COMM_SELF, 1,
            "Incorrect guess for loccols. Expected %d. Got %d. Size: %d x %d.", loccols, lcols, M, N);
        if(Cstart!=cstart) SETERRQ4(PETSC_COMM_SELF, 1,
            "Incorrect guess for Cstart. Expected %d. Got %d. Size: %d x %d.",  Cstart, cstart, M, N);
    }

    /*  For each of the intra-block terms, perform the associated matrix multiplications of operators
        residing on the same blocks */
    {
        std::vector< Mat > OpProdLL(TermsLL.size()), OpProdRR(TermsRR.size());

        #define GetBlockMat(BLOCK,OP,ISITE)\
            (OP==OpSp ? BLOCK.Sp(ISITE) : (OP==OpSm ? BLOCK.Sm(ISITE) : (OP==OpSz ? BLOCK.Sz(ISITE) : NULL)))

        /* Left intra-block */
        for(size_t it = 0; it < TermsLL.size(); ++it)
        {
            const Mat& A = GetBlockMat(LeftBlock,TermsLL[it].Iop,TermsLL[it].Isite);
            const Mat& B = GetBlockMat(LeftBlock,TermsLL[it].Jop,TermsLL[it].Jsite);
            ierr = MatMatMult(A, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT, OpProdLL.data()+it); CHKERRQ(ierr);
            ierr = MatScale(OpProdLL[it], TermsLL[it].a); CHKERRQ(ierr);
        }
        ierr = VerifySzAssumption(OpProdLL, SideLeft); CHKERRQ(ierr);
        ierr = MatKronEyeAdd(OpProdLL, SideLeft, MatOut); CHKERRQ(ierr);
        for (Mat& mat: OpProdLL){ ierr = MatDestroy(&mat); CHKERRQ(ierr); }

        /* Right intra-block */
        for(size_t it = 0; it < TermsRR.size(); ++it)
        {
            const Mat& A = GetBlockMat(RightBlock,TermsRR[it].Iop,TermsRR[it].Isite);
            const Mat& B = GetBlockMat(RightBlock,TermsRR[it].Jop,TermsRR[it].Jsite);
            ierr = MatMatMult(A, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT, OpProdRR.data()+it); CHKERRQ(ierr);
            ierr = MatScale(OpProdRR[it], TermsRR[it].a); CHKERRQ(ierr);
        }
        ierr = VerifySzAssumption(OpProdRR, SideRight); CHKERRQ(ierr);
        ierr = MatKronEyeAdd(OpProdRR, SideRight, MatOut); CHKERRQ(ierr);
        for (Mat& mat: OpProdRR){ ierr = MatDestroy(&mat); CHKERRQ(ierr); }

        #undef GetBlockMat
    }

    /*  Destroy Sm in advance to avoid clashes with modifications in Sp */
    if(CreateSmL){
        ierr = LeftBlock.DestroySm(); CHKERRQ(ierr);
    }
    if(CreateSmR){
        ierr = RightBlock.DestroySm(); CHKERRQ(ierr);
    }
    return(0);
}
