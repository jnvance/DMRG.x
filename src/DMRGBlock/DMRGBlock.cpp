#include "DMRGBlock.hpp"
#include <numeric> // partial_sum
#include <iostream>
#include <../src/mat/impls/aij/seq/aij.h>    /* Mat_SeqAIJ */
#include <../src/mat/impls/aij/mpi/mpiaij.h> /* Mat_MPIAIJ */

/* External functions taken from MiscTools.cpp */
PETSC_EXTERN int64_t ipow(int64_t base, uint8_t exp);
PETSC_EXTERN PetscErrorCode MatSpinOneHalfSzCreate(const MPI_Comm& comm, Mat& Sz);
PETSC_EXTERN PetscErrorCode MatSpinOneHalfSpCreate(const MPI_Comm& comm, Mat& Sp);
PETSC_EXTERN PetscErrorCode InitSingleSiteOperator(const MPI_Comm& comm, const PetscInt dim, Mat* mat);
PETSC_EXTERN PetscErrorCode MatEnsureAssembled(const Mat& matin);

/** Internal macro for checking the initialization state of the block object */
#define CheckInit(func) if (PetscUnlikely(!init))\
    SETERRQ1(mpi_comm, PETSC_ERR_ARG_CORRUPT, "%s was called but block was not yet initialized.",func);

/** Internal macro for checking that a column index belongs in the magnetization block boundaries */
#define CheckIndex(row, col, cstart, cend) if((col) < (cstart) || (col) >= (cend))\
    SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "On row %d, index %d out of bounds [%d,%d) ",\
        (row), (col), (cstart), (cend));

PetscErrorCode Block_SpinOneHalf::Initialize(
    const MPI_Comm& comm_in,
    const PetscInt& num_sites_in,
    const PetscInt& num_states_in)
{
    PetscErrorCode ierr = 0;

    /*  Check whether to do verbose logging  */
    ierr = PetscOptionsGetBool(NULL,NULL,"-verbose",&verbose,NULL); CHKERRQ(ierr);

    /*  Initialize attributes  */
    mpi_comm = comm_in;
    ierr = MPI_Comm_rank(mpi_comm, &mpi_rank); CPP_CHKERRQ(ierr);
    ierr = MPI_Comm_size(mpi_comm, &mpi_size); CPP_CHKERRQ(ierr);

    /*  Initial number of sites and number of states  */
    num_sites = num_sites_in;
    if(num_states_in == PETSC_DEFAULT){
        num_states = ipow(loc_dim, num_sites);
    } else{
        num_states = num_states_in;
    }
    /** If num_states_in is PETSC_DEFAULT, the number of states is calculated exactly from the number of sites */

    /*  Initialize array of operator matrices  */
    Sz.resize(num_sites);
    Sp.resize(num_sites);
    Sm.resize(num_sites);

    /*  Initialize switch  */
    init = PETSC_TRUE;

    /** When creating a block for one site, the single-site operators are initialized using the default
        values and matrix operators for spin-1/2 defined in Block_SpinOneHalf::loc_dim, Block_SpinOneHalf::loc_qn_list and Block_SpinOneHalf::loc_qn_size */
    if (num_sites == 1)
    {
        /*  Create the spin operators for the single site  */
        ierr = MatSpinOneHalfSzCreate(mpi_comm, Sz[0]); CHKERRQ(ierr);
        ierr = MatSpinOneHalfSpCreate(mpi_comm, Sp[0]); CHKERRQ(ierr);

        /*  Initialize the magnetization sectors using the defaults for one site */
        ierr = Magnetization.Initialize(mpi_comm, loc_qn_list, loc_qn_size); CHKERRQ(ierr);

        /*  Check whether sector initialization was done right  */
        ierr = CheckSectors(); CHKERRQ(ierr);
    }
    /** When more than one site is requested, all operator matrices are created with the correct sizes based on the
        number of states */
    else if(num_sites > 1)
    {
        for(PetscInt isite = 0; isite < num_sites; ++isite)
        {
            ierr = InitSingleSiteOperator(mpi_comm, num_states, &Sz[isite]); CHKERRQ(ierr);
            ierr = InitSingleSiteOperator(mpi_comm, num_states, &Sp[isite]); CHKERRQ(ierr);
        }
    }
    else
        SETERRQ1(mpi_comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid input num_sites_in > 0. Given %d.", num_sites_in);

    return ierr;
}

PetscErrorCode Block_SpinOneHalf::Initialize(
    const MPI_Comm& comm_in,
    const PetscInt& num_sites_in,
    const std::vector<PetscReal>& qn_list_in,
    const std::vector<PetscInt>& qn_size_in)
{
    PetscErrorCode ierr = 0;

    if(PetscUnlikely(num_sites_in == 1))
        SETERRQ(mpi_comm, PETSC_ERR_ARG_OUTOFRANGE,
            "Invalid input num_sites_in cannot be equal to 1. Call a different Initialize() function.");

    QuantumNumbers Magnetization_temp;
    ierr = Magnetization_temp.Initialize(comm_in, qn_list_in, qn_size_in); CHKERRQ(ierr);
    PetscInt num_states_in = Magnetization_temp.NumStates();

    ierr = Initialize(comm_in, num_sites_in, num_states_in); CHKERRQ(ierr);
    Magnetization = Magnetization_temp;

    return ierr;
}

PetscErrorCode Block_SpinOneHalf::CheckOperatorArray(const Op_t& OpType) const
{
    PetscErrorCode ierr = 0;

    PetscInt label = 0;
    const Mat *Op;
    switch(OpType) {
        case OpSm: Op = &Sm[0]; break;
        case OpSz: Op = &Sz[0]; break;
        case OpSp: Op = &Sp[0]; break;
        default: SETERRQ(mpi_comm, PETSC_ERR_ARG_WRONG, "Incorrect operator type.");
        /** @throw PETSC_ERR_ARG_WRONG The operator type is incorrect */
    }

    /*  Check the size of each matrix and make sure that it
        matches the number of basis states  */
    PetscInt M, N;
    for(PetscInt isite = 0; isite < num_sites; ++isite)
    {
        if(!Op[isite])
            /** @throw PETSC_ERR_ARG_CORRUPT Matrix not yet created */
            SETERRQ2(mpi_comm, PETSC_ERR_ARG_CORRUPT, "%s[%d] matrix not yet created.", label, isite);

        ierr = MatGetSize(Op[isite], &M, &N); CHKERRQ(ierr);
        if (M != N)
            /** @throw PETSC_ERR_ARG_WRONG Matrix not square */
            SETERRQ2(mpi_comm, PETSC_ERR_ARG_WRONG, "%s[%d] matrix not square.", label, isite);

        if (M != num_states)
            /** @throw PETSC_ERR_ARG_WRONG Matrix dimension does not match the number of states */
            SETERRQ4(mpi_comm, PETSC_ERR_ARG_WRONG, "%s[%d] matrix dimension does not match "
                "the number of states. Expected %d. Got %d.", label, isite, num_states, M);
    }

    return ierr;
}


PetscErrorCode Block_SpinOneHalf::CheckOperators() const
{
    PetscErrorCode ierr = 0;
    CheckInit(__FUNCTION__); /** @throw PETSC_ERR_ARG_CORRUPT Block not yet initialized */

    ierr = CheckOperatorArray(OpSz); CHKERRQ(ierr);
    ierr = CheckOperatorArray(OpSp); CHKERRQ(ierr);

    if (init_Sm){
        ierr = CheckOperatorArray(OpSm); CHKERRQ(ierr);
    }

    return ierr;
}


PetscErrorCode Block_SpinOneHalf::CheckSectors() const
{
    PetscErrorCode ierr = 0;
    CheckInit(__FUNCTION__); /** @throw PETSC_ERR_ARG_CORRUPT Block not yet initialized */

    /*  Check whether the Magnetization object has been initialized correctly */
    ierr = Magnetization.CheckInitialized(); CHKERRQ(ierr);

    /*  The last element of qn_offset must match the total number of states  */
    PetscInt magNumStates = Magnetization.NumStates();

    if(num_states != magNumStates)
        /** @throw PETSC_ERR_ARG_WRONG The number of states in the Magnetization object
            and the internal value do not match */
        SETERRQ2(mpi_comm, PETSC_ERR_ARG_WRONG, "The number of states in the Magnetization object "
            "and the internal value do not match. " "Expected %d. Got %d.", num_states, magNumStates);

    return ierr;
}


PetscErrorCode Block_SpinOneHalf::MatCheckOperatorBlocks(const Op_t& OpType, const PetscInt& isite) const
{
    PetscErrorCode ierr = 0;

    /* Decipher inputs */
    Mat matin;
    if(isite >= num_sites) /** @throw PETSC_ERR_ARG_WRONG The input isite is out of bounds */
        SETERRQ2(mpi_comm, PETSC_ERR_ARG_OUTOFRANGE, "Input isite (%d) out of bounds [0,%d).", isite, num_sites);
    switch(OpType) {
        case OpSm: matin = Sm[isite]; break;
        case OpSz: matin = Sz[isite]; break;
        case OpSp: matin = Sp[isite]; break;
        default: SETERRQ(mpi_comm, PETSC_ERR_ARG_WRONG, "Incorrect operator type.");
        /** @throw PETSC_ERR_ARG_WRONG The operator type is incorrect */
    }
    /* Ensure that the matrix is assembled */
    ierr = MatEnsureAssembled(matin); CHKERRQ(ierr);

    /*  Check whether sector initialization was done right  */
    ierr = CheckSectors(); CHKERRQ(ierr);

    /* Get row and column layout */
    PetscInt rstart = matin->rmap->rstart;
    PetscInt lrows  = matin->rmap->n;
    PetscInt cstart = matin->cmap->rstart;
    // PetscInt lcols  = matin->cmap->n;
    PetscInt nrows  = matin->rmap->N;

    /* Check the matrix type */
    PetscBool matin_is_mpiaij;
    ierr = PetscObjectTypeCompare((PetscObject)matin, MATMPIAIJ, &matin_is_mpiaij); CHKERRQ(ierr);

    /* Do specific tasks for MATMPIAIJ using the diagonal structure */
    if(matin_is_mpiaij){
        /* Extract diagonal (A) and off-diagonal (B) sequential matrices */
        Mat_MPIAIJ *mat = (Mat_MPIAIJ*)matin->data;
        PetscInt *cmap = mat->garray;

        PetscInt nzA, nzB, *cA=nullptr, *cB=nullptr;

        /* Determine the starting block */
        PetscBool flg;
        PetscInt row_BlockIdx, col_GlobIdxStart, col_GlobIdxEnd;
        const std::vector<PetscInt>& qn_offset = Magnetization.Offsets();

        /* Ensure that empty processes do nothing */
        if(!(0 <= rstart && rstart < nrows)) return ierr;

        /* Calculate block boundaries */
        ierr = Magnetization.GlobalIdxToBlockIdx(rstart, row_BlockIdx); CHKERRQ(ierr); /* Call this function once */
        ierr = Magnetization.OpBlockToGlobalRange(row_BlockIdx, OpType, col_GlobIdxStart, col_GlobIdxEnd, flg); CHKERRQ(ierr);

        for(PetscInt lrow = 0; lrow < lrows ; ++lrow)
        {
            /* Decide whether to move to next BlockIdx for the current row */
            if(lrow+rstart >= qn_offset[row_BlockIdx+1]){
                ++row_BlockIdx;
                ierr = Magnetization.OpBlockToGlobalRange(row_BlockIdx, OpType, col_GlobIdxStart, col_GlobIdxEnd, flg); CHKERRQ(ierr);
            }

            ierr  = (*mat->A->ops->getrow)(mat->A, lrow, &nzA, &cA, nullptr);CHKERRQ(ierr);
            ierr  = (*mat->B->ops->getrow)(mat->B, lrow, &nzB, &cB, nullptr);CHKERRQ(ierr);

            if(!flg && nzA!=0 && nzB!=0)
                /** @throw PETSC_ERR_ARG_WRONG The current row should have no entries since it is not a valid quantum
                    number block */
                SETERRQ1(PETSC_COMM_SELF, 1, "Row %d should have no entries.", lrow+rstart);

            /* Check first and last element assuming entries are sorted */
            if(nzA){
                CheckIndex(lrow+rstart, cA[0] + cstart,     col_GlobIdxStart, col_GlobIdxEnd);
                CheckIndex(lrow+rstart, cA[nzA-1] + cstart, col_GlobIdxStart, col_GlobIdxEnd);
            }

            if(nzB){
                CheckIndex(lrow+rstart, cmap[cB[0]],     col_GlobIdxStart, col_GlobIdxEnd);
                CheckIndex(lrow+rstart, cmap[cB[nzB-1]], col_GlobIdxStart, col_GlobIdxEnd);
            }
        }
    }
    else{
        /** @throw PETSC_ERR_SUP This checking has been implemented specifically for MATMPIAIJ only */
        SETERRQ(mpi_comm, PETSC_ERR_SUP, "Implemented only for MATMPIAIJ.");
    }

    return ierr;
}


PetscErrorCode Block_SpinOneHalf::CheckOperatorBlocks() const
{
    PetscErrorCode ierr = 0;
    CheckInit(__FUNCTION__); /** @throw PETSC_ERR_ARG_CORRUPT Block not yet initialized */

    /* Check all operator matrices */
    ierr = CheckOperators(); CHKERRQ(ierr);

    /* Check operator blocks of Sz matrices */
    for(PetscInt isite = 0; isite < num_sites; ++isite){
        ierr = MatCheckOperatorBlocks(OpSz, isite); CHKERRQ(ierr);
    }

    /* Check operator blocks of Sp matrices */
    for(PetscInt isite = 0; isite < num_sites; ++isite){
        ierr = MatCheckOperatorBlocks(OpSp, isite); CHKERRQ(ierr);
    }

    return ierr;
}


PetscErrorCode Block_SpinOneHalf::CreateSm()
{
    PetscErrorCode ierr = 0;

    if(init_Sm) SETERRQ(mpi_comm, 1, "Sm was previously initialized. Call DestroySm() first.");

    ierr = CheckOperatorArray(OpSp); CHKERRQ(ierr);
    for(PetscInt isite = 0; isite < num_sites; ++isite){
        ierr = MatHermitianTranspose(Sp[isite], MAT_INITIAL_MATRIX, &Sm[isite]); CHKERRQ(ierr);
    }
    init_Sm = PETSC_TRUE;

    return ierr;
}


PetscErrorCode Block_SpinOneHalf::DestroySm()
{
    PetscErrorCode ierr = 0;
    if(!init_Sm) SETERRQ1(mpi_comm, 1, "%s was called but Sm was not yet initialized. ",__FUNCTION__);

    for(PetscInt isite = 0; isite < num_sites; ++isite){
        ierr = MatDestroy(&Sm[isite]); CHKERRQ(ierr);
    }
    init_Sm = PETSC_FALSE;

    return ierr;
}


PetscErrorCode Block_SpinOneHalf::Destroy()
{
    PetscErrorCode ierr = 0;
    CheckInit(__FUNCTION__); /** @throw PETSC_ERR_ARG_CORRUPT Block not yet initialized */

    /*  Destroy operator matrices  */
    for(PetscInt isite = 0; isite < num_sites; ++isite){
        ierr = MatDestroy(&Sz[isite]); CHKERRQ(ierr);
        ierr = MatDestroy(&Sp[isite]); CHKERRQ(ierr);
    }

    if (init_Sm){
        ierr = DestroySm(); CHKERRQ(ierr);
    }

    /*  Destroy arrays */
    init = PETSC_FALSE;

    return ierr;
}
