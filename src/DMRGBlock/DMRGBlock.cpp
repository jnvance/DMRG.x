#include "DMRGBlock.hpp"
#include <numeric> // partial_sum
#include <iostream>
#include <sstream>
#include <iomanip>

#include <../src/mat/impls/aij/seq/aij.h>    /* Mat_SeqAIJ */
#include <../src/mat/impls/aij/mpi/mpiaij.h> /* Mat_MPIAIJ */

/* External functions taken from MiscTools.cpp */
PETSC_EXTERN int64_t ipow(int64_t base, uint8_t exp);
PETSC_EXTERN PetscErrorCode MatSpinOneHalfSzCreate(const MPI_Comm& comm, Mat& Sz);
PETSC_EXTERN PetscErrorCode MatSpinOneHalfSpCreate(const MPI_Comm& comm, Mat& Sp);
PETSC_EXTERN PetscErrorCode InitSingleSiteOperator(const MPI_Comm& comm, const PetscInt dim, Mat* mat);
PETSC_EXTERN PetscErrorCode MatEnsureAssembled(const Mat& matin);
PETSC_EXTERN PetscErrorCode MatEnsureAssembled_MultipleMats(const std::vector<Mat>& matrices);
PETSC_EXTERN PetscErrorCode MatEyeCreate(const MPI_Comm& comm, const PetscInt& dim, Mat& eye);
PETSC_EXTERN PetscErrorCode Makedir(const std::string& dir_name);

/** Internal macro for checking the initialization state of the block object */
#define CheckInit(func) if (PetscUnlikely(!init))\
    SETERRQ1(mpi_comm, PETSC_ERR_ARG_CORRUPT, "%s was called but block was not yet initialized.",func);

/** Internal macro for checking that a column index belongs in the magnetization block boundaries */
#define CheckIndex(row, col, cstart, cend) if((col) < (cstart) || (col) >= (cend))\
    SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "On row %d, index %d out of bounds [%d,%d) ",\
        (row), (col), (cstart), (cend));

PetscErrorCode Block::SpinOneHalf::Initialize(
    const MPI_Comm& comm_in)
{
    PetscErrorCode ierr;
    if(mpi_init) SETERRQ(mpi_comm,1,"This initializer should only be called once.");
    mpi_comm = comm_in;
    ierr = MPI_Comm_rank(mpi_comm, &mpi_rank); CPP_CHKERRQ(ierr);
    ierr = MPI_Comm_size(mpi_comm, &mpi_size); CPP_CHKERRQ(ierr);
    mpi_init = PETSC_TRUE;
    return(0);
}


PetscErrorCode Block::SpinOneHalf::Initialize(
    const MPI_Comm& comm_in,
    const PetscInt& num_sites_in,
    const PetscInt& num_states_in)
{
    PetscErrorCode ierr = 0;

    /*  Check whether to do verbose logging  */
    ierr = PetscOptionsGetBool(NULL,NULL,"-verbose",&verbose,NULL); CHKERRQ(ierr);

    /*  Initialize attributes  */
    if(!mpi_init){
        ierr = Initialize(comm_in); CHKERRQ(ierr);
    } else if (comm_in!=mpi_comm) {
        SETERRQ(PETSC_COMM_SELF,1,"Mismatch in MPI communicators.");
    }

    /*  Initial number of sites and number of states  */
    num_sites = num_sites_in;
    if(num_states_in == PETSC_DEFAULT){
        num_states = ipow(loc_dim, num_sites);
    } else{
        num_states = num_states_in;
    }
    /** If num_states_in is PETSC_DEFAULT, the number of states is calculated exactly from the number of sites */

    /*  Initialize array of operator matrices  */
    SzData.resize(num_sites);
    SpData.resize(num_sites);
    SmData.resize(num_sites);

    /*  Initialize switch  */
    init = PETSC_TRUE;

    /** When creating a block for one site, the single-site operators are initialized using the default
        values and matrix operators for spin-1/2 defined in Block::SpinOneHalf::loc_dim, Block::SpinOneHalf::loc_qn_list and Block::SpinOneHalf::loc_qn_size */
    if (num_sites == 1)
    {
        /*  Create the spin operators for the single site  */
        ierr = MatSpinOneHalfSzCreate(mpi_comm, SzData[0]); CHKERRQ(ierr);
        ierr = MatSpinOneHalfSpCreate(mpi_comm, SpData[0]); CHKERRQ(ierr);

        /*  Also initialize the single-site Hamiltonian which is defaulted to zero */
        ierr = InitSingleSiteOperator(mpi_comm, num_states, &H); CHKERRQ(ierr);
        ierr = MatEnsureAssembled(H); CHKERRQ(ierr);
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
            ierr = InitSingleSiteOperator(mpi_comm, num_states, &SzData[isite]); CHKERRQ(ierr);
            ierr = InitSingleSiteOperator(mpi_comm, num_states, &SpData[isite]); CHKERRQ(ierr);
        }
    }
    else
        SETERRQ1(mpi_comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid input num_sites_in > 0. Given %d.", num_sites_in);

    return ierr;
}

PetscErrorCode Block::SpinOneHalf::Initialize(
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

PetscErrorCode Block::SpinOneHalf::Initialize(
    const PetscInt& num_sites_in,
    const QuantumNumbers& qn_in)
{
    PetscErrorCode ierr = 0;

    ierr = qn_in.CheckInitialized(); CHKERRQ(ierr);
    ierr = Initialize(qn_in.MPIComm(), num_sites_in, qn_in.NumStates()); CHKERRQ(ierr);
    Magnetization = qn_in;

    return ierr;
}

PetscErrorCode Block::SpinOneHalf::CheckOperatorArray(const Op_t& OpType) const
{
    PetscErrorCode ierr = 0;

    PetscInt label = 0;
    const Mat *Op;
    switch(OpType) {
        case OpSm: Op = SmData.data(); break;
        case OpSz: Op = SzData.data(); break;
        case OpSp: Op = SpData.data(); break;
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


PetscErrorCode Block::SpinOneHalf::CheckOperators() const
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


PetscErrorCode Block::SpinOneHalf::CheckSectors() const
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


PetscErrorCode Block::SpinOneHalf::MatOpCheckOperatorBlocks(const Op_t& OpType, const PetscInt& isite) const
{
    PetscErrorCode ierr = 0;

    /* Decipher inputs */
    Mat matin;
    if(isite >= num_sites) /** @throw PETSC_ERR_ARG_WRONG The input isite is out of bounds */
        SETERRQ2(mpi_comm, PETSC_ERR_ARG_OUTOFRANGE, "Input isite (%d) out of bounds [0,%d).", isite, num_sites);
    switch(OpType) {
        case OpSm: matin = SmData[isite]; break;
        case OpSz: matin = SzData[isite]; break;
        case OpSp: matin = SpData[isite]; break;
        default: SETERRQ(mpi_comm, PETSC_ERR_ARG_WRONG, "Incorrect operator type.");
        /** @throw PETSC_ERR_ARG_WRONG The operator type is incorrect */
    }

    ierr = MatCheckOperatorBlocks(OpType, matin); CHKERRQ(ierr);

    return ierr;
}


PetscErrorCode Block::SpinOneHalf::MatCheckOperatorBlocks(const Op_t& OpType, const Mat& matin) const
{
    PetscErrorCode ierr = 0;

    /* Ensure that the matrix is assembled */
    ierr = MatEnsureAssembled(matin); CHKERRQ(ierr);

    /*  Check whether sector initialization was done right  */
    ierr = CheckSectors(); CHKERRQ(ierr);

    /* Get row and column layout */
    PetscInt rstart = matin->rmap->rstart;
    PetscInt lrows  = matin->rmap->n;
    PetscInt cstart = matin->cmap->rstart;
    PetscInt nrows  = matin->rmap->N;
    PetscInt ncols  = matin->cmap->N;

    /*  Check the matrix size */
    const PetscInt magNumStates = Magnetization.NumStates();
    if(magNumStates != nrows) SETERRQ2(mpi_comm,1,"Incorrect number of rows. Expected %d. Got %d.", magNumStates, nrows);
    if(magNumStates != nrows) SETERRQ2(mpi_comm,1,"Incorrect number of cols. Expected %d. Got %d.", magNumStates, ncols);

    /* Ensure that empty processes do nothing */
    if(!(0 <= rstart && rstart < nrows)) return ierr;

    /* Check the matrix type */
    PetscBool matin_is_mpiaij, matin_is_mpiaijmkl;
    ierr = PetscObjectTypeCompare((PetscObject)matin, MATMPIAIJ, &matin_is_mpiaij); CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)matin, MATMPIAIJMKL, &matin_is_mpiaijmkl); CHKERRQ(ierr);

    /* Do specific tasks for MATMPIAIJ or MATMPIAIJMKL using the diagonal structure */
    if(matin_is_mpiaij || matin_is_mpiaijmkl){
        /* Extract diagonal (A) and off-diagonal (B) sequential matrices */
        Mat_MPIAIJ *mat = (Mat_MPIAIJ*)matin->data;
        PetscInt *cmap = mat->garray;

        PetscInt nzA, nzB, *cA=nullptr, *cB=nullptr;

        /* Determine the starting block */
        PetscBool flg;
        PetscInt col_GlobIdxStart, col_GlobIdxEnd;

        for(QuantumNumbersIterator Iter(Magnetization, rstart, rstart + lrows); Iter.Loop(); ++Iter)
        {
            const PetscInt lrow = Iter.Steps();
            ierr = Iter.OpBlockToGlobalRange(OpType, col_GlobIdxStart, col_GlobIdxEnd, flg); CHKERRQ(ierr);

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

    ierr = PetscInfo(0, "Operator matrix check satisfied.\n"); CHKERRQ(ierr);
    return ierr;
}


PetscErrorCode Block::SpinOneHalf::CheckOperatorBlocks() const
{
    PetscErrorCode ierr = 0;
    CheckInit(__FUNCTION__); /** @throw PETSC_ERR_ARG_CORRUPT Block not yet initialized */

    /* Check all operator matrices */
    ierr = CheckOperators(); CHKERRQ(ierr);

    /* Check operator blocks of Sz matrices */
    for(PetscInt isite = 0; isite < num_sites; ++isite){
        ierr = MatOpCheckOperatorBlocks(OpSz, isite); CHKERRQ(ierr);
    }

    /* Check operator blocks of Sp matrices */
    for(PetscInt isite = 0; isite < num_sites; ++isite){
        ierr = MatOpCheckOperatorBlocks(OpSp, isite); CHKERRQ(ierr);
    }

    return ierr;
}


PetscErrorCode Block::SpinOneHalf::CreateSm()
{
    PetscErrorCode ierr = 0;

    if(init_Sm) SETERRQ(mpi_comm, 1, "Sm was previously initialized. Call DestroySm() first.");

    ierr = CheckOperatorArray(OpSp); CHKERRQ(ierr);
    for(PetscInt isite = 0; isite < num_sites; ++isite){
        ierr = MatHermitianTranspose(SpData[isite], MAT_INITIAL_MATRIX, &SmData[isite]); CHKERRQ(ierr);
    }
    init_Sm = PETSC_TRUE;

    return ierr;
}


PetscErrorCode Block::SpinOneHalf::DestroySm()
{
    PetscErrorCode ierr = 0;
    if(!init_Sm) SETERRQ1(mpi_comm, 1, "%s was called but Sm was not yet initialized. ",__FUNCTION__);

    for(PetscInt isite = 0; isite < num_sites; ++isite){
        ierr = MatDestroy(&SmData[isite]); CHKERRQ(ierr);
        SmData[isite] = NULL;
    }
    init_Sm = PETSC_FALSE;

    return ierr;
}


PetscErrorCode Block::SpinOneHalf::Destroy()
{
    PetscErrorCode ierr = 0;
    if (PetscUnlikely(!init)) return 0;

    /*  Destroy operator matrices  */
    for(PetscInt isite = 0; isite < num_sites; ++isite){
        ierr = MatDestroy(&SzData[isite]); CHKERRQ(ierr);
        ierr = MatDestroy(&SpData[isite]); CHKERRQ(ierr);
        SzData[isite] = NULL;
        SpData[isite] = NULL;
    }
    ierr = MatDestroy(&H); CHKERRQ(ierr);
    H = NULL;
    if (init_Sm){
        ierr = DestroySm(); CHKERRQ(ierr);
    }
    init = PETSC_FALSE;
    return ierr;
}


PetscErrorCode Block::SpinOneHalf::RotateOperators(const SpinOneHalf& Source, const Mat& RotMatT)
{
    PetscErrorCode ierr = 0;
    CheckInit(__FUNCTION__); /** @throw PETSC_ERR_ARG_CORRUPT Block not yet initialized */

    /*  Since we do not want to rotate the Sm operators separately */
    if(init_Sm){ ierr = DestroySm(); CHKERRQ(ierr); }

    /*  Verify the sizes */
    const PetscInt NumStatesOrig = Source.NumStates();
    const PetscInt NumSitesOrig = Source.NumSites();
    PetscInt NRows_RT, NCols_RT;
    ierr = MatGetSize(RotMatT, &NRows_RT, &NCols_RT); CHKERRQ(ierr);
    if(NCols_RT != NumStatesOrig)
        SETERRQ2(mpi_comm, 1, "RotMatT incorrect number of cols. Expected %d. Got %d.", NCols_RT, NumStatesOrig);
    if(NRows_RT != num_states)
        SETERRQ2(mpi_comm, 1, "RotMatT incorrect number of rows. Expected %d. Got %d.", NRows_RT, num_states);
    if(NumSitesOrig != num_sites)
        SETERRQ2(mpi_comm, 1, "RotMatT incorrect number of sites. Expected %d. Got %d.", NumSitesOrig, num_sites);

    /*  Get the method from command line */
    enum RotMethod { mmmmult=0, matptap=1 };
    RotMethod method = mmmmult;

    #if defined(PETSC_USE_COMPLEX)
        if(method == matptap) SETERRQ(mpi_comm,1,"The method matptap cannot be used for complex scalars.");
    #endif

    Mat RotMat;
    if( method==mmmmult || method==matptap){
        ierr = MatHermitianTranspose(RotMatT, MAT_INITIAL_MATRIX, &RotMat); CHKERRQ(ierr);
    } else {
        SETERRQ(mpi_comm,1,"Not implemented.");
    }

    #if defined(PETSC_USE_DEBUG)
    {
        PetscBool flg = PETSC_FALSE;
        ierr = PetscOptionsGetBool(NULL,NULL,"-print_UUT", &flg, NULL); CHKERRQ(ierr);
        if(flg){
            Mat UUT;
            ierr = MatMatMult(RotMat, RotMatT, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &UUT); CHKERRQ(ierr);
            ierr = MatPeek(UUT,"UUT"); CHKERRQ(ierr);
            ierr = MatDestroy(&UUT); CHKERRQ(ierr);
        }
    }
    #endif

    /*  Destroy previously created operators */
    for(PetscInt isite = 0; isite < num_sites; ++isite)
    {
        ierr = MatDestroy(&SpData[isite]); CHKERRQ(ierr);
        ierr = MatDestroy(&SzData[isite]); CHKERRQ(ierr);
    }

    /*  Perform the rotation on all operators */
    if( method==mmmmult)
    {
        for(PetscInt isite = 0; isite < num_sites; ++isite)
        {
            ierr = MatMatMatMult(RotMatT, Source.Sp(isite), RotMat, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &SpData[isite]); CHKERRQ(ierr);
            ierr = MatMatMatMult(RotMatT, Source.Sz(isite), RotMat, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &SzData[isite]); CHKERRQ(ierr);
        }
        ierr = MatMatMatMult(RotMatT, Source.H, RotMat, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &H); CHKERRQ(ierr);
    } else {
        SETERRQ(mpi_comm,1,"Not implemented.");
    }
    ierr = MatDestroy(&RotMat); CHKERRQ(ierr);
    ierr = CheckOperatorBlocks(); CHKERRQ(ierr);
    return(0);
}

PetscErrorCode Block::SpinOneHalf::AssembleOperators()
{
    PetscErrorCode ierr;
    ierr = MatEnsureAssembled_MultipleMats(SzData); CHKERRQ(ierr);
    ierr = MatEnsureAssembled_MultipleMats(SpData); CHKERRQ(ierr);
    if(H){ ierr = MatEnsureAssembled(H); CHKERRQ(ierr);}
    return(0);
}


PetscErrorCode Block::SpinOneHalf::InitializeSave(
    const std::string& save_dir_in
    )
{
    PetscErrorCode ierr = 0;
    PetscBool flg = PETSC_FALSE;
    if(save_dir_in.empty()) SETERRQ(mpi_comm,1,"Save dir cannot be empty.");
    if(!mpi_init) SETERRQ(mpi_comm,1,"MPI Initialization must be completed first.");
    ierr = PetscTestDirectory(save_dir_in.c_str(), 'r', &flg); CHKERRQ(ierr);
    if(!flg) SETERRQ1(mpi_comm,1,"Directory %s does not exist.",save_dir_in.c_str());
    save_dir = save_dir_in;
    /* If the last character is not a slash then add one */
    if(save_dir.back()!='/') save_dir += '/';
    init_save = PETSC_TRUE;
    return(0);
}


std::string OpFilename(const std::string& RootDir, const std::string& OpName, const size_t& isite = 0){
    std::ostringstream oss;
    oss << RootDir << OpName << "_" << std::setfill('0') << std::setw(9) << isite << ".mat";
    return oss.str();
}


PetscErrorCode Block::SpinOneHalf::SaveOperator(const std::string& OpName, const size_t& isite, Mat& Op){
    PetscErrorCode ierr;
    PetscViewer binv;
    ierr = PetscViewerBinaryOpen(mpi_comm, OpFilename(save_dir,OpName,isite).c_str(), FILE_MODE_WRITE, &binv); CHKERRQ(ierr);
    ierr = MatView(Op, binv); CHKERRQ(ierr);
    ierr = MatDestroy(&Op); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&binv); CHKERRQ(ierr);
    return(0);
}


PetscErrorCode Block::SpinOneHalf::RetrieveOperator(const std::string& OpName, const size_t& isite, Mat& Op){
    PetscErrorCode ierr;
    PetscViewer binv;
    ierr = PetscViewerBinaryOpen(mpi_comm, OpFilename(save_dir,OpName,isite).c_str(), FILE_MODE_READ, &binv); CHKERRQ(ierr);
    ierr = MatCreate(mpi_comm, &Op); CHKERRQ(ierr);
    ierr = MatSetFromOptions(Op); CHKERRQ(ierr);
    ierr = MatLoad(Op, binv); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&binv); CHKERRQ(ierr);
    return(0);
}


PetscErrorCode Block::SpinOneHalf::SaveAndDestroy()
{
    CheckInit(__FUNCTION__); /** @throw PETSC_ERR_ARG_CORRUPT Block not yet initialized */
    if(!init_save) SETERRQ(mpi_comm,1,"InitializeSave() must be called first.");
    PetscErrorCode ierr;
    for(size_t isite = 0; isite < num_sites; ++isite){
        ierr = SaveOperator("Sz",isite,SzData[isite]); CHKERRQ(ierr);
    }
    for(size_t isite = 0; isite < num_sites; ++isite){
        ierr = SaveOperator("Sp",isite,SpData[isite]); CHKERRQ(ierr);
    }
    if(H){
        ierr = SaveOperator("H",0,H); CHKERRQ(ierr);
    }
    ierr = Destroy(); CHKERRQ(ierr);
    init = PETSC_FALSE;
    return(0);
}


PetscErrorCode Block::SpinOneHalf::Retrieve()
{
    if(!init_save) SETERRQ(mpi_comm,1,"InitializeSave() must be called first.");
    if(init) SETERRQ(mpi_comm,1,"SaveAndDestroy() must be called first.");
    PetscErrorCode ierr;
    PetscBool flg = PETSC_FALSE;
    for(size_t isite = 0; isite < num_sites; ++isite){
        ierr = RetrieveOperator("Sz",isite,SzData[isite]); CHKERRQ(ierr);
    }
    for(size_t isite = 0; isite < num_sites; ++isite){
        ierr = RetrieveOperator("Sp",isite,SpData[isite]); CHKERRQ(ierr);
    }
    ierr = PetscTestFile(OpFilename(save_dir,"H",0).c_str(), 'r', &flg); CHKERRQ(ierr);
    if(flg){
        ierr = SaveOperator("H",0,H); CHKERRQ(ierr);
    }
    init = PETSC_TRUE;
    return(0);
}
