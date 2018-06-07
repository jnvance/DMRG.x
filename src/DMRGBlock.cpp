#include "DMRGBlock.hpp"
#include <numeric> // partial_sum
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>

#include <../src/mat/impls/aij/seq/aij.h>    /* Mat_SeqAIJ */
#include <../src/mat/impls/aij/mpi/mpiaij.h> /* Mat_MPIAIJ */

/* External functions taken from MiscTools.cpp */
PETSC_EXTERN int64_t ipow(int64_t base, uint8_t exp);
PETSC_EXTERN PetscErrorCode InitSingleSiteOperator(const MPI_Comm& comm, const PetscInt dim, Mat* mat);
PETSC_EXTERN PetscErrorCode MatEnsureAssembled(const Mat& matin);
PETSC_EXTERN PetscErrorCode MatEnsureAssembled_MultipleMats(const std::vector<Mat>& matrices);
PETSC_EXTERN PetscErrorCode MatEyeCreate(const MPI_Comm& comm, const PetscInt& dim, Mat& eye);
PETSC_EXTERN PetscErrorCode Makedir(const std::string& dir_name);

/** Internal macro for checking the initialization state of the block object */
#define CheckInit(func) if (PetscUnlikely(!init))\
    SETERRQ1(mpi_comm, PETSC_ERR_ARG_CORRUPT, "%s was called but block was not yet initialized.",func);

#define CheckInitOnce(func) if (PetscUnlikely(!init_once))\
    SETERRQ1(mpi_comm, PETSC_ERR_ARG_CORRUPT, "%s was called but Initialize was never called before.",func);

/** Internal macro for checking that a column index belongs in the magnetization block boundaries */
#define CheckIndex(row, col, cstart, cend) if((col) < (cstart) || (col) >= (cend))\
    SETERRQ4(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "On row %d, index %d out of bounds [%d,%d) ",\
        (row), (col), (cstart), (cend));

PetscErrorCode Block::SpinBase::Initialize(
    const MPI_Comm& comm_in)
{
    PetscErrorCode ierr;
    if(mpi_init) SETERRQ(mpi_comm,1,"This initializer should only be called once.");
    mpi_comm = comm_in;
    ierr = MPI_Comm_rank(mpi_comm, &mpi_rank); CHKERRQ(ierr);
    ierr = MPI_Comm_size(mpi_comm, &mpi_size); CHKERRQ(ierr);
    mpi_init = PETSC_TRUE;
    return(0);
}


PetscErrorCode Block::SpinBase::Initialize(
    const MPI_Comm& comm_in,
    const PetscInt& num_sites_in,
    const PetscInt& num_states_in,
    const PetscBool& init_ops
    )
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
        num_states = ipow(loc_dim(), num_sites);
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
    init_once = PETSC_TRUE;

    /** When creating a block for one site, the single-site operators are initialized using the default
        values and matrix operators for spin-1/2 defined in Block::SpinBase::loc_dim, Block::SpinBase::loc_qn_list and Block::SpinBase::loc_qn_size */
    if (num_sites == 1)
    {
        /*  Create the spin operators for the single site  */
        ierr = MatSpinSzCreate(SzData[0]); CHKERRQ(ierr);
        ierr = MatSpinSpCreate(SpData[0]); CHKERRQ(ierr);

        /*  Also initialize the single-site Hamiltonian which is defaulted to zero */
        ierr = InitSingleSiteOperator(mpi_comm, num_states, &H); CHKERRQ(ierr);
        ierr = MatEnsureAssembled(H); CHKERRQ(ierr);
        /*  Initialize the magnetization sectors using the defaults for one site */
        ierr = Magnetization.Initialize(mpi_comm, loc_qn_list(), loc_qn_size()); CHKERRQ(ierr);

        /*  Check whether sector initialization was done right  */
        ierr = CheckSectors(); CHKERRQ(ierr);
    }
    /** When more than one site is requested, all operator matrices are created with the correct sizes based on the
        number of states */
    else if(num_sites > 1)
    {
        if(init_ops)
        {
            for(PetscInt isite = 0; isite < num_sites; ++isite)
            {
                ierr = InitSingleSiteOperator(mpi_comm, num_states, &SzData[isite]); CHKERRQ(ierr);
                ierr = InitSingleSiteOperator(mpi_comm, num_states, &SpData[isite]); CHKERRQ(ierr);
            }
        }
    }
    else
        SETERRQ1(mpi_comm, PETSC_ERR_ARG_OUTOFRANGE, "Invalid input num_sites_in > 0. Given %d.", num_sites_in);

    /* Initialize options for operator rotation */
    ierr = PetscOptionsGetInt(NULL,NULL,"-rot_nsubcomm",&nsubcomm,NULL); CHKERRQ(ierr);
    if(nsubcomm < 1 || nsubcomm > mpi_size)
        SETERRQ1(mpi_comm, 1, "-rot_nsubcomm must be in the range [1, mpi_rank]. Got %D.", nsubcomm);

    /* Require that the size of comm_world be a multiple of the number of subcommunicators */
    if(mpi_size % nsubcomm)
        SETERRQ2(mpi_comm, 1, "The size of mpi_comm (%d) must be a multiple of nsubcomm (%D).", mpi_size, nsubcomm);

    /* Determine the color of each lattice site. The Hamiltonian is processed at isite=num_sites */
    site_color.resize((PetscMPIInt)(num_sites+1));
    for(PetscMPIInt isite = 0; isite < num_sites+1; ++isite){
        site_color[isite] = isite % nsubcomm;
    }

    #if defined(PETSC_USE_COMPLEX)
        if(rot_method == matptap) SETERRQ(mpi_comm,1,"The method matptap cannot be used for complex scalars.");
    #endif

    return ierr;
}


PetscErrorCode Block::SpinBase::Initialize(
    const MPI_Comm& comm_in,
    const PetscInt& num_sites_in,
    const std::vector<PetscReal>& qn_list_in,
    const std::vector<PetscInt>& qn_size_in,
    const PetscBool& init_ops
    )
{
    PetscErrorCode ierr = 0;

    if(PetscUnlikely(num_sites_in == 1))
        SETERRQ(mpi_comm, PETSC_ERR_ARG_OUTOFRANGE,
            "Invalid input num_sites_in cannot be equal to 1. Call a different Initialize() function.");

    QuantumNumbers Magnetization_temp;
    ierr = Magnetization_temp.Initialize(comm_in, qn_list_in, qn_size_in); CHKERRQ(ierr);
    PetscInt num_states_in = Magnetization_temp.NumStates();

    ierr = Initialize(comm_in, num_sites_in, num_states_in, init_ops); CHKERRQ(ierr);
    Magnetization = Magnetization_temp;

    return ierr;
}


PetscErrorCode Block::SpinBase::Initialize(
    const PetscInt& num_sites_in,
    const QuantumNumbers& qn_in)
{
    PetscErrorCode ierr = 0;

    ierr = qn_in.CheckInitialized(); CHKERRQ(ierr);
    ierr = Initialize(qn_in.MPIComm(), num_sites_in, qn_in.NumStates()); CHKERRQ(ierr);
    Magnetization = qn_in;

    return ierr;
}


PetscErrorCode Block::SpinBase::InitializeFromDisk(
    const MPI_Comm& comm_in,
    const std::string& block_path_in)
{
    PetscErrorCode ierr;
    PetscInt num_sites_in, num_states_in, num_sectors_in;
    std::vector<PetscReal> qn_list_in;
    std::vector<PetscInt> qn_size_in;

    PetscMPIInt mpi_rank_in;
    ierr = MPI_Comm_rank(comm_in,&mpi_rank_in); CHKERRQ(ierr);

    std::string block_path = block_path_in;
    if(block_path.back()!='/') block_path += '/';

    if(!mpi_rank_in)
    {
        /* Read data for Block */
        std::ifstream infofile((block_path + "BlockInfo.dat").c_str());
        std::map< std::string, PetscInt > infomap;
        std::string line;
        if (!infofile.is_open())
            perror("error while opening file");
        while (infofile && std::getline(infofile, line)) {
            std::string key;
            PetscInt val;
            std::istringstream iss(line);
            iss >> key >> val;
            infomap[key] = val;
        }
        if (infofile.bad())
            perror("error while reading file");

        infofile.close();

        /* Verify compatibility */
        {
            if(infomap.at("NumBytesPetscInt") != sizeof(PetscInt))
                SETERRQ2(PETSC_COMM_SELF,1,"Incompatible NumBytesPetscInt. Expected %lld. Got %lld.",
                    PetscInt(sizeof(PetscInt)), infomap.at("NumBytesPetscInt"));

            if(infomap.at("NumBytesPetscScalar") != sizeof(PetscScalar))
                SETERRQ2(PETSC_COMM_SELF,1,"Incompatible NumBytesPetscScalar. Expected %lld. Got %lld.",
                    PetscInt(sizeof(PetscScalar)), infomap.at("NumBytesPetscScalar"));

            #if defined(PETSC_USE_COMPLEX)
                PetscInt PetscUseComplex = 1;
            #else
                PetscInt PetscUseComplex = 0;
            #endif

            if(infomap.at("PetscUseComplex") != PetscUseComplex)
                SETERRQ2(PETSC_COMM_SELF,1,"Incompatible PetscUseComplex. Expected %lld. Got %lld.",
                    PetscInt(PetscUseComplex), infomap.at("PetscUseComplex"));
        }

        /* Assign variables */
        num_sites_in = infomap.at("NumSites");
        num_states_in = infomap.at("NumStates");
        num_sectors_in = infomap.at("NumSectors");
    }

    ierr = MPI_Bcast(&num_sites_in, 1, MPIU_INT, 0, comm_in); CHKERRQ(ierr);
    ierr = MPI_Bcast(&num_states_in, 1, MPIU_INT, 0, comm_in); CHKERRQ(ierr);
    ierr = MPI_Bcast(&num_sectors_in, 1, MPIU_INT, 0, comm_in); CHKERRQ(ierr);
    qn_list_in.resize(num_sectors_in);
    qn_size_in.resize(num_sectors_in);

    if(num_sectors_in > 0)
    {
        if(!mpi_rank_in)
        {
            /* Read data for quantum numbers */
            std::string filename = block_path + "QuantumNumbers.dat";
            std::ifstream qnfile(filename.c_str());
            std::string line;
            if (!qnfile.is_open())
                perror("error while opening file");
            PetscInt ctr = 0;
            while (qnfile && std::getline(qnfile, line)) {
                std::istringstream iss(line);
                iss >> qn_size_in.at(ctr) >> qn_list_in.at(ctr);
                ++ctr;
            }
            if (qnfile.bad())
                perror("error while reading file");
            if(ctr!=num_sectors_in)
                SETERRQ3(PETSC_COMM_SELF,1,"Incorrect number of data points in %s. "
                    "Expected %lld. Got %lld.", filename.c_str(), num_sectors_in, ctr);

            qnfile.close();
        }

        ierr = MPI_Bcast(qn_size_in.data(), num_sectors_in, MPIU_INT, 0, comm_in); CHKERRQ(ierr);
        ierr = MPI_Bcast(qn_list_in.data(), num_sectors_in, MPIU_REAL, 0, comm_in); CHKERRQ(ierr);
    }
    else
    {
        SETERRQ(comm_in,1,"NumSectors cannot be zero.");
    }

    /* Initialize block object but do not initialize operators */
    ierr = Initialize(comm_in, num_sites_in, qn_list_in, qn_size_in, PETSC_FALSE); CHKERRQ(ierr);

    /* Read-in the operators */
    std::string save_dir_temp = save_dir;
    save_dir = block_path;
    ierr = Retrieve_NoChecks(); CHKERRQ(ierr);
    save_dir = save_dir_temp;

    /* Check operator validity */
    ierr = CheckOperatorBlocks(); CHKERRQ(ierr);

    return(0);
}


PetscErrorCode Block::SpinBase::CheckOperatorArray(const Op_t& OpType) const
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


PetscErrorCode Block::SpinBase::CheckOperators() const
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


PetscErrorCode Block::SpinBase::CheckSectors() const
{
    PetscErrorCode ierr = 0;
    CheckInitOnce(__FUNCTION__); /** @throw PETSC_ERR_ARG_CORRUPT Block not yet initialized */

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


PetscErrorCode Block::SpinBase::MatOpGetNNZs(
    const Op_t& OpType,
    const PetscInt& isite,
    std::vector<PetscInt>& nnzs
    ) const
{
    PetscErrorCode ierr = 0;

    /* TODO: Generalize this piece of redundant code */

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

    ierr = MatGetNNZs(matin, nnzs); CHKERRQ(ierr);
    return(0);
}


PetscErrorCode Block::SpinBase::MatGetNNZs(
    const Mat& matin,
    std::vector<PetscInt>& nnzs
    ) const
{
    nnzs.clear();
    PetscInt rstart, rend;
    PetscErrorCode ierr = MatGetOwnershipRange(matin, &rstart, &rend); CHKERRQ(ierr);
    PetscInt lrows = rend - rstart;
    nnzs.resize(lrows);
    PetscInt ncols;
    for(PetscInt irow=rstart; irow<rend; ++irow)
    {
        ierr = MatGetRow(matin, irow, &ncols, NULL, NULL); CHKERRQ(ierr);
        nnzs[irow-rstart] = ncols;
        ierr = MatRestoreRow(matin, irow, &ncols, NULL, NULL); CHKERRQ(ierr);
    }
    return(0);
}


PetscErrorCode Block::SpinBase::MatOpCheckOperatorBlocks(const Op_t& OpType, const PetscInt& isite) const
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


PetscErrorCode Block::SpinBase::MatCheckOperatorBlocks(const Op_t& OpType, const Mat& matin) const
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
    else
    {
        MatType type;
        ierr = MatGetType(matin, &type);

        /** @throw PETSC_ERR_SUP This checking has been implemented specifically for MATMPIAIJ only */
        SETERRQ1(mpi_comm, PETSC_ERR_SUP, "Implemented only for MATMPIAIJ. Got %s.", type);
    }

    ierr = PetscInfo(0, "Operator matrix check satisfied.\n"); CHKERRQ(ierr);
    return ierr;
}


PetscErrorCode Block::SpinBase::CheckOperatorBlocks() const
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


PetscErrorCode Block::SpinBase::CreateSm()
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


PetscErrorCode Block::SpinBase::DestroySm()
{
    PetscErrorCode ierr = 0;
    if(!init_Sm && !init) return(0);
    if(!init_Sm) SETERRQ1(mpi_comm, 1, "%s was called but Sm was not yet initialized. ",__FUNCTION__);

    for(PetscInt isite = 0; isite < num_sites; ++isite){
        ierr = MatDestroy(&SmData[isite]); CHKERRQ(ierr);
        SmData[isite] = NULL;
    }
    init_Sm = PETSC_FALSE;

    return ierr;
}


PetscErrorCode Block::SpinBase::Destroy()
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

/* Note: May modify save state of Source */
PetscErrorCode Block::SpinBase::RotateOperators(SpinBase& Source, const Mat& RotMatT_in)
{
    PetscErrorCode ierr = 0;
    Mat RotMatT, RotMat, *SpData_loc, *SzData_loc, H_loc;
    MPI_Comm subcomm;
    PetscMPIInt sub_rank, sub_size, sub_color=0;

    CheckInit(__FUNCTION__); /** @throw PETSC_ERR_ARG_CORRUPT Block not yet initialized */

    /*  Since we do not want to rotate the Sm operators separately */
    if(init_Sm){ ierr = DestroySm(); CHKERRQ(ierr); }

    /*  Verify the sizes */
    const PetscInt NumStatesOrig = Source.NumStates();
    const PetscInt NumSitesOrig = Source.NumSites();
    PetscInt NRows_RT, NCols_RT;
    ierr = MatGetSize(RotMatT_in, &NRows_RT, &NCols_RT); CHKERRQ(ierr);
    if(NCols_RT != NumStatesOrig)
        SETERRQ2(mpi_comm, 1, "RotMatT_in incorrect number of cols. Expected %d. Got %d.", NCols_RT, NumStatesOrig);
    if(NRows_RT != num_states)
        SETERRQ2(mpi_comm, 1, "RotMatT_in incorrect number of rows. Expected %d. Got %d.", NRows_RT, num_states);
    if(NumSitesOrig != num_sites)
        SETERRQ2(mpi_comm, 1, "RotMatT_in incorrect number of sites. Expected %d. Got %d.", NumSitesOrig, num_sites);

    if(nsubcomm > 1)
    {
        /*  Require that save_initialization must have been called first since this operation requires
            reading the matrices from disk to the subcommunicators */
        if(!init_save) SETERRQ(mpi_comm,1,"InitializeSave() must be called first before using this feature.");
        ierr = MatCreateRedundantMatrix(RotMatT_in, nsubcomm, MPI_COMM_NULL, MAT_INITIAL_MATRIX, &RotMatT); CHKERRQ(ierr);
        ierr = PetscObjectGetComm((PetscObject)RotMatT, &subcomm); CHKERRQ(ierr);
        ierr = MPI_Comm_rank(subcomm, &sub_rank); CHKERRQ(ierr);
        ierr = MPI_Comm_size(subcomm, &sub_size); CHKERRQ(ierr);
        sub_color = mpi_rank / sub_size;

        /*  Require source and destination matrices to be saved to disk so that each subcommunicator can retrieve this data */
        ierr = Source.EnsureSaved(); CHKERRQ(ierr);
    }
    else
    {
        RotMatT = RotMatT_in;
        ierr = Source.EnsureRetrieved(); CHKERRQ(ierr);
    }

    if(rot_method==mmmmult){
        ierr = MatHermitianTranspose(RotMatT, MAT_INITIAL_MATRIX, &RotMat); CHKERRQ(ierr);
    } else {
        SETERRQ(mpi_comm,1,"Not implemented.");
    }

    /*  Destroy previously created operators */
    for(PetscInt isite = 0; isite < num_sites; ++isite)
    {
        ierr = MatDestroy(&SpData[isite]); CHKERRQ(ierr);
        ierr = MatDestroy(&SzData[isite]); CHKERRQ(ierr);
    }
    ierr = MatDestroy(&H); CHKERRQ(ierr);

    /* Load source matrices to an array of Mat's */
    ierr = PetscCalloc1(num_sites, &SpData_loc); CHKERRQ(ierr);
    ierr = PetscCalloc1(num_sites, &SzData_loc); CHKERRQ(ierr);

    if(nsubcomm > 1)
    {
        /* Retrieve source matrices manually */
        for(PetscInt isite = 0; isite < num_sites; ++isite){
            if(site_color[isite]!=sub_color) continue;
            ierr = Source.RetrieveOperator("Sp", isite, SpData_loc[isite], subcomm); CHKERRQ(ierr);
            ierr = Source.RetrieveOperator("Sz", isite, SzData_loc[isite], subcomm); CHKERRQ(ierr);
        }
        if(site_color[num_sites]==sub_color){
            ierr = Source.RetrieveOperator("H", 0, H_loc, subcomm); CHKERRQ(ierr);
        }
    }
    else
    {
        /* Copy source matrices (pointers) into local data */
        for(PetscInt isite = 0; isite < num_sites; ++isite) SpData_loc[isite] = Source.Sp(isite);
        for(PetscInt isite = 0; isite < num_sites; ++isite) SzData_loc[isite] = Source.Sz(isite);
        H_loc = Source.H;
    }

    /*  Perform the rotation on all operators */
    if(rot_method==mmmmult)
    {
        for(PetscInt isite = 0; isite < num_sites; ++isite)
        {
            if(site_color[isite]!=sub_color) continue;
            ierr = MatMatMatMult(RotMatT, SpData_loc[isite], RotMat, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &SpData[isite]); CHKERRQ(ierr);
            ierr = MatMatMatMult(RotMatT, SzData_loc[isite], RotMat, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &SzData[isite]); CHKERRQ(ierr);
        }
        if(site_color[num_sites]==sub_color)
        {
            ierr = MatMatMatMult(RotMatT, H_loc, RotMat, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &H); CHKERRQ(ierr);
        }
    }
    else
    {
        SETERRQ(mpi_comm,1,"Not implemented.");
    }

    if(nsubcomm > 1)
    {
        /* Save data back to disk */
        for(PetscInt isite = 0; isite < num_sites; ++isite)
        {
            if(site_color[isite]!=sub_color) continue;
            ierr = SaveOperator("Sz", isite, SzData[isite], subcomm); CHKERRQ(ierr);
            ierr = SaveOperator("Sp", isite, SpData[isite], subcomm); CHKERRQ(ierr);
        }
        if(site_color[num_sites]==sub_color)
        {
            ierr = SaveOperator("H", 0, H, subcomm); CHKERRQ(ierr);
            ierr = MatDestroy(&H_loc); CHKERRQ(ierr);
        }
        /*  Ensure that the current block is returned to a saved state */
        for(PetscInt isite = 0; isite < num_sites; ++isite)
        {
            ierr = MatDestroy(&SpData[isite]); CHKERRQ(ierr);
            ierr = MatDestroy(&SzData[isite]); CHKERRQ(ierr);
            ierr = MatDestroy(&SpData_loc[isite]); CHKERRQ(ierr);
            ierr = MatDestroy(&SzData_loc[isite]); CHKERRQ(ierr);
        }
        ierr = MatDestroy(&RotMatT); CHKERRQ(ierr);
        ierr = MatDestroy(&H); CHKERRQ(ierr);
        ierr = Destroy(); CHKERRQ(ierr);
        init = PETSC_FALSE;
        saved = PETSC_TRUE;
        retrieved = PETSC_FALSE;
    }
    else
    {
        ierr = CheckOperatorBlocks(); CHKERRQ(ierr);
    }

    ierr = PetscFree(SpData_loc); CHKERRQ(ierr);
    ierr = PetscFree(SzData_loc); CHKERRQ(ierr);
    ierr = MatDestroy(&RotMat); CHKERRQ(ierr);

    return(0);
}

PetscErrorCode Block::SpinBase::AssembleOperators()
{
    PetscErrorCode ierr;
    ierr = MatEnsureAssembled_MultipleMats(SzData); CHKERRQ(ierr);
    ierr = MatEnsureAssembled_MultipleMats(SpData); CHKERRQ(ierr);
    if(H){ ierr = MatEnsureAssembled(H); CHKERRQ(ierr);}
    return(0);
}


PetscErrorCode Block::SpinBase::InitializeSave(
    const std::string& save_dir_in
    )
{
    PetscErrorCode ierr = 0;
    PetscBool flg = PETSC_FALSE;
    if(save_dir_in.empty()) SETERRQ(mpi_comm,1,"Save dir cannot be empty.");
    if(!mpi_init) SETERRQ(mpi_comm,1,"MPI Initialization must be completed first.");
    if(!mpi_rank){
        ierr = PetscTestDirectory(save_dir_in.c_str(), 'r', &flg); CHKERRQ(ierr);
    }
    ierr = MPI_Bcast(&flg, 1, MPI_INT, 0, mpi_comm); CHKERRQ(ierr);
    if(!flg) SETERRQ1(mpi_comm,1,"Directory %s does not exist. Please verify that -scratch_dir is specified correctly.",save_dir_in.c_str());
    save_dir = save_dir_in;
    /* If the last character is not a slash then add one */
    if(save_dir.back()!='/') save_dir += '/';
    init_save = PETSC_TRUE;
    return(0);
}


PetscErrorCode Block::SpinBase::SetDiskStorage(
    const std::string& read_dir_in,
    const std::string& write_dir_in
    )
{
    if(init_save == PETSC_TRUE)
        SETERRQ(mpi_comm,1,"InitializeSave() and SetDiskStorage() cannot both be used "
            "on the same block.");

    read_dir  = read_dir_in;
    if(read_dir.back()!='/') read_dir += '/';

    write_dir = write_dir_in;
    if(write_dir.back()!='/') write_dir += '/';

    /** The blocks are going to be read from the `read_dir_in` directory
        during the first retrieval and then `save_dir_in` for succeeding reads.
        This comes in useful for the sweep stage of the
        DMRG algorithm, when we want to read the resulting operators of the previous sweep
        and write the files back into a different subdirectory for the current sweep.
        The switch is done by Retrieve_NoChecks().
     */
    save_dir  = read_dir;

    /** This function also resets the counter in the number of reads performed
        in the PetscInt Block::SpinBase::num_reads parameter */
    num_reads = 0;
    disk_set = PETSC_TRUE;
    return(0);
}


std::string OpFilename(const std::string& RootDir, const std::string& OpName, const size_t& isite = 0){
    std::ostringstream oss;
    oss << RootDir << OpName << "_" << std::setfill('0') << std::setw(9) << isite << ".mat";
    return oss.str();
}


PetscErrorCode Block::SpinBase::SaveOperator(
    const std::string& OpName,
    const size_t& isite,
    Mat& Op,
    const MPI_Comm& comm_in)
{
    PetscErrorCode ierr;
    PetscViewer binv;
    if(!Op) SETERRQ(PETSC_COMM_SELF,1,"Input matrix is null.");
    ierr = PetscViewerBinaryOpen(comm_in, OpFilename(save_dir,OpName,isite).c_str(), FILE_MODE_WRITE, &binv); CHKERRQ(ierr);
    ierr = MatView(Op, binv); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&binv); CHKERRQ(ierr);
    ierr = MatDestroy(&Op); CHKERRQ(ierr);
    Op = NULL;
    return(0);
}


PetscErrorCode Block::SpinBase::SaveBlockInfo()
{
    CheckInit(__FUNCTION__);
    if(!init_save && !disk_set)
        SETERRQ(mpi_comm,1,"InitializeSave() or SetDiskStorage() must be called first.");

    if(mpi_rank) return(0);

    /* Save block information */
    {
        std::ofstream infofile;
        infofile.open((save_dir + "BlockInfo.dat").c_str());

        #if defined(PETSC_USE_COMPLEX)
            PetscInt PetscUseComplex = 1;
        #else
            PetscInt PetscUseComplex = 0;
        #endif

        #define SaveInfo(KEY,VALUE) infofile << std::left << std::setfill(' ') \
                << std::setw(30) << KEY << " " << VALUE << std::endl;
        SaveInfo("NumBytesPetscInt",    sizeof(PetscInt));
        SaveInfo("NumBytesPetscScalar", sizeof(PetscScalar));
        SaveInfo("PetscUseComplex",     PetscUseComplex);
        SaveInfo("NumSites",            num_sites);
        SaveInfo("NumStates",           num_states);
        SaveInfo("NumSectors",          Magnetization.NumSectors());
        #undef SaveInfo

        infofile.close();
    }

    /* Save quantum numbers information */
    {
        std::ofstream qnfile;
        qnfile.open((save_dir + "QuantumNumbers.dat").c_str());
        std::vector<PetscInt>  qn_size = Magnetization.Sizes();
        std::vector<PetscReal> qn_list = Magnetization.List();
        PetscInt num_sectors = Magnetization.NumSectors();
        for(PetscInt idx=0; idx<num_sectors; idx++)
            qnfile  << qn_size.at(idx) << " "
                    << qn_list.at(idx) << std::endl;
        qnfile.close();
    }

    return(0);
}


PetscErrorCode Block::SpinBase::RetrieveOperator(
    const std::string& OpName,
    const size_t& isite,
    Mat& Op,
    const MPI_Comm& comm_in)
{
    PetscErrorCode ierr;
    /* Separately handle the case of Sm operators since they are never saved */
    if(OpName=="Sm")
    {
        Mat OpSp;
        ierr = RetrieveOperator("Sp",isite,OpSp,comm_in); CHKERRQ(ierr);
        ierr = MatHermitianTranspose(OpSp,MAT_INITIAL_MATRIX,&Op); CHKERRQ(ierr);
        ierr = MatDestroy(&OpSp); CHKERRQ(ierr);
        return(0);
    }
    PetscViewer binv;
    MPI_Comm comm = (comm_in==MPI_COMM_NULL) ? mpi_comm : comm_in;
    ierr = PetscViewerBinaryOpen(comm, OpFilename(save_dir,OpName,isite).c_str(), FILE_MODE_READ, &binv); CHKERRQ(ierr);
    ierr = MatCreate(comm, &Op); CHKERRQ(ierr);
    /* NOTE: Added this fix for the conflicting cases of mpiaij operators and seqaij rotation matrices */
    if(comm_in==MPI_COMM_NULL){
        ierr = MatSetFromOptions(Op); CHKERRQ(ierr);
    }
    ierr = MatLoad(Op, binv); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&binv); CHKERRQ(ierr);
    return(0);
}


PetscErrorCode Block::SpinBase::SaveAndDestroy()
{
    CheckInit(__FUNCTION__); /** @throw PETSC_ERR_ARG_CORRUPT Block not yet initialized */
    if(!init_save && !disk_set)
        SETERRQ(mpi_comm,1,"InitializeSave() or SetDiskStorage() must be called first.");

    PetscErrorCode ierr;
    for(PetscInt isite = 0; isite < num_sites; ++isite){
        ierr = SaveOperator("Sz",isite,SzData[isite],mpi_comm); CHKERRQ(ierr);
    }
    for(PetscInt isite = 0; isite < num_sites; ++isite){
        ierr = SaveOperator("Sp",isite,SpData[isite],mpi_comm); CHKERRQ(ierr);
    }
    if(H){
        ierr = SaveOperator("H",0,H,mpi_comm); CHKERRQ(ierr);
    }
    ierr = SaveBlockInfo(); CHKERRQ(ierr);
    ierr = Destroy(); CHKERRQ(ierr);
    init = PETSC_FALSE;
    saved = PETSC_TRUE;
    retrieved = PETSC_FALSE;
    return(0);
}


PetscErrorCode Block::SpinBase::Retrieve()
{
    if(!init_save && !disk_set)
        SETERRQ(mpi_comm,1,"InitializeSave() or SetDiskStorage() must be called first.");

    if(init) SETERRQ(mpi_comm,1,"Destroy() must be called first.");
    PetscErrorCode ierr;
    ierr = Retrieve_NoChecks(); CHKERRQ(ierr);
    init = PETSC_TRUE;
    saved = PETSC_FALSE;
    retrieved = PETSC_TRUE;
    return(0);
}


PetscErrorCode Block::SpinBase::Retrieve_NoChecks()
{
    PetscErrorCode ierr;
    PetscBool flg = PETSC_FALSE;
    for(PetscInt isite = 0; isite < num_sites; ++isite){
        ierr = RetrieveOperator("Sz",isite,SzData[isite],mpi_comm); CHKERRQ(ierr);
    }
    for(PetscInt isite = 0; isite < num_sites; ++isite){
        ierr = RetrieveOperator("Sp",isite,SpData[isite],mpi_comm); CHKERRQ(ierr);
    }
    ierr = PetscTestFile(OpFilename(save_dir,"H",0).c_str(), 'r', &flg); CHKERRQ(ierr);
    if(flg){
        ierr = RetrieveOperator("H",0,H,mpi_comm); CHKERRQ(ierr);
    }

    /** If read and write paths were set using SetDiskStorage(), then after the first read
        the value of `save_dir` is swapped from `read_dir` to `write_dir`.
     */
    if(num_reads==0 && disk_set)
    {
        save_dir = write_dir;
    }
    ++num_reads;
    return(0);
}


PetscErrorCode Block::SpinBase::EnsureSaved()
{
    if(!init || !(init_save || disk_set) || saved) return(0);
    PetscErrorCode ierr = SaveAndDestroy(); CHKERRQ(ierr);
    return(0);
}


PetscErrorCode Block::SpinBase::EnsureRetrieved()
{
    if(init || !(init_save || disk_set) || retrieved) return(0);
    PetscErrorCode ierr = Retrieve(); CHKERRQ(ierr);
    return(0);
}


/*--------------- SpinOneHalf Functions ---------------*/

PetscErrorCode Block::SpinOneHalf::MatSpinSzCreate(Mat& Sz)
{
    if(!MPIInitialized()) SETERRQ(PETSC_COMM_SELF,1,"Block's MPI communicator not initialized.");
    PetscErrorCode  ierr;

    ierr = InitSingleSiteOperator(MPIComm(), loc_dim(), &Sz); CHKERRQ(ierr);

    PetscInt locrows, Istart;
    ierr = PreSplitOwnership(MPIComm(), loc_dim(), locrows, Istart); CHKERRQ(ierr);
    PetscInt Iend = Istart + locrows;

    /**
        This is represented by the matrix
        \f{align}{
              S^z &= \frac{1}{2}
                \begin{pmatrix}
                  1  &  0  \\
                  0  & -1
                \end{pmatrix}

        \f}
     */
    if (Istart <= 0 && 0 < Iend){
        ierr = MatSetValue(Sz, 0, 0, +0.5, INSERT_VALUES); CHKERRQ(ierr);
    }
    if (Istart <= 1 && 1 < Iend){
        ierr = MatSetValue(Sz, 1, 1, -0.5, INSERT_VALUES); CHKERRQ(ierr);
    }

    ierr = MatAssemblyBegin(Sz, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Sz, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    return(0);
}


PetscErrorCode Block::SpinOneHalf::MatSpinSpCreate(Mat& Sp)
{
    if(!MPIInitialized()) SETERRQ(PETSC_COMM_SELF,1,"Block's MPI communicator not initialized.");
    PetscErrorCode  ierr;

    ierr = InitSingleSiteOperator(MPIComm(), loc_dim(), &Sp); CHKERRQ(ierr);

    PetscInt locrows, Istart;
    ierr = PreSplitOwnership(MPIComm(), loc_dim(), locrows, Istart); CHKERRQ(ierr);
    PetscInt Iend = Istart + locrows;

    /**
        This is represented by the matrix
        \f{align}{
              S^+ &=
                \begin{pmatrix}
                  0  &  1  \\
                  0  &  0
                \end{pmatrix},
        \f}
     */
    if (Istart <= 0 && 0 < Iend){
        ierr = MatSetValue(Sp, 0, 1, +1.0, INSERT_VALUES); CHKERRQ(ierr);
    }

    ierr = MatAssemblyBegin(Sp, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Sp, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    return(0);
}


/*--------------- SpinOne Functions ---------------*/

PetscErrorCode Block::SpinOne::MatSpinSzCreate(Mat& Sz)
{
    if(!MPIInitialized()) SETERRQ(PETSC_COMM_SELF,1,"Block's MPI communicator not initialized.");
    PetscErrorCode  ierr;

    ierr = InitSingleSiteOperator(MPIComm(), loc_dim(), &Sz); CHKERRQ(ierr);

    PetscInt locrows, Istart;
    ierr = PreSplitOwnership(MPIComm(), loc_dim(), locrows, Istart); CHKERRQ(ierr);
    PetscInt Iend = Istart + locrows;

    /**
        This is represented by the matrix
        \f{align}{
              S^z &=
                \begin{pmatrix}
                  1  &  0  &  0 \\
                  0  &  0  &  0 \\
                  0  &  0  & -1
                \end{pmatrix}
        \f}
     */
    if (Istart <= 0 && 0 < Iend){
        ierr = MatSetValue(Sz, 0, 0, +1.0, INSERT_VALUES); CHKERRQ(ierr);
    }
    if (Istart <= 2 && 2 < Iend){
        ierr = MatSetValue(Sz, 2, 2, -1.0, INSERT_VALUES); CHKERRQ(ierr);
    }

    ierr = MatAssemblyBegin(Sz, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Sz, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    return(0);
}


PetscErrorCode Block::SpinOne::MatSpinSpCreate(Mat& Sp)
{
    if(!MPIInitialized()) SETERRQ(PETSC_COMM_SELF,1,"Block's MPI communicator not initialized.");
    PetscErrorCode  ierr;

    ierr = InitSingleSiteOperator(MPIComm(), loc_dim(), &Sp); CHKERRQ(ierr);

    PetscInt locrows, Istart;
    ierr = PreSplitOwnership(MPIComm(), loc_dim(), locrows, Istart); CHKERRQ(ierr);
    PetscInt Iend = Istart + locrows;

    const PetscScalar Sqrt2 = PetscSqrtScalar(2.0);

    /**
        This is represented by the matrix
        \f{align}{
              S^+ &= \sqrt{2}
                \begin{pmatrix}
                  0  &  1  &  0  \\
                  0  &  0  &  1  \\
                  0  &  0  &  0
                \end{pmatrix},
        \f}
     */
    if (Istart <= 0 && 0 < Iend){
        ierr = MatSetValue(Sp, 0, 1, Sqrt2, INSERT_VALUES); CHKERRQ(ierr);
    }
    if (Istart <= 1 && 1 < Iend){
        ierr = MatSetValue(Sp, 1, 2, Sqrt2, INSERT_VALUES); CHKERRQ(ierr);
    }

    ierr = MatAssemblyBegin(Sp, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Sp, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    return(0);
}
