#ifndef __DMRG_BLOCK_HPP__
#define __DMRG_BLOCK_HPP__

/**
    @defgroup   DMRGBlockContainer   DMRGBlockContainer
    @brief      Implementation of the DMRGBlockContainer class
    @addtogroup DMRGBlockContainer
    @{ */

#include <petsctime.h>
#include <slepceps.h>
#include <petscmat.h>
#include <vector>
#include <map>
#include <set>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <string>
#include "DMRGKron.hpp"

PETSC_EXTERN PetscErrorCode Makedir(const std::string& dir_name);

#define CheckInitialization(init,mpi_comm) if(!init) SETERRQ(mpi_comm,1,"DMRGBlockContainer object not initialized. Call Initialize() first.");
#define PrintLines() printf("-----------------------------------------\n")
#define PRINTLINES() printf("=========================================\n")
#define PrintBlocks(LEFT,RIGHT) printf(" [%lld]-* *-[%lld]\n", LLD(LEFT), LLD(RIGHT))
#define OpIdxToStr(OPTYPE,IDX) (OpToStr(OPTYPE)+std::to_string(IDX))
#define GetOpMats(OPMATS,CORRSIDEOPS,OPID) (OPMATS.at(OpIdxToStr(CORRSIDEOPS.at(OPID).OpType,CORRSIDEOPS.at(OPID).idx)))

/** Storage for information on resulting eigenpairs of the reduced density matrices */
struct Eigen_t
{
    PetscScalar eigval; /**< Eigenvalue */
    PetscInt    seqIdx; /**< Index of the EPS and matrix objects in the vector sequence */
    PetscInt    epsIdx; /**< Index in the EPS object */
    PetscInt    blkIdx; /**< Index in the block's magnetization sectors */
};

/** Comparison operator for sorting Eigen_t objects by decreasing eigenvalues */
bool greater_eigval(const Eigen_t &e1, const Eigen_t &e2) { return e1.eigval > e2.eigval; }

/** Comparison operator for sorting Eigen_t objects by increasing blkIdx (decreasing qn's) */
bool less_blkIdx(const Eigen_t &e1, const Eigen_t &e2) { return e1.blkIdx < e2.blkIdx; }

/** Defines a single operator to be used for performing measurements */
struct Op {
    Op_t        OpType;     /**< Operator type */
    PetscInt    idx;        /**< Site index */

    /** Print out information regarding this operator */
    PetscErrorCode PrintInfo() const {
        std::cout << "  Op" << OpIdxToStr(OpType, idx) << std::endl;
        return(0);
    }
};

/** Contains and manipulates the system and environment blocks used in a single DMRG run */
template<class Block, class Hamiltonian> class DMRGBlockContainer
{

private:

    /** Provides an alias of Side_t to follow the Sys-Env convention */
    typedef enum
    {
        BlockSys = 0,
        BlockEnv = 1
    }
    Block_t;

    /** Lists the types of steps that can be performed */
    typedef enum
    {
        WarmupStep,
        SweepStep,
        NullStep=-1
    }
    Step_t;

    /** Storage for basic data to be saved corresponding to a call of SingleDMRGStep() */
    struct StepData
    {
        PetscInt    NumSites_Sys;       /**< Number of sites in the input sys block */
        PetscInt    NumSites_Env;       /**< Number of sites in the input env block */
        PetscInt    NumSites_SysEnl;    /**< Number of sites in the enlarged sys block */
        PetscInt    NumSites_EnvEnl;    /**< Number of sites in the enlarged env block */
        PetscInt    NumStates_Sys;      /**< Number of states in the input sys block */
        PetscInt    NumStates_Env;      /**< Number of states in the input env block */
        PetscInt    NumStates_SysEnl;   /**< Number of states in the enlarged sys block */
        PetscInt    NumStates_EnvEnl;   /**< Number of states in the enlarged env block */
        PetscInt    NumStates_SysRot;   /**< Number of states in the rotated sys block */
        PetscInt    NumStates_EnvRot;   /**< Number of states in the rotated env block */
        PetscInt    NumStates_H;        /**< Number of states used in constructing the superblock Hamiltonian */
        PetscScalar GSEnergy;           /**< Ground state energy */
        PetscReal   TruncErr_Sys;       /**< Truncation error for the system block */
        PetscReal   TruncErr_Env;       /**< Truncation error for the environment block */
    };

    /** Storage for timings to be saved corresponding to a call of SingleDMRGStep() */
    struct TimingsData
    {
        PetscLogDouble  tEnlr;          /**< Enlargement of each block */
        PetscLogDouble  tKron;          /**< Construction of the Hamiltonian using Kronecker product routines */
        PetscLogDouble  tDiag;          /**< Diagonalization of the Hamiltonian using EPS routines */
        PetscLogDouble  tRdms;          /**< Construction of the reduced density matrices from the ground state vector */
        PetscLogDouble  tRotb;          /**< Rotation of the block operators */
        PetscLogDouble  Total;          /**< Total time */
    };

    /** Describes an n-point correlator whose expectation value will be measured at the end of each sweep.
        Thus, Sys and Env here have reflection symmetry. */
    struct Correlator
    {
        PetscInt            idx;        /**< Index of the correlator in the sequence */
        std::vector< Op >   SysOps;     /**< List of operators residing on the system block */
        std::vector< Op >   EnvOps;     /**< List of operators residing on the environment block */
        std::string         name;       /**< Short name of the correlator */
        std::string         desc1;      /**< Descriptor 1: Using 2d indices */
        std::string         desc2;      /**< Descriptor 2: Using 1d indices */
        std::string         desc3;      /**< Descriptor 3: Using 1d indices separated into sys and env blocks */

        /** Prints some information regarding the correlator */
        PetscErrorCode PrintInfo() const {
            std::cout << "  Correlator " << idx << ": " << name << std::endl;
            std::cout << "    " << desc1 << std::endl;
            std::cout << "    " << desc2 << std::endl;
            std::cout << "    " << desc3 << std::endl;
            return(0);
        }
    };

    /** Context struct for results of the basis transformation that may be useful in solving the correlators */
    struct BasisTransformation
    {
        std::map< PetscInt, Mat >   rdmd_list;      /**< diagonal blocks of the reduced density matrix */
        Mat                         RotMatT;        /**< rotation matrix for the block */
        QuantumNumbers              QN;             /**< quantum numbers context for the block */
        PetscReal                   TruncErr;       /**< total weights of discarded states for the block */

        /** Destructor that safely deallocates the PETSc matrix objects */
        ~BasisTransformation()
        {
            PetscErrorCode ierr = 0;
            for(auto& rdmd: rdmd_list){
                ierr = MatDestroy(&(rdmd.second)); CPP_CHKERR(ierr);
            }
            ierr = MatDestroy(&RotMatT); CPP_CHKERR(ierr);
        }
    };

public:

    /** The constructor only takes in the MPI communicator; Initialize() has to be called next. */
    explicit DMRGBlockContainer(const MPI_Comm& mpi_comm): mpi_comm(mpi_comm){}

    /** Initializes the container object with blocks of one site on each of the system and environment.

        @par Options Database
        Command-line arguments:
         - `-verbose <bool>`
         - `-dry_run <bool>`
         - `-no_symm <bool>`
         - `-do_shell <bool>`
         - `-scratch_dir <string>`
         - `-do_scratch_dir <bool>`
         - `-data_dir <string>`
         - `-do_save_prealloc <bool>`
         - `-mstates <int>`
         - `-mwarmup <int>`
         - `-nsweeps <int>`
         - `-msweeps <int>`

     */
    PetscErrorCode Initialize()
    {
        if(init) SETERRQ(mpi_comm,1,"DMRG object has already been initialized.");
        PetscErrorCode ierr = 0;

        /*  Get MPI attributes */
        ierr = MPI_Comm_size(mpi_comm, &mpi_size); CHKERRQ(ierr);
        ierr = MPI_Comm_rank(mpi_comm, &mpi_rank); CHKERRQ(ierr);

        /*  Initialize Hamiltonian object */
        ierr = Ham.SetFromOptions(); CHKERRQ(ierr);

        /*  Initialize SingleSite which is used as added site */
        ierr = SingleSite.Initialize(mpi_comm, 1, PETSC_DEFAULT); CHKERRQ(ierr);

        num_sites = Ham.NumSites();

        if((num_sites) < 2) SETERRQ1(mpi_comm,1,"There must be at least two total sites. Got %lld.", LLD(num_sites));
        if((num_sites) % 2) SETERRQ1(mpi_comm,1,"Total number of sites must be even. Got %lld.", LLD(num_sites));

        /*  Get some info from command line */
        ierr = PetscOptionsGetBool(NULL,NULL,"-verbose",&verbose,NULL); CHKERRQ(ierr);
        ierr = PetscOptionsGetBool(NULL,NULL,"-no_symm",&no_symm,NULL); CHKERRQ(ierr);
        ierr = PetscOptionsGetBool(NULL,NULL,"-do_shell",&do_shell,NULL); CHKERRQ(ierr);
        ierr = PetscOptionsGetBool(NULL,NULL,"-dry_run",&dry_run,NULL); CHKERRQ(ierr);
        /*  The scratch space to save temporary data*/
        char path[512];
        PetscBool opt_do_scratch_dir;
        ierr = PetscOptionsGetString(NULL,NULL,"-scratch_dir",path,512,&opt_do_scratch_dir); CHKERRQ(ierr);
        do_scratch_dir = opt_do_scratch_dir;
        ierr = PetscOptionsGetBool(NULL,NULL,"-do_scratch_dir",&do_scratch_dir,NULL); CHKERRQ(ierr);
        if(do_scratch_dir){
            scratch_dir = std::string(path);
            if(scratch_dir.back()!='/') scratch_dir += '/';
        }

        /* The location to save basic data */
        memset(path,0,sizeof(path));
        std::string data_dir;
        PetscBool opt_data_dir;
        ierr = PetscOptionsGetString(NULL,NULL,"-data_dir",path,512,&opt_data_dir); CHKERRQ(ierr);
        if(opt_data_dir){
            data_dir = std::string(path);
            if(data_dir.back()!='/') data_dir += '/';
        }
        ierr = PetscFOpen(mpi_comm, (data_dir+std::string("DMRGSteps.json")).c_str(), "w", &fp_step); CHKERRQ(ierr);
        ierr = SaveStepHeaders(); CHKERRQ(ierr);
        if(!mpi_rank) fprintf(fp_step,"[\n");

        ierr = PetscFOpen(mpi_comm, (data_dir+std::string("Timings.json")).c_str(), "w", &fp_timings); CHKERRQ(ierr);
        ierr = SaveTimingsHeaders(); CHKERRQ(ierr);
        if(!mpi_rank) fprintf(fp_timings,"[\n");

        ierr = PetscFOpen(mpi_comm, (data_dir+std::string("EntanglementSpectra.json")).c_str(), "w", &fp_entanglement); CHKERRQ(ierr);
        if(!mpi_rank) fprintf(fp_entanglement,"[\n");

        ierr = PetscFOpen(mpi_comm, (data_dir+std::string("DMRGRun.json")).c_str(), "w", &fp_data); CHKERRQ(ierr);
        if(!mpi_rank){
            fprintf(fp_data,"{\n");
            Ham.SaveOut(fp_data);
            fprintf(fp_data,",\n");
        }

        ierr = PetscFOpen(mpi_comm, (data_dir+std::string("Correlations.json")).c_str(), "w", &fp_corr); CHKERRQ(ierr);

        /* do_save_prealloc: default value is FALSE */
        ierr = PetscOptionsGetBool(NULL,NULL,"-do_save_prealloc",&do_save_prealloc,NULL); CHKERRQ(ierr);
        if(do_save_prealloc){
            ierr = PetscFOpen(mpi_comm, (data_dir+std::string("HamiltonianPrealloc.json")).c_str(), "w", &fp_prealloc); CHKERRQ(ierr);
            if(!mpi_rank) fprintf(fp_prealloc,"[\n");
        }

        /*  Print some info to stdout */
        if(!mpi_rank){
            printf( "=========================================\n"
                    "DENSITY MATRIX RENORMALIZATION GROUP\n"
                    "-----------------------------------------\n");
            Ham.PrintOut();
            printf( "-----------------------------------------\n");
            printf( "DIRECTORIES\n");
            if(do_scratch_dir) printf(
                    "  Scratch: %s\n", scratch_dir.c_str());
            printf( "  Data:    %s\n", opt_data_dir ? data_dir.c_str() : "." );
            printf( "=========================================\n");
        }

        /*  Setup the modes for performing the warmup and sweeps */
        {
            PetscBool   opt_mstates = PETSC_FALSE,
                        opt_mwarmup = PETSC_FALSE,
                        opt_nsweeps = PETSC_FALSE,
                        opt_msweeps = PETSC_FALSE,
                        opt_maxnsweeps = PETSC_FALSE;
            PetscInt mstates, num_msweeps=1000;
            msweeps.resize(num_msweeps);

            ierr = PetscOptionsGetInt(NULL,NULL,"-mstates",&mstates,&opt_mstates); CHKERRQ(ierr);
            ierr = PetscOptionsGetInt(NULL,NULL,"-mwarmup",&mwarmup,&opt_mwarmup); CHKERRQ(ierr);
            ierr = PetscOptionsGetInt(NULL,NULL,"-nsweeps",&nsweeps,&opt_nsweeps); CHKERRQ(ierr);
            ierr = PetscOptionsGetIntArray(NULL,NULL,"-msweeps",&msweeps.at(0),&num_msweeps,&opt_msweeps); CHKERRQ(ierr);
            msweeps.resize(num_msweeps);

            PetscInt num_maxnsweeps = 1000;
            maxnsweeps.resize(num_maxnsweeps);
            ierr = PetscOptionsGetIntArray(NULL,NULL,"-maxnsweeps",&maxnsweeps.at(0),&num_maxnsweeps,&opt_maxnsweeps); CHKERRQ(ierr);
            maxnsweeps.resize(num_maxnsweeps);

            /** @note
                @parblock
                This function also enforces some restrictions on command line inputs:
                - either `-mstates` or `-mwarmup` has to be specified
                @endparblock */
            if(!opt_mstates && !opt_mwarmup)
                SETERRQ(mpi_comm,1,"Either -mstates or -mwarmup has to be specified.");

            /** @parblock
                - `-mstates` and `-mwarmup` are redundant and the value of the latter takes precedence over the other
                @endparblock */
            if(opt_mstates && !opt_mwarmup)
                mwarmup = mstates;

            /** @parblock
                - `-nsweeps` and `-msweeps` are incompatible and only one of them can be specified at a time
                @endparblock */
            if(opt_nsweeps && opt_msweeps)
                SETERRQ(mpi_comm,1,"-msweeps and -nsweeps cannot both be specified at the same time.");

            /** @parblock
                - `-nsweeps` and `-maxnsweeps` are incompatible and only one of them can be specified at a time
                @endparblock */
            if(opt_nsweeps && opt_msweeps)
                SETERRQ(mpi_comm,1,"-nsweeps and -maxnsweeps cannot both be specified at the same time.");

            /** @parblock
                - `-nsweeps` and `-maxnsweeps` are incompatible and only one of them can be specified at a time
                @endparblock */
            if(opt_maxnsweeps && (num_maxnsweeps != num_msweeps))
                SETERRQ2(mpi_comm,1,"-msweeps and -maxnsweeps must have the same number of items. "
                    "Got %lld and %lld, respectively.", num_msweeps, num_maxnsweeps);

            /** @note
                @parblock
                The following criteria is used to decide the kind of sweeps to be performed:
                - if `-nsweeps` is specified use SWEEP_MODE_NSWEEPS
                @endparblock */
            if(opt_nsweeps && !opt_msweeps)
            {
                sweep_mode = SWEEP_MODE_NSWEEPS;
            }
            else if(opt_msweeps && !opt_nsweeps)
            {
                if(opt_maxnsweeps)
                {
                    /** @parblock
                        - if `-msweeps` and `-maxnsweeps` is specified use SWEEP_MODE_TOLERANCE_TEST
                        @endparblock */
                    sweep_mode = SWEEP_MODE_TOLERANCE_TEST;
                }
                else
                {
                    /** @parblock
                        - if `-msweeps` is specified, and not `-maxnsweeps`, use SWEEP_MODE_MSWEEPS
                        @endparblock */
                    sweep_mode = SWEEP_MODE_MSWEEPS;
                }
            }
            else if(!opt_msweeps && !opt_nsweeps)
            {
                sweep_mode = SWEEP_MODE_NULL;
            }
            else
            {
                SETERRQ(mpi_comm,1,"Invalid parameters specified for choosing sweep mode.");
            }

            /* printout some info */
            if(!mpi_rank){
                std::cout
                    << "WARMUP\n"
                    << "  NumStates to keep:           " << mwarmup << "\n"
                    << "SWEEP\n"
                    << "  Sweep mode:                  " << SweepModeToString.at(sweep_mode)
                    << std::endl;

                if(sweep_mode==SWEEP_MODE_NSWEEPS)
                {
                    std::cout << "  Number of sweeps:            " << nsweeps << std::endl;
                }
                else if(sweep_mode==SWEEP_MODE_MSWEEPS)
                {
                    std::cout << "  NumStates to keep:          ";
                    for(const PetscInt& mstates: msweeps) std::cout << " " << mstates;
                    std::cout << std::endl;
                }
                else if(sweep_mode==SWEEP_MODE_TOLERANCE_TEST)
                {
                    std::cout << "  NumStates to keep, maxiter: ";
                    for(size_t i = 0; i < msweeps.size(); ++i)
                        std::cout << " (" << msweeps.at(i) << "," << maxnsweeps.at(i) << ")";
                    std::cout << std::endl;
                }
                else if(sweep_mode==SWEEP_MODE_NULL)
                {

                }

                PRINTLINES();
            }
        }

        LoopType = WarmupStep; /*??????*/
        init = PETSC_TRUE;
        return(0);
    }

    /** Calls the Destroy() method and deallocates container object */
    ~DMRGBlockContainer()
    {
        PetscErrorCode ierr = Destroy(); CPP_CHKERR(ierr);
    }

    /** Destroys the container object */
    PetscErrorCode Destroy()
    {
        if(!init) return(0);
        PetscInt ierr = 0;
        ierr = SingleSite.Destroy(); CHKERRQ(ierr);
        for(Block blk: sys_blocks) { ierr = blk.Destroy(); CHKERRQ(ierr); }
        for(Block blk: env_blocks) { ierr = blk.Destroy(); CHKERRQ(ierr); }

        if(!mpi_rank && !data_tabular) fprintf(fp_step,"\n]\n");
        if(!mpi_rank && data_tabular) fprintf(fp_step,"\n  ]\n}\n");
        ierr = PetscFClose(mpi_comm, fp_step); CHKERRQ(ierr);

        if(!mpi_rank && !data_tabular) fprintf(fp_timings,"\n]\n");
        if(!mpi_rank && data_tabular) fprintf(fp_timings,"\n  ]\n}\n");
        ierr = PetscFClose(mpi_comm, fp_timings); CHKERRQ(ierr);

        if(!mpi_rank) fprintf(fp_entanglement,"\n]\n");
        ierr = PetscFClose(mpi_comm, fp_entanglement); CHKERRQ(ierr);

        ierr = SaveLoopsData();
        if(!mpi_rank) fprintf(fp_data,"\n}\n");
        ierr = PetscFClose(mpi_comm, fp_data); CHKERRQ(ierr);

        if(!mpi_rank) fprintf(fp_corr,"\n  ]\n}\n");
        ierr = PetscFClose(mpi_comm, fp_corr); CHKERRQ(ierr);

        if(do_save_prealloc){
            if(!mpi_rank) fprintf(fp_prealloc,"\n]\n");
            ierr = PetscFClose(mpi_comm, fp_prealloc); CHKERRQ(ierr);
        }

        init = PETSC_FALSE;
        return(0);
    }

    /** Sets up measurement of correlation functions at the end of each sweep. Sites are numbered according
        to the superblock. This function is called once for each measurement. */
    PetscErrorCode SetUpCorrelation(
        const std::vector< Op >& OpList,
        const std::string& name,
        const std::string& desc
        )
    {
        CheckInitialization(init,mpi_comm);

        /* Verify that the function is called before warm up and after initialization */
        if(LoopType==SweepStep)
            SETERRQ(mpi_comm,1,"Setup correlation functions should be called before starting the sweeps.");

        /* Generate the measurement object */
        Correlator m;
        m.idx = measurements.size();
        m.name = name;
        m.desc1 = desc;
        m.desc2 += "< ";
        for(const Op& op: OpList) m.desc2 += OpToStr(op.OpType) + "_{" + std::to_string(op.idx) + "} ";
        m.desc2 += ">";

        /* Convert the input operators list into a measurement struct with site numbering according to blocks */
        for(const Op& op: OpList){
            if(0 <= op.idx && op.idx < num_sites/2){
                m.SysOps.push_back(op);
            }
            else if(num_sites/2 <= op.idx && op.idx < num_sites ){
                m.EnvOps.push_back({op.OpType, num_sites - 1 - op.idx});
            }
            else {
                SETERRQ2(mpi_comm,1,"Operator index must be in the range [0,%D). Got %D.", num_sites, op.idx);
            }
        }

        /* Reflection symmetry: if the resulting SysOps is empty, swap with EnvOps */
        if(m.SysOps.empty())
        {
            m.SysOps = m.EnvOps;
            m.EnvOps.clear();
        }

        /* Also printout the description in terms of local block indices */
        m.desc3 += "< ( ";
        for(const Op& op: m.SysOps) m.desc3 += OpToStr(op.OpType) + "_{" + std::to_string(op.idx) + "} ";
        if(m.SysOps.size()==0) m.desc3 += "1 ";
        m.desc3 += ") ⊗ ( ";
        for(const Op& op: m.EnvOps) m.desc3 += OpToStr(op.OpType) + "_{" + std::to_string(op.idx) + "} ";
        if(m.EnvOps.size()==0) m.desc3 += "1 ";
        m.desc3 += ") >";

        /* Printout some information */
        // if(!mpi_rank) m.PrintInfo();

        measurements.push_back(m);
        return(0);
    }

    /** Performs the warmup stage of DMRG.
        The system and environment blocks are grown until both reach the maximum number which is half the total number
        of sites. All created system blocks are stored and will be represented by at most `MStates` number of basis states */
    PetscErrorCode Warmup()
    {
        CheckInitialization(init,mpi_comm);
        if(dry_run) return(0);

        PetscErrorCode ierr = 0;
        if(warmed_up) SETERRQ(mpi_comm,1,"Warmup has already been called, and it can only be called once.");
        if(!mpi_rank) printf("WARMUP\n");

        /*  Initialize array of blocks */
        num_sys_blocks = num_sites - 1;
        sys_blocks.resize(num_sys_blocks);

        /*  Initialize directories for saving the block operators */
        if(do_scratch_dir){
            PetscBool flg;
            ierr = PetscTestDirectory(scratch_dir.c_str(), 'r', &flg); CHKERRQ(ierr);
            if(!flg) SETERRQ1(mpi_comm,1,"Directory %s does not exist. Please verify that -scratch_dir is specified correctly.",scratch_dir.c_str());
            if(!mpi_rank){
                for(PetscInt iblock = 0; iblock < num_sys_blocks; ++iblock){
                    std::string path = BlockDir("Sys",iblock);
                    ierr = Makedir(path); CHKERRQ(ierr);
                }
                ierr = Makedir(BlockDir("SysEnl",0)); CHKERRQ(ierr);
                ierr = Makedir(BlockDir("EnvEnl",0)); CHKERRQ(ierr);
            }
            for(PetscInt iblock = 0; iblock < num_sys_blocks; ++iblock){
                std::string path = BlockDir("Sys",iblock);
                ierr = sys_blocks[iblock].Initialize(mpi_comm); CHKERRQ(ierr);
                ierr = sys_blocks[iblock].InitializeSave(path); CHKERRQ(ierr);
            }
        }

        /*  Initialize the 0th system block with one site  */
        ierr = sys_blocks[sys_ninit++].Initialize(mpi_comm, 1, PETSC_DEFAULT); CHKERRQ(ierr);

        /*  Create a set of small but exact initial blocks */
        {
            if(num_sites % 2) SETERRQ1(mpi_comm,1,"Total number of sites must be even. Got %d.", num_sites);
            if(AddSite.NumSites() != 1) SETERRQ1(mpi_comm,1,"Routine assumes an additional site of 1. Got %d.", AddSite.NumSites());

            /*  Number of sites in a single cluster, whose multiples form a full lattice ensuring that the total size is even */
            PetscInt nsites_cluster = Ham.NumEnvSites();
            if (nsites_cluster % 2) nsites_cluster *= 2;

            /*  Prepare an exact representation of blocks of sites incremented up to the cluster size */
            if(!mpi_rank){
                if(verbose) PrintLines();
                printf(" Preparing initial blocks.\n");
            }
            while(sys_ninit < nsites_cluster){
                PetscInt NumSitesTotal = sys_blocks[sys_ninit-1].NumSites() + AddSite.NumSites();
                ierr = KronEye_Explicit(sys_blocks[sys_ninit-1], AddSite, Ham.H(NumSitesTotal), sys_blocks[sys_ninit]); CHKERRQ(ierr);
                ++sys_ninit;
            }

            {
                if(!mpi_rank && verbose) printf("  sys_ninit: %lld\n", LLD(sys_ninit));
                for(PetscInt isys = 0; isys < sys_ninit; ++isys){
                    if(!mpi_rank && verbose) printf("   block %lld, num_sites %lld\n", LLD(isys), LLD(sys_blocks[isys].NumSites()));
                }
            }

            if(sys_ninit >= num_sites/2)
            {
                SETERRQ(mpi_comm,1,"No DMRG Steps were performed since all site operators were created exactly. "
                    " Please change the system dimensions.");
            }

            /*  Continuously enlarge the system block until it reaches half the total system size and use the largest
                available environment block that forms a full lattice (multiple of nsites_cluster) */
            LoopType = WarmupStep;
            StepIdx = 0;
            while(sys_ninit < num_sites/2)
            {
                PetscInt full_cluster = (((sys_ninit+2) / nsites_cluster)+1) * nsites_cluster;
                PetscInt env_numsites = full_cluster - sys_ninit - 2;

                /* Increment env_numsites up to the highest number of env_blocks available */
                PetscInt env_add = ((sys_ninit - env_numsites) / nsites_cluster) * nsites_cluster;
                env_numsites += env_add;
                full_cluster += env_add;

                if(env_numsites < 1 || env_numsites > sys_ninit)
                    SETERRQ1(mpi_comm,1,"Incorrect number of sites. Got %lld.",  LLD(env_numsites));

                if(!mpi_rank){
                    if(verbose) PrintLines();
                    printf(" %s  %lld/%lld/%lld\n", "WARMUP",  LLD(LoopIdx),  LLD(StepIdx),  LLD(GlobIdx));
                    PrintBlocks(sys_ninit,env_numsites);
                }
                if(do_scratch_dir){
                    std::set< PetscInt > SysIdx = {sys_ninit-1, env_numsites-1};
                    ierr = SysBlocksActive(SysIdx); CHKERRQ(ierr);
                }
                ierr = SingleDMRGStep(
                    sys_blocks[sys_ninit-1],  sys_blocks[env_numsites-1], mwarmup,
                    sys_blocks[sys_ninit],    sys_blocks[env_numsites], PetscBool(sys_ninit+1==num_sites/2)); CHKERRQ(ierr);

                ++sys_ninit;

                #if defined(PETSC_USE_DEBUG)
                    if(!mpi_rank && verbose) printf("  Number of system blocks: %lld\n", LLD(sys_ninit));
                #endif
            }
        }

        if(sys_ninit != num_sites/2)
            SETERRQ2(mpi_comm,1,"Expected sys_ninit = num_sites/2 = %lld. Got %lld.",LLD(num_sites/2), LLD(sys_ninit));
        /* Destroy environment blocks (if any) */
        for(PetscInt ienv = 0; ienv < env_ninit; ++ienv){
            ierr = env_blocks[0].Destroy(); CHKERRQ(ierr);
        }
        env_ninit = 0;
        warmed_up = PETSC_TRUE;

        if(verbose){
            PetscPrintf(mpi_comm,
                "  Initialized system blocks: %lld\n"
                "  Target number of sites:    %lld\n\n", LLD(sys_ninit), LLD(num_sites));
        }
        if(!mpi_rank) PRINTLINES();
        ++LoopIdx;
        return(0);
    }

    /** Performs the sweep stage of DMRG. */
    PetscErrorCode Sweeps()
    {
        if(dry_run) return(0);
        PetscErrorCode ierr;
        if(sweep_mode==SWEEP_MODE_NSWEEPS)
        {
            for(PetscInt isweep = 0; isweep < nsweeps; ++isweep)
            {
                ierr = SingleSweep(mwarmup); CHKERRQ(ierr);
            }
        }
        else if(sweep_mode==SWEEP_MODE_MSWEEPS)
        {
            for(const PetscInt& mstates: msweeps)
            {
                ierr = SingleSweep(mstates); CHKERRQ(ierr);
            }
        }
        else if(sweep_mode==SWEEP_MODE_TOLERANCE_TEST)
        {
            for(size_t imstates = 0; imstates < msweeps.size(); ++imstates)
            {
                PetscInt mstates  = msweeps.at(imstates);
                PetscInt max_iter = maxnsweeps.at(imstates);
                PetscInt iter = 0;
                if(max_iter==0) continue;

                PetscScalar prev_gse;
                PetscReal   max_trn;
                PetscReal   diff_gse;
                bool        cont;

                do
                {
                    prev_gse = gse;
                    ierr = SingleSweep(mstates); CHKERRQ(ierr);
                    diff_gse = PetscAbsScalar(gse-prev_gse);
                    max_trn = *std::max_element(trunc_err.begin(), trunc_err.end());
                    max_trn = std::max(max_trn,0.0);
                    iter++;
                    cont = (iter<max_iter) && (diff_gse > max_trn);

                    if(!mpi_rank)
                    {
                        std::cout
                            << "SWEEP_MODE_TOLERANCE_TEST\n"
                            << "  Iterations / Max Iterations:       " << iter << "/" << max_iter << "\n"
                            << "  Difference in ground state energy: " << diff_gse << "\n"
                            << "  Largest truncation error:          " << max_trn << "\n"
                            << "  " << (cont?"CONTINUE":"BREAK")
                            << std::endl;
                        PRINTLINES();
                    }
                }
                while(cont);
            }
        }
        else if(sweep_mode==SWEEP_MODE_NULL)
        {

        }
        else
        {
            SETERRQ(mpi_comm,1,"Invalid sweep mode.");
        }

        return(0);
    }

    /** Performs a single sweep from center to right, and back to center */
    PetscErrorCode SingleSweep(
        const PetscInt& MStates, /**< [in] the maximum number of states to keep after each truncation */
        const PetscInt& MinBlock = PETSC_DEFAULT /**< [in] the minimum block length when performing sweeps. Defaults to 1 */
        )
    {
        CheckInitialization(init,mpi_comm);

        PetscErrorCode ierr;
        if(!warmed_up) SETERRQ(mpi_comm,1,"Warmup must be called first before performing sweeps.");
        if(!mpi_rank) printf("SWEEP MStates=%lld\n", LLD(MStates));

        /*  Setup the attributes that will contain some information about this sweep */
        trunc_err.clear();

        /*  Set a minimum number of blocks (min_block). Decide whether to set it statically or let
            the number correspond to the least number of sites needed to exactly build MStates. */
        PetscInt min_block = MinBlock==PETSC_DEFAULT ? 1 : MinBlock;
        if(min_block < 1) SETERRQ1(mpi_comm,1,"MinBlock must at least be 1. Got %d.", min_block);

        /*  Starting from the midpoint, perform a center to right sweep */
        LoopType = SweepStep;
        StepIdx = 0;
        for(PetscInt iblock = num_sites/2; iblock < num_sites - min_block - 2; ++iblock)
        {
            const PetscInt  insys  = iblock-1,   inenv  = num_sites - iblock - 3;
            const PetscInt  outsys = iblock,     outenv = num_sites - iblock - 2;
            if(!mpi_rank){
                if(verbose) PrintLines();
                printf(" %s  %lld/%lld/%lld\n", "SWEEP", LLD(LoopIdx), LLD(StepIdx), LLD(GlobIdx));
                PrintBlocks(insys+1,inenv+1);
            }
            if(do_scratch_dir){
                std::set< PetscInt > SysIdx = {insys, inenv};
                ierr = SysBlocksActive(SysIdx); CHKERRQ(ierr);
            }
            ierr = SingleDMRGStep(sys_blocks[insys],  sys_blocks[inenv], MStates,
                                    sys_blocks[outsys], sys_blocks[outenv]); CHKERRQ(ierr);
        }

        /*  Since we ASSUME REFLECTION SYMMETRY, the remainder of the sweep can be done as follows:
            Starting from the right-most min_block, perform a right to left sweep up to the MIDPOINT */
        for(PetscInt iblock = min_block; iblock < num_sites/2; ++iblock)
        {
            const PetscInt  insys  = num_sites - iblock - 3,    inenv  = iblock-1;
            const PetscInt  outsys = num_sites - iblock - 2,    outenv = iblock;
            if(!mpi_rank){
                if(verbose) PrintLines();
                printf(" %s  %lld/%lld/%lld\n", "SWEEP", LLD(LoopIdx), LLD(StepIdx), LLD(GlobIdx));
                PrintBlocks(insys+1,inenv+1);
            }
            if(do_scratch_dir){
                std::set< PetscInt > SysIdx = {insys, inenv};
                ierr = SysBlocksActive(SysIdx); CHKERRQ(ierr);
            }
            ierr = SingleDMRGStep(sys_blocks[insys],  sys_blocks[inenv], MStates,
                                    sys_blocks[outsys], sys_blocks[outenv], PetscBool(outsys==outenv)); CHKERRQ(ierr);
        }
        sweeps_mstates.push_back(MStates);
        ++LoopIdx;
        if(!mpi_rank) PRINTLINES();

        return(0);
    };

    /** Accesses the specified system block */
    const Block& SysBlock(const PetscInt& BlockIdx) const {
        if(BlockIdx >= sys_ninit) throw std::runtime_error("Attempted to access uninitialized system block.");
        return sys_blocks[BlockIdx];
    }

    /** Accesses the specified environment block */
    const Block& EnvBlock(const PetscInt& BlockIdx) const {
        if(BlockIdx >= env_ninit) throw std::runtime_error("Attempted to access uninitialized environment block.");
        return env_blocks[BlockIdx];
    }

    /** Accesses the 0th environment block */
    const Block& EnvBlock() const{ return env_blocks[0]; }

    /** Returns that number of sites recorded in the Hamiltonian object */
    PetscInt NumSites() const { return num_sites; }

    /** Returns whether verbose printing is active */
    PetscBool Verbose() const { return verbose; }

    /** Const reference to the Hamiltonian object for extracting its parameters */
    const Hamiltonian& HamiltonianRef() const { return Ham; }

private:

    /** Determines the type of sweep to be performed */
    typedef enum
    {
        SWEEP_MODE_NULL,            /**< Do not perform any sweep */
        SWEEP_MODE_NSWEEPS,         /**< Perform N sweeps with a fixed number of kept states */
        SWEEP_MODE_MSWEEPS,         /**< Perform sweeps with a varying number of kept states */
        SWEEP_MODE_TOLERANCE_TEST   /**< Perform sweeps with a certain number of kept states until the
                                         a tolerance is reached, then change the number of kept states */
    } SweepMode_t;

    /** Gives the equivalent string for the sweep mode */
    const std::map< SweepMode_t, std::string > SweepModeToString = {
        {SWEEP_MODE_NULL, "SWEEP_MODE_NULL"},
        {SWEEP_MODE_NSWEEPS, "SWEEP_MODE_NSWEEPS"},
        {SWEEP_MODE_MSWEEPS, "SWEEP_MODE_MSWEEPS"},
        {SWEEP_MODE_TOLERANCE_TEST, "SWEEP_MODE_TOLERANCE_TEST"}
    };

    /** MPI Communicator */
    MPI_Comm    mpi_comm = PETSC_COMM_SELF;

    /** MPI rank in mpi_comm */
    PetscMPIInt mpi_rank;

    /** MPI size of mpi_comm */
    PetscMPIInt mpi_size;

    /** Tells whether the object is initialized */
    PetscBool   init = PETSC_FALSE;

    /** Tells whether to printout info during certain function calls */
    PetscBool   verbose = PETSC_FALSE;

    /** Tells whether to skip the warmup and sweep stages even when called */
    PetscBool dry_run = PETSC_FALSE;

    /** Tells whether the object was initialized using Initialize() */
    PetscBool   warmed_up = PETSC_FALSE;

    /** Tells whether no quantum number symmetries will be implemented */
    PetscBool   no_symm = PETSC_FALSE;

    /** Stores the mode of sweeps to be performed */
    SweepMode_t sweep_mode = SWEEP_MODE_NULL;

    /** Number of states requested for warmup */
    PetscInt    mwarmup = 0;

    /** Number of sweeps to be performed with the number of states specified by mwarmup (for SWEEP_MODE_NSWEEPS only) */
    PetscInt    nsweeps = 0;

    /** Number of states for each sweep (for SWEEP_MODE_MSWEEPS and SWEEP_MODE_TOLERANCE_TEST only) */
    std::vector< PetscInt > msweeps;

    /** Maximum number of sweeps for each state in msweeps (for SWEEP_MODE_TOLERANCE_TEST only) */
    std::vector< PetscInt > maxnsweeps;

    /** Records the number of states requested for each sweep, where each entry is a single
        call to Sweep() */
    std::vector<PetscInt> sweeps_mstates;

    /** Total number of sites */
    PetscInt    num_sites;

    /** Number of system blocks to be stored.
        Usually it is the maximum number of system sites (num_sites - 1) */
    PetscInt    num_sys_blocks;

    /** Number of environment blocks to be stored.
        Usually it is only 1 since the environment block will be re-used */
    PetscInt    num_env_blocks = 1;

    /** Array of system blocks each of which will be kept
        all throughout the simulation */
    std::vector< Block > sys_blocks;

    /** Number of initialized blocks in SysBlocks */
    PetscInt    sys_ninit = 0;

    /** Environment blocks to be used only during warmup.
        For our purposes, this will contain only one block which will
        continuously be enlarged after each iteration */
    std::vector< Block > env_blocks;

    /** Number of initialized blocks in EnvBlocks */
    PetscInt    env_ninit = 0;

    /** Container for the Hamiltonian and geometry */
    Hamiltonian Ham;

    /** Single site that is added to each block
        during the block enlargement procedure */
    Block SingleSite;

    /** Reference to the block of site/s added during enlargement */
    Block& AddSite = SingleSite;

    /** Directory in which the blocks will be saved */
    std::string scratch_dir = ".";

    /** Tells whether to save and retrieve blocks to reduce memory usage at runtime.
        This is automatically set when indicating -scratch_dir */
    PetscBool do_scratch_dir = PETSC_TRUE;

    /** Tells whether data should be saved in tabular form, instead of verbose json */
    PetscBool data_tabular = PETSC_TRUE;

    /** Tells whether to save preallocation data for the superblock Hamiltonian */
    PetscBool do_save_prealloc = PETSC_FALSE;

    /** Whether to create an implicit MATSHELL matrix for the superblock Hamiltonian */
    PetscBool do_shell = PETSC_TRUE;

    /** File to store basic data (energy, number of sites, etc) */
    FILE *fp_step = NULL;

    /** File to store timings data for each section of a single iteration */
    FILE *fp_timings = NULL;

    /** File to store the entanglement spectrum of the reduced density matrices for a single iteration */
    FILE *fp_entanglement = NULL;

    /** File to store timings data for each section of a single iteration */
    FILE *fp_data = NULL;

    /** File to store preallocation data of the superblock Hamiltonian */
    FILE *fp_prealloc = NULL;

    /** File to store timings data for each section of a single iteration */
    FILE *fp_corr = NULL;

    /** Global index key which must be unique for each record */
    PetscInt GlobIdx = 0;

    /** The type of step performed, whether as part of warmup or sweep */
    Step_t   LoopType = NullStep;

    /** Counter for this loop (the same counter for warmup and sweeps) */
    PetscInt LoopIdx = 0;

    /** Counter for the step inside this loop */
    PetscInt StepIdx = 0;

    /** Stores the measurements to be performed at the end of each sweep */
    std::vector< Correlator > measurements;

    /** Tells whether the headers for the correlators have been printed */
    PetscBool corr_headers_printed = PETSC_FALSE;

    /** Tells whether a single entry for the correlators have been printed */
    PetscBool corr_printed_first = PETSC_FALSE;

    /** Stores the ground state energy at the end of a single dmrg step */
    PetscScalar gse = 0.0;

    /** Stores the truncation error at the end of a single dmrg step */
    std::vector<PetscReal>  trunc_err;

    /** Performs a single DMRG iteration taking in a system and environment block, adding one site
        to each and performing a truncation to at most MStates */
    PetscErrorCode SingleDMRGStep(
        Block& SysBlock,            /**< [in] the old system (left) block */
        Block& EnvBlock,            /**< [in] the old environment (right) block */
        const PetscInt& MStates,    /**< [in] the maximum number of states to keep */
        Block& SysBlockOut,         /**< [out] the new system (left) block */
        Block& EnvBlockOut,         /**< [out] the new environment (right) block */
        PetscBool do_measurements=PETSC_FALSE /**< [in] whether to do measurements for this step */
        )
    {
        PetscErrorCode ierr;
        PetscLogDouble t0, tenlr, tkron, tdiag, trdms, trotb;
        TimingsData timings_data;
        ierr = PetscTime(&t0); CHKERRQ(ierr);

        /* Fill-in data from input blocks */
        StepData step_data;
        step_data.NumSites_Sys = SysBlock.NumSites();
        step_data.NumSites_Env = EnvBlock.NumSites();
        step_data.NumStates_Sys = SysBlock.NumStates();
        step_data.NumStates_Env = EnvBlock.NumStates();

        /* Check whether the system and environment blocks are the same */
        Mat H = nullptr; /* Hamiltonian matrix */
        const PetscBool flg = PetscBool(&SysBlock==&EnvBlock);

        /* (Block) Add one site to each block */
        Block SysBlockEnl, EnvBlockEnl;
        PetscInt NumSitesSysEnl = SysBlock.NumSites() + AddSite.NumSites();
        const std::vector< Hamiltonians::Term > TermsSys = Ham.H(NumSitesSysEnl);
        ierr = KronEye_Explicit(SysBlock, AddSite, TermsSys, SysBlockEnl); CHKERRQ(ierr);
        ierr = SysBlock.EnsureSaved(); CHKERRQ(ierr);
        if(!flg){
            PetscInt NumSitesEnvEnl = EnvBlock.NumSites() + AddSite.NumSites();
            const std::vector< Hamiltonians::Term > TermsEnv = Ham.H(NumSitesEnvEnl);
            ierr = KronEye_Explicit(EnvBlock, AddSite, TermsEnv, EnvBlockEnl); CHKERRQ(ierr);
            ierr = EnvBlock.EnsureSaved(); CHKERRQ(ierr);
            ierr = EnvBlockEnl.InitializeSave(BlockDir("EnvEnl",0)); CHKERRQ(ierr);
        } else {
            EnvBlockEnl = SysBlockEnl;
        }
        ierr = SysBlockEnl.InitializeSave(BlockDir("SysEnl",0)); CHKERRQ(ierr);

        ierr = PetscTime(&tenlr); CHKERRQ(ierr);
        timings_data.tEnlr = tenlr-t0;
        if(!mpi_rank && verbose) printf("* Add One Site:          %12.6f s\n", timings_data.tEnlr);

        step_data.NumSites_SysEnl = SysBlockEnl.NumSites();
        step_data.NumSites_EnvEnl = EnvBlockEnl.NumSites();
        step_data.NumStates_SysEnl = SysBlockEnl.NumStates();
        step_data.NumStates_EnvEnl = EnvBlockEnl.NumStates();

        #if defined(PETSC_USE_DEBUG)
        {
            PetscBool flg = PETSC_FALSE;
            ierr = PetscOptionsGetBool(NULL,NULL,"-print_qn",&flg,NULL); CHKERRQ(ierr);
            if(flg){
                /* Print the enlarged system block's quantum numbers for each state */
                ierr = PetscPrintf(mpi_comm,"  SysBlockEnl  "); CHKERRQ(ierr);
                ierr = SysBlockEnl.Magnetization.PrintQNs(); CHKERRQ(ierr);
                ierr = PetscPrintf(mpi_comm,"  EnvBlockEnl  "); CHKERRQ(ierr);
                ierr = EnvBlockEnl.Magnetization.PrintQNs(); CHKERRQ(ierr);
            }
        }
        #endif

        /* Prepare the Hamiltonian taking both enlarged blocks together */
        PetscInt NumSitesTotal = SysBlockEnl.NumSites() + EnvBlockEnl.NumSites();
        const std::vector< Hamiltonians::Term > Terms = Ham.H(NumSitesTotal);

        /* Set the QN sectors as an option */
        std::vector<PetscReal> QNSectors = {0};
        if(no_symm) {
            QNSectors = {};
        }
        KronBlocks_t KronBlocks(SysBlockEnl, EnvBlockEnl, QNSectors, fp_prealloc, GlobIdx);
        step_data.NumStates_H = KronBlocks.NumStates();
        #if defined(PETSC_USE_DEBUG)
        {
            PetscBool flg = PETSC_FALSE;
            ierr = PetscOptionsGetBool(NULL,NULL,"-print_H_kron",&flg,NULL); CHKERRQ(ierr);
            if(flg && !mpi_rank){
                std::cout << "***** Kron_Explicit *****" << std::endl;
                std::cout << "SysBlockEnl  qn_list:   ";
                for(auto i: SysBlockEnl.Magnetization.List()) std::cout << i << "   ";
                std::cout << std::endl;

                std::cout << "SysBlockEnl  qn_size:   ";
                for(auto i: SysBlockEnl.Magnetization.Sizes()) std::cout << i << "   ";
                std::cout << std::endl;

                std::cout << "SysBlockEnl  qn_offset: ";
                for(auto i: SysBlockEnl.Magnetization.Offsets()) std::cout << i << "   ";
                std::cout << std::endl;

                std::cout << std::endl;

                std::cout << "EnvBlockEnl qn_list:   ";
                for(auto i: EnvBlockEnl.Magnetization.List()) std::cout << i << "   ";
                std::cout << std::endl;

                std::cout << "EnvBlockEnl qn_size:   ";
                for(auto i: EnvBlockEnl.Magnetization.Sizes()) std::cout << i << "   ";
                std::cout << std::endl;

                std::cout << "EnvBlockEnl qn_offset: ";
                for(auto i: EnvBlockEnl.Magnetization.Offsets()) std::cout << i << "   ";
                std::cout << std::endl;

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
            if(flg){
                if(!mpi_rank){std::cout << "***** SysBlockEnl *****" << std::endl;}
                for(const Mat& mat: SysBlockEnl.Sz())
                {
                    MatPeek(mat,"Sz");
                }
                for(const Mat& mat: SysBlockEnl.Sp())
                {
                    MatPeek(mat,"Sp");
                }
                if(!mpi_rank){std::cout << "***** EnvBlockEnl *****" << std::endl;}
                for(const Mat& mat: EnvBlockEnl.Sz())
                {
                    MatPeek(mat,"Sz");
                }
                for(const Mat& mat: EnvBlockEnl.Sp())
                {
                    MatPeek(mat,"Sp");
                }
                if(!mpi_rank){std::cout << "***********************" << std::endl;}
            }
        }
        #endif

        ierr = KronBlocks.KronSumSetRedistribute(PETSC_TRUE); CHKERRQ(ierr);
        ierr = KronBlocks.KronSumSetToleranceFromOptions(); CHKERRQ(ierr);
        ierr = KronBlocks.KronSumSetShellMatrix(do_shell); CHKERRQ(ierr);
        ierr = KronBlocks.KronSumConstruct(Terms, H); CHKERRQ(ierr);
        if(!H) SETERRQ(mpi_comm,1,"H is null.");

        ierr = PetscTime(&tkron); CHKERRQ(ierr);
        timings_data.tKron = tkron-tenlr;
        if(!mpi_rank && verbose)  printf("* Build Superblock H:    %12.6f s\n", timings_data.tKron);

        #if defined(PETSC_USE_DEBUG)
        {
            PetscBool flg = PETSC_FALSE;
            ierr = PetscOptionsGetBool(NULL,NULL,"-print_H",&flg,NULL); CHKERRQ(ierr);
            if(flg){ ierr = MatPeek(H,"H"); CHKERRQ(ierr); }
            flg = PETSC_FALSE;
            ierr = PetscOptionsGetBool(NULL,NULL,"-print_H_terms",&flg,NULL); CHKERRQ(ierr);
            if(flg){
                if(!mpi_rank) printf(" H(%lld)\n", LLD(NumSitesTotal));
                for(const Hamiltonians::Term& term: Terms)
                {
                    if(!mpi_rank) printf("%.2f %2s(%2lld) %2s(%2lld)\n", term.a, (OpString.find(term.Iop)->second).c_str(), LLD(term.Isite),
                        (OpString.find(term.Jop)->second).c_str(), LLD(term.Jsite) );
                }
            }
            ierr = MPI_Barrier(mpi_comm); CHKERRQ(ierr);
        }
        #endif

        /* Solve for the ground state */

        #if defined(PETSC_USE_COMPLEX)
            SETERRQ(mpi_comm,PETSC_ERR_SUP,"This function is only implemented for scalar-type=real.");
            /*  Using both gsv_r and gsv_i but assuming that gsv_i = 0 */
        #endif

        Vec gsv_r, gsv_i;
        PetscScalar gse_r, gse_i;
        ierr = MatCreateVecs(H, &gsv_r, nullptr); CHKERRQ(ierr);
        ierr = MatCreateVecs(H, &gsv_i, nullptr); CHKERRQ(ierr);
        {
            EPS eps;
            ierr = EPSCreate(mpi_comm, &eps); CHKERRQ(ierr);
            ierr = EPSSetOperators(eps, H, nullptr); CHKERRQ(ierr);
            ierr = EPSSetProblemType(eps, EPS_HEP); CHKERRQ(ierr);
            ierr = EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL); CHKERRQ(ierr);
            ierr = EPSSetOptionsPrefix(eps,"H_"); CHKERRQ(ierr);
            ierr = EPSSetFromOptions(eps); CHKERRQ(ierr);
            ierr = EPSSolve(eps); CHKERRQ(ierr);
            ierr = EPSGetEigenpair(eps, 0, &gse_r, &gse_i, gsv_r, gsv_i); CHKERRQ(ierr);
            ierr = EPSDestroy(&eps); CHKERRQ(ierr);
        }
        step_data.GSEnergy = gse_r;

        if(do_shell){
            ierr = MatDestroy_KronSumShell(&H); CHKERRQ(ierr);
        }
        ierr = MatDestroy(&H); CHKERRQ(ierr);

        ierr = PetscTime(&tdiag); CHKERRQ(ierr);
        timings_data.tDiag = tdiag-tkron;
        if(!mpi_rank && verbose) printf("* Solve Ground State:    %12.6f s\n", timings_data.tDiag);

        #if defined(PETSC_USE_DEBUG)
        {
            PetscBool flg = PETSC_FALSE;
            ierr = PetscOptionsGetBool(NULL,NULL,"-print_H_gs",&flg,NULL); CHKERRQ(ierr);
            if(flg){
                ierr = PetscPrintf(mpi_comm, "\n Ground State Energy: %g + %gj\n", gse_r, gse_i); CHKERRQ(ierr);
                ierr = VecPeek(gsv_r, " gsv_r"); CHKERRQ(ierr);
            }
        }
        #endif

        if(no_symm){
            ierr = MPI_Barrier(mpi_comm); CHKERRQ(ierr);
            SETERRQ(mpi_comm,PETSC_ERR_SUP,"Unsupported option: no_symm.");
        }

        /*  Calculate the reduced density matrices in block-diagonal form, and from this we can calculate the
            (transposed) rotation matrix */
        BasisTransformation *BT_L, *BT_R;
        BT_L = new BasisTransformation;
        BT_R = new BasisTransformation;
        ierr = GetTruncation(KronBlocks, gsv_r, MStates, *BT_L, *BT_R); CHKERRQ(ierr);

        /* TODO: Add an option to accept flg for redundant blocks */
        /* TODO: Retrieve reduced density matrices from this function */

        /* TODO: Perform evaluation of inter-block correlation functions here */
        /* TODO: Perform evaluation of intra-block correlation functions */

        ierr = CalculateCorrelations_BlockDiag(KronBlocks, gsv_r, *BT_L, do_measurements); CHKERRQ(ierr);

        ierr = VecDestroy(&gsv_r); CHKERRQ(ierr);
        ierr = VecDestroy(&gsv_i); CHKERRQ(ierr);

        /* (Block) Initialize the new blocks
            copy enlarged blocks to out blocks but overwrite the matrices */
        ierr = SysBlockOut.Destroy(); CHKERRQ(ierr);
        ierr = EnvBlockOut.Destroy(); CHKERRQ(ierr);

        ierr = PetscTime(&trdms); CHKERRQ(ierr);
        timings_data.tRdms = trdms-tdiag;
        if(!mpi_rank && verbose) printf("* Eigendec. of RDMs:     %12.6f s\n", timings_data.tRdms);

        ierr = SysBlockOut.Initialize(SysBlockEnl.NumSites(), BT_L->QN); CHKERRQ(ierr);
        ierr = SysBlockOut.RotateOperators(SysBlockEnl, BT_L->RotMatT); CHKERRQ(ierr);
        ierr = SysBlockEnl.Destroy(); CHKERRQ(ierr);
        if(!flg){
            ierr = EnvBlockOut.Initialize(EnvBlockEnl.NumSites(), BT_R->QN); CHKERRQ(ierr);
            ierr = EnvBlockOut.RotateOperators(EnvBlockEnl, BT_R->RotMatT); CHKERRQ(ierr);
            ierr = EnvBlockEnl.Destroy(); CHKERRQ(ierr);
        }

        step_data.NumStates_SysRot = SysBlockOut.NumStates();
        step_data.NumStates_EnvRot = EnvBlockOut.NumStates();
        step_data.TruncErr_Sys = BT_L->TruncErr;
        step_data.TruncErr_Env = BT_R->TruncErr;

        #if defined(PETSC_USE_DEBUG)
        {
            PetscBool flg = PETSC_FALSE;
            ierr = PetscOptionsGetBool(NULL,NULL,"-print_qn",&flg,NULL); CHKERRQ(ierr);
            if(flg){
                /* Print the enlarged system block's quantum numbers for each state */
                ierr = PetscPrintf(mpi_comm,"  SysBlockOut  "); CHKERRQ(ierr);
                ierr = SysBlockOut.Magnetization.PrintQNs(); CHKERRQ(ierr);
                ierr = PetscPrintf(mpi_comm,"  EnvBlockOut  "); CHKERRQ(ierr);
                ierr = EnvBlockOut.Magnetization.PrintQNs(); CHKERRQ(ierr);
            }
        }
        #endif

        ierr = PetscTime(&trotb); CHKERRQ(ierr);
        timings_data.tRotb = trotb-trdms;
        if(!mpi_rank && verbose) printf("* Rotation of Operators: %12.6f s\n", timings_data.tRotb);

        timings_data.Total = trotb - t0;
        timings_data.Total += (timings_data.Total < 0) * 86400.0; /* Just in case it transitions from a previous day */

        if(!mpi_rank && verbose){
            const PetscReal pEnlr = 100*(timings_data.tEnlr)/timings_data.Total;
            const PetscReal pKron = 100*(timings_data.tKron)/timings_data.Total;
            const PetscReal pDiag = 100*(timings_data.tDiag)/timings_data.Total;
            const PetscReal pRdms = 100*(timings_data.tRdms)/timings_data.Total;
            const PetscReal pRotb = 100*(timings_data.tRotb)/timings_data.Total;
            printf("\n");
            printf("  Sys Block In:\n");
            printf("    NumStates:      %lld\n", LLD(SysBlock.Magnetization.NumStates()));
            printf("    NumSites:       %lld\n", LLD(SysBlock.NumSites()));
            printf("  Env Block In:\n");
            printf("    NumStates:      %lld\n", LLD(EnvBlock.Magnetization.NumStates()));
            printf("    NumSites:       %lld\n", LLD(EnvBlock.NumSites()));
            printf("  Sys Block Enl:\n");
            printf("    NumStates:      %lld\n", LLD(SysBlockEnl.Magnetization.NumStates()));
            printf("    NumSites:       %lld\n", LLD(SysBlockEnl.NumSites()));
            printf("  Env Block Enl:\n");
            printf("    NumStates:      %lld\n", LLD(EnvBlockEnl.Magnetization.NumStates()));
            printf("    NumSites:       %lld\n", LLD(EnvBlockEnl.NumSites()));
            printf("  Superblock:\n");
            printf("    NumStates:      %lld\n", LLD(KronBlocks.NumStates()));
            printf("    NumSites:       %lld\n", LLD(NumSitesTotal));
            printf("    Energy:         %-10.10g\n", gse_r);
            printf("    Energy/site:    %-10.10g\n", gse_r/PetscReal(NumSitesTotal));
            printf("  Sys Block Out\n"
                   "    NumStates:      %lld\n"
                   "    TrunError:      %g\n", LLD(BT_L->QN.NumStates()), BT_L->TruncErr);
            printf("  Env Block Out\n"
                   "    NumStates:      %lld\n"
                   "    TrunError:      %g\n", LLD(BT_R->QN.NumStates()), BT_R->TruncErr);
            printf("\n");
            printf("  Total Time:              %12.6f s\n", timings_data.Total);
            printf("    Add One Site:          %12.6f s \t%6.2f %%\n", timings_data.tEnlr, pEnlr);
            printf("    Build Superblock H:    %12.6f s \t%6.2f %%\n", timings_data.tKron, pKron);
            printf("    Solve Ground State:    %12.6f s \t%6.2f %%\n", timings_data.tDiag, pDiag);
            printf("    Eigendec. of RDMs:     %12.6f s \t%6.2f %%\n", timings_data.tRdms, pRdms);
            printf("    Rotation of Operators: %12.6f s \t%6.2f %%\n", timings_data.tRotb, pRotb);
            printf("\n");
        }

        /* Store some results to class attributes */
        gse = gse_r;
        trunc_err.push_back(BT_L->TruncErr);

        /* Save data */
        ierr = SaveStepData(step_data); CHKERRQ(ierr);
        ierr = SaveTimingsData(timings_data); CHKERRQ(ierr);

        /* Delete context */
        delete BT_L;
        delete BT_R;

        /* Increment counters */
        ++GlobIdx;
        ++StepIdx;
        return(0);
    }

    /** Obtain the rotation matrix for the truncation step from the ground state vector */
    PetscErrorCode GetTruncation(
        const KronBlocks_t& KronBlocks, /**< [in] Context for quantum numbers aware Kronecker product */
        const Vec& gsv_r,               /**< [in] Real part of the superblock ground state vector */
        const PetscInt& MStates,        /**< [in] the maximum number of states to keep */
        BasisTransformation& BT_L,      /**< [out] basis transformation context for the system (left) block */
        BasisTransformation& BT_R       /**< [out] basis transformation context for the environment (right) block */
        )
    {
        PetscErrorCode ierr;

        if(no_symm) SETERRQ(mpi_comm,PETSC_ERR_SUP,"Unsupported option: no_symm.");
        #if defined(PETSC_USE_COMPLEX)
            SETERRQ(mpi_comm,PETSC_ERR_SUP,"This function is only implemented for scalar-type=real.");
            /* Error due to re-use of *v buffer for *vT */
        #endif

        /*  Send the whole vector to the root process */
        Vec gsv_r_loc;
        VecScatter ctx;
        ierr = VecScatterCreateToZero(gsv_r, &ctx, &gsv_r_loc); CHKERRQ(ierr);
        ierr = VecScatterBegin(ctx, gsv_r, gsv_r_loc, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
        ierr = VecScatterEnd(ctx, gsv_r, gsv_r_loc, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

        #if defined(PETSC_USE_DEBUG)
        PetscBool flg = PETSC_FALSE;
        ierr = PetscOptionsGetBool(NULL,NULL,"-print_trunc",&flg,NULL); CHKERRQ(ierr);
        if(false){
            for(PetscMPIInt irank = 0; irank < mpi_size; ++irank){
                if(irank==mpi_rank){std::cout << "[" << mpi_rank << "]<<" << std::endl;

                    ierr = PetscPrintf(PETSC_COMM_SELF, "gsv_r_loc\n"); CHKERRQ(ierr);
                    ierr = VecView(gsv_r_loc, PETSC_VIEWER_STDOUT_SELF); CHKERRQ(ierr);

                std::cout << ">>[" << mpi_rank << "]" << std::endl;}\
            ierr = MPI_Barrier(mpi_comm); CHKERRQ(ierr);}
        }
        #endif

        std::vector< Eigen_t > eigen_L, eigen_R;        /* Container for eigenvalues of the RDMs */
        std::vector< EPS > eps_list_L, eps_list_R;      /* Container for EPS objects */
        std::vector< Vec > rdmd_vecs_L, rdmd_vecs_R;    /* Container for the corresponding vectors */

        /*  Do eigendecomposition on root process  */
        if(!mpi_rank)
        {
            /*  Verify the vector length  */
            PetscInt size;
            ierr = VecGetSize(gsv_r_loc, &size);
            if(KronBlocks.NumStates() != size) SETERRQ2(PETSC_COMM_SELF,1,"Incorrect vector length. "
                "Expected %d. Got %d.", KronBlocks.NumStates(), size);

            #if defined(PETSC_USE_DEBUG)
                if(flg) printf("\n\n");
            #endif

            PetscScalar *v;
            ierr = VecGetArray(gsv_r_loc, &v); CHKERRQ(ierr);

            /*  Loop through the L-R pairs forming the target sector in KronBlocks */
            for(PetscInt idx = 0; idx < KronBlocks.size(); ++idx)
            {
                const PetscInt Istart = KronBlocks.Offsets(idx);
                const PetscInt Iend   = KronBlocks.Offsets(idx+1);
                const PetscInt Idx_L  = KronBlocks.LeftIdx(idx);
                const PetscInt Idx_R  = KronBlocks.RightIdx(idx);
                const PetscInt N_L    = KronBlocks.LeftBlockRef().Magnetization.Sizes(Idx_L);
                const PetscInt N_R    = KronBlocks.RightBlockRef().Magnetization.Sizes(Idx_R);

                /*  Verify the segment length */
                if(Iend - Istart != N_L * N_R) SETERRQ2(PETSC_COMM_SELF,1, "Incorrect segment length. "
                    "Expected %d. Got %d.", N_L * N_R, Iend - Istart);

                /*  Initialize and fill sequential dense matrices containing the diagonal blocks of the
                    reduced density matrices */
                Mat Psi, PsiT, rdmd_L, rdmd_R;
                ierr = MatCreateSeqDense(PETSC_COMM_SELF, N_R, N_L, &v[Istart], &PsiT); CHKERRQ(ierr);
                ierr = MatHermitianTranspose(PsiT, MAT_INITIAL_MATRIX, &Psi); CHKERRQ(ierr);
                ierr = MatMatMult(Psi, PsiT, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &rdmd_L);
                ierr = MatMatMult(PsiT, Psi, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &rdmd_R);
                /*  TODO: Check for possible bleeding of memory due to ownership of *v */
                ierr = MatDestroy(&Psi); CHKERRQ(ierr);
                ierr = MatDestroy(&PsiT); CHKERRQ(ierr);

                /*  Verify the sizes of the reduced density matrices */
                {
                    PetscInt Nrows, Ncols;
                    ierr = MatGetSize(rdmd_L, &Nrows, &Ncols); CHKERRQ(ierr);
                    if(Nrows != N_L) SETERRQ2(PETSC_COMM_SELF,1,"Incorrect Nrows in L. Expected %d. Got %d.", N_L, Nrows);
                    if(Ncols != N_L) SETERRQ2(PETSC_COMM_SELF,1,"Incorrect Ncols in L. Expected %d. Got %d.", N_L, Ncols);
                    ierr = MatGetSize(rdmd_R, &Nrows, &Ncols); CHKERRQ(ierr);
                    if(Nrows != N_R) SETERRQ2(PETSC_COMM_SELF,1,"Incorrect Nrows in R. Expected %d. Got %d.", N_R, Nrows);
                    if(Ncols != N_R) SETERRQ2(PETSC_COMM_SELF,1,"Incorrect Ncols in R. Expected %d. Got %d.", N_R, Ncols);
                }

                /*  Solve the full eigenspectrum of the reduced density matrices */
                EPS eps_L, eps_R;
                ierr = EigRDM_BlockDiag(rdmd_L, idx, Idx_L, eigen_L, eps_L); CHKERRQ(ierr);
                ierr = EigRDM_BlockDiag(rdmd_R, idx, Idx_R, eigen_R, eps_R); CHKERRQ(ierr);

                #if defined(PETSC_USE_DEBUG)
                if(flg){
                    printf(" KB QN: %-6g  Left :%3lld  Right: %3lld\n", KronBlocks.QN(idx), LLD(Idx_L), LLD(Idx_R))   ;
                    ierr = MatPeek(rdmd_L, "rdmd_L"); CHKERRQ(ierr);
                    ierr = MatPeek(rdmd_R, "rdmd_R"); CHKERRQ(ierr);
                    printf("\n");
                }
                #endif

                eps_list_L.push_back(eps_L);
                eps_list_R.push_back(eps_R);
                BT_L.rdmd_list[Idx_L] = rdmd_L;
                BT_R.rdmd_list[Idx_R] = rdmd_R;

                /*  Prepare the vectors for getting the eigenvectors */
                Vec v_L, v_R;
                ierr = MatCreateVecs(rdmd_L, NULL, &v_L); CHKERRQ(ierr);
                rdmd_vecs_L.push_back(v_L);
                ierr = MatCreateVecs(rdmd_R, NULL, &v_R); CHKERRQ(ierr);
                rdmd_vecs_R.push_back(v_R);
            }

            #if defined(PETSC_USE_DEBUG)
            if(flg){
                printf("\nBefore sorting\n");
                for(const Eigen_t& eig: eigen_L) printf(" L: %-16.10g seq: %-5lld eps: %-5lld blk: %-5lld\n",
                    eig.eigval, LLD(eig.seqIdx), LLD(eig.epsIdx), LLD(eig.blkIdx));
                printf("\n");
                for(const Eigen_t& eig: eigen_R) printf(" R: %-16.10g seq: %-5lld eps: %-5lld blk: %-5lld\n",
                    eig.eigval, LLD(eig.seqIdx), LLD(eig.epsIdx), LLD(eig.blkIdx));
                printf("\n\n");
            }
            #endif

            /*  Dump unsorted (grouped) entanglement spectra to file */
            ierr = SaveEntanglementSpectra(
                eigen_L, KronBlocks.LeftBlockRef().Magnetization.ListRef(),
                eigen_R, KronBlocks.RightBlockRef().Magnetization.ListRef()); CHKERRQ(ierr);

            /*  Sort the eigenvalue lists in descending order */
            std::stable_sort(eigen_L.begin(), eigen_L.end(), greater_eigval);
            std::stable_sort(eigen_R.begin(), eigen_R.end(), greater_eigval);

            #if defined(PETSC_USE_DEBUG)
            if(flg){
                printf("\nAfter sorting\n");
                for(const Eigen_t& eig: eigen_L) printf(" L: %-16.10g seq: %-5lld eps: %-5lld blk: %-5lld\n",
                    eig.eigval, LLD(eig.seqIdx), LLD(eig.epsIdx), LLD(eig.blkIdx));
                printf("\n");
                for(const Eigen_t& eig: eigen_R) printf(" R: %-16.10g seq: %-5lld eps: %-5lld blk: %-5lld\n",
                    eig.eigval, LLD(eig.seqIdx), LLD(eig.epsIdx), LLD(eig.blkIdx));
                printf("\n\n");
            }
            #endif
            ierr = VecRestoreArray(gsv_r_loc, &v); CHKERRQ(ierr);

        }
        /*  Broadcast the number of eigenstates from 0 to all processes */
        PetscInt NEigenStates_L = eigen_L.size(); /* Number of eigenstates in the left block reduced density matrix */
        PetscInt NEigenStates_R = eigen_R.size(); /* Number of eigenstates in the right block reduced density matrix */
        ierr = MPI_Bcast(&NEigenStates_L, 1, MPIU_INT, 0, PETSC_COMM_WORLD); CHKERRQ(ierr);
        ierr = MPI_Bcast(&NEigenStates_R, 1, MPIU_INT, 0, PETSC_COMM_WORLD); CHKERRQ(ierr);

        /*  Decide how many states to retain */
        const PetscInt m_L = PetscMin(MStates, NEigenStates_L);
        const PetscInt m_R = PetscMin(MStates, NEigenStates_R);

        /*  The number of states present in the enlarged blocks */
        const PetscInt NStates_L = KronBlocks.LeftBlockRef().Magnetization.NumStates();
        const PetscInt NStates_R = KronBlocks.RightBlockRef().Magnetization.NumStates();

        /*  The rotation matrices take have the dimension m x NStates */
        ierr = MatCreate(mpi_comm, &BT_L.RotMatT); CHKERRQ(ierr);
        ierr = MatCreate(mpi_comm, &BT_R.RotMatT); CHKERRQ(ierr);
        Mat& RotMatT_L = BT_L.RotMatT;
        Mat& RotMatT_R = BT_R.RotMatT;
        ierr = MatSetSizes(RotMatT_L, PETSC_DECIDE, PETSC_DECIDE, m_L, NStates_L); CHKERRQ(ierr);
        ierr = MatSetSizes(RotMatT_R, PETSC_DECIDE, PETSC_DECIDE, m_R, NStates_R); CHKERRQ(ierr);
        ierr = MatSetFromOptions(RotMatT_L); CHKERRQ(ierr);
        ierr = MatSetFromOptions(RotMatT_R); CHKERRQ(ierr);
        ierr = MatSetUp(RotMatT_L); CHKERRQ(ierr);
        ierr = MatSetUp(RotMatT_R); CHKERRQ(ierr);

        #if defined(PETSC_USE_DEBUG)
            if(flg && !mpi_rank) printf("    m_L: %-lld  m_R: %-lld\n\n", LLD(m_L), LLD(m_R));
        #endif

        std::vector< PetscReal > qn_list_L, qn_list_R;
        std::vector< PetscInt >  qn_size_L, qn_size_R;
        PetscInt numBlocks_L, numBlocks_R;
        PetscReal& TruncErr_L = BT_L.TruncErr;
        PetscReal& TruncErr_R = BT_R.TruncErr;
        if(!mpi_rank)
        {
            /* Take only the first m states and sort in ascending order of blkIdx */
            eigen_L.resize(m_L);
            eigen_R.resize(m_R);
            std::stable_sort(eigen_L.begin(), eigen_L.end(), less_blkIdx);
            std::stable_sort(eigen_R.begin(), eigen_R.end(), less_blkIdx);

            #if defined(PETSC_USE_DEBUG)
            if(flg) {
                printf("\n\n");
                for(const Eigen_t& eig: eigen_L) printf(" L: %-16.10g seq: %-5lld eps: %-5lld blk: %-5lld\n",
                    eig.eigval, LLD(eig.seqIdx), LLD(eig.epsIdx), LLD(eig.blkIdx));
                printf("\n");
                for(const Eigen_t& eig: eigen_R) printf(" R: %-16.10g seq: %-5lld eps: %-5lld blk: %-5lld\n",
                    eig.eigval, LLD(eig.seqIdx), LLD(eig.epsIdx), LLD(eig.blkIdx));
                printf("\n\n");
            }
            #endif

            /*  Calculate the elements of the rotation matrices and the QN object */
            ierr = FillRotation_BlockDiag(eigen_L, eps_list_L, rdmd_vecs_L, KronBlocks.LeftBlockRef(),  RotMatT_L); CHKERRQ(ierr);
            ierr = FillRotation_BlockDiag(eigen_R, eps_list_R, rdmd_vecs_R, KronBlocks.RightBlockRef(), RotMatT_R); CHKERRQ(ierr);

            /*  Calculate the truncation error */
            TruncErr_L = 1.0;
            for(const Eigen_t &eig: eigen_L) TruncErr_L -= (eig.eigval > 0) * eig.eigval;
            TruncErr_R = 1.0;
            for(const Eigen_t &eig: eigen_R) TruncErr_R -= (eig.eigval > 0) * eig.eigval;

            /*  Calculate the quantum numbers lists */
            {
                std::map< PetscReal, PetscInt > BlockIdxs;
                for(const Eigen_t &eig: eigen_L) BlockIdxs[ eig.blkIdx ] += 1;
                for(const auto& idx: BlockIdxs) qn_list_L.push_back( KronBlocks.LeftBlockRef().Magnetization.List(idx.first) );
                for(const auto& idx: BlockIdxs) qn_size_L.push_back( idx.second );
                numBlocks_L = qn_list_L.size();
            }

            {
                std::map< PetscReal, PetscInt > BlockIdxs;
                for(const Eigen_t &eig: eigen_R) BlockIdxs[ eig.blkIdx ] += 1;
                for(const auto& idx: BlockIdxs) qn_list_R.push_back( KronBlocks.RightBlockRef().Magnetization.List(idx.first) );
                for(const auto& idx: BlockIdxs) qn_size_R.push_back( idx.second );
                numBlocks_R = qn_list_R.size();
            }

            #if defined(PETSC_USE_DEBUG)
            if(flg){
                for(PetscInt i = 0; i < numBlocks_L; ++i) printf("    %g  %lld\n", qn_list_L[i], LLD(qn_size_L[i]));
                printf("\n");
                for(PetscInt i = 0; i < numBlocks_R; ++i) printf("    %g  %lld\n", qn_list_R[i], LLD(qn_size_R[i]));
            }
            #endif
        }

        /*  Broadcast the truncation errors to all processes */
        ierr = MPI_Bcast(&TruncErr_L, 1, MPIU_SCALAR, 0, PETSC_COMM_WORLD); CHKERRQ(ierr);
        ierr = MPI_Bcast(&TruncErr_R, 1, MPIU_SCALAR, 0, PETSC_COMM_WORLD); CHKERRQ(ierr);

        /*  Broadcast the number of quantum blocks */
        ierr = MPI_Bcast(&numBlocks_L, 1, MPIU_INT, 0, PETSC_COMM_WORLD); CHKERRQ(ierr);
        ierr = MPI_Bcast(&numBlocks_R, 1, MPIU_INT, 0, PETSC_COMM_WORLD); CHKERRQ(ierr);

        /*  Broadcast the information on quantum number blocks */
        if(mpi_rank) qn_list_L.resize(numBlocks_L);
        if(mpi_rank) qn_size_L.resize(numBlocks_L);
        ierr = MPI_Bcast(qn_list_L.data(), numBlocks_L, MPIU_REAL, 0, PETSC_COMM_WORLD); CHKERRQ(ierr);
        ierr = MPI_Bcast(qn_size_L.data(), numBlocks_L, MPIU_INT, 0, PETSC_COMM_WORLD); CHKERRQ(ierr);
        if(mpi_rank) qn_list_R.resize(numBlocks_R);
        if(mpi_rank) qn_size_R.resize(numBlocks_R);
        ierr = MPI_Bcast(qn_list_R.data(), numBlocks_R, MPIU_REAL, 0, PETSC_COMM_WORLD); CHKERRQ(ierr);
        ierr = MPI_Bcast(qn_size_R.data(), numBlocks_R, MPIU_INT, 0, PETSC_COMM_WORLD); CHKERRQ(ierr);

        /*  Assemble the rotation matrix */
        ierr = MatAssemblyBegin(RotMatT_L, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyBegin(RotMatT_R, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(RotMatT_L, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(RotMatT_R, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

        #if defined(PETSC_USE_DEBUG)
        if(flg){
            ierr = MatPeek(RotMatT_L, "RotMatT_L"); CHKERRQ(ierr);
            ierr = MatPeek(RotMatT_R, "RotMatT_R"); CHKERRQ(ierr);
        }
        #endif

        ierr = BT_L.QN.Initialize(mpi_comm, qn_list_L, qn_size_L); CHKERRQ(ierr);
        ierr = BT_R.QN.Initialize(mpi_comm, qn_list_R, qn_size_R); CHKERRQ(ierr);

        for(EPS& eps: eps_list_L){
            ierr = EPSDestroy(&eps); CHKERRQ(ierr);
        }
        for(EPS& eps: eps_list_R){
            ierr = EPSDestroy(&eps); CHKERRQ(ierr);
        }
        // for(Mat& mat: rdmd_list_L){
        //     ierr = MatDestroy(&mat); CHKERRQ(ierr);
        // }
        // for(Mat& mat: rdmd_list_R){
        //     ierr = MatDestroy(&mat); CHKERRQ(ierr);
        // }
        for(Vec& vec: rdmd_vecs_L){
            ierr = VecDestroy(&vec); CHKERRQ(ierr);
        }
        for(Vec& vec: rdmd_vecs_R){
            ierr = VecDestroy(&vec); CHKERRQ(ierr);
        }
        ierr = VecScatterDestroy(&ctx); CHKERRQ(ierr);
        ierr = VecDestroy(&gsv_r_loc); CHKERRQ(ierr);

        return(0);
    }

    /** Obtain the eigenspectrum of a diagonal block of the reduced density matrix through an interface to a lapack routine */
    PetscErrorCode EigRDM_BlockDiag(
        const Mat& matin,                   /**< [in] diagonal block matrix */
        const PetscInt& seqIdx,             /**< [in] sequence index */
        const PetscInt& blkIdx,             /**< [in] block index */
        std::vector< Eigen_t >& eigList,    /**< [out] resulting list of eigenstates */
        EPS& eps                            /**< [out] eigensolver context */
        )
    {
        PetscErrorCode ierr;
        /*  Require that input matrix be square */
        PetscInt Nrows, Ncols;
        ierr = MatGetSize(matin, &Nrows, &Ncols); CHKERRQ(ierr);
        if(Nrows!=Ncols) SETERRQ2(PETSC_COMM_SELF,1,"Input must be square matrix. Got size %d x %d.", Nrows, Ncols);

        ierr = EPSCreate(PETSC_COMM_SELF, &eps); CHKERRQ(ierr);
        ierr = EPSSetOperators(eps, matin, nullptr); CHKERRQ(ierr);
        ierr = EPSSetProblemType(eps, EPS_HEP); CHKERRQ(ierr);
        ierr = EPSSetWhichEigenpairs(eps, EPS_LARGEST_REAL); CHKERRQ(ierr);
        ierr = EPSSetType(eps, EPSLAPACK);
        ierr = EPSSetTolerances(eps, 1.0e-16, PETSC_DEFAULT); CHKERRQ(ierr);
        ierr = EPSSolve(eps); CHKERRQ(ierr);

        /*  Verify convergence */
        PetscInt nconv;
        ierr = EPSGetConverged(eps, &nconv); CHKERRQ(ierr);
        if(nconv != Nrows) SETERRQ2(PETSC_COMM_SELF,1,"Incorrect number of converged eigenpairs. "
            "Expected %d. Got %d.", Nrows, nconv); CHKERRQ(ierr);

        /*  Get the converged eigenvalue */
        for(PetscInt epsIdx = 0; epsIdx < nconv; ++epsIdx)
        {
            PetscScalar eigr=0.0, eigi=0.0;
            ierr = EPSGetEigenvalue(eps, epsIdx, &eigr, &eigi); CHKERRQ(ierr);

            /*  Verify that the eigenvalue is real */
            if(eigi != 0.0) SETERRQ1(PETSC_COMM_SELF,1,"Imaginary part of eigenvalue must be zero. "
                "Got %g\n", eigi);

            eigList.push_back({ eigr, seqIdx, epsIdx, blkIdx });
        }
        return(0);
    }

    /** Fills the rotation matrix assumming that the reduced density matrix has a block diagonal structure */
    PetscErrorCode FillRotation_BlockDiag(
        const std::vector< Eigen_t >&   eigen_list,     /**< [in] full list of eigenstates */
        const std::vector< EPS >&       eps_list,       /**< [in] ordered list of EPS contexts */
        const std::vector< Vec >&       rdmd_vecs,      /**< [in] ordered list of corresponding eigenvector containers */
        const Block&                    BlockRef,       /**< [in] reference to the block object to get the magnetization */
        Mat&                            RotMatT         /**< [out] resulting rotation matrix */
        )
    {
        PetscErrorCode ierr;

        #if defined(PETSC_USE_COMPLEX)
            SETERRQ(mpi_comm,PETSC_ERR_SUP,"This function is only implemented for scalar-type=real.");
        #endif

        /*  Allocate space for column indices using the maximum size */
        std::vector< PetscInt> qnsize = BlockRef.Magnetization.Sizes();
        std::vector< PetscInt>::iterator it = std::max_element(qnsize.begin(), qnsize.end());
        PetscInt max_qnsize = PetscInt(*it);
        PetscInt *idx;
        ierr = PetscCalloc1(max_qnsize+1, &idx); CHKERRQ(ierr);

        PetscScalar eigr, eigi, *vals;
        PetscInt prev_blkIdx = -1;
        PetscInt startIdx = 0;
        PetscInt numStates = 0;
        PetscInt rowCtr = 0;
        for(const Eigen_t &eig: eigen_list)
        {
            /*  Retrieve the eigenpair from the Eigen_t object */
            const PetscInt seqIdx = eig.seqIdx;
            ierr = EPSGetEigenpair(eps_list[seqIdx], eig.epsIdx, &eigr, &eigi, rdmd_vecs[seqIdx], nullptr); CHKERRQ(ierr);
            ierr = VecGetArray(rdmd_vecs[seqIdx], &vals); CHKERRQ(ierr);
            /*  Verify that eigr is the same eigenvalue as eig.eigval */
            if(eigr != eig.eigval) SETERRQ2(PETSC_COMM_SELF,1,"Incorrect eigenvalue. Expected %g. Got %g.", eig.eigval, eigr);

            /*  Determine the block indices, updating only when the block index changes */
            if(prev_blkIdx != eig.blkIdx){
                startIdx = BlockRef.Magnetization.Offsets(eig.blkIdx); assert(startIdx!=-1);
                numStates = BlockRef.Magnetization.Sizes(eig.blkIdx); assert(numStates!=-1);
                for(PetscInt i = 0; i < numStates+1; ++i) idx[i] = startIdx + i;
                prev_blkIdx = eig.blkIdx;
            }

            /*  Set the value of the rotation matrix to the values of the eigenvector from the root process */
            ierr = MatSetValues(RotMatT, 1, &rowCtr, numStates, idx, vals, INSERT_VALUES); CHKERRQ(ierr);

            ierr = VecRestoreArray(rdmd_vecs[seqIdx], &vals); CHKERRQ(ierr);
            ++rowCtr;
        }
        ierr = PetscFree(idx); CHKERRQ(ierr);
        return(0);
    }

    /** Calculates the correlation functions. Must be called only at the end of a sweep since corresponding partitioning of the
        lattice sites are embedded in the generation of Measurement objects, and reflection symmetry is assumed by taking in only
        the system block. NOTE: May be generalized if needed. */
    PetscErrorCode CalculateCorrelations_BlockDiag(
        KronBlocks_t& KronBlocks, /**< [in] Kronblocks context of the superblock */
        const Vec& gsv_r,               /**< [in] Real part of the superblock ground state vector */
        const BasisTransformation& BT,  /**< [in] BasisTransformation context containing reduced density matrix */
        const PetscBool flg=PETSC_TRUE  /**< [in] Whether to do the measurements */
        )
    {
        if(!mpi_rank)
        {
            if(!corr_headers_printed)
            {
                fprintf(fp_corr, "{\n");
                fprintf(fp_corr, "  \"info\" :\n");
                fprintf(fp_corr, "  [\n");

                for(size_t icorr=0; icorr<measurements.size(); ++icorr){
                    if(icorr) fprintf(fp_corr, ",\n");
                    Correlator& c = measurements[icorr];
                    fprintf(fp_corr, "    {\n");
                    fprintf(fp_corr, "      \"corrIdx\" : %lld,\n", LLD(c.idx));
                    fprintf(fp_corr, "      \"name\"    : \"%s\",\n",  c.name.c_str());
                    fprintf(fp_corr, "      \"desc1\"   : \"%s\",\n", c.desc1.c_str());
                    fprintf(fp_corr, "      \"desc2\"   : \"%s\",\n", c.desc2.c_str());
                    fprintf(fp_corr, "      \"desc3\"   : \"%s\"\n",  c.desc3.c_str());
                    fprintf(fp_corr, "    }");
                }

                fprintf(fp_corr, "\n");
                fprintf(fp_corr, "  ],\n");
                fprintf(fp_corr, "  \"values\" :\n");
                fprintf(fp_corr, "  [\n");
                fflush(fp_corr);

                corr_headers_printed = PETSC_TRUE;
            }
        }

        if(!flg) return(0);
        PetscErrorCode ierr;
        std::vector< PetscScalar > CorrValues(measurements.size());

        PetscBool debug = PETSC_FALSE; /* FIXME: Remove later */
        if(debug && !mpi_rank) std::cout << "\n\n====" << __FUNCTION__ << "====" << std::endl;

        #if 1
        /* Explicitly build the operators in the Kronecker product space */
        std::vector< Correlator > CorrSysEnv = measurements;
        #else
        /* Separately handle the case of a correlator of operators living only on the system block */
        /*  Classify the correlators accordingly */
        std::vector< Correlator > CorrSys;     /* For operators residing in the system block */
        std::vector< Correlator > CorrSysEnv;  /* For operators residing in the system and environment blocks */

        for(const Correlator& m: measurements){
            if(m.SysOps.size()!=0 && m.EnvOps.size()==0){
                CorrSys.push_back(m);
            }
            else if (m.SysOps.size()==0 && m.EnvOps.size()!=0){
                SETERRQ(mpi_comm,1,"Reflection symmetry is imposed. Empty SysOps and non-empty EnvOps not permitted.");
            }
            else if (m.SysOps.size()!=0 && m.EnvOps.size()!=0){
                CorrSysEnv.push_back(m);
            }
            else {
                /* TODO: Skip correlators with no operators and simply return the  */
            }
        }

        /*---- For correlators in the system block ----*/
        /*  Send the elements of the reduced density matrix (diagonals) to their corresponding processors
            Since each processor has a complete copy of KronBlocks, each one is aware of how much data
            it will receive. */
        if(CorrSys.size() > 0)
        {

            std::vector< PetscInt > Sizes = KronBlocks.LeftBlockRef().Magnetization.Sizes();
            std::vector< PetscInt > Offsets = KronBlocks.LeftBlockRef().Magnetization.Offsets();
            PetscInt NumSectors = KronBlocks.LeftBlockRef().Magnetization.NumSectors();
            PetscInt NumStates = KronBlocks.LeftBlockRef().Magnetization.NumStates();

            /* Verify the sizes of the reduced density matrix diagonals */
            if(!mpi_rank){
                for(PetscInt i=0; i<NumSectors; ++i){
                    PetscInt M,N;
                    if(BT.rdmd_list.find(i)==BT.rdmd_list.end()) continue;
                    ierr = MatGetSize(BT.rdmd_list.at(i), &M, &N); CHKERRQ(ierr);
                    if(M!=N)
                        SETERRQ2(PETSC_COMM_SELF,1,"Matrix must be square. Got %D x %D instead.",M,N);
                    if(M!=Sizes.at(i))
                        SETERRQ2(mpi_comm,1,"Incorrect size. Expected %D. Got %D.", M, Sizes.at(i));
                }
            }

            /* Load only needed operator matrices manually */
            ierr = KronBlocks.LeftBlockRefMod().EnsureSaved(); CHKERRQ(ierr);

            /* TODO: Implement in a subcommunicator */
            MPI_Comm& sub_comm = mpi_comm;
            {
                PetscMPIInt sub_rank, sub_size;
                ierr = MPI_Comm_rank(sub_comm, &sub_rank); CHKERRQ(ierr);
                ierr = MPI_Comm_size(sub_comm, &sub_size); CHKERRQ(ierr);

                /* Prepare an MPI dense matrix for the reduced density matrix */
                Mat rho;
                ierr = MatCreateDense(sub_comm,PETSC_DECIDE,PETSC_DECIDE,NumStates,NumStates,NULL,&rho); CHKERRQ(ierr);
                ierr = MatSetOption(rho,MAT_ROW_ORIENTED,PETSC_TRUE); CHKERRQ(ierr);
                if(!mpi_rank)
                {
                    PetscScalar *v;
                    PetscInt *idx, max_size = 0;
                    for(const PetscInt& s: Sizes) max_size = (max_size < s) ? s : max_size;
                    ierr = PetscCalloc1(max_size,&idx); CHKERRQ(ierr);
                    for(PetscInt iblk=0; iblk<NumSectors; ++iblk)
                    {
                        /*  Manually take care of the possiblity that a blk was not included in the sectors used.
                            Although this is unlikely to happen when imposing symmetry and the target magnetization is zero */
                        if(BT.rdmd_list.find(iblk) == BT.rdmd_list.end()) continue;
                        PetscInt size = Sizes[iblk];
                        PetscInt shift = Offsets[iblk];
                        for(PetscInt is=0; is<size; ++is)
                        {
                            idx[is] = is + shift;
                        }
                        ierr = MatDenseGetArray(BT.rdmd_list.at(iblk), &v); CHKERRQ(ierr);
                        for(PetscInt is=0; is<size; ++is)
                        {
                            PetscInt row = is + shift;
                            ierr = MatSetValues(rho,1,&row,size,idx,v+is*size,INSERT_VALUES); CHKERRQ(ierr);
                        }
                        ierr = MatDenseRestoreArray(BT.rdmd_list.at(iblk), &v); CHKERRQ(ierr);
                    }
                    ierr = PetscFree(idx); CHKERRQ(ierr);
                }
                ierr = MatAssemblyBegin(rho, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
                ierr = MatAssemblyEnd(rho, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

                std::map< std::string, Mat > OpMats;
                std::vector< Mat > AllOpProds; /* Store only the created operator products to avoid duplication */
                std::vector< Mat > OpProds; /* Points to a matrix in AllOpProds or OpMats */

                ierr = CalculateOperatorProducts(sub_comm,CorrSys,BlockSys,
                    KronBlocks.LeftBlockRefMod(),OpMats,AllOpProds,OpProds); CHKERRQ(ierr);

                /* Allocate space to store the local trace of each correlator */
                PetscScalar *tr_rho;
                ierr = PetscCalloc1(CorrSys.size(),&tr_rho); CHKERRQ(ierr);

                /* Obtain the trace of the matrix products with the reduced density matrix */
                PetscInt ncols_rho, ncols_corr, rstart, rend;
                const PetscScalar *vals_rho, *vals_corr;
                const PetscInt *cols_corr;
                ierr = MatGetOwnershipRange(rho,&rstart,&rend); CHKERRQ(ierr);
                for(PetscInt irow=rstart; irow<rend; ++irow)
                {
                    ierr = MatGetRow(rho,irow,&ncols_rho,NULL,&vals_rho); CHKERRQ(ierr);
                    for(size_t icorr=0; icorr<CorrSys.size(); ++icorr)
                    {
                        ierr = MatGetRow(OpProds.at(icorr),irow,&ncols_corr,&cols_corr,&vals_corr); CHKERRQ(ierr);
                        PetscScalar y = 0;
                        for(PetscInt icol=0; icol<ncols_corr; ++icol)
                        {
                            y += vals_corr[icol]*vals_rho[cols_corr[icol]];
                        }
                        tr_rho[icorr] += y;
                        ierr = MatRestoreRow(OpProds.at(icorr),irow,&ncols_corr,&cols_corr,&vals_corr); CHKERRQ(ierr);
                    }
                    ierr = MatRestoreRow(rho,irow,&ncols_rho,NULL,&vals_rho); CHKERRQ(ierr);
                }

                /* Do an MPI_Allreduce to sum all values */
                ierr = MPI_Allreduce(MPI_IN_PLACE, tr_rho, PetscMPIInt(CorrSys.size()), MPIU_SCALAR, MPI_SUM, sub_comm); CHKERRQ(ierr);

                for(size_t icorr=0; icorr<CorrSys.size(); ++icorr)
                {
                    CorrValues[CorrSys[icorr].idx] = tr_rho[icorr];
                }

                ierr = PetscFree(tr_rho); CHKERRQ(ierr);
                for(Mat& op_prod: AllOpProds){
                    ierr = MatDestroy(&op_prod); CHKERRQ(ierr);
                }
                for(auto& op_mat: OpMats){
                    ierr = MatDestroy(&(op_mat.second)); CHKERRQ(ierr);
                }

                ierr = MatDestroy(&rho); CHKERRQ(ierr);
            }
        }
        #endif
        /* TODO: Also calculate CorrSys Quantities using the CorrSysEnv routine */

        /*---- For correlators in the system and environment block ----*/
        if(CorrSysEnv.size() > 0)
        {
            if(debug && !mpi_rank) std::cout << "\nCorrSysEnv" << std::endl;
            if(debug && !mpi_rank) for(const Correlator& m: CorrSysEnv) m.PrintInfo();

            /* Prepare products of the system and environment block operators */
            std::map< std::string, Mat > OpMats;
            std::vector< Mat > AllOpProds; /* Store only the created operator products to avoid duplication */
            std::vector< Mat > OpProdsSys, OpProdsEnv; /* Points to a matrix in AllOpProds or OpMats */

            ierr = KronBlocks.LeftBlockRefMod().EnsureSaved(); CHKERRQ(ierr);
            ierr = KronBlocks.RightBlockRefMod().EnsureSaved(); CHKERRQ(ierr);

            ierr = CalculateOperatorProducts(MPI_COMM_NULL, CorrSysEnv, BlockSys,
                KronBlocks.LeftBlockRefMod(), OpMats, AllOpProds, OpProdsSys); CHKERRQ(ierr);

            ierr = CalculateOperatorProducts(MPI_COMM_NULL, CorrSysEnv, BlockEnv,
                KronBlocks.RightBlockRefMod(), OpMats, AllOpProds, OpProdsEnv); CHKERRQ(ierr);

            for(size_t icorr=0; icorr < CorrSysEnv.size(); ++icorr)
            {
                /*  FIXME: Assume Sz-type inputs.
                    TODO: implement a guesser based on the constituent operator types in the correlator */

                /*  Prepare the kron shell matrix */
                Mat KronOp = NULL;
                Vec Op_Vec;
                PetscScalar Vec_Op_Vec;
                int OpTypeSys = OpSz;
                int OpTypeEnv = OpSz;
                for(const Op& op : CorrSysEnv[icorr].SysOps) OpTypeSys += int(op.OpType);
                for(const Op& op : CorrSysEnv[icorr].EnvOps) OpTypeEnv += int(op.OpType);
                ierr = KronBlocks.KronConstruct(OpProdsSys.at(icorr), Op_t(OpTypeSys),
                                                OpProdsEnv.at(icorr), Op_t(OpTypeEnv), KronOp); CHKERRQ(ierr);
                ierr = MatCreateVecs(KronOp, NULL, &Op_Vec); CHKERRQ(ierr);
                ierr = MatMult(KronOp, gsv_r, Op_Vec); CHKERRQ(ierr);
                ierr = VecDot(Op_Vec, gsv_r, &Vec_Op_Vec); CHKERRQ(ierr);
                ierr = MatDestroy_KronSumShell(&KronOp); CHKERRQ(ierr);
                ierr = MatDestroy(&KronOp); CHKERRQ(ierr);
                ierr = VecDestroy(&Op_Vec); CHKERRQ(ierr);

                CorrValues[CorrSysEnv[icorr].idx] = Vec_Op_Vec;
            }

            for(Mat& op_prod: AllOpProds){
                ierr = MatDestroy(&op_prod); CHKERRQ(ierr);
            }
            for(auto& op_mat: OpMats){
                ierr = MatDestroy(&(op_mat.second)); CHKERRQ(ierr);
            }
        }

        if(debug && !mpi_rank){
            std::cout << "\nValues" << std::endl;
            for(size_t icorr=0; icorr<measurements.size(); ++icorr){
                measurements[icorr].PrintInfo();
                std::cout << "     = " << CorrValues[icorr] << std::endl << std::endl;
            }
        }

        /* Print results to file */
        if(!mpi_rank)
        {
            if(!corr_printed_first)
            {
                corr_printed_first = PETSC_TRUE;
            }
            else
            {
                fprintf(fp_corr, ",\n");
            }
            fprintf(fp_corr, "    [");
            for(size_t icorr=0; icorr<measurements.size(); ++icorr){
                if(icorr) fprintf(fp_corr, ",");
                fprintf(fp_corr, " %g", CorrValues[icorr]);
            }
            fprintf(fp_corr, " ]");
            fflush(fp_corr);
        }

        if(debug && !mpi_rank) std::cout << "\n\n" << std::endl;
        return(0);
    }

    /** Calculates the products of operators that belong to a single block represented by the same basis. */
    PetscErrorCode CalculateOperatorProducts(
        const MPI_Comm& comm_in,                    /**< [in] communicator */
        const std::vector< Correlator >& Corr,      /**< [in] list of correlators */
        const Block_t& BlockType,                   /**< [in] block type (whether BlockSys or BlockEnv) */
        Block& BlockRef,                            /**< [in] reference to the block (non-const since RetrieveOperator
                                                              requires modification of the block object) */
        std::map< std::string, Mat >& OpMats,       /**< [out] maps operator key strings to corresponding matrices */
        std::vector< Mat >& AllOpProds,             /**< [out] list of all operator products generated and to be destroyed */
        std::vector< Mat >& OpProds                 /**< [out] list of operator products requested in Corr */
        )
    {
        PetscErrorCode ierr;
        MPI_Comm comm = (comm_in==MPI_COMM_NULL) ? mpi_comm : comm_in;

        for(size_t icorr=0; icorr<Corr.size(); ++icorr)
        {
            const Correlator& c = Corr.at(icorr);
            const std::vector< Op >& OpsList = (BlockType == BlockSys)?c.SysOps:c.EnvOps;
            for(const Op& o : OpsList)
            {
                std::string key = OpIdxToStr(o.OpType,o.idx);
                if(OpMats.find(key)==OpMats.end())
                {
                    OpMats[key] = NULL;
                    ierr = BlockRef.RetrieveOperator(
                        OpToStr(o.OpType), o.idx, OpMats[key], comm); CHKERRQ(ierr);
                }
            }
            if(OpsList.empty())
            {
                std::string key = "eye";
                if(OpMats.find(key)==OpMats.end())
                {
                    Mat eye = NULL;
                    PetscInt nstates = BlockRef.NumStates();
                    ierr = MatEyeCreate(comm, eye, nstates); CHKERRQ(ierr);
                    OpMats[key] = eye;
                }
            }
        }

        /* Prepare the operator products for each correlator */
        OpProds.resize(Corr.size());
        for(size_t icorr=0; icorr<Corr.size(); ++icorr)
        {
            const Correlator& c = Corr.at(icorr);
            const std::vector< Op >& OpsList = (BlockType == BlockSys)?c.SysOps:c.EnvOps;
            if(OpsList.size()==1)
            {
                /* Point to a matrix in OpMats */
                OpProds.at(icorr) = GetOpMats(OpMats,OpsList,0);
            }
            else if(OpsList.size()>1)
            {
                /* Generate the operator products */
                Mat Prod0=NULL, Prod1=NULL;
                /* Perform the first multiplication */
                ierr = MatMatMult(GetOpMats(OpMats,OpsList,0),GetOpMats(OpMats,OpsList,1),
                    MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Prod1); CHKERRQ(ierr);

                /* Iterate over a swap and multiply*/
                PetscInt nmults = OpsList.size()-1;
                for(PetscInt imult=1; imult<nmults; ++imult)
                {
                    ierr = MatDestroy(&Prod0); CHKERRQ(ierr);
                    Prod0 = Prod1;
                    Prod1 = NULL;
                    ierr = MatMatMult(Prod0,GetOpMats(OpMats,OpsList,imult+1),MAT_INITIAL_MATRIX,PETSC_DEFAULT,&Prod1); CHKERRQ(ierr);
                }
                ierr = MatDestroy(&Prod0); CHKERRQ(ierr);
                AllOpProds.push_back(Prod1);
                OpProds.at(icorr) = Prod1;
            }
            else if(OpsList.empty())
            {
                /* Populate with an identity instead of throwing an error */
                OpProds.at(icorr) = OpMats["eye"];
            }
            else
            {
                SETERRQ1(mpi_comm,1,"%s should be non-empty.",(BlockType == BlockSys)?"SysOps":"EnvOps");
            }
        }

        return(0);
    }

    /** Ensure that required blocks are loaded while unrequired blocks are saved */
    PetscErrorCode SysBlocksActive(const std::set< PetscInt >& SysIdx)
    {
        PetscErrorCode ierr;
        PetscInt sys_idx = 0;
        std::set< PetscInt >::iterator act_it;
        for(act_it = SysIdx.begin(); act_it != SysIdx.end(); ++act_it){
            for(PetscInt idx = sys_idx; idx < *act_it; ++idx){
                ierr = sys_blocks[idx].EnsureSaved(); CHKERRQ(ierr);
            }
            ierr = sys_blocks[*act_it].EnsureRetrieved(); CHKERRQ(ierr);
            sys_idx = *act_it+1;
        }
        for(PetscInt idx = sys_idx; idx < sys_ninit; ++idx){
            ierr = sys_blocks[idx].EnsureSaved(); CHKERRQ(ierr);
        }
        return(0);
    }

    /** Returns the path to the directory for the storage of a specific system block */
    std::string BlockDir(const std::string& BlockType, const PetscInt& iblock)
    {
        std::ostringstream oss;
        oss << scratch_dir << BlockType << "_" << std::setfill('0') << std::setw(9) << iblock;
        return oss.str();
    }

    /* Save headers for tabular step */
    PetscErrorCode SaveStepHeaders()
    {
        if(mpi_rank || !data_tabular) return(0);
        fprintf(fp_step,"{\n");
        fprintf(fp_step,"  \"headers\" : [");
        fprintf(fp_step,    "\"GlobIdx\", ");                                   /* 01 */
        fprintf(fp_step,    "\"LoopType\", ");                                  /* 02 */
        fprintf(fp_step,    "\"LoopIdx\", ");                                   /* 03 */
        fprintf(fp_step,    "\"StepIdx\", ");                                   /* 04 */
        fprintf(fp_step,    "\"NSites_Sys\", ");                                /* 05 */
        fprintf(fp_step,    "\"NSites_Env\", ");                                /* 06 */
        fprintf(fp_step,    "\"NSites_SysEnl\", ");                             /* 07 */
        fprintf(fp_step,    "\"NSites_EnvEnl\", ");                             /* 08 */
        fprintf(fp_step,    "\"NStates_Sys\", ");                               /* 09 */
        fprintf(fp_step,    "\"NStates_Env\", ");                               /* 10 */
        fprintf(fp_step,    "\"NStates_SysEnl\", ");                            /* 11 */
        fprintf(fp_step,    "\"NStates_EnvEnl\", ");                            /* 12 */
        fprintf(fp_step,    "\"NStates_SysRot\", ");                            /* 13 */
        fprintf(fp_step,    "\"NStates_EnvRot\", ");                            /* 14 */
        fprintf(fp_step,    "\"NumStates_H\", ");                               /* 15 */
        fprintf(fp_step,    "\"TruncErr_Sys\", ");                              /* 16 */
        fprintf(fp_step,    "\"TruncErr_Env\", ");                              /* 17 */
        fprintf(fp_step,    "\"GSEnergy\"");                                    /* 18 */
        fprintf(fp_step,"  ],\n");
        fprintf(fp_step,"  \"table\" : ");
        fflush(fp_step);
        return(0);
    }

    /** Save step data to file */
    PetscErrorCode SaveStepData(
        const StepData& data
        )
    {
        if(mpi_rank) return(0);
        fprintf(fp_step,"%s", GlobIdx ? ",\n" : "");
        if(data_tabular){
            fprintf(fp_step,"    [ ");
            fprintf(fp_step,"%lld, ",   LLD(GlobIdx));                          /* 01 */
            fprintf(fp_step,"%s, ",     LoopType ? "\"Sweep\"" : "\"Warmup\""); /* 02 */
            fprintf(fp_step,"%lld, ",   LLD(LoopIdx));                          /* 03 */
            fprintf(fp_step,"%lld, ",   LLD(StepIdx));                          /* 04 */
            fprintf(fp_step,"%lld, ",   LLD(data.NumSites_Sys));                /* 05 */
            fprintf(fp_step,"%lld, ",   LLD(data.NumSites_Env));                /* 06 */
            fprintf(fp_step,"%lld, ",   LLD(data.NumSites_SysEnl));             /* 07 */
            fprintf(fp_step,"%lld, ",   LLD(data.NumSites_EnvEnl));             /* 08 */
            fprintf(fp_step,"%lld, ",   LLD(data.NumStates_Sys));               /* 09 */
            fprintf(fp_step,"%lld, ",   LLD(data.NumStates_Env));               /* 10 */
            fprintf(fp_step,"%lld, ",   LLD(data.NumStates_SysEnl));            /* 11 */
            fprintf(fp_step,"%lld, ",   LLD(data.NumStates_EnvEnl));            /* 12 */
            fprintf(fp_step,"%lld, ",   LLD(data.NumStates_SysRot));            /* 13 */
            fprintf(fp_step,"%lld, ",   LLD(data.NumStates_EnvRot));            /* 14 */
            fprintf(fp_step,"%lld, ",   LLD(data.NumStates_H));                 /* 15 */
            fprintf(fp_step,"%.12g, ",  data.TruncErr_Sys);                     /* 16 */
            fprintf(fp_step,"%.12g, ",  data.TruncErr_Env);                     /* 17 */
            fprintf(fp_step,"%.12g",    data.GSEnergy);                         /* 18 */
            fprintf(fp_step,"]");
            fflush(fp_step);
            return(0);
        }
        fprintf(fp_step,"  {\n");
        fprintf(fp_step,"    \"GlobIdx\": %lld,\n",        LLD(GlobIdx));
        fprintf(fp_step,"    \"LoopType\": \"%s\",\n",     LoopType ? "Sweep" : "Warmup");
        fprintf(fp_step,"    \"LoopIdx\": %lld,\n",        LLD(LoopIdx));
        fprintf(fp_step,"    \"StepIdx\": %lld,\n",        LLD(StepIdx));
        fprintf(fp_step,"    \"NSites_Sys\": %lld,\n",     LLD(data.NumSites_Sys));
        fprintf(fp_step,"    \"NSites_Env\": %lld,\n",     LLD(data.NumSites_Env));
        fprintf(fp_step,"    \"NSites_SysEnl\": %lld,\n",  LLD(data.NumSites_SysEnl));
        fprintf(fp_step,"    \"NSites_EnvEnl\": %lld,\n",  LLD(data.NumSites_EnvEnl));
        fprintf(fp_step,"    \"NStates_Sys\": %lld,\n",    LLD(data.NumStates_Sys));
        fprintf(fp_step,"    \"NStates_Env\": %lld,\n",    LLD(data.NumStates_Env));
        fprintf(fp_step,"    \"NStates_SysEnl\": %lld,\n", LLD(data.NumStates_SysEnl));
        fprintf(fp_step,"    \"NStates_EnvEnl\": %lld,\n", LLD(data.NumStates_EnvEnl));
        fprintf(fp_step,"    \"NStates_SysRot\": %lld,\n", LLD(data.NumStates_SysRot));
        fprintf(fp_step,"    \"NStates_EnvRot\": %lld,\n", LLD(data.NumStates_EnvRot));
        fprintf(fp_step,"    \"TruncErr_Sys\": %.20g,\n",  data.TruncErr_Sys);
        fprintf(fp_step,"    \"TruncErr_Env\": %.20g,\n",  data.TruncErr_Env);
        fprintf(fp_step,"    \"GSEnergy\": %.20g\n",       data.GSEnergy);
        fprintf(fp_step,"  }");
        fflush(fp_step);
        return(0);
    }

    /* Save headers for tabular step */
    PetscErrorCode SaveTimingsHeaders()
    {
        if(mpi_rank || !data_tabular) return(0);
        fprintf(fp_timings,"{\n");
        fprintf(fp_timings,"  \"headers\" : [");
        fprintf(fp_timings,"\"GlobIdx\", ");
        fprintf(fp_timings,"\"Total\", ");
        fprintf(fp_timings,"\"Enlr\", ");
        fprintf(fp_timings,"\"Kron\", ");
        fprintf(fp_timings,"\"Diag\", ");
        fprintf(fp_timings,"\"Rdms\", ");
        fprintf(fp_timings,"\"Rotb\" ");
        fprintf(fp_timings,"],\n");
        fprintf(fp_timings,"  \"table\" : ");
        fflush(fp_timings);
        return(0);
    }

    /** Save timings data to file */
    PetscErrorCode SaveTimingsData(
        const TimingsData& data
        )
    {
        if(mpi_rank) return(0);
        fprintf(fp_timings,"%s", GlobIdx ? ",\n" : "");
        if(data_tabular){
            fprintf(fp_timings,"    [ ");
            fprintf(fp_timings,"%lld, ", LLD(GlobIdx));
            fprintf(fp_timings,"%.9g, ", data.Total);
            fprintf(fp_timings,"%.9g, ", data.tEnlr);
            fprintf(fp_timings,"%.9g, ", data.tKron);
            fprintf(fp_timings,"%.9g, ", data.tDiag);
            fprintf(fp_timings,"%.9g, ", data.tRdms);
            fprintf(fp_timings,"%.9g ",  data.tRotb);
            fprintf(fp_timings,"]");
            fflush(fp_timings);
            return(0);
        }
        fprintf(fp_timings,"  {\n");
        fprintf(fp_timings,"    \"Total\": %.9g,\n", data.Total);
        fprintf(fp_timings,"    \"Enlr\":  %.9g,\n", data.tEnlr);
        fprintf(fp_timings,"    \"Kron\":  %.9g,\n", data.tKron);
        fprintf(fp_timings,"    \"Diag\":  %.9g,\n", data.tDiag);
        fprintf(fp_timings,"    \"Rdms\":  %.9g,\n", data.tRdms);
        fprintf(fp_timings,"    \"Rotb\":  %.9g\n",  data.tRotb);
        fprintf(fp_timings,"  }");
        fflush(fp_timings);
        return(0);
    }

    /** Save the entanglement spectra to file */
    PetscErrorCode SaveEntanglementSpectra(
        const std::vector< Eigen_t >& eigen_L,
        const std::vector< PetscReal >& qn_L,
        const std::vector< Eigen_t >& eigen_R,
        const std::vector< PetscReal >& qn_R
        )
    {
        if(mpi_rank) return(0);
        fprintf(fp_entanglement, "%s", GlobIdx ? ",\n" : "");
        fprintf(fp_entanglement, "  {\n");
        fprintf(fp_entanglement, "    \"GlobIdx\": %lld,\n", LLD(GlobIdx));
        fprintf(fp_entanglement, "    \"Sys\": [\n");
        {
            PetscInt idx_prev = 999999999;
            for(const Eigen_t &eig: eigen_L){
                if(idx_prev!=eig.blkIdx){
                    if(idx_prev != 999999999) fprintf(fp_entanglement," ]},\n");
                    fprintf(fp_entanglement,"      {");
                    fprintf(fp_entanglement,"\"sector\": %g, \"vals\": [ %g",qn_L[eig.blkIdx], eig.eigval);
                } else {
                    fprintf(fp_entanglement,", %g", eig.eigval);
                }
                idx_prev = eig.blkIdx;
            }
            fprintf(fp_entanglement," ]}\n");
        }
        fprintf(fp_entanglement,"    ],\n");
        fprintf(fp_entanglement,"    \"Env\": [\n");
        {
            PetscInt idx_prev = 999999999;
            for(const Eigen_t &eig: eigen_R){
                if(idx_prev!=eig.blkIdx){
                    if(idx_prev != 999999999) fprintf(fp_entanglement," ]},\n");
                    fprintf(fp_entanglement,"      {");
                    fprintf(fp_entanglement,"\"sector\": %g, \"vals\": [ %g",qn_R[eig.blkIdx], eig.eigval);
                } else {
                    fprintf(fp_entanglement,", %g", eig.eigval);
                }
                idx_prev = eig.blkIdx;
            }
            fprintf(fp_entanglement," ]}\n");
        }
        fprintf(fp_entanglement,"    ]\n");
        fprintf(fp_entanglement,"  }");
        fflush(fp_entanglement);
        return(0);
    }

    PetscErrorCode SaveLoopsData()
    {
        if(mpi_rank) return(0);

        fprintf(fp_data,"  \"Warmup\": {\n");
        fprintf(fp_data,"    \"MStates\": %lld\n", LLD(mwarmup));
        fprintf(fp_data,"  },\n");
        fprintf(fp_data,"  \"Sweeps\": {\n");
        fprintf(fp_data,"    \"MStates\": [");

        PetscInt nsweeps = sweeps_mstates.size();
        if(nsweeps>0) fprintf(fp_data," %lld", LLD(sweeps_mstates[0]));
        for(PetscInt i=1;i<nsweeps;++i) fprintf(fp_data,", %lld", LLD(sweeps_mstates[i]));

        fprintf(fp_data," ]\n");
        fprintf(fp_data,"  }");
        fflush(fp_data);
        return(0);
    }

};

#undef OpIdxToStr
#undef GetOpMats

/**
    @}
 */

#endif // __DMRG_BLOCK_HPP__
