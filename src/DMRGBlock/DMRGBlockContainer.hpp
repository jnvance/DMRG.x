#ifndef __DMRG_BLOCK_HPP__
#define __DMRG_BLOCK_HPP__

/**
    @defgroup   DMRGBlockContainer   DMRGBlockContainer
    @brief      Implementation of the DMRGBlockContainer class
    @addtogroup DMRGBlockContainer
    @{ */

#include <slepceps.h>
#include <petscmat.h>
#include <vector>
#include <map>

#include "DMRGKron.hpp"

/** Provides an alias of Side_t to follow the Sys-Env convention */
typedef enum
{
    BlockSys = 0,
    BlockEnv = 1
}
Block_t;

/** Contains and manipulates the system and environment blocks used in a single DMRG run */
template<class Block, class Hamiltonian> class DMRGBlockContainer
{

public:

    /** Initializes the container object with blocks of one site on each of the system and environment */
    DMRGBlockContainer(const MPI_Comm& mpi_comm): mpi_comm(mpi_comm)
    {
        PetscInt ierr = 0;

        /*  Get MPI attributes */
        ierr = MPI_Comm_size(mpi_comm, &mpi_size); assert(!ierr);
        ierr = MPI_Comm_rank(mpi_comm, &mpi_rank); assert(!ierr);

        /*  Initialize SingleSite which is used as added site */
        ierr = SingleSite.Initialize(mpi_comm, 1, PETSC_DEFAULT); assert(!ierr);

        num_sites = Ham.NumSites();
        num_sys_blocks = num_sites - 1;

        if((num_sites) < 2) throw std::runtime_error("There must be at least two total sites.");
        if((num_sites) % 2)  throw std::runtime_error("Total number of sites must be even.");

        sys_blocks.resize(num_sys_blocks);
        env_blocks.resize(num_env_blocks);

        /*  Initialize the 0th system block and the environment with one site each  */
        ierr = sys_blocks[sys_ninit++].Initialize(mpi_comm, 1, PETSC_DEFAULT); assert(!ierr);
        ierr = env_blocks[env_ninit++].Initialize(mpi_comm, 1, PETSC_DEFAULT); assert(!ierr);

        /*  Get some info from command line */
        ierr = PetscOptionsGetBool(NULL,NULL,"-verbose",&verbose,NULL); assert(!ierr);
        ierr = SetFromOptions(); assert(!ierr);
    }

    /** Destroys all created blocks */
    ~DMRGBlockContainer()
    {
        PetscInt ierr = 0;
        ierr = SingleSite.Destroy(); assert(!ierr);
        for(Block blk: sys_blocks) { ierr = blk.Destroy(); assert(!ierr); }
        for(Block blk: env_blocks) { ierr = blk.Destroy(); assert(!ierr); }
    }

    /** Get parameters from command line options */
    PetscErrorCode SetFromOptions()
    {
        PetscErrorCode ierr;
        ierr = Ham.SetFromOptions(); CHKERRQ(ierr);
        return(0);
    }

    /** Performs the warmup stage of DMRG.
        The system and environment blocks are grown until both reach the maximum number which is half the total number
        of sites. All created system blocks and only the last environment block are stored all of which will
        be represented by `MStates` number of basis states */
    PetscErrorCode Warmup(
        const PetscInt& MStates /**< [in] the maximum number of states to keep after each truncation */
        )
    {
        PetscErrorCode ierr = 0;

        if(warmed_up) SETERRQ(mpi_comm,1,"Warmup has already been called, and it can only be called once.");
        /* Continuously enlarge the system block until it reaches half the total system size */
        while(sys_ninit < num_sites/2)
        {
            if(!mpi_rank && verbose) printf("sys_ninit = %d  env_ninit = %d\n", sys_ninit, env_ninit);

            Block env_temp;
            ierr = SingleDMRGStep( sys_blocks[sys_ninit-1], env_blocks[0], MStates,
                sys_blocks[sys_ninit], env_temp); CHKERRQ(ierr);
            ierr = env_blocks[0].Destroy(); CHKERRQ(ierr);
            env_blocks[0] = env_temp;

            ++sys_ninit;
        }
        warmed_up = PETSC_TRUE;

        if(!mpi_rank && verbose) printf("sys_ninit = %d   num_sites = %d\n", sys_ninit, num_sites);

        return ierr;
    }

    /** Destroys the container object */
    PetscErrorCode Destroy();

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

private:

    /** MPI Communicator */
    MPI_Comm    mpi_comm = PETSC_COMM_SELF;

    /** MPI rank in mpi_comm */
    PetscMPIInt mpi_rank;

    /** MPI size of mpi_comm */
    PetscMPIInt mpi_size;

    /** Tells whether to printout info during certain function calls */
    PetscBool   verbose = PETSC_FALSE;

    /** Tells whether the object was initialized using Initialize() */
    PetscBool   warmed_up = PETSC_FALSE;

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

    /** Adds one site to BlockIn producing BlockOut */
    PetscErrorCode EnlargeBlock(
        const Block_t& BlockType,   /**< [in] the source block type (`BlockSys` or `BlockEnv`) */
        const PetscInt& BlockIdx,   /**< [in] the index of the source block */
        Block& BlockOut             /**< [out] the output enlarged block */
        );

    PetscErrorCode SingleDMRGStep(
        Block& SysBlock,            /**< [in] the old system (left) block */
        Block& EnvBlock,            /**< [in] the old environment (right) block */
        const PetscInt& MStates,    /**< [in] the maximum number of states to keep */
        Block& SysBlockOut,         /**< [out] the new system (left) block */
        Block& EnvBlockOut          /**< [out] the new environment (right) block */
        )
    {
        PetscErrorCode ierr;

        /* Check whether the system and environment blocks are the same */
        Mat H = nullptr; /* Hamiltonian matrix */
        const PetscBool flg = PetscBool(&SysBlock==&EnvBlock);

        if(!mpi_rank && verbose) printf("flg=%s\n", flg?"TRUE":"FALSE");

        /* (Block) Add one site to each block */
        Block SysBlockEnl, EnvBlockEnl;
        ierr = KronEye_Explicit(SysBlock, AddSite, SysBlockEnl); CHKERRQ(ierr);
        if(!flg){
            ierr = KronEye_Explicit(EnvBlock, AddSite, EnvBlockEnl); CHKERRQ(ierr);
        } else {
            EnvBlockEnl = SysBlockEnl;
        }

        /* Prepare the Hamiltonian taking both enlarged blocks together */
        PetscInt NumSitesTotal = SysBlockEnl.NumSites() + EnvBlockEnl.NumSites();
        const std::vector< Hamiltonians::Term > Terms = Ham.H(NumSitesTotal);
        KronBlocks_t KronBlocks(SysBlockEnl, EnvBlockEnl, {0});
        ierr = KronBlocks.KronSumConstruct(Terms, H); CHKERRQ(ierr);




#if 1
        if(!mpi_rank) printf(" H(%d)\n", NumSitesTotal);
        for(const Hamiltonians::Term& term: Terms)
        {
            if(!mpi_rank) printf("%.2f %2s(%2d) %2s(%2d)\n", term.a, (OpString.find(term.Iop)->second).c_str(), term.Isite,
                (OpString.find(term.Jop)->second).c_str(), term.Jsite );
        }
#endif


        /* Solve for the ground state */

        ierr = MatDestroy(&H); CHKERRQ(ierr);
        /* Calculate the reduced density matrices */


        /* (Block) Get the eigendecomposition of the reduced density matrices */


        /* (Block) Sort the eigenvalues and initialize the quantum number list and sizes */


        /* (Block) Calculate the rotation matrices */


        /* (Block) Initialize the new blocks
            copy enlarged blocks to out blocks but overwrite the matrices */
        ierr = SysBlockOut.Destroy(); CHKERRQ(ierr);
        ierr = EnvBlockOut.Destroy(); CHKERRQ(ierr);

        SysBlockOut = SysBlockEnl;
        if(!flg){
            EnvBlockOut = EnvBlockEnl;
        }

        return(0);
    }

};

/**
    @}
 */

#endif // __DMRG_BLOCK_HPP__
