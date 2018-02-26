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

#if defined(PETSC_USE_DEBUG)
#include <iostream>
#endif

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

        /*  Initialize Hamiltonian object */
        ierr = Ham.SetFromOptions(); assert(!ierr);

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

    #define PrintBlocks(LEFT,RIGHT) printf(" [%d]-* *-[%d]\n",(LEFT),(RIGHT))

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

        if(!mpi_rank && verbose) printf("WARMUP\n");

        /* Continuously enlarge the system block until it reaches half the total system size */

        while(sys_ninit < num_sites/2)
        {
            if(!mpi_rank && verbose) PrintBlocks(sys_ninit,sys_ninit);

            Block env_temp;
            ierr = SingleDMRGStep(
                sys_blocks[sys_ninit-1],  env_blocks[0], MStates,
                sys_blocks[sys_ninit], env_temp); CHKERRQ(ierr);
            ierr = env_blocks[0].Destroy(); CHKERRQ(ierr);
            env_blocks[0] = env_temp;

            ++sys_ninit;
        }
        warmed_up = PETSC_TRUE;
        if(sys_ninit != num_sites/2)
            SETERRQ2(mpi_comm,1,"Expected sys_ninit = num_sites/2 = %d. Got %d.",num_sites/2, sys_ninit);
        /* Destroy environment block */
        ierr = env_blocks[0].Destroy(); CHKERRQ(ierr);
        env_ninit = 0;

        if(verbose) PetscPrintf(mpi_comm, "Initialized system blocks: %d\n"
            "Total number of sites: %d\n\n", sys_ninit, num_sites);

        return ierr;
    }

    PetscErrorCode Sweep(
        const PetscInt& MStates,
        const PetscInt& MinBlock = PETSC_DEFAULT
        )
    {
        PetscErrorCode ierr;
        if(!warmed_up) SETERRQ(mpi_comm,1,"Warmup must be called first before performing sweeps.");

        /*  TODO: Set a minimum number of blocks (min_block). Decide whether to set it statically or let
            the number correspond to the least number of sites needed to exactly build MStates. */
        PetscInt min_block = MinBlock==PETSC_DEFAULT ? 1 : MinBlock;
        if(min_block < 1) SETERRQ1(mpi_comm,1,"MinBlock must at least be 1. Got %d.", min_block);

        if(!mpi_rank && verbose) printf("SWEEP MStates=%d\n", MStates);

        /*  Starting from the midpoint, perform a center to right sweep */
        for(PetscInt iblock = num_sites/2; iblock < num_sites - min_block - 2; ++iblock)
        {
            const PetscInt  insys  = iblock-1,   inenv  = num_sites - iblock - 3;
            const PetscInt  outsys = iblock,     outenv = num_sites - iblock - 2;
            if(!mpi_rank && verbose) PrintBlocks(insys+1,inenv+1);
            ierr = SingleDMRGStep(sys_blocks[insys],  sys_blocks[inenv], MStates,
                                    sys_blocks[outsys], sys_blocks[outenv]); CHKERRQ(ierr);
        }

        /*  Since we ASSUME REFLECTION SYMMETRY, the remainder of the sweep can be done as follows:
            Starting from the right-most min_block, perform a right to left sweep up to the MIDPOINT */
        for(PetscInt iblock = min_block; iblock < num_sites/2; ++iblock)
        {
            const PetscInt  insys  = num_sites - iblock - 3,    inenv  = iblock-1;
            const PetscInt  outsys = num_sites - iblock - 2,    outenv = iblock;
            if(!mpi_rank && verbose) PrintBlocks(insys+1,inenv+1);
            ierr = SingleDMRGStep(sys_blocks[insys],  sys_blocks[inenv], MStates,
                                    sys_blocks[outsys], sys_blocks[outenv]); CHKERRQ(ierr);
        }

        /*  FIXME: Does this count as a complete sweep? */

        /*  NOTE: If we do not assume REFLECTION SYMMETRY, then the sweeps should be center to right,
            right to left, then left to center */

        return(0);
    };


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

        /* Set the QN sectors as an option */
        KronBlocks_t KronBlocks(SysBlockEnl, EnvBlockEnl, {0});

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

        ierr = KronBlocks.KronSumConstruct(Terms, H); CHKERRQ(ierr);

        #if defined(PETSC_USE_DEBUG)
        {
            PetscBool flg = PETSC_FALSE;
            ierr = PetscOptionsGetBool(NULL,NULL,"-print_H",&flg,NULL); CHKERRQ(ierr);
            if(flg){ ierr = MatPeek(H,"H"); CHKERRQ(ierr); }
            flg = PETSC_FALSE;
            ierr = PetscOptionsGetBool(NULL,NULL,"-print_H_terms",&flg,NULL); CHKERRQ(ierr);
            if(flg){
                if(!mpi_rank) printf(" H(%d)\n", NumSitesTotal);
                for(const Hamiltonians::Term& term: Terms)
                {
                    if(!mpi_rank) printf("%.2f %2s(%2d) %2s(%2d)\n", term.a, (OpString.find(term.Iop)->second).c_str(), term.Isite,
                        (OpString.find(term.Jop)->second).c_str(), term.Jsite );
                }
            }
        }
        #endif

        /* Solve for the ground state */

        #if defined(PETSC_USE_COMPLEX)
            SETERRQ(mpi_comm,PETSC_ERR_SUP,"This function is only implemented for scalar-type=real.");
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
            ierr = EPSSetFromOptions(eps); CHKERRQ(ierr);
            ierr = EPSSolve(eps); CHKERRQ(ierr);
            ierr = EPSGetEigenpair(eps, 0, &gse_r, &gse_i, gsv_r, gsv_i); CHKERRQ(ierr);
            ierr = EPSDestroy(&eps); CHKERRQ(ierr);
        }
        ierr = MatDestroy(&H); CHKERRQ(ierr);

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

        /* Calculate the reduced density matrices */


        ierr = VecDestroy(&gsv_r); CHKERRQ(ierr);
        ierr = VecDestroy(&gsv_i); CHKERRQ(ierr);
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
