#ifndef __DMRG_BLOCK_HPP__
#define __DMRG_BLOCK_HPP__

/**
    @defgroup   DMRGBlockContainer   DMRGBlockContainer
    @brief      Implementation of the J1J2_SpinOneHalf_SquareLattice class
    @addtogroup DMRGBlockContainer
    @{ */

#include <slepceps.h>
#include <petscmat.h>
#include <vector>
#include <map>

#include "DMRGBlock.hpp"
#include "linalg_tools.hpp"

/** Contains and manipulates the system and environment blocks used in a single DMRG run
    and deals specifically with a J1-J2 Hamiltonian on a square lattice

    @remarks __TODO:__ Insert some brief description of the J1-J2 Hamiltonian and some use cases of this code
 */
class J1J2_SpinOneHalf_SquareLattice
{

private:

    /** MPI Communicator */
    MPI_Comm    mpi_comm = PETSC_COMM_WORLD;

    /** MPI rank in mpi_comm */
    PetscMPIInt mpi_rank;

    /** MPI size of mpi_comm */
    PetscMPIInt mpi_size;

    /** Tells whether to printout info during certain function calls */
    PetscBool   verbose = PETSC_FALSE;

    /** Tells whether the object was initialized using Initialize() */
    PetscBool   initialized = PETSC_FALSE;

    /** Coupling strength for nearest-neighbor interactions */
    PetscScalar J1 = 1.0;

    /** Coupling strength for next-nearest-neighbor interactions */
    PetscScalar J2 = 1.0;

    /** Target maximum number of states after truncation */
    PetscInt    mstates = 20;

    /** Length along (growing) longitudinal direction */
    PetscInt    Lx = 4;

    /** Length along (fixed) transverse direction */
    PetscInt    Ly = 3;

    /** Total number of sites */
    PetscInt    num_sites;

    /** Number of system blocks to store, usually Lx*Ly-1 */
    PetscInt    num_blocks;

    /** Array of system blocks each of which will be kept
        all throughout the simulation */
    std::vector< Block_SpinOneHalf > sys_blocks;

    /** Number of initialized blocks in SysBlocks */
    PetscInt    sys_blocks_num_init = 0;

    /** Environment block to be used only during warmup */
    Block_SpinOneHalf env_block;

    /** Static block containing single site operators for reference */
    Block_SpinOneHalf SingleSite;

    /** Constant reference to the single site that is added to each block
        during the EnlargeBlock() procedure */
    const Block_SpinOneHalf& AddSite = SingleSite;

public:

    /** Initializes the container object with one site */
    PetscErrorCode Initialize();

    /** Performs one single DMRG step with Sys and Env.
        The new system and environment blocks will each have one site added based on ::AddSite
     */
    PetscErrorCode SingleDMRGStep(
        const Block_SpinOneHalf& Sys,   /**< [in] the old system (left) block */
        const Block_SpinOneHalf& Env,   /**< [in] the old environment (right) block */
        Block_SpinOneHalf& SysOut,      /**< [out] the new system (left) block */
        Block_SpinOneHalf& EnvOut       /**< [out] the new environment (right) block */
        );

    /** Adds one site to BlockIn producing BlockOut */
    PetscErrorCode EnlargeBlock(
        const Block_SpinOneHalf& BlockIn,   /**< [in] input block */
        const Side_t& AddSide,              /**< [in] side on which to add a site (SideLeft or SideRight) */
        Block_SpinOneHalf& BlockOut         /**< [in] output block */
        );

    /** Destroys the container object */
    PetscErrorCode Destroy();

    /** Accesses a specified system block */
    const Block_SpinOneHalf& SysBlock(const PetscInt& iblock) const
    {
        assert(initialized);
        assert(iblock < sys_blocks_num_init);
        assert(sys_blocks[iblock].Initialized());
        return sys_blocks[iblock];
    }

    /** Accesses the environment block */
    const Block_SpinOneHalf& EnvBlock() const
    {
        assert(initialized);
        assert(env_block.Initialized());
        return env_block;
    }

};

/**
    @}
 */

#endif // __DMRG_BLOCK_HPP__
