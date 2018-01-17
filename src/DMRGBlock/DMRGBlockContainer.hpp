#ifndef __DMRG_BLOCK_HPP__
#define __DMRG_BLOCK_HPP__

#include <slepceps.h>
#include <petscmat.h>
#include <vector>
#include <map>

#include "DMRGBlock.hpp"
#include "linalg_tools.hpp"


class Heisenberg_SpinOneHalf_SquareLattice
{

private:

    /*------ MPI and Misc ------*/

    /** MPI Communicator */
    MPI_Comm    mpi_comm = PETSC_COMM_WORLD;

    /** MPI rank in mpi_comm */
    PetscMPIInt mpi_rank;

    /** MPI size of mpi_comm */
    PetscMPIInt mpi_size;

    /** Tells whether to printout info during certain function calls */
    PetscBool   verbose = PETSC_FALSE;

    PetscBool   initialized = PETSC_FALSE;

    /*------ Coupling Constants ------*/

    /** Coupling strength for nearest-neighbor interactions */
    PetscScalar J1 = 1.0;

    /** Coupling strength for next-nearest-neighbor interactions */
    PetscScalar J2 = 1.0;

    /*------ DMRG Attributes ------*/

    /** Target maximum number of states after truncation */
    PetscInt    mstates = 20;


    /*------ Geometry ------*/

    /** Length along (growing) longitudinal direction */
    PetscInt    Lx = 4;

    /** Length along (fixed) transverse direction */
    PetscInt    Ly = 3;

    /** Total number of sites */
    PetscInt    num_sites;

    /*------ Blocks ------*/

    /** Number of system blocks to store, usually Lx*Ly-1 */
    PetscInt    num_blocks;

    /** Array of system blocks each of which will be kept
        all throughout the simulation */
    Block_SpinOneHalf *sys_blocks;

    /** Number of initialized blocks in SysBlocks */
    PetscInt    sys_blocks_num_init = 0;

    /** Environment block to be used only during warmup */
    Block_SpinOneHalf env_block;

    /** Static block containing single site operators for reference */
    Block_SpinOneHalf SingleSite;

    /** Constant reference to added site */
    const Block_SpinOneHalf& AddSite = SingleSite;

public:

    /** Initializes the container object with one site */
    PetscErrorCode Initialize();

    /** Performs one single DMRG step with Sys and Env */
    PetscErrorCode SingleDMRGStep(
        const Block_SpinOneHalf& Sys,
        const Block_SpinOneHalf& Env,
        Block_SpinOneHalf& SysOut);

    /** Adds one site to BlockIn producing BlockOut */
    PetscErrorCode EnlargeBlock(
        const Block_SpinOneHalf& BlockIn,
        const Side_t& AddSide,
        Block_SpinOneHalf& BlockOut);

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

#endif // __DMRG_BLOCK_HPP__
