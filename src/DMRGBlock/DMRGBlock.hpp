#ifndef __DMRG_SITE_HPP
#define __DMRG_SITE_HPP

#include <petscmat.h>
#include "kron.hpp"
#include "linalg_tools.hpp"

/** Defines the offset of each operator */
typedef enum {
    OpSm=-1,
    OpSz=0,
    OpSp=+1
} Op_t;


class Block_SpinOneHalf
{

private:

    /*------ Basic attributes ------*/

    /** MPI Communicator */
    MPI_Comm        mpi_comm;

    /** MPI rank in mpi_comm */
    PetscMPIInt     mpi_rank;

    /** MPI size of mpi_comm */
    PetscMPIInt     mpi_size;

    /** Local dimension of a single site */
    const PetscInt loc_dim = 2;

    /** Sz sectors of a single site */
    const std::vector<PetscScalar> loc_qn_list = {+0.5, -0.5};

    /** Number of states in each sector in a single site */
    const std::vector<PetscInt> loc_qn_size = {1, 1};

    /** Tells whether the block was initialized */
    PetscBool init = PETSC_FALSE;

    /** Tells whether to printout info during certain function calls */
    PetscBool verbose = PETSC_FALSE;

    /** Number of sites in the block */
    PetscInt num_sites;

    /** Number of basis states in the Hilbert space */
    PetscInt num_states;

    /** Tells whether the Sm matrices have been initialized */
    PetscBool init_Sm = PETSC_FALSE;

public:
    /*------ Magnetization Sectors ------*/

    /** Number of Sz sectors in the Hilbert space */
    PetscInt num_sectors;

    /** List of Sz quantum numbers */
    std::vector<PetscReal> qn_list;

    /** Offset for each quantum number block */
    std::vector<PetscInt> qn_offset;

    /** Number of states in each quantum number block */
    std::vector<PetscInt> qn_size;


    /*------ Misc Functions ------*/

    /** Determines whether the operator arrays have been successfully filled with matrices */
    PetscErrorCode CheckOperatorArray(Mat *Op, const char* label) const;

    /** Indicates whether block has been initialized before us */
    PetscBool Initialized() const { return init; }

    /*------ Operator Matrices ------*/

    /** Matrix representation of the Hamiltonian operator */
    Mat     H = nullptr;

    /** List of $S^z$ operators */
    Mat*    Sz = nullptr;

    /** List of $S^+$ operators */
    Mat*    Sp = nullptr;

    /** List of $S^-$ operators */
    Mat*    Sm = nullptr;

    /** Initialize block object with input attributes and array of matrix operators */
    PetscErrorCode Initialize(const MPI_Comm& comm_in, PetscInt num_sites_in, PetscInt num_states_in);

    /** Checks whether all operators have been initialized and have correct dimensions */
    PetscErrorCode CheckOperators() const;

    /** Checks whether sector indexing was done properly */
    PetscErrorCode CheckSectors() const;

    /** Checks whether blocks follow the correct sector indices */
    PetscErrorCode CheckOperatorBlocks() const; /* TODO implementation */

    /** Extracts the block structure for each operator */
    PetscErrorCode GetOperatorBlocks(Op_t Operator); /* TODO implementation */

    /** Creates the Sm matrices on the fly */
    PetscErrorCode CreateSm();

    /** Destroys the Sm matrices on the fly */
    PetscErrorCode DestroySm();

    /** Destroys all operator matrices and frees memory */
    PetscErrorCode Destroy();

};

#endif // __DMRG_SITE_HPP
