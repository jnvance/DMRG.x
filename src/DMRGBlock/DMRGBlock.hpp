#ifndef __DMRG_SITE_HPP
#define __DMRG_SITE_HPP

#include <petscmat.h>
#include "kron.hpp"
#include "linalg_tools.hpp"

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
    PetscErrorCode CheckOperatorArray(Mat *Op, const char* label);

public:

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
    PetscErrorCode CheckOperators();

    /** Checks whether sector indexing was done properly */
    // PetscErrorCode CheckSectors();

    /** Creates the Sm matrices on the fly */
    PetscErrorCode CreateSm();

    /** Destroys the Sm matrices on the fly */
    PetscErrorCode DestroySm();

    /** Destroys all operator matrices and frees memory */
    PetscErrorCode Destroy();

};

#endif // __DMRG_SITE_HPP
