#ifndef __DMRG_SITE_HPP
#define __DMRG_SITE_HPP

/**
    @defgroup   DMRGBlock   Block
    @brief      Implementation of the Block_SpinOneHalf class which contains the data and methods
                for a block of spin sites
    @addtogroup DMRGBlock
    @{ */

#include <petscmat.h>
#include "QuantumNumbers.hpp"
#include "kron.hpp"
#include "linalg_tools.hpp"

/** Identifies the three possible spin operators and also represents the shift associated
    to its action on the quantum number blocks */
typedef enum
{
    OpSm = -1,  /**< \f$ S^- \f$ operator */
    OpSz = 0,   /**< \f$ S^z \f$ operator */
    OpSp = +1   /**< \f$ S^+ \f$ operator */
} Op_t;

/** Identifies the sides of the DMRG block */
typedef enum
{
    SideLeft = 0,   /**< left block */
    SideRight = 1   /**< right block */
} Side_t;

/** Contains the matrix representations of the operators of a block of spin-1/2 sites, the associated
    magnetization sectors and some useful information and checking functions */
class Block_SpinOneHalf
{

private:

    /** MPI Communicator */
    MPI_Comm        mpi_comm;

    /** MPI rank in mpi_comm */
    PetscMPIInt     mpi_rank;

    /** MPI size of mpi_comm */
    PetscMPIInt     mpi_size;

    /** Local dimension of a single site.
        @remarks __NOTE:__ Default for spin-1/2 */
    const PetscInt loc_dim = 2;

    /** Sz sectors of a single site.
        @remarks __NOTE:__ Default for spin-1/2 */
    const std::vector<PetscScalar> loc_qn_list = {+0.5, -0.5};

    /** Number of states in each sector in a single site.
        @remarks __NOTE:__ Default for spin-1/2 */
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

    /** Initializes block object with input attributes and array of matrix operators.
        @post Arrays of operator matrices are initialized to the correct number of sites and states.
        @remarks __TODO:__ Consider interfacing this to the object constructor.
    */
    PetscErrorCode Initialize(
        const MPI_Comm& comm_in,    /**< [in] MPI communicator */
        PetscInt num_sites_in,      /**< [in] Number of sites */
        PetscInt num_states_in      /**< [in] Number of states (or PETSC_DEFAULT) */
        );

    /** Destroys all operator matrices and frees memory.
        @remarks __TODO:__ Consider interfacing this to the object desctructor */
    PetscErrorCode Destroy();

    /** Stores the information on the magnetization Sectors */
    QuantumNumbers Magnetization;

    /** Determines whether the operator arrays have been successfully filled with matrices.
        @remarks __TODO:__ Change the interface to take in only Op_t */
    PetscErrorCode CheckOperatorArray(
        Mat *Op,            /**< [in] pointer to the array of operator matrices */
        const char* label   /**< [in] string identifying the operator matrices */
        ) const;

    /** Indicates whether block has been initialized before using it */
    PetscBool Initialized() const { return init; }

    /** Gets the communicator associated to the block */
    MPI_Comm MPIComm() const { return mpi_comm; }

    /** Gets the number of sites that are currently initialized */
    PetscInt NumSites() const {return num_sites; }

    /** Gets the number of states that are currently used */
    PetscInt NumStates() const {return num_states; }

    /** Matrix representation of the Hamiltonian operator */
    Mat     H = nullptr;

    /** Array of matrices representing \f$S^z\f$ operators */
    Mat*    Sz = nullptr;

    /** Array of matrices representing \f$S^+\f$ operators */
    Mat*    Sp = nullptr;

    /** Array of matrices representing \f$S^-\f$ operators */
    Mat*    Sm = nullptr;

    /** Checks whether all operators have been initialized and have correct dimensions */
    PetscErrorCode CheckOperators() const;

    /** Checks whether sector indexing was done properly */
    PetscErrorCode CheckSectors() const;

    /** Checks the block indexing in the matrix operator op_t on site isite.
        @pre Implemented only for MPIAIJ matrices */
    PetscErrorCode MatCheckOperatorBlocks(
        const Op_t& op_t,       /**< [in] operator type */
        const PetscInt& isite   /**< [in] site index */
        ) const;

    /** Checks whether all matrix blocks follow the correct sector indices using MatCheckOperatorBlocks() */
    PetscErrorCode CheckOperatorBlocks() const;

    /** Extracts the block structure for each operator.
        @remark __TODO:__ Implementation */
    PetscErrorCode GetOperatorBlocks(Op_t Operator);

    /** Creates the Sm matrices on the fly */
    PetscErrorCode CreateSm();

    /** Destroys the Sm matrices on the fly */
    PetscErrorCode DestroySm();

};

/**
    @}
 */

#endif // __DMRG_SITE_HPP
