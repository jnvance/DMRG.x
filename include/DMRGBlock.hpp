#ifndef __DMRG_SITE_HPP
#define __DMRG_SITE_HPP

/**
    @defgroup   DMRGBlock   DMRGBlock
    @brief      Implementation of the Block_SpinOneHalf class which contains the data and methods
                for a block of spin sites
    @addtogroup DMRGBlock
    @{ */

#include <petscmat.h>
#include "QuantumNumbers.hpp"
#include "MiscTools.hpp"
#include "kron.hpp"
#include "linalg_tools.hpp"
#include <string>
#include <map>

/** Identifies the three possible spin operators and also represents the shift associated
    to its action on the quantum number blocks */
typedef enum
{
    OpSm = -1,  /**< \f$ S^- \f$ operator */
    OpSz = 0,   /**< \f$ S^z \f$ operator */
    OpSp = +1,  /**< \f$ S^+ \f$ operator */
    OpEye= +2   /**< Identity operator */
} Op_t;

/** Lists down the names for each operator */
static const std::map<Op_t, std::string> OpString =
{
    {OpSm, "Sm"},
    {OpSz, "Sz"},
    {OpSp, "Sp"}
};

/** Identifies the sides of the DMRG block */
typedef enum
{
    SideLeft = 0,   /**< left block */
    SideRight = 1   /**< right block */
} Side_t;

static const std::vector<Op_t> BasicOpTypes = { OpSz, OpSp };
static const std::vector<Side_t> SideTypes = { SideLeft, SideRight };

/** Contains the definitions for blocks of spin sites */
namespace Block {

    /** Contains the matrix representations of the operators of a block of spin-1/2 sites, the associated
        magnetization sectors and some useful information and checking functions */
    class SpinOneHalf
    {

    private:

        /** MPI Communicator */
        MPI_Comm        mpi_comm = PETSC_COMM_SELF;

        /** MPI rank in mpi_comm */
        PetscMPIInt     mpi_rank;

        /** MPI size of mpi_comm */
        PetscMPIInt     mpi_size;

        /** Local dimension of a single site.
            @remarks __NOTE:__ Default for spin-1/2 */
        PetscInt loc_dim = 2;

        /** Sz sectors of a single site.
            @remarks __NOTE:__ Default for spin-1/2 */
        std::vector<PetscScalar> loc_qn_list = {+0.5, -0.5};

        /** Number of states in each sector in a single site.
            @remarks __NOTE:__ Default for spin-1/2 */
        std::vector<PetscInt> loc_qn_size = {1, 1};

        /** Tells whether the block was initialized */
        PetscBool init = PETSC_FALSE;

        /** Tells whether the block's MPI attributes were initialized */
        PetscBool mpi_init = PETSC_FALSE;

        /** Tells whether to printout info during certain function calls */
        PetscBool verbose = PETSC_FALSE;

        /** Number of sites in the block */
        PetscInt num_sites;

        /** Number of basis states in the Hilbert space */
        PetscInt num_states;

        /** Tells whether the Sm matrices have been initialized */
        PetscBool init_Sm = PETSC_FALSE;

        /** Array of matrices representing \f$S^z\f$ operators */
        std::vector<Mat> SzData;

        /** Array of matrices representing \f$S^+\f$ operators */
        std::vector<Mat> SpData;

        /** Array of matrices representing \f$S^-\f$ operators */
        std::vector<Mat> SmData;

        /** Whether saving the block matrices to file has been initialized correctly */
        PetscBool init_save = PETSC_FALSE;

        /** Whether the block matrices have been saved */
        PetscBool saved = PETSC_FALSE;

        /** Whether the block matrices have been retrieved */
        PetscBool retrieved = PETSC_FALSE;

        /** Root directory to save the matrix blocks */
        std::string save_dir;

        /** Saves a single operator */
        PetscErrorCode SaveOperator(
            const std::string& OpName,
            const size_t& isite,
            Mat& Op
            );

        /** Retrieves a single operator */
        PetscErrorCode RetrieveOperator(
            const std::string& OpName,
            const size_t& isite,
            Mat& Op
            );

    public:

        /** Initializes block object's MPI attributes */
        PetscErrorCode Initialize(
            const MPI_Comm& comm_in       /**< [in] MPI communicator */
        );

        /** Initializes block object with input attributes and array of matrix operators.
            @post Arrays of operator matrices are initialized to the correct number of sites and states.
            @remarks __TODO:__ Consider interfacing this to the object constructor.
        */
        PetscErrorCode Initialize(
            const MPI_Comm& comm_in,      /**< [in] MPI communicator */
            const PetscInt& num_sites_in, /**< [in] Number of sites */
            const PetscInt& num_states_in /**< [in] Number of states (or PETSC_DEFAULT) */
            );

        /** Initializes block object with input attributes and array of matrix operators.
            @post Arrays of operator matrices are initialized to the correct number of sites and states.
            @post Magnetization object is initialized as well
            @remarks __TODO:__ Consider interfacing this to the object constructor.
        */
        PetscErrorCode Initialize(
            const MPI_Comm& comm_in,
            const PetscInt& num_sites_in,
            const std::vector<PetscReal>& qn_list_in,
            const std::vector<PetscInt>& qn_size_in
            );

        /** Initializes block object with an initialized quantum numbers object */
        PetscErrorCode Initialize(
            const PetscInt& num_sites_in,
            const QuantumNumbers& qn_in
            );

        /** Initializes the writing of the block matrices to file */
        PetscErrorCode InitializeSave(
            const std::string& save_dir_in
            );

        /** Tells whether InitializeSave() has been properly called */
        PetscBool SaveInitialized() const { return init_save; }

        /** Save all the matrix operators to file and destroy the current storage */
        PetscErrorCode SaveAndDestroy();

        /** Retrieve all the matrix operators that were written to file by SaveAndDestroy() */
        PetscErrorCode Retrieve();

        /** Ensures that the block matrices have been saved if the block is initialized, otherwise does nothing */
        PetscErrorCode EnsureSaved();

        /** Ensures that the block matrices have been retrieved if the block is initialized, otherwise does nothing */
        PetscErrorCode EnsureRetrieved();

        /** Destroys all operator matrices and frees memory.
            @remarks __TODO:__ Consider interfacing this to the object desctructor */
        PetscErrorCode Destroy();

        /** Stores the information on the magnetization Sectors */
        QuantumNumbers Magnetization;

        /** Determines whether the operator arrays have been successfully filled with matrices.
            @remarks __TODO:__ Change the interface to take in only Op_t */
        PetscErrorCode CheckOperatorArray(
            const Op_t& OpType  /**< [in] operator type */
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

        Mat Sz(const PetscInt& Isite) const {
            if(Isite >= num_sites) throw std::runtime_error("Attempted to access non-existent site.");
            return SzData[Isite];
        }

        Mat Sp(const PetscInt& Isite) const {
            if(Isite >= num_sites) throw std::runtime_error("Attempted to access non-existent site.");
            return SpData[Isite];
        }

        Mat Sm(const PetscInt& Isite) const {
            if(Isite >= num_sites) throw std::runtime_error("Attempted to access non-existent site.");
            if(!init_Sm) throw std::runtime_error("Sm matrices were not initialized on this block.");
            return SmData[Isite];
        }

        const std::vector<Mat>& Sz() const { return SzData; }

        const std::vector<Mat>& Sp() const { return SpData; }

        const std::vector<Mat>& Sm() const { return SmData; }

        /** Checks whether all operators have been initialized and have correct dimensions */
        PetscErrorCode CheckOperators() const;

        /** Checks whether sector indexing was done properly */
        PetscErrorCode CheckSectors() const;

        /** Checks the block indexing in the matrix operator op_t on site isite.
            @pre Implemented only for MPIAIJ matrices */
        PetscErrorCode MatOpCheckOperatorBlocks(
            const Op_t& op_t,       /**< [in] operator type */
            const PetscInt& isite   /**< [in] site index */
            ) const;

        /** Checks the block indexing in the matrix operator op_t on specified matrix matin.
            @pre Implemented only for MPIAIJ matrices */
        PetscErrorCode MatCheckOperatorBlocks(
            const Op_t& op_t,       /**< [in] operator type */
            const Mat& matin        /**< [in] matrix to be checked */
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

        /** Rotates all operators from a source block using the given transposed rotation matrix */
        PetscErrorCode RotateOperators(
            const SpinOneHalf& Source,  /**< [in] Block containing the original operators */
            const Mat& RotMatT          /**< [in] Transposed rotation matrix */
            );

        /** Ensures that all operators are assembled */
        PetscErrorCode AssembleOperators();
    };

}

/**
    @}
 */

#endif // __DMRG_SITE_HPP
