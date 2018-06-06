#ifndef __DMRG_SITE_HPP
#define __DMRG_SITE_HPP

/**
    @defgroup   DMRGBlock   DMRGBlock
    @brief      Implementation of the Block classes which contain the data and methods
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

/** Convert the input Op_t to a C-style string */
#define OpToCStr(OP)    ((OpString.find(OP)->second).c_str())

/** Convert the input Op_t to a C++ string */
#define OpToStr(OP)     (OpString.find(OP)->second)

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

    /** Base class for the implementation of a block of spin sites. */
    class SpinBase
    {

    private:

        /** MPI Communicator */
        MPI_Comm        mpi_comm = PETSC_COMM_SELF;

        /** MPI rank in mpi_comm */
        PetscMPIInt     mpi_rank;

        /** MPI size of mpi_comm */
        PetscMPIInt     mpi_size;

        /** Tells whether the block was initialized */
        PetscBool init = PETSC_FALSE;

        /** Tells whether the block was initialized at least once before */
        PetscBool init_once = PETSC_FALSE;

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
            Mat& Op,
            const MPI_Comm& comm_in
            );

        /** Private function to retrieve operators without checking for save initialization. */
        PetscErrorCode Retrieve_NoChecks();

        /* Number of subcommunicators to be used when performing the rotation. */
        PetscInt nsubcomm = 1;

        /* Tells the subcommunicator to use for rotating this site */
        std::vector< PetscMPIInt > site_color;

        /** List of possible methods for performing basis transformation */
        enum RotMethod { mmmmult=0, matptap=1 };

        /** Method to use for performing basis transformation. TODO: Get the method from command line */
        RotMethod rot_method = mmmmult;

    protected:

        /** Returns PETSC_TRUE if the MPI communicator was initialized properly. */
        PetscBool MPIInitialized() const { return mpi_init; }

    public:

        /** Local dimension of a single site. */
        virtual PetscInt loc_dim() const = 0;

        /** Sz sectors of a single site. */
        virtual std::vector<PetscScalar> loc_qn_list() const = 0;

        /** Number of states in each sector in a single site. */
        virtual std::vector<PetscInt> loc_qn_size() const = 0;

        /** Creates the single-site \f$ S^z \f$ operator. */
        virtual PetscErrorCode MatSpinSzCreate(Mat& Sz) = 0;

        /** Creates the single-site raising operator \f$ S^+ \f$,
            from which we can define \f$ S^- = (S^+)^\dagger \f$. */
        virtual PetscErrorCode MatSpinSpCreate(Mat& Sp) = 0;

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

        /** Initializes block object from data located in the directory `block_path`. */
        PetscErrorCode InitializeFromDisk(
            const MPI_Comm& comm_in,
            const std::string& block_path
            );

        /** Initializes the writing of the block matrices to file */
        PetscErrorCode InitializeSave(
            const std::string& save_dir_in
            );

        /** Tells whether InitializeSave() has been properly called */
        PetscBool SaveInitialized() const { return init_save; }

        /** Save all the matrix operators to file and destroy the current storage */
        PetscErrorCode SaveAndDestroy();

        /** Save some information about the block that could be used to reconstruct it later.

            Information to be saved:
              - Number of sites
              - Number of states
              - QuantumNumbers list and sizes
         */
        PetscErrorCode SaveBlockInfo();

        /** Retrieves a single operator */
        PetscErrorCode RetrieveOperator(
            const std::string& OpName,
            const size_t& isite,
            Mat& Op,
            const MPI_Comm& comm_in=MPI_COMM_NULL
            );

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

        /** Returns the saved state of all operators */
        PetscBool Saved() const { return saved; }

        /** Gets the communicator associated to the block */
        MPI_Comm MPIComm() const { return mpi_comm; }

        /** Gets the number of sites that are currently initialized */
        PetscInt NumSites() const {return num_sites; }

        /** Gets the number of states that are currently used */
        PetscInt NumStates() const {return num_states; }

        /** Matrix representation of the Hamiltonian operator */
        Mat     H = nullptr;

        /** Returns the matrix pointer to the \f$ S^z \f$ operator at site `Isite`. */
        Mat Sz(const PetscInt& Isite) const {
            if(Isite >= num_sites) throw std::runtime_error("Attempted to access non-existent site.");
            return SzData[Isite];
        }

        /** Returns the matrix pointer to the \f$ S^+ \f$ operator at site `Isite`. */
        Mat Sp(const PetscInt& Isite) const {
            if(Isite >= num_sites) throw std::runtime_error("Attempted to access non-existent site.");
            return SpData[Isite];
        }

        /** Returns the matrix pointer to the \f$ S^- \f$ operator at site `Isite`, if available. */
        Mat Sm(const PetscInt& Isite) const {
            if(Isite >= num_sites) throw std::runtime_error("Attempted to access non-existent site.");
            if(!init_Sm) throw std::runtime_error("Sm matrices were not initialized on this block.");
            return SmData[Isite];
        }

        /** Returns the list of matrix pointer to the \f$ S^z \f$ operators. */
        const std::vector<Mat>& Sz() const { return SzData; }

        /** Returns the list of matrix pointer to the \f$ S^+ \f$ operators. */
        const std::vector<Mat>& Sp() const { return SpData; }

        /** Returns the list of matrix pointer to the \f$ S^- \f$ operators. */
        const std::vector<Mat>& Sm() const { return SmData; }

        /** Checks whether all operators have been initialized and have correct dimensions */
        PetscErrorCode CheckOperators() const;

        /** Checks whether sector indexing was done properly */
        PetscErrorCode CheckSectors() const;

        /** Returns the number of non-zeros in each row for an operator acting on a given site */
        PetscErrorCode MatOpGetNNZs(
            const Op_t& OpType,
            const PetscInt& isite,
            std::vector<PetscInt>& nnzs
            ) const;

        /** Returns the number of non-zeros in each row for a given matrix */
        PetscErrorCode MatGetNNZs(
            const Mat& matin,
            std::vector<PetscInt>& nnzs
            ) const;

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
            SpinBase& Source,           /**< [in] Block containing the original operators (may modify save state) */
            const Mat& RotMatT          /**< [in] Transposed rotation matrix */
            );

        /** Ensures that all operators are assembled */
        PetscErrorCode AssembleOperators();
    };

    /** Specific implementation for a block of spin-\f$ 1/2 \f$ sites */
    class SpinOneHalf: public SpinBase
    {
    public:
        /** Returns the local dimension of a single site \f$d=2\f$. */
        PetscInt loc_dim() const { return 2; }

        /** Returns the \f$ S^z \f$ sectors of a single site \f$\{+1/2,-1/2\}\f$ */
        std::vector<PetscScalar> loc_qn_list() const { return std::vector<PetscScalar>({+0.5, -0.5}); }

        /** Returns the number of states in each sector in a single site \f$\{1,1\}\f$. */
        std::vector<PetscInt> loc_qn_size() const { return std::vector<PetscInt>({1, 1}); }

        /** Creates the single-site \f$ S^z \f$ operator. */
        PetscErrorCode MatSpinSzCreate(Mat& Sz);

        /** Creates the single-site raising operator \f$ S^+ \f$,
            from which we can define \f$ S^- = (S^+)^\dagger \f$. */
        PetscErrorCode MatSpinSpCreate(Mat& Sp);
    };

    /** Specific implementation for a block of spin-\f$ 1 \f$ sites */
    class SpinOne: public SpinBase
    {
    public:
        /** Returns the local dimension of a single site \f$d=3\f$. */
        PetscInt loc_dim() const { return 3; }

        /** Returns the \f$ S^z \f$ sectors of a single site \f$\{+1/2,-1/2\}\f$ */
        std::vector<PetscScalar> loc_qn_list() const { return std::vector<PetscScalar>({+1, 0, -1}); }

        /** Returns the number of states in each sector in a single site \f$\{1,1\}\f$. */
        std::vector<PetscInt> loc_qn_size() const { return std::vector<PetscInt>({1, 1, 1}); }

        /** Creates the single-site \f$ S^z \f$ operator. */
        PetscErrorCode MatSpinSzCreate(Mat& Sz);

        /** Creates the single-site raising operator \f$ S^+ \f$,
            from which we can define \f$ S^- = (S^+)^\dagger \f$. */
        PetscErrorCode MatSpinSpCreate(Mat& Sp);
    };

}

/**
    @}
 */

#endif // __DMRG_SITE_HPP
