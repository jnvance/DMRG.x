#ifndef __DMRG_BLOCK_HPP__
#define __DMRG_BLOCK_HPP__

/**
    @defgroup   DMRGBlock   DMRGBlock
    @brief      Implementation of the SpinBase class which contain the data and methods
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

/** Identifies the spin types implemented for this block */
typedef enum
{
    SpinOneHalf  = 102, /**< Spin one-half. `SpinTypeKey = 102` */
    SpinOne      = 101, /**< Spin one. `SpinTypeKey = 101` */
    SpinNull     = -2   /**< Null input spin */
} Spin_t;

/** Maps a string to its corresponding spin type */
static const std::map<std::string,Spin_t> SpinTypes = {
    {"1/2",         SpinOneHalf },
    {"1",           SpinOne     }
};

/** Maps a spin type to its corresponding string */
static const std::map<Spin_t,std::string> SpinTypesString = {
    {SpinOneHalf,   "1/2"       },
    {SpinOne,       "1"         }
};

static const std::vector<Op_t> BasicOpTypes = { OpSz, OpSp };
static const std::vector<Side_t> SideTypes = { SideLeft, SideRight };

/** Contains the definitions for blocks of spin sites.
    Definition of spin operators were taken from this
    [site](http://easyspin.org/documentation/spinoperators.html). */
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

        /** Type of spin contained in the block */
        Spin_t          spin_type = SpinOneHalf;

        /** Type of spin contained in the block expressed as a string */
        std::string     spin_type_str = "1/2";

        /** Stores the local dimension of a single site */
        PetscInt        _loc_dim = 2;

        /** Stores the Sz sectors of a single site. */
        std::vector<PetscScalar>    _loc_qn_list = {+0.5, -0.5};

        /** Stores the number of states in each sector in a single site. */
        std::vector<PetscInt>       _loc_qn_size = {1, 1};

        /** Tells whether the block was initialized */
        PetscBool init = PETSC_FALSE;

        /** Tells whether the block was initialized at least once before */
        PetscBool init_once = PETSC_FALSE;

        /** Tells whether the block's MPI attributes were initialized */
        PetscBool mpi_init = PETSC_FALSE;

        /** Tells whether to printout info during certain function calls */
        PetscBool verbose = PETSC_FALSE;

        /** Tells whether to printout IO functions */
        PetscBool log_io = PETSC_FALSE;

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

        /** Whether SetDiskStorage() has properly set the block storage locations. */
        PetscBool disk_set = PETSC_FALSE;

        /** Whether the block matrices have been saved */
        PetscBool saved = PETSC_FALSE;

        /** Whether the block matrices have been retrieved */
        PetscBool retrieved = PETSC_FALSE;

        /** Root directory to read from and write the matrix blocks into */
        std::string save_dir;

        /** Root directory to read the matrix blocks from during the first access */
        std::string read_dir;

        /** Root directory to write the matrix blocks into */
        std::string write_dir;

        /** Number of reads from file made for this block.

            @todo Its value is incremented by a call to Retrieve_NoChecks() via
            Retrieve() and InitializeFromDisk(), and reset to zero by SetDiskStorage().
         */
        PetscInt num_reads = 0;

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

        /** Method to use for performing basis transformation.
            @todo Get the method from command line and expand choices */
        RotMethod rot_method = mmmmult;

    protected:

        /** Returns PETSC_TRUE if the MPI communicator was initialized properly. */
        PetscBool MPIInitialized() const { return mpi_init; }

    public:

        /** Local dimension of a single site. */
        virtual PetscInt loc_dim() const
        {
            return _loc_dim;
        }

        /** Sz sectors of a single site. */
        virtual std::vector<PetscScalar> loc_qn_list() const
        {
            return _loc_qn_list;
        }

        /** Number of states in each sector in a single site. */
        virtual std::vector<PetscInt> loc_qn_size() const
        {
            return _loc_qn_size;
        }

        /** Creates the single-site \f$ S^z \f$ operator. */
        virtual PetscErrorCode MatSpinSzCreate(Mat& Sz);

        /** Creates the single-site raising operator \f$ S^+ \f$,
            from which we can define \f$ S^- = (S^+)^\dagger \f$. */
        virtual PetscErrorCode MatSpinSpCreate(Mat& Sp);

        /** Initializes block object's MPI attributes */
        PetscErrorCode Initialize(
            const MPI_Comm& comm_in       /**< [in] MPI communicator */
        );

        /** Initializes block object with input attributes and array of matrix operators.
            @post Arrays of operator matrices are initialized to the correct number of sites and states.
            @todo Consider interfacing this to the object constructor.

            @par Options:
            - `-verbose`
            - `-log_io`
        */
        PetscErrorCode Initialize(
            const MPI_Comm& comm_in,      /**< [in] MPI communicator */
            const PetscInt& num_sites_in, /**< [in] Number of sites */
            const PetscInt& num_states_in,/**< [in] Number of states (or PETSC_DEFAULT) */
            const PetscBool& init_ops = PETSC_TRUE
            /**< [in] Whether to initialize operators automatically when `num_sites_in` > 1 */
            );

        /** Initializes block object with input attributes and array of matrix operators.
            @post Arrays of operator matrices are initialized to the correct number of sites and states.
            @post Magnetization object is initialized as well
            @remarks __TODO:__ Consider interfacing this to the object constructor.
        */
        PetscErrorCode Initialize(
            const MPI_Comm& comm_in,                    /**< [in] MPI communicator */
            const PetscInt& num_sites_in,               /**< [in] Number of sites */
            const std::vector<PetscReal>& qn_list_in,   /**< [in] List of quantum numbers in each sector */
            const std::vector<PetscInt>& qn_size_in,    /**< [in] List of number of states in each sector */
            const PetscBool& init_ops = PETSC_TRUE
            /**< [in] Whether to initialize operators automatically when `num_sites_in` > 1 */
            );

        /** Initializes block object with an initialized quantum numbers object */
        PetscErrorCode Initialize(
            const PetscInt& num_sites_in,   /**< [in] Number of sites */
            const QuantumNumbers& qn_in     /**< [in] QuantumNumbers object to associate with this block */
            );

        /** Initializes block object from data located in the directory `block_path`. */
        PetscErrorCode InitializeFromDisk(
            const MPI_Comm& comm_in,        /**< [in] MPI communicator */
            const std::string& block_path   /**< [in] Directory storing the block object's data */
            );

        /** Initializes the writing of the block matrices to file.
            The directory `save_dir_in` may be set only once. If this directory changes throughout
            the lifetime of a block, or if the read directory is different from the write directory
            consider using SetDiskStorage() instead. */
        PetscErrorCode InitializeSave(
            const std::string& save_dir_in  /**< [in] Directory to store block object's data */
            );

        /** Tells where to read from and save the operators and data about the block. */
        PetscErrorCode SetDiskStorage(
            const std::string& read_dir_in,     /**< [in] Directory to initially read-in block object's data */
            const std::string& write_dir_in     /**< [in] Directory to store block object's data */
            );

        /** Returns the value of save_dir where the block data will be read from/written */
        std::string SaveDir() const { return save_dir; }

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

}

/**
    @}
 */

#endif // __DMRG_BLOCK_HPP__
