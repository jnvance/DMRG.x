#ifndef __QUANTUM_NUMBERS_HPP
#define __QUANTUM_NUMBERS_HPP

/**
    @defgroup   QuantumNumbers   QuantumNumbers
    @brief      Implementation of the QuantumNumbers class containing information on quantum number blocks

    @remarks __TODO:__ Write an explainer on the block structure of the indexing of basis states
 */


#include <petscmat.h>
#include <vector>
#include <cassert>

/**
    @addtogroup QuantumNumbers
    @{ */

/** Contains information on quantum numbers and associated index ranges. */
class QuantumNumbers
{

private:

    /** MPI Communicator */
    MPI_Comm mpi_comm = PETSC_COMM_SELF;

    /** Number of Sz sectors in the Hilbert space */
    PetscInt num_sectors;

    /** List of Sz quantum numbers */
    std::vector<PetscReal> qn_list;

    /** Offset for each quantum number block */
    std::vector<PetscInt> qn_offset;

    /** Number of states in each quantum number block */
    std::vector<PetscInt> qn_size;

    /** Number of basis states in the Hilbert space */
    PetscInt num_states;

    /** Tells whether initialized previously */
    PetscBool initialized = PETSC_FALSE;

public:

    /** Initializes the quantum number object.
        @remarks __TODO:__ Consider interfacing this to the object constructor */
    PetscErrorCode Initialize(
        const MPI_Comm& mpi_comm_in,                /**< [in] MPI communicator */
        const std::vector<PetscReal>& qn_list_in,   /**< [in] list of quantum numbers */
        const std::vector<PetscInt>& qn_size_in     /**< [in] size of each quantum number block */
        );

    /** Returns the number of quantum number sectors */
    PetscInt NumSectors() const
    {
        assert(initialized);
        return num_sectors;
    }

    /** Returns the list of quantum numbers */
    std::vector<PetscReal> List() const
    {
        assert(initialized);
        return qn_list;
    }

    /** Returns the offsets for each quantum number block */
    std::vector<PetscInt> Offsets() const
    {
        assert(initialized);
        return qn_offset;
    }

    /** Returns the number of basis states in each quantum number block */
    std::vector<PetscInt> Sizes() const
    {
        assert(initialized);
        return qn_size;
    }

    /** Returns the total number of states */
    PetscInt NumStates() const
    {
        assert(initialized);
        return num_states;
    }

    /** Maps the quantum number block index to the global indices [start,end) */
    PetscErrorCode BlockIdxToGlobalRange(
        const PetscInt& BlockIdx,   /**< [in]  Index of the quantum number block */
        PetscInt& GlobIdxStart,     /**< [out] Inclusive lower bound index */
        PetscInt& GlobIdxEnd        /**< [out] Exclusive upper bound index */
        ) const;

    /** Maps the shifted quantum number block index to the global indices [start,end)
        The value of flg is set to PETSC_TRUE if the output block exists, PETSC_FALSE otherwise */
    PetscErrorCode OpBlockToGlobalRange(
        const PetscInt& BlockIdx,   /**< [in]  Index of the quantum number block */
        const PetscInt& BlockShift, /**< [in]  Shift in quantum number associated to the operator */
        PetscInt& GlobIdxStart,     /**< [out] Inclusive lower bound index */
        PetscInt& GlobIdxEnd,       /**< [out] Exclusive upper bound index */
        PetscBool& flg              /**< [out] Indicates whether range is non-zero */
        ) const;

    /** Maps the quantum number value to the global indices [start,end) */
    PetscErrorCode QNToGlobalRange(
        const PetscReal& QNValue,   /**< [in]  Value of the quantum number */
        PetscInt& GlobIdxStart,     /**< [out] Inclusive lower bound index */
        PetscInt& GlobIdxEnd        /**< [out] Exclusive upper bound index */
        ) const;

    /** Maps the global index of a basis state to its block index */
    PetscErrorCode GlobalIdxToBlockIdx(
        const PetscInt& GlobIdx,    /**< [in]  Global index */
        PetscInt& BlockIdx          /**< [out] Block index */
        ) const;

    /** Maps the global index of a basis state to its quantum number */
    PetscErrorCode GlobalIdxToQN(
        const PetscInt& GlobIdx,    /**< [in]  Global index */
        PetscReal& QNValue          /**< [out] Value of the quantum number */
        ) const;

};

/**
    @}
 */

#endif
