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

    /** Checks whether quantum number object was properly initialized */
    PetscErrorCode CheckInitialized() const
    {
        if(PetscUnlikely(!initialized))
            SETERRQ(mpi_comm, PETSC_ERR_ARG_CORRUPT, "QuantumNumbers object not yet initialized.");
        else
            return 0;
    }

    /** Gets the communicator associated to the block */
    MPI_Comm MPIComm() const { return mpi_comm; }

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

    /** Returns the quantum number associated to the given index */
    PetscReal List(const PetscInt& idx) const
    {
        assert(initialized);
        if(0 <= idx && idx < num_sectors)
            return qn_list[idx];
        else
            return -1;
    }

    /** Returns the offsets for each quantum number block */
    std::vector<PetscInt> Offsets() const
    {
        assert(initialized);
        return qn_offset;
    }

    /** Returns the offsets for the quantum number block with a specified index */
    PetscInt Offsets(const PetscInt& idx) const
    {
        assert(initialized);
        if(0 <= idx && idx < num_sectors)
            return qn_offset[idx];
        else
            return -1;
    }

    /** Returns the number of basis states in each quantum number block */
    std::vector<PetscInt> Sizes() const
    {
        assert(initialized);
        return qn_size;
    }

    /** Returns the number of basis states in each quantum number block */
    PetscInt Sizes(const PetscInt& idx) const
    {
        assert(initialized);
        if(0 <= idx && idx < num_sectors)
            return qn_size[idx];
        else
            return -1;
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

    /** Maps the shifted quantum number block index to the global indices [start,end).
        The value of flg is set to PETSC_TRUE if the output block exists, PETSC_FALSE otherwise */
    PetscErrorCode OpBlockToGlobalRange(
        const PetscInt& BlockIdx,   /**< [in]  Index of the quantum number block */
        const PetscInt& BlockShift, /**< [in]  Shift in quantum number associated to the operator */
        PetscInt& GlobIdxStart,     /**< [out] Inclusive lower bound index */
        PetscInt& GlobIdxEnd,       /**< [out] Exclusive upper bound index */
        PetscBool& flg              /**< [out] Indicates whether range is non-zero */
        ) const;

    /** Maps the shifted quantum number block index to the global index start in the range [start,end).
        The value of flg is set to PETSC_TRUE if the output block exists, PETSC_FALSE otherwise */
    PetscInt OpBlockToGlobalRangeStart(
        const PetscInt& BlockIdx,   /**< [in]  Index of the quantum number block */
        const PetscInt& BlockShift, /**< [in]  Shift in quantum number associated to the operator */
        PetscBool& flg              /**< [out] Indicates whether range is non-zero */
        ) const
    {
        PetscErrorCode ierr;
        PetscInt GlobIdxStart, GlobIdxEnd;
        ierr = OpBlockToGlobalRange(BlockIdx, BlockShift, GlobIdxStart, GlobIdxEnd, flg);
        assert(!ierr);
        return GlobIdxStart;
    }

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

    /** Maps the global index of a basis state to its block index and local index in the block */
    PetscErrorCode GlobalIdxToBlockIdx(
        const PetscInt& GlobIdx,    /**< [in]  Global index */
        PetscInt& BlockIdx,         /**< [out] Block index */
        PetscInt& LocIdx            /**< [out] Local index in the block */
        ) const;

    /** Maps the global index of a basis state to its quantum number */
    PetscErrorCode GlobalIdxToQN(
        const PetscInt& GlobIdx,    /**< [in]  Global index */
        PetscReal& QNValue          /**< [out] Value of the quantum number */
        ) const;

    /** Maps the block index of a basis state to its global index */
    PetscErrorCode BlockIdxToGlobalIdx(
        const PetscInt& BlockIdx,   /**< [in]  Block index */
        const PetscInt& LocIdx,     /**< [in]  Local index in the block */
        PetscInt& GlobIdx           /**< [out] Global index */
        ) const;

    /** Maps the block index of a basis state to its global index and returns the value directly */
    PetscInt BlockIdxToGlobalIdx(
        const PetscInt& BlockIdx,   /**< [in]  Block index */
        const PetscInt& LocIdx      /**< [in]  Local index in the block */
        ) const;

};


/** Iterates over a range of basis indices with information on each of its quantum number. */
class QuantumNumbersIterator
{
private:

    /** Reference to the Quantum Numbers object on which this iterator is based on */
    const QuantumNumbers& QN;

    /** Starting index in the range [Istart, Iend) */
    PetscInt istart_ = 0;

    /** The final excluded index of the range [Istart, Iend) */
    PetscInt iend_ = 0;

    /* Stores the value of the current index */
    PetscInt idx_ = 0;

    /** The block index associated with Idx */
    PetscInt blockidx_ = 0;

    /** The local index in the block associated with Idx */
    PetscInt locidx_ = 0;


public:

    typedef QuantumNumbersIterator Self_t;

    /** Initialize an iterator through all quantum numbers */
    QuantumNumbersIterator(
        const QuantumNumbers& QN_in     /**< [in] base QuantumNumbers object */
        ):
        QN(QN_in),
        iend_(QN.NumStates())
    {}

    /** Initialize an iterator through a range of indices */
    QuantumNumbersIterator(
        const QuantumNumbers& QN_in,  /**< [in] Base QuantumNumbers object */
        const PetscInt& GlobIdxStart, /**< [in] Inclusive lower bound index */
        const PetscInt& GlobIdxEnd    /**< [in] Exclusive upper bound index */
        ):
        QN(QN_in),
        istart_(GlobIdxStart),
        iend_(GlobIdxEnd),
        idx_(istart_)
    {
        PetscErrorCode ierr;
        if(istart_==iend_) {}
        else
        {
            ierr = QN.GlobalIdxToBlockIdx(istart_, blockidx_);
            assert(!ierr);
        }
    }

    /* TODO: Initialize an iterator through a range of quantum number blocks */

    /** Gets the current state index */
    PetscInt Idx() const {return idx_;}

    /** Gets the current quantum number block index */
    PetscInt BlockIdx() const {return blockidx_;}

    /** Gets the first quantum number block index */
    PetscInt IdxStart() const {return istart_;}

    /** Gets the first quantum number block index */
    PetscInt IdxEnd() const {return iend_;}

    /** Gets the current local index in the quantum number block */
    PetscInt LocIdx() const {return idx_ - QN.Offsets()[blockidx_];}

    /** Determines whether the end of the range has not yet been reached */
    bool Loop() const {return idx_ < iend_;}

    /** Gets the number of steps incremented from the starting index */
    PetscInt Steps() const {return idx_-istart_;}

    /** Overloading the ++ increment */
    Self_t operator++()
    {
        ++idx_;
        if(idx_ >= QN.Offsets()[blockidx_+1]) ++blockidx_;
        return *this;
    }

    /** Interfaced function OpBlockToGlobalRange from the Quantum Numbers class with current block index as input */
    PetscErrorCode OpBlockToGlobalRange(
        const PetscInt& BlockShift,
        PetscInt& GlobIdxStart,
        PetscInt& GlobIdxEnd,
        PetscBool& flg
        ) const
    {
        PetscInt ierr;
        ierr = QN.OpBlockToGlobalRange(blockidx_, BlockShift, GlobIdxStart, GlobIdxEnd, flg); CHKERRQ(ierr);
        return(0);
    }

};


/**
    @}
 */

#endif
