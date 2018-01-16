#ifndef __QUANTUM_NUMBERS_HPP
#define __QUANTUM_NUMBERS_HPP

#include <petscmat.h>
#include <vector>
#include <cassert>

#define DMRG_ERR_OUTOFBOUNDS        1001        /* An input index went out of bounds */

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

    /** Initializes the object */
    PetscErrorCode Initialize(
        const MPI_Comm& mpi_comm_in,
        const std::vector<PetscReal>& qn_list_in,
        const std::vector<PetscInt>& qn_size_in);

    /** Accesses the number of quantum number sectors */
    PetscInt NumSectors() const {
        assert(initialized);
        return num_sectors;
    }

    /** Accesses the list of quantum numbers */
    std::vector<PetscReal> List() const {
        assert(initialized);
        return qn_list;
    }

    /** Accesses the offsets for each quantum number block */
    std::vector<PetscInt> Offsets() const {
        assert(initialized);
        return qn_offset;
    }

    /** Accesses the number of basis states in each quantum number block */
    std::vector<PetscInt> Sizes() const {
        assert(initialized);
        return qn_size;
    }

    /** Accesses the total number of states */
    PetscInt NumStates() const {
        assert(initialized);
        return num_states;
    }

    /** Maps the quantum number block index to the global indices [start,end) */
    PetscErrorCode BlockIdxToGlobalRange(
        const PetscInt& BlockIdx,
        PetscInt& GlobIdxStart,
        PetscInt& GlobIdxEnd
        ) const;

    /** Maps the shifted quantum number block index to the global indices [start,end)
        The value of flg is set to PETSC_TRUE if the output block exists, PETSC_FALSE otherwise */
    PetscErrorCode OpBlockToGlobalRange(
        const PetscInt& BlockIdx,
        const PetscInt& BlockShift,
        PetscInt& GlobIdxStart,
        PetscInt& GlobIdxEnd,
        PetscBool& flg
        ) const;

    /** Maps the quantum number value to the global indices [start,end) */
    PetscErrorCode QNToGlobalRange(
        const PetscReal& QNValue,
        PetscInt& GlobIdxStart,
        PetscInt& GlobIdxEnd
        ) const;

    /** Maps the global index to block index */
    PetscErrorCode GlobalIdxToBlockIdx(
        const PetscInt& GlobIdx,
        PetscInt& BlockIdx
        ) const;

    /** Maps the global index to quantum number */
    PetscErrorCode GlobalIdxToQN(
        const PetscInt& GlobIdx,
        PetscReal& QNValue
        ) const;

};

#endif
