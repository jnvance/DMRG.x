#ifndef __QUANTUM_NUMBERS_HPP
#define __QUANTUM_NUMBERS_HPP

#include <petscmat.h>
#include <vector>
#include <cassert>

class QuantumNumbers
{

private:

    MPI_Comm mpi_comm;

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

    /** Whether initialized previously */
    PetscBool initialized = PETSC_FALSE;

public:

    PetscErrorCode Initialize(
        MPI_Comm mpi_comm_in,
        std::vector<PetscReal> qn_list_in,
        std::vector<PetscInt> qn_size_in);

    PetscInt NumSectors() const {
        assert(initialized);
        return num_sectors;
    }
    std::vector<PetscReal> List() const {
        assert(initialized);
        return qn_list;
    }
    std::vector<PetscInt> Offsets() const {
        assert(initialized);
        return qn_offset;
    }
    std::vector<PetscInt> Sizes() const {
        assert(initialized);
        return qn_size;
    }
    PetscInt NumStates() const {
        assert(initialized);
        return num_states;
    }
};

#endif
