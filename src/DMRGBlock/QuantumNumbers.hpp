#ifndef __QUANTUM_NUMBERS_HPP
#define __QUANTUM_NUMBERS_HPP

#include <petscmat.h>
#include <vector>
#include <cassert>

class QuantumNumbers
{

private:

    /** MPI Communicator */
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

    /** Tells whether initialized previously */
    PetscBool initialized = PETSC_FALSE;

public:

    /** Initializes the object */
    PetscErrorCode Initialize(
        MPI_Comm mpi_comm_in,
        std::vector<PetscReal> qn_list_in,
        std::vector<PetscInt> qn_size_in);

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

};

#endif
