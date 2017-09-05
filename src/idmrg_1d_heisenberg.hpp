#include "dmrg.hpp"
#include <iostream>
#include <vector>
#include <unordered_map>

/*
    The DMRG Class is inherited by a class that specifies the Hamiltonian
*/
class iDMRG_Heisenberg: public iDMRG
{
protected:

    /** XX-coupling constant */
    PetscScalar J;

    /** Z-coupling constant */
    PetscScalar Jz;

    /** Whether to perform targetting of magnetization sector */
    PetscBool do_target_Sz = PETSC_FALSE;

    /** Target magnetization */
    PetscReal Mz;

public:

    PetscErrorCode SetParameters(PetscScalar J_in, PetscScalar Jz_in, PetscReal Mz_in, PetscBool do_target_Sz_in);

    /*
        Overload base class implementation
        with the Heisenberg Hamiltonian
    */
    PetscErrorCode BuildBlockLeft() final;
    PetscErrorCode BuildBlockRight() final;
    PetscErrorCode BuildSuperBlock() final;
};
