#include <vector>
#include "dmrg.hpp"


/*
    The DMRG Class is inherited by a class that specifies the Hamiltonian
*/
class iDMRG_Heisenberg: public iDMRG
{
protected:

    PetscScalar J;
    PetscScalar Jz;

public:

    PetscErrorCode SetParameters(PetscScalar J_in, PetscScalar Jz_in);

    /*
        Overload base class implementation
        with the Heisenberg Hamiltonian
    */
    PetscErrorCode BuildBlockLeft() final;
    PetscErrorCode BuildBlockRight() final;
    PetscErrorCode BuildSuperBlock() final;
};
