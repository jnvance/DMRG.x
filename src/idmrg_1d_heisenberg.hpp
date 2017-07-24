#include "dmrg.hpp"


/*
    The DMRG Class is inherited by a class that specifies the Hamiltonian
*/
class iDMRG_Heisenberg: public iDMRG
{
    // PetscInt local_dim_ = 2;

public:

    /*
        Overload base class implementation
        with the Heisenberg Hamiltonian
    */
    PetscErrorCode BuildBlockLeft() final;
    PetscErrorCode BuildBlockRight() final;
    PetscErrorCode BuildSuperBlock() final;

};
