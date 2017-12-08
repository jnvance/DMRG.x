#include <petscsys.h>
#include <slepceps.h>
#include "DMRGBlock.hpp"

PETSC_EXTERN PetscErrorCode Kron_Explicit(
    const Block_SpinOneHalf& LeftBlock,
    const Block_SpinOneHalf& RightBlock,
    Block_SpinOneHalf& BlockOut,
    PetscBool BuildHamiltonian
    )
{
    PetscErrorCode ierr = 0;


    return ierr;
}
