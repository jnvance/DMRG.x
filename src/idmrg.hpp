#ifndef __IDMRG_HPP__
#define __IDMRG_HPP__

#include <slepceps.h>
#include "dmrgblock.hpp"
// #include "linalg_tools.hpp"

class iDMRG
{

protected:

    PetscInt    final_nsites_;
    PetscInt    nsteps_;
    PetscInt    iter_;

    DMRGBlock   BlockLeft_;
    DMRGBlock   BlockRight_;

    Mat         superblock_H_ = NULL;
    PetscBool   superblock_set_ = PETSC_FALSE;

    /* Ground state */
    PetscScalar *gse_r_list_, *gse_i_list_;
    Vec         gsv_r_, gsv_i_;
    PetscBool   groundstate_solved_ = PETSC_FALSE;

    Mat         dm_left;
    Mat         dm_right;

    MPI_Comm    comm_;

    /* Matrices of the single-site operators */
    Mat eye1_, Sz1_, Sp1_, Sm1_;

public:

    PetscErrorCode init(MPI_Comm);
    PetscErrorCode destroy();

    PetscInt LengthBlockLeft(){ return BlockLeft_.length(); }
    PetscInt LengthBlockRight(){ return BlockRight_.length(); }

    /* Block enlargement to be implemented in inherited classes */
    virtual PetscErrorCode BuildBlockLeft()=0;
    virtual PetscErrorCode BuildBlockRight()=0;
    virtual PetscErrorCode BuildSuperBlock()=0;

    /* Solve states */
    PetscErrorCode SolveGroundState(PetscReal& gse_r, PetscReal& gse_i, PetscReal& error);

    /* Miscellaneous functions */
    PetscErrorCode MatPeekOperators();
    PetscErrorCode MatSaveOperators();

};

#endif // __IDMRG_HPP__
