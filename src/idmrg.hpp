#ifndef __IDMRG_HPP__
#define __IDMRG_HPP__

#include <slepceps.h>
#include "dmrgblock.hpp"

/*
    @defgroup   idmrg   iDMRG
    @brief      Implementation of the iDMRG class

 */

class iDMRG
{

protected:

    PetscInt    final_nsites_;
    PetscInt    nsteps_;
    PetscInt    iter_;

    DMRGBlock   BlockLeft_;
    DMRGBlock   BlockRight_;

    Mat         superblock_H_ = nullptr;
    PetscBool   superblock_set_ = PETSC_FALSE;

    /* Ground state */
    Vec         gsv_r_ = nullptr;
    Vec         gsv_i_ = nullptr;
    PetscBool   groundstate_solved_ = PETSC_FALSE;

    Mat         dm_left = nullptr;
    Mat         dm_right = nullptr;

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

    /* From ground state, construct the left and right reduced density matrices */
    PetscErrorCode BuildReducedDensityMatrices();


    /* Miscellaneous functions */
    PetscErrorCode MatPeekOperators();
    PetscErrorCode MatSaveOperators();

};

#endif // __IDMRG_HPP__
