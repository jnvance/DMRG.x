#ifndef __IDMRG_HPP__
#define __IDMRG_HPP__

#include <slepceps.h>
#include "dmrgblock.hpp"

/**
    @defgroup   idmrg   iDMRG
    @brief      Implements the iDMRG class

    @addtogroup idmrg
    @{
 */

/**
    Contains the objects needed to perform the infinite-size DMRG

    _Note:_ This class is currently implemented in 1D

    _TODO:_ Further generalizations of the class must be done as more site operators
    are introduced for other spin systems.
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

    /**
        Real part of ground state eigenvector, or the full complex-valued vector.

        When compiled with real PetscScalar (the default configuration), this contains only the real part of the vector.
        Otherwise, if the current PetscArch is compiled with the flag `--with-scalar-type=complex`,
        this vector contains the full complex-valued vector.
     */
    Vec         gsv_r_ = nullptr;

    /**
        Imaginary part of the ground state eigenvector.
        When compiled with real PetscScalar, this contains only the real part of the vector.
        Otherwise, with complex PetscScalar, this is ignored during the entire program.
     */
    Vec         gsv_i_ = nullptr;

    /**
        Tells whether SolveGroundState() has been succesfully run. Also it indicates whether
        the groundstate in gsv_r and/or gsv_i have been succesfully solved
     */
    PetscBool   groundstate_solved_ = PETSC_FALSE;

    /** Density matrix for the left block */
    Mat         dm_left = nullptr;

    /** Density matrix for the right block */
    Mat         dm_right = nullptr;

    MPI_Comm    comm_ = PETSC_COMM_WORLD; /** MPI communicator for distributed arrays */

    Mat eye1_;  /**< 2x2 identity matrix */
    Mat Sz1_;   /**< Single-site \f$ S_z \f$ operator as a 2x2 matrix */
    Mat Sp1_;   /**< Single-site \f$ S_+ \f$ operator as a 2x2 matrix */
    Mat Sm1_;   /**< Single-site \f$ S_- \f$ operator as a 2x2 matrix */

public:

    /**
        Explicit initializer
     */
    PetscErrorCode init(MPI_Comm = PETSC_COMM_WORLD);

    /**
        Explicit destructor
     */
    PetscErrorCode destroy();

    PetscInt LengthBlockLeft(){ return BlockLeft_.length(); }
    PetscInt LengthBlockRight(){ return BlockRight_.length(); }

    /* Block enlargement to be implemented in inherited classes */
    virtual PetscErrorCode BuildBlockLeft()=0;
    virtual PetscErrorCode BuildBlockRight()=0;
    virtual PetscErrorCode BuildSuperBlock()=0;

    /**
        Solve the eigenenergy and eigenvector of the ground state.
     */
    PetscErrorCode SolveGroundState(PetscReal& gse_r, PetscReal& gse_i, PetscReal& error);

    /* From ground state, construct the left and right reduced density matrices */
    PetscErrorCode BuildReducedDensityMatrices();


    /* Miscellaneous functions */
    PetscErrorCode MatPeekOperators();
    PetscErrorCode MatSaveOperators();

};

/** @}*/

#endif // __IDMRG_HPP__
