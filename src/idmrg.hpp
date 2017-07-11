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

    _TODO:_ Implement iteration memeber using final_nsites_, nsteps_ and iter_
 */
class iDMRG
{

protected:

    /**
        Target number of sites.
    */
    PetscInt    final_nsites_;

    /**
        Target number of steps.
    */
    PetscInt    nsteps_;

    /**
        Completed number of steps.
    */
    PetscInt    iter_;

    /**
        DMRGBlock object representing the left block of sites
     */
    DMRGBlock   BlockLeft_;

    /**
        DMRGBlock object representing the right block of sites
     */
    DMRGBlock   BlockRight_;

    /**
        Matrix operator containing the superblock Hamiltonian
     */
    Mat         superblock_H_ = nullptr;

    /**
        Tells whether the superblock Hamiltonian has been successfully constructed
     */
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

        Note: The current implementation only handles the complex PetscScalar and will continue to
        do so. Thus, this object will be removed in the future.
     */
    Vec         gsv_i_ = nullptr;

    /**
        Tells whether SolveGroundState() has been succesfully run. Also it indicates whether
        the groundstate in gsv_r and/or gsv_i have been succesfully solved
     */
    PetscBool   groundstate_solved_ = PETSC_FALSE;

    /**
        Stores the ground state vector as a C-style or row-based matrix which may be populated
        using VecReshapeToLocalMat
     */
    Mat         gsv_mat_seq;


    /**
        Density matrix for the left block
     */
    Mat         dm_left = nullptr;

    /**
        Density matrix for the right block
     */
    Mat         dm_right = nullptr;

    /**
        Tells whether BuildReducedDensityMatrices() has been succesfully run
        and dm_left and dm_right are in the correct state
     */
    PetscBool   dm_solved = PETSC_FALSE;

    /**
        Tells whether SVDReducedDensityMatrices() has been succesfully run
        and the SVD of the reduced density matrices has been solved
     */
    PetscBool   dm_svd = PETSC_FALSE;

    /**
        MPI communicator for distributed arrays
    */
    MPI_Comm    comm_ = PETSC_COMM_WORLD;

    /**
        2x2 identity matrix
    */
    Mat eye1_;
    /**
        Single-site \f$ S_z \f$ operator as a 2x2 matrix
    */
    Mat Sz1_;
    /**
        Single-site \f$ S_+ \f$ operator as a 2x2 matrix
    */
    Mat Sp1_;
    /**
        Single-site \f$ S_- \f$ operator as a 2x2 matrix
    */
    Mat Sm1_;

public:

    /**
        Explicit initializer
     */
    PetscErrorCode init(MPI_Comm = PETSC_COMM_WORLD);

    /**
        Explicit destructor
     */
    PetscErrorCode destroy();

    /**
        Returns the number of sites in the left block;
     */
    PetscInt LengthBlockLeft()
    {
        return BlockLeft_.length();
    }

    /**
        Returns the number of sites in the right block;
     */
    PetscInt LengthBlockRight()
    {
        return BlockRight_.length();
    }

    /**
        Left block enlargement. To be implemented in inherited classes
     */
    virtual PetscErrorCode BuildBlockLeft()=0;

    /**
        Right block enlargement. To be implemented in inherited classes
     */
    virtual PetscErrorCode BuildBlockRight()=0;

    /**
        Right block enlargement. To be implemented in inherited classes
     */
    virtual PetscErrorCode BuildSuperBlock()=0;

    /**
        Solve the eigenenergy and eigenvector of the ground state.
     */
    PetscErrorCode SolveGroundState(PetscReal& gse_r, PetscReal& gse_i, PetscReal& error);

    /**
        From ground state, construct the left and right reduced density matrices
     */
    PetscErrorCode BuildReducedDensityMatrices();

    /**
        Get the SVD of the left and right reduced density matrices.
     */
    PetscErrorCode SVDReducedDensityMatrices();

    /**
        Printout operator matrices to standard output
     */
    PetscErrorCode MatPeekOperators();

    /**
        Save operator matrices to subfolder
     */
    PetscErrorCode MatSaveOperators();

};

/** @}*/

#endif // __IDMRG_HPP__
