#ifndef __IDMRG_HPP__
#define __IDMRG_HPP__

#include <slepceps.h>
#include <petsctime.h>
#include "dmrgblock.hpp"

#ifdef __TIMINGS
    #define DMRG_TIMINGS_START(FUNC_NAME) \
        PetscLogDouble funct_time0 ## FUNC_NAME, funct_time ## FUNC_NAME; \
        ierr = PetscTime(&funct_time0 ## FUNC_NAME); CHKERRQ(ierr);

    #define DMRG_TIMINGS_END(FUNC_NAME) \
        ierr = PetscTime(&funct_time ## FUNC_NAME); CHKERRQ(ierr); \
        funct_time ## FUNC_NAME = funct_time ## FUNC_NAME - funct_time0 ## FUNC_NAME; \
        ierr = PetscFPrintf(PETSC_COMM_WORLD, fp_timings, "%10d      %-50s %.20g\n", iter_, FUNC_NAME, funct_time ## FUNC_NAME);
#else
    #define DMRG_TIMINGS_START(FUNC_NAME)
    #define DMRG_TIMINGS_END(FUNC_NAME)
#endif


#ifdef __DMRG_SUB_TIMINGS

    #define DMRG_SUB_TIMINGS_START(SECTION_LABEL) \
        PetscLogDouble subfunct_time0 ## SECTION_LABEL, subfunct_time ## SECTION_LABEL; \
        ierr = PetscTime(&subfunct_time0 ## SECTION_LABEL); CHKERRQ(ierr);

    #define DMRG_SUB_TIMINGS_END(SECTION_LABEL) \
        ierr = PetscTime(&subfunct_time ## SECTION_LABEL); CHKERRQ(ierr); \
        subfunct_time ## SECTION_LABEL = subfunct_time ## SECTION_LABEL - subfunct_time0 ## SECTION_LABEL; \
        ierr = PetscPrintf(PETSC_COMM_WORLD, "%8s %-50s %.20g\n", "", SECTION_LABEL, subfunct_time ## SECTION_LABEL);

    /* Inspect accumulated timings for a section of code inside a loop */

    #define DMRG_SUB_TIMINGS_ACCUM_INIT(SECTION_LABEL) \
        PetscLogDouble subfunct_time0 ## SECTION_LABEL, subfunct_time1 ## SECTION_LABEL, subfunct_time ## SECTION_LABEL = 0.0;

    #define DMRG_SUB_TIMINGS_ACCUM_START(SECTION_LABEL) \
        ierr = PetscTime(&subfunct_time0 ## SECTION_LABEL); CHKERRQ(ierr);

    #define DMRG_SUB_TIMINGS_ACCUM_END(SECTION_LABEL) \
        ierr = PetscTime(&subfunct_time1 ## SECTION_LABEL); CHKERRQ(ierr); \
        subfunct_time ## SECTION_LABEL += subfunct_time1 ## SECTION_LABEL - subfunct_time0 ## SECTION_LABEL; \

    #define DMRG_SUB_TIMINGS_ACCUM_PRINT(SECTION_LABEL) \
        ierr = PetscPrintf(PETSC_COMM_WORLD, "%8s %-50s %.20g\n", "", SECTION_LABEL, subfunct_time ## SECTION_LABEL);

#else
    #define DMRG_SUB_TIMINGS_INIT(SECTION_LABEL)
    #define DMRG_SUB_TIMINGS_START(SECTION_LABEL)
    #define DMRG_SUB_TIMINGS_END(SECTION_LABEL)
    #define DMRG_SUB_TIMINGS_ACCUM_INIT(SECTION_LABEL)
    #define DMRG_SUB_TIMINGS_ACCUM_START(SECTION_LABEL)
    #define DMRG_SUB_TIMINGS_ACCUM_END(SECTION_LABEL)
    #define DMRG_SUB_TIMINGS_ACCUM_PRINT(SECTION_LABEL)
#endif


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
        Dimension of the local hilbert space
    */
    PetscInt    local_dim_;

    /**
        Target number of sites.
    */
    PetscInt    final_nsites_;

    /**
        Target number of states.
    */
    PetscInt    mstates_;

    /**
        Target number of steps.
    */
    PetscInt    nsteps_;

    /**
        Flag for when parameters have been set
    */
    PetscBool   parameters_set = PETSC_FALSE;

    /**
        Completed number of steps.
    */
    PetscInt    iter_ = 0;

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
        Counts the number of truncations performed
     */
    PetscInt    ntruncations_ = 0;

    /**
        Rotation matrix formed from the singular vectors of the largest
        singular values of dm_left
     */
    Mat         U_left_ = nullptr;

    /**
        Rotation matrix formed from the singular vectors of the largest
        singular values of dm_right
     */
    Mat         U_right_ = nullptr;

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

    /**
        Internal function to check whether parameters have been set
    */
    PetscErrorCode CheckSetParameters();

public:

    /**
        Explicit initializer
     */
    PetscErrorCode init(MPI_Comm comm = PETSC_COMM_WORLD, PetscInt nsites = 100, PetscInt mstates = 20);

    /**
        Explicit destructor
     */
    PetscErrorCode destroy();

    /**
        Returns the number of sites in the left block
     */
    PetscInt LengthBlockLeft()
    {
        return BlockLeft_.length();
    }

    /**
        Returns the number of sites in the right block
     */
    PetscInt LengthBlockRight()
    {
        return BlockRight_.length();
    }

    /**
        Returns the total number of sites
     */
     PetscInt TotalLength()
    {
        return LengthBlockLeft() + LengthBlockRight();
    }

    /**
        Returns the total basis size
     */
    PetscInt TotalBasisSize()
    {
        if(BlockLeft_.is_valid() && BlockRight_.is_valid()){
            return BlockLeft_.basis_size() * BlockRight_.basis_size();
        } else {
            return -1;
        }
    }

    /**
        Returns the target number of sites
     */
    PetscInt TargetLength()
    {
        return final_nsites_;
    }

    /**
        Returns the target number of states
     */
    PetscInt mstates()
    {
        return mstates_;
    }

    /**
        Returns the number of dimensions of the local Hilbert space
     */
    PetscInt local_dim()
    {
        return local_dim_;
    }

    /**
     *  Reference to iteration number
     */
    PetscInt& iter()
    {
        return iter_;
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
        Construct the rotation matrices for truncating the block and spin operators.
     */
    PetscErrorCode GetRotationMatrices();

    /**

     */
    PetscErrorCode TruncateOperators();

    /**
        Printout operator matrices to standard output
     */
    PetscErrorCode MatPeekOperators();

    /**
        Save operator matrices to subfolder
     */
    PetscErrorCode MatSaveOperators();

    /**
        Timings
     */
    FILE *fp_timings;

};

/** @}*/

#endif // __IDMRG_HPP__
