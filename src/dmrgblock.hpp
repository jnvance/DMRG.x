#ifndef __DMRGBLOCK_HPP__
#define __DMRGBLOCK_HPP__

/**
    @defgroup   dmrgblock   DMRG Block
    @brief      Implementation of the DMRGBlock class

 */


#include <slepceps.h>
#include <petscmat.h>


/**
    @addtogroup dmrgblock
    @{
 */

/**
    Contains the matrix representations of the operators of a block of spins
    and some useful information on its state.

    _Note:_ This class is currently implemented with the Heisenberg model in mind.

    _TODO:_ Further generalizations of the class must be done as more site operators
    are introduced for other spin systems.
 */
class DMRGBlock
{
    Mat             H_;             /**< Hamiltonian of entire block */
    Mat             Sz_;            /**< Sz operator of rightmost/leftmost spin */
    Mat             Sp_;            /**< S+ operator of rightmost/leftmost spin */

    PetscInt        length_;        /**< Number of sites within the blocks */
    PetscInt        basis_size_;    /**< Number of states used to represent the operators */

    /** MPI communicator for distributed arrays */
    MPI_Comm        comm_ = PETSC_COMM_WORLD;

    /** Default length for 1D spin-1/2 Heisenberg model */
    static const PetscInt DMRGBLOCK_DEFAULT_LENGTH = 1;

    /** Default basis size for 1D spin-1/2 Heisenberg model */
    static const PetscInt DMRGBLOCK_DEFAULT_BASIS_SIZE = 2;

public:

    /** Constructor with spin-\f$1/2\f$ defaults */
    DMRGBlock() :
        length_(DMRGBLOCK_DEFAULT_LENGTH),
        basis_size_(DMRGBLOCK_DEFAULT_LENGTH),
        comm_(PETSC_COMM_WORLD) { }

    ~DMRGBlock(){}

    /** Explicit initializer */
    PetscErrorCode init(MPI_Comm comm = PETSC_COMM_WORLD,
                        PetscInt length = DMRGBLOCK_DEFAULT_LENGTH,
                        PetscInt basis_size = DMRGBLOCK_DEFAULT_BASIS_SIZE);

    /** Explicit destructor */
    PetscErrorCode destroy();

    /**
        Returns a reference to the block Hamiltonian matrix.
        Set to `nullptr` when not in use.
     */
    Mat H()
    {
        return H_;
    }

    /**
        Returns a reference to the \f$S_z\f$ matrix.
        Set to `nullptr` when not in use.
     */
    Mat Sz()
    {
        return Sz_;
    }

    /**
        Returns a reference to the \f$S_+\f$ matrix.
        Set to `nullptr` when not in use.
     */
    Mat Sp()
    {
        return Sp_;
    }

    /**
        Update operators simultaneously

        @param[in]   H_new  Replaces \f$H\f$ matrix
        @param[in]   Sz_new Replaces \f$S_z\f$ matrix
        @param[in]   Sp_new Replaces \f$S_+\f$ matrix

     */
    PetscErrorCode  update_operators(Mat H_new, Mat Sz_new, Mat Sp_new);

    /**
        Update \f$H\f$ matrix

        @param[in]   H_new  Replaces \f$H\f$ matrix
     */
    PetscErrorCode  update_H(Mat H_new);

    /**
        Update \f$S_z\f$ matrix

        @param[in]   Sz_new  Replaces \f$S_z\f$ matrix
     */
    PetscErrorCode  update_Sz(Mat Sz_new);

    /**
        Update \f$S_+\f$ matrix

        @param[in]   Sp_new  Replaces \f$S_+\f$ matrix
    */
    PetscErrorCode  update_Sp(Mat Sp_new);

    /**
        Returns the number of sites within the blocks
     */
    PetscInt length()
    {
        return length_;
    }

    /**
        Returns the number of states used to represent the operators
     */
    PetscInt basis_size()
    {
        return basis_size_;
    }

    /**
        Updates the length of the block
     */
    void length(PetscInt _length)
    {
        length_ = _length;
    }

};

/**
    @}
 */

#endif // __DMRGBLOCK_HPP__
