#ifndef __DMRGBLOCK_HPP__
#define __DMRGBLOCK_HPP__

/** @defgroup DMRG

    @brief Implementation of the DMRGBlock and iDMRG classes

 */


#include <slepceps.h>
#include <petscmat.h>


#define     DMRGBLOCK_DEFAULT_LENGTH        1
#define     DMRGBLOCK_DEFAULT_BASIS_SIZE    2
#define     DMRG_DEFAULT_MPI_COMM           PETSC_COMM_WORLD

/** @defgroup DMRG

 */
class DMRGBlock
{
    Mat             H_;     /* Hamiltonian of entire block */
    Mat             Sz_;    /* Operators of rightmost/leftmost spin */
    Mat             Sp_;

    PetscInt        length_;
    PetscInt        basis_size_;

    MPI_Comm        comm_;

public:

    DMRGBlock() :   length_(DMRGBLOCK_DEFAULT_LENGTH),
                    basis_size_(DMRGBLOCK_DEFAULT_BASIS_SIZE),
                    comm_(DMRG_DEFAULT_MPI_COMM)
                    {}  // constructor with spin 1/2 defaults

    ~DMRGBlock(){}

    // PetscErrorCode  init(MPI_Comm, PetscInt, PetscInt);     /* explicit initializer */
    PetscErrorCode init( MPI_Comm comm = DMRG_DEFAULT_MPI_COMM,
                                PetscInt length = DMRGBLOCK_DEFAULT_LENGTH,
                                PetscInt basis_size = DMRGBLOCK_DEFAULT_BASIS_SIZE);

    PetscErrorCode  destroy();                              /* explicit destructor */

    Mat  H()        {return H_;}
    Mat  Sz()       {return Sz_;}
    Mat  Sp()       {return Sp_;}

    PetscErrorCode  update_operators(Mat H_new, Mat Sz_new, Mat Sp_new);

    PetscErrorCode  update_H(Mat H_new);
    PetscErrorCode  update_Sz(Mat Sz_new);
    PetscErrorCode  update_Sp(Mat Sp_new);


    PetscInt        length(){return length_;}
    PetscInt        basis_size(){return basis_size_;}

    void            length(PetscInt _length){length_ = _length;}

};


/*--------------------------------------------------------------------------------*/


#endif // __DMRGBLOCK_HPP__
