#ifndef __DMRG_SITE_HPP
#define __DMRG_SITE_HPP

#include <petscmat.h>
#include "kron.hpp"
#include "linalg_tools.hpp"


class Block_SpinOneHalf
{

private:

    MPI_Comm comm;
    PetscMPIInt rank;

    static const PetscInt loc_dim = 2;
    PetscInt mat_op_dim;
    PetscBool init_ = PETSC_FALSE;
    PetscBool verbose = PETSC_FALSE;

    /* FIXME: Optimize matrices for MKL, specify type or use one global matrix type */
    // MatType

    Mat H_  = nullptr;
    Mat Sz_ = nullptr;
    Mat Sp_ = nullptr;
    Mat Sm_ = nullptr;

    /* Will be useful in fDMRG
       represents the matrix rotated only once
     */
    #if 0
    Mat Sz_conn = nullptr;
    Mat Sp_conn = nullptr;
    Mat Sm_conn = nullptr;
    #endif

    PetscErrorCode InitH();
    PetscErrorCode InitSz();
    PetscErrorCode InitSp();

public:

    PetscErrorCode Initialize(const MPI_Comm& comm_in);
    PetscErrorCode Destroy();

    PetscErrorCode UpdateH(Mat& H_in);

    PetscErrorCode CreateSm();
    PetscErrorCode DestroySm();

    PetscErrorCode OpKronEye(PetscInt eye_dim = loc_dim);
    PetscErrorCode EyeKronOp(PetscInt eye_dim = loc_dim);

    PetscErrorCode CheckOperators();

    const Mat& H(){ return H_; }
    const Mat& Sz(){ return Sz_; }
    const Mat& Sp(){ return Sp_; }
    const Mat& Sm(){
        // if(!Sm_) CPP_CHKERRQ_MSG(1, "Sm not yet created.");
        return Sm_;
    }
    PetscInt LocDim() const { return loc_dim; }
    PetscInt MatOpDim() const {
        // PetscErrorCode ierr = 0;
        // ierr = CheckOperators(); CPP_CHKERRQ_MSG(ierr,"Site::CheckOperators error.");
        return mat_op_dim; }
    PetscBool Initialized() const { return init_; }
    MPI_Comm Comm() const { return comm; }
    /**
        Container for the magnetization sectors of a single site
     */
    const std::vector<PetscScalar> single_site_sectors = {0.5, -0.5};

};

#endif // __DMRG_SITE_HPP
