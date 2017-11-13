#include "DMRGBlock.hpp"



/* TODO: Merge into one function differentiated by setting of values */
/* FIXME: Optimize matrices for MKL, specify type or use one global matrix type */

PetscErrorCode Block_SpinOneHalf::InitH()
{
    PetscErrorCode ierr = 0;

    if (H_) SETERRQ(comm, 1, "Sz was previously initialized.");

    ierr = MatCreate(comm, &H_); CHKERRQ(ierr);
    ierr = MatSetSizes(H_, PETSC_DECIDE, PETSC_DECIDE, loc_dim, loc_dim); CHKERRQ(ierr);
    ierr = MatSetFromOptions(H_); CHKERRQ(ierr);
    ierr = MatSetUp(H_); CHKERRQ(ierr);

    ierr = MatAssemblyBegin(H_, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(H_, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    if(verbose && !rank) printf(">>> site::%s\n",__FUNCTION__);

    return ierr;
}


PetscErrorCode Block_SpinOneHalf::InitSz()
{
    PetscErrorCode ierr = 0;

    if (Sz_) SETERRQ(comm, 1, "Sz was previously initialized.");

    ierr = MatCreate(comm, &Sz_); CHKERRQ(ierr);
    ierr = MatSetSizes(Sz_, PETSC_DECIDE, PETSC_DECIDE, loc_dim, loc_dim); CHKERRQ(ierr);
    ierr = MatSetFromOptions(Sz_); CHKERRQ(ierr);
    ierr = MatSetUp(Sz_); CHKERRQ(ierr);

    ierr = MatSetValue(Sz_, 0, 0, +0.5, INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatSetValue(Sz_, 1, 1, -0.5, INSERT_VALUES); CHKERRQ(ierr);

    ierr = MatAssemblyBegin(Sz_, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Sz_, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    if(verbose && !rank) printf(">>> site::%s\n",__FUNCTION__);

    return ierr;
}


PetscErrorCode Block_SpinOneHalf::InitSp()
{
    PetscErrorCode ierr = 0;

    if (Sp_) SETERRQ(comm, 1, "Sz was previously initialized.");

    ierr = MatCreate(comm, &Sp_); CHKERRQ(ierr);
    ierr = MatSetSizes(Sp_, PETSC_DECIDE, PETSC_DECIDE, loc_dim, loc_dim); CHKERRQ(ierr);
    ierr = MatSetFromOptions(Sp_); CHKERRQ(ierr);
    ierr = MatSetUp(Sp_); CHKERRQ(ierr);

    ierr = MatSetValue(Sp_, 0, 1, +1.0, INSERT_VALUES); CHKERRQ(ierr);

    ierr = MatAssemblyBegin(Sp_, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Sp_, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    if(verbose && !rank) printf(">>> site::%s\n",__FUNCTION__);

    return ierr;
}


PetscErrorCode Block_SpinOneHalf::Initialize(const MPI_Comm& comm_in)
{
    PetscErrorCode ierr = 0;

    /* Check whether to do verbose logging */
    ierr = PetscOptionsGetBool(NULL,NULL,"-verbose",&verbose,NULL); CHKERRQ(ierr);

    /* Initialize attributes*/
    comm = comm_in;
    ierr = MPI_Comm_rank(comm, &rank); CPP_CHKERRQ(ierr);

    /* Initialize matrices */
    ierr = InitH(); CHKERRQ(ierr);
    ierr = InitSz(); CHKERRQ(ierr);
    ierr = InitSp(); CHKERRQ(ierr);
    init_ = PETSC_TRUE;
    ierr = CheckOperators(); CHKERRQ(ierr);

    if(verbose && !rank) printf(">>> site::%s\n",__FUNCTION__);

    return ierr;
}


PetscErrorCode Block_SpinOneHalf::Destroy()
{
    PetscErrorCode ierr = 0;

    ierr = CheckOperators(); CHKERRQ(ierr);
    ierr = MatDestroy(&H_); CPP_CHKERRQ(ierr);
    ierr = MatDestroy(&Sz_); CPP_CHKERRQ(ierr);
    ierr = MatDestroy(&Sp_); CPP_CHKERRQ(ierr);
    ierr = MatDestroy(&Sm_); CPP_CHKERRQ(ierr);

    init_ = PETSC_FALSE;

    if(verbose && !rank) printf(">>> site::%s\n",__FUNCTION__);

    return ierr;
}


PetscErrorCode Block_SpinOneHalf::UpdateH(Mat& H_new)
{
    PetscErrorCode  ierr = 0;

    if (H_ == H_new) return ierr;
    Mat H_temp = H_;
    H_ = H_new;
    H_new = nullptr;
    ierr = MatDestroy(&H_temp); CHKERRQ(ierr);

    return ierr;
}


PetscErrorCode Block_SpinOneHalf::CreateSm()
{
    PetscErrorCode ierr = 0;

    if(!init_) SETERRQ(comm, 1, "Site was not initialized.");
    if(!Sp_) SETERRQ(comm, 1, "Sp not initialized.");

    LINALG_TOOLS__MATASSEMBLY_FINAL(Sp_);
    ierr = MatHermitianTranspose(Sp_, MAT_INITIAL_MATRIX, &Sm_); CHKERRQ(ierr);

    if(verbose && !rank) printf(">>> site::%s\n",__FUNCTION__);

    return ierr;
}


PetscErrorCode Block_SpinOneHalf::DestroySm()
{
    PetscErrorCode ierr = 0;

    if(!init_) SETERRQ(comm, 1, "Site was not initialized.");
    ierr = MatDestroy(&Sm_); CHKERRQ(ierr);
    Sm_ = nullptr;

    return ierr;
}


PetscErrorCode Block_SpinOneHalf::CheckOperators()
{
    PetscErrorCode ierr = 0;

    /* Check initialization */
    if(!init_) SETERRQ(comm, 1, "Site was not initialized.");
    if(!Sz_) SETERRQ(comm, 1, "Sz not initialized.");
    if(!Sp_) SETERRQ(comm, 1, "Sp not initialized.");

    /* Check sizes */
    PetscInt M_Sz, N_Sz, M_Sp, N_Sp;
    ierr = MatGetSize(Sz_, &M_Sz, &N_Sz); CHKERRQ(ierr);
    ierr = MatGetSize(Sp_, &M_Sp, &N_Sp); CHKERRQ(ierr);
    if(M_Sz != N_Sz) SETERRQ2(comm, 1, "Sz not square. Current size: (%d,%d)", M_Sz, N_Sz);
    if(M_Sp != N_Sp) SETERRQ2(comm, 1, "Sp not square. Current size: (%d,%d)", M_Sp, N_Sp);
    if(M_Sz != M_Sp) SETERRQ2(comm, 1, "Size of Sz (%d) different from that of Sp (%d).", M_Sz, M_Sp);

    /* Set sizes as operator dimension */
    mat_op_dim = M_Sz;

    return ierr;
}


PetscErrorCode Block_SpinOneHalf::OpKronEye(PetscInt eye_dim)
{
    PetscErrorCode ierr = 0;

    ierr = CheckOperators(); CHKERRQ(ierr);

    Mat eye, mat_kron;
    ierr = MatEyeCreate(comm, eye, eye_dim); CHKERRQ(ierr);

    #define OP_KRON_EYE(MATRIX) \
    ierr = MatKronProd(1.0, MATRIX, eye, mat_kron); CHKERRQ(ierr); \
    ierr = MatAssemblyBegin(MATRIX, MAT_FINAL_ASSEMBLY); \
    ierr = MatAssemblyEnd(MATRIX, MAT_FINAL_ASSEMBLY); \
    ierr = MatDestroy(&MATRIX); CHKERRQ(ierr); \
    MATRIX = mat_kron; \
    mat_kron = nullptr;

    OP_KRON_EYE(Sz_)
    OP_KRON_EYE(Sp_)
    #undef OP_KRON_EYE

    ierr = MatDestroy(&eye);
    ierr = CheckOperators(); CHKERRQ(ierr);

    return ierr;
}


PetscErrorCode Block_SpinOneHalf::EyeKronOp(PetscInt eye_dim)
{
    PetscErrorCode ierr = 0;

    ierr = CheckOperators(); CHKERRQ(ierr);

    Mat eye, mat_kron;
    ierr = MatEyeCreate(comm, eye, eye_dim); CHKERRQ(ierr);

    #define EYE_KRON_OP(MATRIX) \
    ierr = MatKronProd(1.0, eye, MATRIX, mat_kron); CHKERRQ(ierr); \
    ierr = MatAssemblyBegin(MATRIX, MAT_FINAL_ASSEMBLY); \
    ierr = MatAssemblyEnd(MATRIX, MAT_FINAL_ASSEMBLY); \
    ierr = MatDestroy(&MATRIX); CHKERRQ(ierr); \
    MATRIX = mat_kron; \
    mat_kron = nullptr;

    EYE_KRON_OP(Sz_)
    EYE_KRON_OP(Sp_)
    #undef EYE_KRON_OP

    ierr = MatDestroy(&eye);
    ierr = CheckOperators(); CHKERRQ(ierr);

    return ierr;
}


