#include "idmrg.hpp"
#include "linalg_tools.hpp"


PetscErrorCode iDMRG::init(MPI_Comm comm)
{
    PetscErrorCode  ierr = 0;
    comm_ = comm;

    /* Initialize block objects */
    ierr = BlockLeft_.init(comm_); CHKERRQ(ierr);
    ierr = BlockRight_.init(comm_); CHKERRQ(ierr);

    /* Initialize single-site operators */
    MatEyeCreate(comm, eye1_, 2);
    MatSzCreate(comm, Sz1_);
    MatSpCreate(comm, Sp1_);
    MatTranspose(Sp1_, MAT_INITIAL_MATRIX, &Sm1_);

    return ierr;
}


PetscErrorCode iDMRG::destroy()
{
    PetscErrorCode  ierr = 0;

    /* Destroy block objects */
    ierr = BlockLeft_.destroy(); CHKERRQ(ierr);
    ierr = BlockRight_.destroy(); CHKERRQ(ierr);

    /* Destroy single-site operators */
    MatDestroy(&eye1_);
    MatDestroy(&Sz1_);
    MatDestroy(&Sp1_);
    MatDestroy(&Sm1_);
    MatDestroy(&superblock_H_); /* Do a check whether matrix is in the correct state */

    eye1_ = nullptr;
    Sz1_ = nullptr;
    Sp1_ = nullptr;
    Sm1_ = nullptr;
    superblock_H_ = nullptr;

    return ierr;
}


PetscErrorCode iDMRG::SolveGroundState(PetscReal& gse_r, PetscReal& gse_i, PetscReal& error)
{
    PetscErrorCode ierr = 0;

    /*
        Checkpoint whether superblock Hamiltonian has been set and assembled
    */
    if (superblock_set_ == PETSC_FALSE)
        SETERRQ(comm_, 1, "Superblock Hamiltonian has not been set with BuildSuperBlock().");

    PetscBool assembled;
    ierr = MatAssembled(superblock_H_, &assembled); CHKERRQ(ierr);
    if (assembled == PETSC_FALSE){
        ierr = MatAssemblyBegin(superblock_H_, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(superblock_H_, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }

    /*
        Solve the eigensystem using SLEPC EPS
    */

    EPS eps;
    ierr = EPSCreate(comm_, &eps); CHKERRQ(ierr);
    ierr = EPSSetOperators(eps, superblock_H_, nullptr); CHKERRQ(ierr);
    ierr = EPSSetProblemType(eps, EPS_HEP); CHKERRQ(ierr);
    ierr = EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL);
    ierr = EPSSetDimensions(eps, 1, PETSC_DECIDE, PETSC_DECIDE);

    ierr = EPSSetFromOptions(eps); CHKERRQ(ierr);
    ierr = EPSSolve(eps); CHKERRQ(ierr);

    PetscInt nconv;
    ierr = EPSGetConverged(eps,&nconv);CHKERRQ(ierr);

    if (gsv_r_) VecDestroy(&gsv_r_);
    if (gsv_i_) VecDestroy(&gsv_i_);
    ierr = MatCreateVecs(superblock_H_,nullptr,&gsv_r_); CHKERRQ(ierr);

    /* TODO: Verify that this works */
    #if defined(PETSC_USE_COMPLEX)
        gsv_i_ = nullptr;
    #else
        ierr = MatCreateVecs(superblock_H_,nullptr,&gsv_i_); CHKERRQ(ierr);
    #endif

    PetscScalar kr, ki;

    if (nconv>0)
    {
        /*
            Get converged eigenpairs: 0-th eigenvalue is stored in gse_r (real part) and
            gse_i (imaginary part)

            Note on EPSGetEigenpair():

            If the eigenvalue is real, then eigi and Vi are set to zero. If PETSc is configured
            with complex scalars the eigenvalue is stored directly in eigr (eigi is set to zero)
            and the eigenvector in Vr (Vi is set to zero).
        */

        #if defined(PETSC_USE_COMPLEX)
            ierr = EPSGetEigenpair(eps, 0, &kr, &ki, gsv_r_, nullptr); CHKERRQ(ierr);
            gse_r = PetscRealPart(kr);
            gse_i = PetscImaginaryPart(kr);
        #else
            ierr = EPSGetEigenpair(eps, 0, &kr, &ki, gsv_r_, gsv_i_); CHKERRQ(ierr);
            gse_r = kr;
            gse_i = ki;
        #endif

        ierr = EPSComputeError(eps, 0, EPS_ERROR_RELATIVE, &error);CHKERRQ(ierr);
        groundstate_solved_ = PETSC_TRUE;
        superblock_set_ = PETSC_FALSE;
    }
    else
    {
        PetscPrintf(PETSC_COMM_WORLD,"Warning: EPS did not converge.");
    }

    MatDestroy(&superblock_H_);
    EPSDestroy(&eps);

    return ierr;
}


PetscErrorCode iDMRG::BuildReducedDensityMatrices()
{
    PetscErrorCode  ierr = 0;
    /*
        Determine whether ground state has been solved with SolveGroundState()
     */
    if(groundstate_solved_ == PETSC_FALSE)
        SETERRQ(comm_, 1, "Ground state not yet solved.");
    /*
        Collect information regarding the basis size of the
        left and right blocks
     */
    PetscInt size_left, size_right;
    ierr = MatGetSize(BlockLeft_.H(), &size_left, nullptr); CHKERRQ(ierr);
    ierr = MatGetSize(BlockRight_.H(), &size_right, nullptr); CHKERRQ(ierr);
    /*
        Collect entire groundstate vector to all processes
     */
    #ifdef PETSC_USE_COMPLEX
        ierr = VecToMatMultHC(comm_, gsv_r_, nullptr, dm_left, size_left, size_right, PETSC_TRUE);
        CHKERRQ(ierr);
    #else
        SETERRQ(comm_, 1, "Not implemented for real scalars.");
    #endif

    #ifdef __TESTING
        ierr = VecWrite(comm_, gsv_r_, "data/gsv_r_.dat"); CHKERRQ(ierr);
    #endif
    /*
        Toggle switches
    */
    groundstate_solved_ = PETSC_FALSE;
    dm_solved = PETSC_TRUE;
    /*
        Destroy ground state vectors
    */
    if (gsv_r_) VecDestroy(&gsv_r_); gsv_r_ = nullptr;
    if (gsv_i_) VecDestroy(&gsv_i_); gsv_i_ = nullptr;
    return ierr;
}


PetscErrorCode iDMRG::SVDReducedDensityMatrices()
{
    PetscErrorCode  ierr = 0;

    if(! (dm_left && dm_solved))
        SETERRQ(comm_, 1, "Reduced density matrices not yet solved.");

    MatGetSVD(comm_, dm_left);

    #ifdef __TESTING
        ierr = MatWrite(comm_, dm_left, "data/dm_left.dat"); CHKERRQ(ierr);
    #endif

    dm_solved = PETSC_FALSE;
    dm_svd    = PETSC_TRUE;

    MatDestroy(&dm_left);
    return ierr;
}


PetscErrorCode iDMRG::MatPeekOperators()
{
    PetscErrorCode  ierr = 0;

    PetscPrintf(comm_, "\nLeft Block Operators\nBlock Length = %d\n", BlockLeft_.length());
    ierr = MatPeek(comm_, BlockLeft_.H(), "H (left)");
    ierr = MatPeek(comm_, BlockLeft_.Sz(), "Sz (left)");
    ierr = MatPeek(comm_, BlockLeft_.Sp(), "Sp (left)");

    PetscPrintf(comm_, "\nRight Block Operators\nBlock Length = %d\n", BlockRight_.length());
    ierr = MatPeek(comm_, BlockRight_.H(), "H (right)");
    ierr = MatPeek(comm_, BlockRight_.Sz(), "Sz (right)");
    ierr = MatPeek(comm_, BlockRight_.Sp(), "Sp (right)");

    if (superblock_H_ && (superblock_set_ == PETSC_TRUE)){
        PetscPrintf(comm_, "\nSuperblock\nBlock Length = %d\n", BlockLeft_.length() + BlockRight_.length());
        ierr = MatPeek(comm_, superblock_H_, "H (superblock)"); CHKERRQ(ierr);
    }

    return ierr;
}


PetscErrorCode iDMRG::MatSaveOperators()
{
    PetscErrorCode  ierr = 0;

    ierr = MatWrite(comm_, BlockLeft_.H(), "data/H_left.dat"); CHKERRQ(ierr);
    ierr = MatWrite(comm_, BlockLeft_.Sz(), "data/Sz_left.dat"); CHKERRQ(ierr);
    ierr = MatWrite(comm_, BlockLeft_.Sp(), "data/Sp_left.dat"); CHKERRQ(ierr);

    ierr = MatWrite(comm_, BlockRight_.H(), "data/H_right.dat"); CHKERRQ(ierr);
    ierr = MatWrite(comm_, BlockRight_.Sz(), "data/Sz_right.dat"); CHKERRQ(ierr);
    ierr = MatWrite(comm_, BlockRight_.Sp(), "data/Sp_right.dat"); CHKERRQ(ierr);

    if (superblock_H_ && (superblock_set_ == PETSC_TRUE)){
        ierr = MatWrite(comm_, superblock_H_, "data/H_superblock.dat"); CHKERRQ(ierr);
    }

    return ierr;
}
