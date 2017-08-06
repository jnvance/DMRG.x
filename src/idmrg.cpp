#include "idmrg.hpp"
#include "linalg_tools.hpp"


#undef __FUNCT__
#define __FUNCT__ "iDMRG::init"
PetscErrorCode iDMRG::init(MPI_Comm comm, PetscInt nsites, PetscInt mstates)
{
    PetscErrorCode  ierr = 0;
    DMRG_TIMINGS_START(__FUNCT__);

    comm_ = comm;
    mstates_ = mstates;
    final_nsites_ = nsites;

    /* Initialize block objects */
    ierr = BlockLeft_.init(comm_); CHKERRQ(ierr);
    ierr = BlockRight_.init(comm_); CHKERRQ(ierr);

    /* Initialize single-site operators */
    MatEyeCreate(comm, eye1_, 2);
    MatSzCreate(comm, Sz1_);
    MatSpCreate(comm, Sp1_);
    MatTranspose(Sp1_, MAT_INITIAL_MATRIX, &Sm1_);

    /* Initialize log file for timings */
    #ifdef __TIMINGS
        ierr = PetscFOpen(PETSC_COMM_WORLD, "timings.dat", "w", &fp_timings); CHKERRQ(ierr);
    #endif

    DMRG_TIMINGS_END(__FUNCT__);
    return ierr;
}

#undef __FUNCT__
#define __FUNCT__ "iDMRG::destroy"
PetscErrorCode iDMRG::destroy()
{
    PetscErrorCode  ierr = 0;
    DMRG_TIMINGS_START(__FUNCT__);


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

    DMRG_TIMINGS_END(__FUNCT__);
    /*
        Close log files after ending timings
        otherwise, this causes a segmentation fault
     */
    #ifdef __TIMINGS
        ierr = PetscFClose(PETSC_COMM_WORLD, fp_timings); CHKERRQ(ierr);
    #endif

    return ierr;
}



#undef __FUNCT__
#define __FUNCT__ "iDMRG::SolveGroundState"
PetscErrorCode iDMRG::SolveGroundState(PetscReal& gse_r, PetscReal& gse_i, PetscReal& error)
{
    PetscErrorCode ierr = 0;
    DMRG_TIMINGS_START(__FUNCT__);

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
    // ierr = EPSSetType(eps, EPSKRYLOVSCHUR);
    // ierr = EPSSetDimensions(eps, 4, PETSC_DECIDE, PETSC_DECIDE);
    // ierr = EPSSetTolerances(eps, 1.0e-20, 200);

    ierr = EPSSetFromOptions(eps); CHKERRQ(ierr);
    ierr = EPSSolve(eps); CHKERRQ(ierr);

    PetscInt nconv;
    ierr = EPSGetConverged(eps,&nconv);CHKERRQ(ierr);

    if (gsv_r_) VecDestroy(&gsv_r_);
    if (gsv_i_) VecDestroy(&gsv_i_);
    ierr = MatCreateVecs(superblock_H_, &gsv_r_, nullptr); CHKERRQ(ierr);

    /* TODO: Verify that this works */
    #if defined(PETSC_USE_COMPLEX)
        gsv_i_ = nullptr;
    #else
        ierr = MatCreateVecs(superblock_H_,&gsv_i_,nullptr); CHKERRQ(ierr);
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
        // superblock_set_ = PETSC_FALSE; // See note below
    }
    else
    {
        PetscPrintf(PETSC_COMM_WORLD,"Warning: EPS did not converge.");
    }

    #ifdef __TESTING
        #define __SAVE_SUPERBLOCK
    #endif

    #ifdef __SAVE_SUPERBLOCK
        char filename[PETSC_MAX_PATH_LEN];
        sprintf(filename,"data/superblock_H_%06d.dat",iter());
        MatWrite(superblock_H_,filename);
        sprintf(filename,"data/gsv_r_%06d.dat",iter());
        VecWrite(gsv_r_,filename);
        #ifndef PETSC_USE_COMPLEX
            sprintf(filename,"data/gsv_i_%06d.dat",iter());
            VecWrite(gsv_i_,filename);
        #endif
    #endif // __SAVE_SUPERBLOCK

    /*
        Retain superblock_H_ matrix
        Destroy it only when it is needed to be rebuilt
    */
    // MatDestroy(&superblock_H_);

    EPSDestroy(&eps);

    DMRG_TIMINGS_END(__FUNCT__);
    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "iDMRG::BuildReducedDensityMatrices"
PetscErrorCode iDMRG::BuildReducedDensityMatrices()
{
    PetscErrorCode  ierr = 0;
    DMRG_TIMINGS_START(__FUNCT__);

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
    ierr = VecReshapeToLocalMat(gsv_r_, gsv_mat_seq, size_left, size_right); CHKERRQ(ierr);
    ierr = MatMultSelfHC(gsv_mat_seq, dm_left, PETSC_TRUE); CHKERRQ(ierr);
    ierr = MatMultSelfHC(gsv_mat_seq, dm_right, PETSC_FALSE); CHKERRQ(ierr);

    /*
        Toggle switches
    */
    groundstate_solved_ = PETSC_FALSE;
    dm_solved = PETSC_TRUE;
    /*
        Destroy ground state vectors and matrix
    */
    if (gsv_r_) VecDestroy(&gsv_r_); gsv_r_ = nullptr;
    if (gsv_i_) VecDestroy(&gsv_i_); gsv_i_ = nullptr;
    if (gsv_mat_seq) MatDestroy(&gsv_mat_seq); gsv_mat_seq = nullptr;

    DMRG_TIMINGS_END(__FUNCT__);
    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "iDMRG::GetRotationMatrices"
PetscErrorCode iDMRG::GetRotationMatrices()
{
    PetscErrorCode  ierr = 0;
    DMRG_TIMINGS_START(__FUNCT__);


    if(!(dm_left && dm_right && dm_solved))
        SETERRQ(comm_, 1, "Reduced density matrices not yet solved.");

    PetscScalar trunc_error_left, trunc_error_right;
    FILE *fp_left = nullptr, *fp_right = nullptr;

    #ifdef __TESTING
        char filename[PETSC_MAX_PATH_LEN];
        sprintf(filename,"data/dm_left_singularvalues_%06d.dat",iter());
        ierr = PetscFOpen(PETSC_COMM_WORLD, filename, "w", &fp_left); CHKERRQ(ierr);
        sprintf(filename,"data/dm_right_singularvalues_%06d.dat",iter());
        ierr = PetscFOpen(PETSC_COMM_WORLD, filename, "w", &fp_right); CHKERRQ(ierr);
    #endif

    #ifdef __SVD_USE_EPS
        ierr = EPSLargestEigenpairs(dm_left, mstates_, trunc_error_left, U_left_,fp_left); CHKERRQ(ierr);
        ierr = EPSLargestEigenpairs(dm_right, mstates_, trunc_error_right, U_right_,fp_right); CHKERRQ(ierr);
    #else
        ierr = SVDLargestStates(dm_left, mstates_, trunc_error_left, U_left_,fp_left); CHKERRQ(ierr);
        ierr = SVDLargestStates(dm_right, mstates_, trunc_error_right, U_right_,fp_right); CHKERRQ(ierr);
    #endif

    #ifdef __TESTING
        ierr = PetscFClose(PETSC_COMM_WORLD, fp_left); CHKERRQ(ierr);
        ierr = PetscFClose(PETSC_COMM_WORLD, fp_right); CHKERRQ(ierr);
    #endif


    #ifdef __PRINT_TRUNCATION_ERROR
        ierr = PetscPrintf(comm_,
            "%12sTruncation error (left):  %12e\n",
            " ", trunc_error_left);

        ierr = PetscPrintf(comm_,
            "%12sTruncation error (right): %12e\n",
            " ", trunc_error_right); CHKERRQ(ierr);
    #endif

    dm_solved = PETSC_FALSE;
    dm_svd = PETSC_TRUE;

    #ifdef __TESTING
        sprintf(filename,"data/dm_left_%06d.dat",iter());
        ierr = MatWrite(dm_left, filename); CHKERRQ(ierr);
        sprintf(filename,"data/dm_right_%06d.dat",iter());
        ierr = MatWrite(dm_right, filename); CHKERRQ(ierr);
        sprintf(filename,"data/U_left_%06d.dat",iter());
        ierr = MatWrite(U_left_, filename); CHKERRQ(ierr);
        sprintf(filename,"data/U_right_%06d.dat",iter());
        ierr = MatWrite(U_right_, filename); CHKERRQ(ierr);
    #endif

    if (dm_left)   {ierr = MatDestroy(&dm_left); CHKERRQ(ierr);}
    if (dm_right)  {ierr = MatDestroy(&dm_right); CHKERRQ(ierr);}

    DMRG_TIMINGS_END(__FUNCT__);
    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "iDMRG::TruncateOperators"
PetscErrorCode iDMRG::TruncateOperators()
{
    PetscErrorCode ierr = 0;
    DMRG_TIMINGS_START(__FUNCT__);

    if(!(dm_svd && U_left_ && U_right_))
        SETERRQ(comm_, 1, "SVD of reduced density matrices not yet solved.");

    /* Save operator state before rotation */
    #ifdef __CHECK_ROTATION
        char filename[PETSC_MAX_PATH_LEN];

        sprintf(filename,"data/H_left_pre_%06d.dat",iter());
        MatWrite(BlockLeft_.H(), filename);

        sprintf(filename,"data/Sz_left_pre_%06d.dat",iter());
        MatWrite(BlockLeft_.Sz(), filename);

        sprintf(filename,"data/Sp_left_pre_%06d.dat",iter());
        MatWrite(BlockLeft_.Sp(), filename);

        sprintf(filename,"data/H_right_pre_%06d.dat",iter());
        MatWrite(BlockRight_.H(), filename);

        sprintf(filename,"data/Sz_right_pre_%06d.dat",iter());
        MatWrite(BlockRight_.Sz(), filename);

        sprintf(filename,"data/Sp_right_pre_%06d.dat",iter());
        MatWrite(BlockRight_.Sp(), filename);

    #endif // __CHECK_ROTATION


    /* Rotation */
    Mat mat_temp = nullptr;
    Mat U_hc = nullptr;

    ierr = MatHermitianTranspose(U_left_, MAT_INITIAL_MATRIX, &U_hc); CHKERRQ(ierr);

    ierr = MatMatMatMult(U_hc, BlockLeft_.H(), U_left_, MAT_INITIAL_MATRIX, PETSC_DECIDE, &mat_temp); CHKERRQ(ierr);
    ierr = BlockLeft_.update_H(mat_temp); CHKERRQ(ierr);
    ierr = MatMatMatMult(U_hc, BlockLeft_.Sz(), U_left_, MAT_INITIAL_MATRIX, PETSC_DECIDE, &mat_temp); CHKERRQ(ierr);
    ierr = BlockLeft_.update_Sz(mat_temp); CHKERRQ(ierr);
    ierr = MatMatMatMult(U_hc, BlockLeft_.Sp(), U_left_, MAT_INITIAL_MATRIX, PETSC_DECIDE, &mat_temp); CHKERRQ(ierr);
    ierr = BlockLeft_.update_Sp(mat_temp); CHKERRQ(ierr);

    ierr = MatDestroy(&U_hc); CHKERRQ(ierr);

    ierr = MatHermitianTranspose(U_right_, MAT_INITIAL_MATRIX, &U_hc); CHKERRQ(ierr);

    ierr = MatMatMatMult(U_hc, BlockRight_.H(), U_right_, MAT_INITIAL_MATRIX, PETSC_DECIDE, &mat_temp); CHKERRQ(ierr);
    ierr = BlockRight_.update_H(mat_temp); CHKERRQ(ierr);
    ierr = MatMatMatMult(U_hc, BlockRight_.Sz(), U_right_, MAT_INITIAL_MATRIX, PETSC_DECIDE, &mat_temp); CHKERRQ(ierr);
    ierr = BlockRight_.update_Sz(mat_temp); CHKERRQ(ierr);
    ierr = MatMatMatMult(U_hc, BlockRight_.Sp(), U_right_, MAT_INITIAL_MATRIX, PETSC_DECIDE, &mat_temp); CHKERRQ(ierr);
    ierr = BlockRight_.update_Sp(mat_temp); CHKERRQ(ierr);

    ierr = MatDestroy(&U_hc); CHKERRQ(ierr);

    if(mat_temp)    {ierr = MatDestroy(&mat_temp); CHKERRQ(ierr);}
    if(U_left_)     {ierr = MatDestroy(&U_left_); CHKERRQ(ierr);}
    if(U_right_)    {ierr = MatDestroy(&U_right_); CHKERRQ(ierr);}


    /* Save operator state after rotation */

    #ifdef __CHECK_ROTATION
        sprintf(filename,"data/H_left_post_%06d.dat",iter());
        MatWrite(BlockLeft_.H(), filename);

        sprintf(filename,"data/Sz_left_post_%06d.dat",iter());
        MatWrite(BlockLeft_.Sz(), filename);

        sprintf(filename,"data/Sp_left_post_%06d.dat",iter());
        MatWrite(BlockLeft_.Sp(), filename);

        sprintf(filename,"data/H_right_post_%06d.dat",iter());
        MatWrite(BlockRight_.H(), filename);

        sprintf(filename,"data/Sz_right_post_%06d.dat",iter());
        MatWrite(BlockRight_.Sz(), filename);

        sprintf(filename,"data/Sp_right_post_%06d.dat",iter());
        MatWrite(BlockRight_.Sp(), filename);
    #endif // __CHECK_ROTATION
    #undef __CHECK_ROTATION


    DMRG_TIMINGS_END(__FUNCT__);
    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "iDMRG::MatPeekOperators"
PetscErrorCode iDMRG::MatPeekOperators()
{
    PetscErrorCode  ierr = 0;
    DMRG_TIMINGS_START(__FUNCT__);


    PetscPrintf(comm_, "\nLeft Block Operators\nBlock Length = %d\n", BlockLeft_.length());
    ierr = MatPeek(BlockLeft_.H(), "H (left)");
    ierr = MatPeek(BlockLeft_.Sz(), "Sz (left)");
    ierr = MatPeek(BlockLeft_.Sp(), "Sp (left)");

    PetscPrintf(comm_, "\nRight Block Operators\nBlock Length = %d\n", BlockRight_.length());
    ierr = MatPeek(BlockRight_.H(), "H (right)");
    ierr = MatPeek(BlockRight_.Sz(), "Sz (right)");
    ierr = MatPeek(BlockRight_.Sp(), "Sp (right)");

    if (superblock_H_ && (superblock_set_ == PETSC_TRUE)){
        PetscPrintf(comm_, "\nSuperblock\nBlock Length = %d\n", BlockLeft_.length() + BlockRight_.length());
        ierr = MatPeek(superblock_H_, "H (superblock)"); CHKERRQ(ierr);
    }

    DMRG_TIMINGS_END(__FUNCT__);
    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "iDMRG::MatSaveOperators"
PetscErrorCode iDMRG::MatSaveOperators()
{
    PetscErrorCode  ierr = 0;
    DMRG_TIMINGS_START(__FUNCT__);

    char filename[PETSC_MAX_PATH_LEN];
    char extended[PETSC_MAX_PATH_LEN];

    if (superblock_set_==PETSC_TRUE){
        sprintf(extended,"_ext_");
    } else {
        sprintf(extended,"_");
    }

    sprintf(filename,"data/H_left%s%06d.dat",extended,iter());
    ierr = MatWrite(BlockLeft_.H(), filename); CHKERRQ(ierr);

    sprintf(filename,"data/Sz_left%s%06d.dat",extended,iter());
    ierr = MatWrite(BlockLeft_.Sz(), filename); CHKERRQ(ierr);

    sprintf(filename,"data/Sp_left%s%06d.dat",extended,iter());
    ierr = MatWrite(BlockLeft_.Sp(), filename); CHKERRQ(ierr);

    sprintf(filename,"data/H_right%s%06d.dat",extended,iter());
    ierr = MatWrite(BlockRight_.H(), filename); CHKERRQ(ierr);

    sprintf(filename,"data/Sz_right%s%06d.dat",extended,iter());
    ierr = MatWrite(BlockRight_.Sz(), filename); CHKERRQ(ierr);

    sprintf(filename,"data/Sp_right%s%06d.dat",extended,iter());
    ierr = MatWrite(BlockRight_.Sp(), filename); CHKERRQ(ierr);

    if (superblock_H_ && (superblock_set_ == PETSC_TRUE)){
        sprintf(filename,"data/H_superblock_%06d.dat",iter());
        ierr = MatWrite(superblock_H_, filename); CHKERRQ(ierr);
    }

    DMRG_TIMINGS_END(__FUNCT__);
    return ierr;
}
