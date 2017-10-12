#include "idmrg.hpp"
#include "linalg_tools.hpp"


#undef __FUNCT__
#define __FUNCT__ "iDMRG::init"
PetscErrorCode iDMRG::init(MPI_Comm comm, PetscInt nsites, PetscInt mstates)
{
    PetscErrorCode  ierr = 0;
    DMRG_TIMINGS_START(__FUNCT__);

    comm_ = comm;

    ierr = MPI_Comm_rank(comm_, &rank_); CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm_, &nprocs_); CHKERRQ(ierr);

    mstates_ = mstates;
    final_nsites_ = nsites;

    /* Initialize block objects */
    ierr = BlockLeft_.init(comm_); CHKERRQ(ierr);
    ierr = BlockRight_.init(comm_); CHKERRQ(ierr);

    /* Initialize single-site operators */
    ierr = MatEyeCreate(comm, eye1_, 2); CHKERRQ(ierr);
    ierr = MatSzCreate(comm, Sz1_); CHKERRQ(ierr);
    ierr = MatSpCreate(comm, Sp1_); CHKERRQ(ierr);
    ierr = MatTranspose(Sp1_, MAT_INITIAL_MATRIX, &Sm1_); CHKERRQ(ierr);

    /*
        Initialize single-site sectors
        TODO: Transfer definition to spin-dependent class
    */
    single_site_sectors = {0.5, -0.5};
    BlockLeft_.basis_sector_array = single_site_sectors;
    BlockRight_.basis_sector_array = single_site_sectors;

    sector_indices = {};

    #ifdef __TESTING
        #define PRINT_VEC(stdvectorpetscscalar) \
            for (std::vector<PetscScalar>::const_iterator i = stdvectorpetscscalar.begin(); \
                i != stdvectorpetscscalar.end(); ++i) printf("%f\n",PetscRealPart(*i)); \
                printf("\n");
    #else
        #define PRINT_VEC(stdvectorpetscscalar)
    #endif
    /* For debugging */
    // PRINT_VEC(single_site_sectors)
    // PRINT_VEC(BlockLeft_.basis_sector_array)
    // PRINT_VEC(BlockRight_.basis_sector_array)

    #undef PRINT_VEC

    /*
        Check whether to perform SVD on a subset of processes
    */
    ierr = PetscOptionsGetInt(NULL,NULL,"-svd_nsubcomm",&svd_nsubcomm,NULL); CHKERRQ(ierr);
    if(svd_nsubcomm > 1)
    {
        /* Get information on MPI */
        PetscMPIInt nprocs;
        MPI_Comm_size(comm_, &nprocs);

        /* Check whether splitting is viable */
        if(nprocs % svd_nsubcomm)
            SETERRQ2(comm_, 1,  "The number of processes in comm (%d) must be divisible "
                                "by svd_nsubcomm (%d).", nprocs, svd_nsubcomm);

        /* Set object-wide flag */
        do_svd_commsplit = PETSC_TRUE;
    }

    /*
        Check whether to perform SVD on root process
    */
    ierr = PetscOptionsGetBool(NULL,NULL,"-do_svd_on_root",&do_svd_on_root,NULL); CHKERRQ(ierr);


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
    /*
     * Destroy block objects
     */
    ierr = BlockLeft_.destroy(); CHKERRQ(ierr);
    ierr = BlockRight_.destroy(); CHKERRQ(ierr);
    /*
     * Destroy single-site operators
     */
    LINALG_TOOLS__MATDESTROY(eye1_);
    LINALG_TOOLS__MATDESTROY(Sz1_);
    LINALG_TOOLS__MATDESTROY(Sp1_);
    LINALG_TOOLS__MATDESTROY(Sm1_);
    LINALG_TOOLS__MATDESTROY(superblock_H_);
    LINALG_TOOLS__VECDESTROY(gsv_r_);
    LINALG_TOOLS__VECDESTROY(gsv_i_);

    /*
     * Close log files after ending timings otherwise,
     * this causes a segmentation fault
     */
    DMRG_TIMINGS_END(__FUNCT__);

    #ifdef __TIMINGS
        ierr = PetscFClose(PETSC_COMM_WORLD, fp_timings); CHKERRQ(ierr);
    #endif

    return ierr;
}

PetscErrorCode iDMRG::SetTargetSz(PetscReal Sz_in, PetscBool do_target_Sz_in)
{
    PetscErrorCode ierr = 0;
    if(target_Sz_set==PETSC_TRUE)
        SETERRQ(comm_,1,"Target Sz has been set.");
    target_Sz = Sz_in;
    do_target_Sz = do_target_Sz_in;
    target_Sz_set=PETSC_TRUE;
    return ierr;
}

#undef __FUNCT__
#define __FUNCT__ "iDMRG::CheckSetParameters"
PetscErrorCode iDMRG::CheckSetParameters()
{
    PetscErrorCode  ierr = 0;

    if (parameters_set == PETSC_FALSE)
    {
        SETERRQ(comm_, 1, "Parameters not yet set.");
    }

    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "iDMRG::SolveGroundState"
PetscErrorCode iDMRG::SolveGroundState(PetscReal& gse_r, PetscReal& gse_i, PetscReal& error)
{
    PetscErrorCode ierr = 0;
    DMRG_TIMINGS_START(__FUNCT__);
    DMRG_SUB_TIMINGS_START(__FUNCT__);

    /*
        Checkpoint whether superblock Hamiltonian has been set and assembled
    */
    if (superblock_set_ == PETSC_FALSE)
        SETERRQ(comm_, 1, "Superblock Hamiltonian has not been set with BuildSuperBlock().");

    LINALG_TOOLS__MATASSEMBLY_FINAL(superblock_H_);

    PetscInt superblock_H_size;
    ierr = MatGetSize(superblock_H_, nullptr, &superblock_H_size); CHKERRQ(ierr);

    /*
        Solve the eigensystem using SLEPC EPS
    */
    EPS eps;

    ierr = EPSCreate(comm_, &eps); CHKERRQ(ierr);
    ierr = EPSSetOperators(eps, superblock_H_, nullptr); CHKERRQ(ierr);
    ierr = EPSSetProblemType(eps, EPS_HEP); CHKERRQ(ierr);
    ierr = EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL); CHKERRQ(ierr);
    // ierr = EPSSetType(eps, EPSKRYLOVSCHUR); CHKERRQ(ierr);
    // ierr = EPSSetDimensions(eps, 1, PETSC_DECIDE, PETSC_DECIDE); CHKERRQ(ierr);

    /*
        If compatible, use previously solved ground state vector as initial guess
     */
    if ((!do_target_Sz) && gsv_r_ && superblock_H_ && ntruncations_ > 1)
    {
        PetscInt gsv_size;

        ierr = VecGetSize(gsv_r_, &gsv_size); CHKERRQ(ierr);

        if(gsv_size==superblock_H_size){
            ierr = EPSSetInitialSpace(eps, 1, &gsv_r_); CHKERRQ(ierr);
        }
    }

    ierr = EPSSetFromOptions(eps); CHKERRQ(ierr);

    #define __EPS_SOLVE__ "    EPSSolve"
    DMRG_SUB_TIMINGS_START(__EPS_SOLVE__)
    ierr = EPSSolve(eps); CHKERRQ(ierr);
    DMRG_SUB_TIMINGS_END(__EPS_SOLVE__)

    PetscInt nconv;

    ierr = EPSGetConverged(eps,&nconv);CHKERRQ(ierr);
    if (nconv < 1)
        SETERRQ(comm_,1, "EPS did not converge.");

    if (gsv_r_){ ierr = VecDestroy(&gsv_r_); CHKERRQ(ierr); }
    if (gsv_i_){ ierr = VecDestroy(&gsv_i_); CHKERRQ(ierr); }

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

        ierr = EPSComputeError(eps, 0, EPS_ERROR_RELATIVE, &error); CHKERRQ(ierr);
        groundstate_solved_ = PETSC_TRUE;
        // superblock_set_ = PETSC_FALSE; // See note below
    }
    else
    {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"Warning: EPS did not converge."); CHKERRQ(ierr);
    }

    #ifdef __TESTING
        #define __SAVE_SUPERBLOCK
    #endif

    #ifdef __SAVE_SUPERBLOCK
        char filename[PETSC_MAX_PATH_LEN];
        sprintf(filename,"data/superblock_H_%06d.dat",iter());
        ierr = MatWrite(superblock_H_,filename); CHKERRQ(ierr);
        sprintf(filename,"data/gsv_r_%06d.dat",iter());
        ierr = VecWrite(gsv_r_,filename); CHKERRQ(ierr);
        #ifndef PETSC_USE_COMPLEX
            sprintf(filename,"data/gsv_i_%06d.dat",iter());
            ierr = VecWrite(gsv_i_,filename); CHKERRQ(ierr);
        #endif
    #endif // __SAVE_SUPERBLOCK

    /*
        Retain superblock_H_ matrix
        Destroy it only when it is needed to be rebuilt
        Destroy EPS object
    */
    ierr = EPSDestroy(&eps); CHKERRQ(ierr);

    DMRG_SUB_TIMINGS_END(__FUNCT__);
    DMRG_TIMINGS_END(__FUNCT__);
    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "iDMRG::BuildReducedDensityMatrices"
PetscErrorCode iDMRG::BuildReducedDensityMatrices()
{
    PetscErrorCode  ierr = 0;
    DMRG_TIMINGS_START(__FUNCT__);
    DMRG_SUB_TIMINGS_START(__FUNCT__);

    /*
        Determine whether ground state has been solved with SolveGroundState()
     */
    if(groundstate_solved_ == PETSC_FALSE)
        SETERRQ(comm_, 1, "Ground state not yet solved.");

    if (do_target_Sz) {

        /* Clear rho_block_dict for both blocks */
        if(BlockLeft_.rho_block_dict.size())
            for (auto item: BlockLeft_.rho_block_dict)
                MatDestroy(&item.second);

        if(BlockRight_.rho_block_dict.size())
            for (auto item: BlockRight_.rho_block_dict)
                MatDestroy(&item.second);

        BlockLeft_.rho_block_dict.clear();
        BlockRight_.rho_block_dict.clear();

        /* Using VecScatter gather all elements of gsv */
        Vec         vec = gsv_r_;
        Vec         vec_seq;
        VecScatter  ctx;

        ierr = VecScatterCreateToAll(vec, &ctx, &vec_seq); CHKERRQ(ierr);
        ierr = VecScatterBegin(ctx, vec, vec_seq, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);
        ierr = VecScatterEnd(ctx, vec, vec_seq, INSERT_VALUES, SCATTER_FORWARD); CHKERRQ(ierr);

        PetscInt size_left, size_right, size_right2;

        Mat psi0_sector = nullptr;

        for(auto elem: sector_indices)
        {
            const PetscScalar&      sys_enl_Sz = elem.first;
            const PetscScalar       env_enl_Sz = target_Sz - sys_enl_Sz;
            std::vector<PetscInt>&  indices    = elem.second;

            if(indices.size() > 0)
            {
                auto& sys_enl_basis_by_sector = BlockLeft_.basis_by_sector;
                auto& env_enl_basis_by_sector = BlockRight_.basis_by_sector;
                size_left  = sys_enl_basis_by_sector[sys_enl_Sz].size();
                size_right = indices.size() / size_left;
                size_right2= env_enl_basis_by_sector[target_Sz - sys_enl_Sz].size();

                if(size_right != size_right2)
                    SETERRQ(comm_, 1, "Right block dimension mismatch.");

                if((size_t)(size_left*size_right) != indices.size())
                    SETERRQ(comm_, 1, "Reshape dimension mismatch.");

                ierr = LocalVecReshapeToLocalMat(
                    vec_seq, psi0_sector, size_left, size_right, indices); CHKERRQ(ierr);

                /* Decide whether to do build RDMs serially on 0th process */

                MPI_Comm comm = comm_;
                if (do_svd_on_root)
                {
                    if (rank_ != 0) continue;
                    comm = PETSC_COMM_SELF;
                }

                ierr = MatMultSelfHC_AIJ(comm, psi0_sector, dm_left, PETSC_TRUE); CHKERRQ(ierr);
                ierr = MatMultSelfHC_AIJ(comm, psi0_sector, dm_right, PETSC_FALSE); CHKERRQ(ierr);

                BlockLeft_.rho_block_dict[sys_enl_Sz] = dm_left;
                BlockRight_.rho_block_dict[env_enl_Sz] = dm_right;

                dm_left = nullptr;
                dm_right = nullptr;

                if(psi0_sector) {
                    ierr = MatDestroy(&psi0_sector); CHKERRQ(ierr);
                    psi0_sector = nullptr;
                }
            }
        }

        ierr = VecScatterDestroy(&ctx); CHKERRQ(ierr);
        ierr = VecDestroy(&vec_seq); CHKERRQ(ierr);

    } else {
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

        ierr = MatMultSelfHC_AIJ(comm_, gsv_mat_seq, dm_left, PETSC_TRUE); CHKERRQ(ierr);
        ierr = MatMultSelfHC_AIJ(comm_, gsv_mat_seq, dm_right, PETSC_FALSE); CHKERRQ(ierr);

        /*
            Destroy temporary matrices
        */
        if (gsv_mat_seq) MatDestroy(&gsv_mat_seq); gsv_mat_seq = nullptr;

    }

    /*
        Toggle switches
    */
    groundstate_solved_ = PETSC_FALSE;
    dm_solved = PETSC_TRUE;

    DMRG_SUB_TIMINGS_END(__FUNCT__)
    DMRG_TIMINGS_END(__FUNCT__);
    return ierr;
}

/**
    Define the tuple type representing a possible eigenstate
    0 - PetscReal - eigenvalue
    1 - PetscInt - index of the SVD object and the corresponding reduced density matrix
    2 - PetscInt - index of eigenstate in SVD_OBJECT
    3 - PetscScalar - value of Sz_sector acting as key for element 4
    4 - std::vector<PetscInt> - current sector basis from basis by sector
*/
typedef std::tuple<PetscReal, PetscInt, PetscInt, PetscScalar, std::vector<PetscInt>> eigenstate_t;


/** Comparison function for eigenstates in descending order */
bool compare_descending_eigenstates(eigenstate_t a, eigenstate_t b)
{
    return (std::get<0>(a)) > (std::get<0>(b));
}


PetscErrorCode GetRotationMatrices_targetSz(
    const PetscInt mstates,
    DMRGBlock& block,
    Mat& mat,
    PetscReal& truncation_error)
{
    PetscErrorCode ierr = 0;

    /* Get information on MPI */

    PetscMPIInt     nprocs, rank;
    MPI_Comm comm = PETSC_COMM_WORLD;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    const std::unordered_map<PetscScalar,std::vector<PetscInt>>&
        sys_enl_basis_by_sector = block.basis_by_sector;

    std::vector< SVD_OBJECT >   svd_list(block.rho_block_dict.size());
    std::vector< Vec >          vec_list(block.rho_block_dict.size());
    std::vector< eigenstate_t > possible_eigenstates;

    /* Diagonalize each block of the reduced density matrix */

    /*********************TIMINGS**********************/
    #ifdef __DMRG_SUB_TIMINGS
        PetscLogDouble svd_total_time0, svd_total_time;
        ierr = PetscTime(&svd_total_time0); CHKERRQ(ierr);
    #endif
    /**************************************************/

    PetscInt counter = 0;
    for (auto elem: block.rho_block_dict)
    {
        /*********************TIMINGS**********************/
        #if defined(__DMRG_SUB_TIMINGS) && (__DMRG_SUB_SVD_TIMINGS)
            PetscLogDouble svd_time0, svd_time;
            ierr = PetscTime(&svd_time0); CHKERRQ(ierr);
        #endif
        /**************************************************/

        /* Keys and values of rho_block_dict */
        PetscScalar         Sz_sector = elem.first;
        Mat&                rho_block = elem.second;

        /* SVD of the reduced density matrices */
        SVD_OBJECT          svd;
        Vec                 Vr;
        PetscScalar         error;
        PetscInt            nconv;

        ierr = MatGetSVD(rho_block, svd, nconv, error, NULL); CHKERRQ(ierr);

        /* Dump svd into map */
        svd_list[counter] = svd;

        /* Create corresponding vector for later use */
        ierr = MatCreateVecs(rho_block, &Vr, nullptr); CHKERRQ(ierr);
        vec_list[counter] = Vr;

        /* Get current sector basis indices */
        std::vector<PetscInt> current_sector_basis = sys_enl_basis_by_sector.at(Sz_sector);

        /* Verify that sizes match */
        PetscInt vec_size;
        ierr = VecGetSize(Vr, &vec_size); CHKERRQ(ierr);
        if((size_t)vec_size!=current_sector_basis.size())
            SETERRQ2(comm,1,"Vector size mismatch. Expected %d from current sector basis. Got %d from Vec.",current_sector_basis.size(),vec_size);

        /* Loop through the eigenstates and dump as tuple to vector */
        for (PetscInt svd_id = 0; svd_id < nconv; ++svd_id)
        {
            PetscReal sigma; /* May require PetscReal */
            ierr = SVDGetSingularTriplet(svd, svd_id, &sigma, nullptr, nullptr); CHKERRQ(ierr);
            eigenstate_t tuple(sigma, counter, svd_id, Sz_sector, current_sector_basis);
            possible_eigenstates.push_back(tuple);

        }

        svd = nullptr;
        Vr = nullptr;
        ++counter;

        /*********************TIMINGS**********************/
        #if defined(__DMRG_SUB_TIMINGS) && (__DMRG_SUB_SVD_TIMINGS)
            ierr = PetscTime(&svd_time); CHKERRQ(ierr);
            svd_time = svd_time - svd_time0;
            ierr = PetscPrintf(PETSC_COMM_WORLD, "%16s SVD %24ssize: %-12d %.20g\n", "","",vec_size, svd_time);
        #endif
        /**************************************************/
    }
    /*********************TIMINGS**********************/
    #ifdef __DMRG_SUB_TIMINGS
        ierr = PetscTime(&svd_total_time); CHKERRQ(ierr);
        svd_total_time = svd_total_time - svd_total_time0;
        ierr = PetscPrintf(PETSC_COMM_WORLD, "%16s %-42s %.20g\n", "","SVD total:", svd_total_time);
    #endif
    /**************************************************/

    /* Sort all possible eigenstates in descending order of eigenvalues */
    std::stable_sort(possible_eigenstates.begin(),possible_eigenstates.end(),compare_descending_eigenstates);

    /* Build the transformation matrix from the `m` overall most significant eigenvectors */

    PetscInt my_m = std::min((PetscInt) possible_eigenstates.size(), (PetscInt) mstates);

    /* Create the transformation matrix */

    PetscInt nrows = block.basis_size();
    PetscInt ncols = my_m;
    ierr = MatCreate(PETSC_COMM_WORLD, &mat); CHKERRQ(ierr);
    ierr = MatSetSizes(mat, PETSC_DECIDE, PETSC_DECIDE, nrows, ncols); CHKERRQ(ierr);
    ierr = MatSetFromOptions(mat); CHKERRQ(ierr);
    ierr = MatSetUp(mat); CHKERRQ(ierr);

    /* Guess the local ownership of resultant matrix */

    PetscInt remrows = nrows % nprocs;
    PetscInt locrows = nrows / nprocs;
    PetscInt Istart = locrows * rank;

    if (rank < remrows){
        locrows += 1;
        Istart += rank;
    } else {
        Istart += remrows;
    }

    // PetscInt Iend = Istart + locrows;

    /* FIXME: Preallocate w/ optimization options */

    PetscInt max_nz_rows = locrows;

    /* Prepare buffers */
    PetscInt*       mat_rows;
    PetscScalar*    mat_vals;
    PetscReal sum_sigma = 0.0;

    ierr = PetscMalloc1(max_nz_rows,&mat_rows); CHKERRQ(ierr);
    ierr = PetscMalloc1(max_nz_rows,&mat_vals); CHKERRQ(ierr);

    std::vector<PetscScalar> new_sector_array(my_m);

    /* Loop through eigenstates and build the rotation matrix */

    /*********************TIMINGS**********************/
    #ifdef __DMRG_SUB_TIMINGS
        PetscLogDouble rot_mat_time0, rot_mat_time;
        ierr = PetscTime(&rot_mat_time0); CHKERRQ(ierr);
    #endif
    /**************************************************/

    for (PetscInt Ieig = 0; Ieig < my_m; ++Ieig)
    {
        /* Unpack tuple */

        eigenstate_t tuple = possible_eigenstates[Ieig];

        PetscReal   sigma     = std::get<0>(tuple);
        PetscInt    block_id  = std::get<1>(tuple);
        PetscInt    svd_id    = std::get<2>(tuple);
        PetscScalar Sz_sector = std::get<3>(tuple);
        std::vector<PetscInt>&  current_sector_basis = std::get<4>(tuple);

        // PetscPrintf(comm, "Sigma: %f\n",sigma);

        sum_sigma += sigma;

        /* Get a copy of the vector's array associated to this process */

        SVD_OBJECT& svd = svd_list[block_id];
        Vec&        Vr  = vec_list[block_id];

        PetscReal sigma_svd;
        ierr = SVDGetSingularTriplet(svd, svd_id, &sigma_svd, Vr, nullptr); CHKERRQ(ierr);
        if(sigma_svd!=sigma)
            SETERRQ2(comm,1,"Eigenvalue mismatch. Expected %f. Got %f.", sigma, sigma_svd);

        /* Get ownership and check sizes */

        PetscInt vec_size, Vstart, Vend;

        ierr = VecGetOwnershipRange(Vr, &Vstart, &Vend);
        // PetscInt Vloc = Vend - Vstart;

        ierr = VecGetSize(Vr,&vec_size); CHKERRQ(ierr);
        if((size_t)vec_size!=current_sector_basis.size())
            SETERRQ2(comm,1,"Vector size mismatch. Expected %d. Got %d.",current_sector_basis.size(),vec_size);


        const PetscScalar *vec_vals;
        ierr = VecGetArrayRead(Vr, &vec_vals);

        /* Loop through current_sector_basis and eigenvectors (depending on ownership ) */
        // for (PetscInt Jsec = 0; Jsec < current_sector_basis.size(); ++Jsec)

        /* Inspection */
        // PetscPrintf(comm_,"\n\n\nsigma = %f\n",sigma);
        // PetscPrintf(comm_,"current_sector_basis = ");
        // for (auto& item: current_sector_basis) PetscPrintf(comm,"%d ", item);
        // PetscPrintf(comm_,"\n");
        // VecPeek(Vr, "Vr");

        // MPI_Barrier(PETSC_COMM_WORLD);

        PetscInt nrows_write = 0;
        for (PetscInt Jsec = Vstart; Jsec < Vend; ++Jsec)
        {
            PetscInt j_idx = current_sector_basis[Jsec];

            mat_rows[nrows_write] = j_idx;
            mat_vals[nrows_write] = vec_vals[Jsec-Vstart];

            // printf("[%d] %3d %3d mat_vals = %f\n",rank,Ieig,Jsec,mat_vals[nrows_write]);
            ++nrows_write;
        }

        new_sector_array[Ieig] = Sz_sector;

        /* Set values over one possibly non-local column */
        ierr = MatSetValues(mat, nrows_write, mat_rows, 1, &Ieig, mat_vals, INSERT_VALUES);

        ierr = VecRestoreArrayRead(Vr, &vec_vals);
    }

    /*********************TIMINGS**********************/
    #ifdef __DMRG_SUB_TIMINGS
        ierr = PetscTime(&rot_mat_time); CHKERRQ(ierr);
        rot_mat_time = rot_mat_time - rot_mat_time0;
        ierr = PetscPrintf(PETSC_COMM_WORLD, "%16s %-42s %.20g\n", "","RotMat Construction:", rot_mat_time);

        ierr = PetscTime(&rot_mat_time0); CHKERRQ(ierr);
    #endif
    /**************************************************/


    /* Output truncation error */
    truncation_error = 1.0 - sum_sigma;

    /* Replace block's sector_array */
    block.basis_sector_array = new_sector_array;

    /* Final assembly of output matrix */
    ierr = MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);\
    ierr = MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    /* Destroy buffers */
    ierr = PetscFree(mat_rows); CHKERRQ(ierr);
    ierr = PetscFree(mat_vals); CHKERRQ(ierr);

    /* Destroy temporary PETSc objects */
    for (auto svd: svd_list){
        ierr = SVDDestroy(&svd); CHKERRQ(ierr);
    }
    for (auto vec: vec_list){
        ierr = VecDestroy(&vec); CHKERRQ(ierr);
    }

    /*********************TIMINGS**********************/
    #ifdef __DMRG_SUB_TIMINGS
        ierr = PetscTime(&rot_mat_time); CHKERRQ(ierr);
        rot_mat_time = rot_mat_time - rot_mat_time0;
        ierr = PetscPrintf(PETSC_COMM_WORLD, "%16s %-42s %.20g\n", "","RotMat Assembly:", rot_mat_time);
    #endif
    /**************************************************/

    return ierr;
}


PetscErrorCode GetRotationMatrices_targetSz_root(
    const PetscInt mstates,
    DMRGBlock& block,
    Mat& mat,
    PetscReal& truncation_error)
{
    PetscErrorCode ierr = 0;

    /* Get information on MPI */
    PetscMPIInt     nprocs, rank;
    MPI_Comm comm = PETSC_COMM_WORLD;
    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &rank);

    /* Do the SVD step only on the root processor */
    PetscInt my_m = 0;
    std::vector< SVD_OBJECT >   svd_list;
    std::vector< Vec >          vec_list;
    std::vector< eigenstate_t > possible_eigenstates;

    if (!rank)
    {
        const std::unordered_map<PetscScalar,std::vector<PetscInt>>&
            sys_enl_basis_by_sector = block.basis_by_sector;

        svd_list.resize(block.rho_block_dict.size());
        vec_list.resize(block.rho_block_dict.size());

        /* Diagonalize each block of the reduced density matrix */

        /*********************TIMINGS**********************/
        #ifdef __DMRG_SUB_TIMINGS
            PetscLogDouble svd_total_time0, svd_total_time;
            ierr = PetscTime(&svd_total_time0); CHKERRQ(ierr);
        #endif
        /**************************************************/

        PetscInt counter = 0;
        for (auto elem: block.rho_block_dict)
        {
            /*********************TIMINGS**********************/
            #if defined(__DMRG_SUB_TIMINGS) && (__DMRG_SUB_SVD_TIMINGS)
                PetscLogDouble svd_time0, svd_time;
                ierr = PetscTime(&svd_time0); CHKERRQ(ierr);
            #endif
            /**************************************************/

            /* Keys and values of rho_block_dict */
            PetscScalar         Sz_sector = elem.first;
            Mat&                rho_block = elem.second;

            /* SVD of the reduced density matrices */
            SVD_OBJECT          svd;
            Vec                 Vr;
            PetscScalar         error;
            PetscInt            nconv;

            ierr = MatGetSVD(rho_block, svd, nconv, error, NULL); CHKERRQ(ierr);

            /* Dump svd into map */
            svd_list[counter] = svd;

            /* Create corresponding vector for later use */
            ierr = MatCreateVecs(rho_block, &Vr, nullptr); CHKERRQ(ierr);
            vec_list[counter] = Vr;

            /* Get current sector basis indices */
            std::vector<PetscInt> current_sector_basis = sys_enl_basis_by_sector.at(Sz_sector);

            /* Verify that sizes match */
            PetscInt vec_size;
            ierr = VecGetSize(Vr, &vec_size); CHKERRQ(ierr);
            if((size_t)vec_size!=current_sector_basis.size())
                SETERRQ2(PETSC_COMM_SELF,1,"Vector size mismatch. Expected %d from current sector basis. Got %d from Vec.",current_sector_basis.size(),vec_size);

            /* Loop through the eigenstates and dump as tuple to vector */
            for (PetscInt svd_id = 0; svd_id < nconv; ++svd_id)
            {
                PetscReal sigma; /* May require PetscReal */
                ierr = SVDGetSingularTriplet(svd, svd_id, &sigma, nullptr, nullptr); CHKERRQ(ierr);
                eigenstate_t tuple(sigma, counter, svd_id, Sz_sector, current_sector_basis);
                possible_eigenstates.push_back(tuple);
            }

            svd = nullptr;
            Vr = nullptr;
            ++counter;

            /*********************TIMINGS**********************/
            #if defined(__DMRG_SUB_TIMINGS) && (__DMRG_SUB_SVD_TIMINGS)
                ierr = PetscTime(&svd_time); CHKERRQ(ierr);
                svd_time = svd_time - svd_time0;
                ierr = PetscPrintf(PETSC_COMM_SELF, "%16s SVD %24ssize: %-12d %.20g\n", "","",vec_size, svd_time);
            #endif
            /**************************************************/
        }
        /*********************TIMINGS**********************/
        #ifdef __DMRG_SUB_TIMINGS
            ierr = PetscTime(&svd_total_time); CHKERRQ(ierr);
            svd_total_time = svd_total_time - svd_total_time0;
            ierr = PetscPrintf(PETSC_COMM_SELF, "%16s %-42s %.20g\n", "","SVD total:", svd_total_time);
        #endif
        /**************************************************/

        /* Sort all possible eigenstates in descending order of eigenvalues */
        std::stable_sort(possible_eigenstates.begin(),possible_eigenstates.end(),compare_descending_eigenstates);

        /* Build the rotation matrix from the `m` overall most significant eigenvectors */
        my_m = std::min((PetscInt) possible_eigenstates.size(), (PetscInt) mstates);

    }

    /* Broadcast my_m to all processes */
    ierr = MPI_Bcast(&my_m, 1, MPIU_INT, 0, PETSC_COMM_WORLD); CHKERRQ(ierr);

    /* Create the global rotation matrix */
    PetscInt nrows = block.basis_size();
    PetscInt ncols = my_m;

    ierr = MatCreate(PETSC_COMM_WORLD, &mat); CHKERRQ(ierr);
    ierr = MatSetSizes(mat, PETSC_DECIDE, PETSC_DECIDE, nrows, ncols); CHKERRQ(ierr);
    ierr = MatSetFromOptions(mat); CHKERRQ(ierr);

    /*********************TIMINGS**********************/
    #ifdef __DMRG_SUB_TIMINGS
        PetscLogDouble rot_mat_time0, rot_mat_time;
        ierr = PetscTime(&rot_mat_time0); CHKERRQ(ierr);
    #endif
    /**************************************************/

    /*
        On rank 0: dump resulting sparse matrix object's column indices and
        values to vectors and scatter values to owning processes.

        Note: Take into account the fact that some rows may be empty and some
        processors may not receive any rows at all.
     */

    /* Calculate number of locally owned rows in resulting matrix */

    const PetscInt remrows = nrows % nprocs;
    const PetscInt locrows = nrows / nprocs + ((rank < remrows) ?  1 : 0 );
    const PetscInt Istart  = nrows / nprocs * rank + ((rank < remrows) ?  rank : remrows);
    const PetscInt Iend = Istart + locrows;

    /* Prepare buffers for scatter and broadcast */
    PetscMPIInt *sendcounts = nullptr;  /* The number of rows to scatter to each process */
    PetscMPIInt *displs = nullptr;      /* The starting row for each process */
    PetscMPIInt recvcount = 0;          /* Number of entries to receive from scatter */
    std::vector<PetscScalar> new_sector_array(my_m); /* Updates the sector array */

    /* Matrix buffers for MatSetValues */
    PetscInt    *mat_cols;
    PetscScalar *mat_vals;

    /* Temporary matrix buffer to be filled up by root */
    std::vector< std::vector< PetscInt >> mat_cols_list;
    std::vector< std::vector< PetscScalar >> mat_vals_list;

    /* Counters for nonzeros in the sparse matrix */
    PetscInt *Dnnz = nullptr, *Onnz = nullptr, *Tnnz = nullptr; /* Number of nonzeros in each row */
    PetscInt *Rdisp = nullptr; /* The displacement for each row */

    if(!rank)
    {
        ierr = PetscCalloc2(nprocs, &sendcounts, nprocs, &displs); CHKERRQ(ierr);

        /* Calculate row and column layout for all processes */
        std::vector<PetscMPIInt> row_to_rank(nrows); /* Maps a row to its owner rank */
        std::vector<PetscInt> Rrange(nprocs+1); /* The range of row ownership in each rank */
        std::vector<PetscInt> Crange(nprocs+1); /* The range of the diagonal columns in each rank */

        PetscInt Iend = 0, Cend=0;
        for (PetscMPIInt Irank = 0; Irank < nprocs; ++Irank)
        {
            /* Guess the local ownership ranges at the receiving process */
            PetscInt remrows = nrows % nprocs;
            PetscInt locrows = nrows / nprocs + ((Irank < remrows) ?  1 : 0 );
            PetscInt Istart  = nrows / nprocs * Irank + ((Irank < remrows) ?  Irank : remrows);

            /* Self-consistency check */
            if(Istart!=Iend) SETERRQ2(PETSC_COMM_SELF,1,"Error in row layout guess. "
                "Expected %d. Got %d.", Iend, Istart);
            Iend = Istart + locrows;
            Rrange[Irank] = Istart;

            for (PetscInt Irow = Istart; Irow < Iend; ++Irow)
                row_to_rank[Irow] = Irank;

            sendcounts[Irank] = (PetscMPIInt) locrows;
            displs[Irank]     = (PetscMPIInt) Istart;

            /* Guess the local diagonal column layout at the receiving process */
            PetscInt remcols = ncols % nprocs;
            PetscInt locdiag = ncols / nprocs + ((Irank < remcols) ? 1 : 0 );
            PetscInt Cstart  = ncols / nprocs * Irank + ((Irank < remcols) ?  Irank : remcols);

            /* Self-consistency check */
            if(Cstart!=Cend) SETERRQ2(PETSC_COMM_SELF,1,"Error in column layout guess. "
                "Expected %d. Got %d.", Cend, Cstart);
            Cend = Cstart + locdiag;

            Crange[Irank] = Cstart;
        }
        /* Also define for the right-most and bottom boundary */
        Crange[nprocs] = ncols;
        Rrange[nprocs] = nrows;

        /* 
            Store the sparse matrix nonzeros as resizable vectors since we do not know
                how many elements go into a row.
            Calculate the preallocation data at root as well.
            Possibly, dump the contents of these vector to individual buffers later on
                or use them directly as MPI send buffers
         */
        mat_cols_list.resize(nrows);
        mat_vals_list.resize(nrows);

        ierr = PetscCalloc4(nrows, &Dnnz, nrows, &Onnz, nrows, &Tnnz, nrows, &Rdisp); CHKERRQ(ierr);

        /* Extraction of values */

        truncation_error = 1.0;
        for (PetscInt Ieig = 0; Ieig < my_m; ++Ieig)
        {
            /* Unpack selected eigenstate tuple */
            eigenstate_t tuple = possible_eigenstates[Ieig];
            PetscReal   sigma     = std::get<0>(tuple);
            PetscInt    block_id  = std::get<1>(tuple);
            PetscInt    svd_id    = std::get<2>(tuple);
            PetscScalar Sz_sector = std::get<3>(tuple);
            std::vector<PetscInt>&  current_sector_basis = std::get<4>(tuple);

            /* Get a copy of the vector's array associated to this process */
            SVD_OBJECT& svd = svd_list[block_id];
            Vec&        Vr  = vec_list[block_id];

            /* Inspect sigma and deduct from truncation error */
            PetscReal sigma_svd;
            ierr = SVDGetSingularTriplet(svd, svd_id, &sigma_svd, Vr, nullptr); CHKERRQ(ierr);
            if(sigma_svd!=sigma)
                SETERRQ2(comm,1,"Eigenvalue mismatch. Expected %f. Got %f.", sigma, sigma_svd);
            truncation_error = truncation_error - sigma;

            /* Get vector size and inspect */
            PetscInt vec_size;
            ierr = VecGetSize(Vr,&vec_size); CHKERRQ(ierr);
            if((size_t)vec_size!=current_sector_basis.size())
                SETERRQ2(comm,1,"Vector size mismatch. Expected %d. Got %d.",current_sector_basis.size(),vec_size);

            /* Read elements of the vector */
            const PetscScalar *vec_vals;
            ierr = VecGetArrayRead(Vr, &vec_vals); CHKERRQ(ierr);

            /* Loop through current_sector_basis and eigenvectors */
            for (size_t Jsec = 0; Jsec < current_sector_basis.size(); ++Jsec)
            {
                PetscInt j_idx = current_sector_basis[Jsec];    /* the row index */
                PetscMPIInt Irank = row_to_rank[j_idx];         /* the rank corresponding to this row */

                /* Determine the preallocation where Ieig is the column index */
                if ( Crange[Irank] <= Ieig && Ieig < Crange[Irank+1] ){
                    Dnnz[j_idx] += 1;
                } else {
                    Onnz[j_idx] += 1;
                }

                /* Dump values and columns to respective buffer */
                mat_cols_list[j_idx].push_back(Ieig);
                mat_vals_list[j_idx].push_back(vec_vals[Jsec]);

            }
            new_sector_array[Ieig] = Sz_sector;
            ierr = VecRestoreArrayRead(Vr, &vec_vals);
        }

        /* Dump data to scatterv send buffer */
        PetscInt tot_nnz = 0;
        for (PetscInt Irow = 0; Irow < nrows; ++Irow)
        {
            Rdisp[Irow] = tot_nnz;
            Tnnz[Irow] = Dnnz[Irow] + Onnz[Irow];
            tot_nnz += Tnnz[Irow];
        }

        /* Checkpoint: Compare preallocation data (Xnnz) with the lengths of the matrix buffer */
        for (size_t Irow = 0; Irow < nrows; ++Irow)
        {
            if((size_t)(Tnnz[Irow]) != mat_cols_list[Irow].size())
                SETERRQ3(PETSC_COMM_SELF, 1, "Error in matrix buffer size in row %d. "
                    "Expected %d from preallocation. Got %d on cols buffer.",
                    Irow, Tnnz[Irow], mat_cols_list[Irow].size());

            if((size_t)(Tnnz[Irow]) != mat_vals_list[Irow].size())
                SETERRQ3(PETSC_COMM_SELF, 1, "Error in matrix buffer size in row %d. "
                    "Expected %d from preallocation. Got %d on vals buffer.",
                    Irow, Tnnz[Irow], mat_cols_list[Irow].size());
        }

        /* Send preallocation initial info */
        ierr = MPI_Scatterv(Dnnz, sendcounts, displs, MPIU_INT, MPI_IN_PLACE, recvcount, MPIU_INT, 0, comm); CHKERRQ(ierr);
        ierr = MPI_Scatterv(Onnz, sendcounts, displs, MPIU_INT, MPI_IN_PLACE, recvcount, MPIU_INT, 0, comm); CHKERRQ(ierr);

        ierr = PetscCalloc2(tot_nnz, &mat_cols, tot_nnz, &mat_vals); CHKERRQ(ierr);

        for (PetscInt Irow = 0; Irow < nrows; ++Irow)
        {
            std::memcpy(&mat_cols[Rdisp[Irow]], mat_cols_list[Irow].data(), Tnnz[Irow] * sizeof(PetscInt));
            /* Uncomment only if memory is an issue: */
            // mat_cols_list[Irow].clear();
            // mat_cols_list[Irow].shrink_to_fit();
        }

        for (PetscInt Irow = 0; Irow < nrows; ++Irow)
        {
            std::memcpy(&mat_vals[Rdisp[Irow]], mat_vals_list[Irow].data(), Tnnz[Irow] * sizeof(PetscScalar));
            /* Uncomment only if memory is an issue: */
            // mat_cols_list[Irow].clear();
            // mat_cols_list[Irow].shrink_to_fit();
        }

        /* Prepare scatterv info */

        /* Number of entries per process */
        ierr = PetscMemzero(sendcounts, nprocs*sizeof(PetscMPIInt)); CHKERRQ(ierr);
        /* Starting entry of each process */
        ierr = PetscMemzero(displs, nprocs*sizeof(PetscMPIInt)); CHKERRQ(ierr);

        tot_nnz = 0;
        for (PetscInt Irank = 0; Irank < nprocs; ++Irank)
        {
            displs[Irank] = (PetscMPIInt)tot_nnz;
            sendcounts[Irank] = 0;
            for (PetscInt Irow = Rrange[Irank]; Irow < Rrange[Irank+1]; ++Irow)
                sendcounts[Irank] += Tnnz[Irow];
            tot_nnz += sendcounts[Irank];
        }

        /* Scatter matrix data */
        ierr = MPI_Scatterv(mat_cols, sendcounts, displs, MPIU_INT, MPI_IN_PLACE, recvcount, MPIU_INT, 0, comm); CHKERRQ(ierr);
        ierr = MPI_Scatterv(mat_vals, sendcounts, displs, MPIU_SCALAR, MPI_IN_PLACE, recvcount, MPIU_SCALAR, 0, comm); CHKERRQ(ierr);

    }
    else
    {
        /* Receive initial preallocation info */

        recvcount = locrows;
        ierr = PetscCalloc4(locrows, &Dnnz, locrows, &Onnz, locrows, &Tnnz, locrows, &Rdisp); CHKERRQ(ierr);
        ierr = PetscCalloc2(1, &sendcounts, 1, &displs); CHKERRQ(ierr); /* Just allocate but will be ignored */

        ierr = MPI_Scatterv(NULL, sendcounts, displs, MPIU_INT, Dnnz, recvcount, MPIU_INT, 0, comm); CHKERRQ(ierr);
        ierr = MPI_Scatterv(NULL, sendcounts, displs, MPIU_INT, Onnz, recvcount, MPIU_INT, 0, comm); CHKERRQ(ierr);

        /* Prepare to receive matrix by scatterv */

        for (PetscInt lrow = 0; lrow < locrows; ++lrow){
            Tnnz[lrow] = Dnnz[lrow] + Onnz[lrow];
        }

        PetscInt tot_nnz = 0;
        for (PetscInt lrow = 0; lrow < locrows; ++lrow){
            Rdisp[lrow] = tot_nnz;
            tot_nnz += Tnnz[lrow];
        }
        recvcount = tot_nnz;

        ierr = PetscCalloc2(tot_nnz, &mat_cols, tot_nnz, &mat_vals); CHKERRQ(ierr);

        /* Receive matrix data */

        ierr = MPI_Scatterv(NULL, sendcounts, displs, MPIU_INT, mat_cols, recvcount, MPIU_INT, 0, comm); CHKERRQ(ierr);
        ierr = MPI_Scatterv(NULL, sendcounts, displs, MPIU_SCALAR, mat_vals, recvcount, MPIU_SCALAR, 0, comm); CHKERRQ(ierr);

    }

    /* Preallocate */
    ierr = MatMPIAIJSetPreallocation(mat, -1, Dnnz, -1, Onnz); CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(mat, -1, Dnnz); CHKERRQ(ierr);

    /* Set matrix properties */
    ierr = MatSetOption(mat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE); CHKERRQ(ierr);
    ierr = MatSetOption(mat, MAT_NO_OFF_PROC_ZERO_ROWS, PETSC_TRUE);
    ierr = MatSetOption(mat, MAT_IGNORE_OFF_PROC_ENTRIES, PETSC_TRUE); CHKERRQ(ierr);
    ierr = MatSetOption(mat, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE); CHKERRQ(ierr);
    ierr = MatSetOption(mat, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE); CHKERRQ(ierr);

    /*********************TIMINGS**********************/
    #ifdef __DMRG_SUB_TIMINGS
        ierr = PetscTime(&rot_mat_time); CHKERRQ(ierr);
        rot_mat_time = rot_mat_time - rot_mat_time0;
        ierr = PetscPrintf(PETSC_COMM_WORLD, "%16s %-42s %.20f\n", "","RotMat Prepare:", rot_mat_time);
        ierr = PetscTime(&rot_mat_time0); CHKERRQ(ierr);
    #endif
    /**************************************************/

    /* Construct final matrix */
    for (PetscInt Irow = Istart; Irow < Iend; ++Irow){
        ierr = MatSetValues(mat, 1, &Irow, Tnnz[Irow-Istart],
            mat_cols+Rdisp[Irow-Istart], mat_vals+Rdisp[Irow-Istart], INSERT_VALUES); CHKERRQ(ierr);
    }

    /* Deallocate buffers */
    ierr = PetscFree2(mat_cols, mat_vals); CHKERRQ(ierr);
    ierr = PetscFree2(sendcounts, displs); CHKERRQ(ierr);
    ierr = PetscFree4(Dnnz, Onnz, Tnnz, Rdisp); CHKERRQ(ierr);

    /*********************TIMINGS**********************/
    #ifdef __DMRG_SUB_TIMINGS
        ierr = PetscTime(&rot_mat_time); CHKERRQ(ierr);
        rot_mat_time = rot_mat_time - rot_mat_time0;
        ierr = PetscPrintf(PETSC_COMM_WORLD, "%16s %-42s %.20f\n", "","RotMat Construction:", rot_mat_time);
        ierr = PetscTime(&rot_mat_time0); CHKERRQ(ierr);
    #endif
    /**************************************************/

    /* Final assembly of output matrix */
    ierr = MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    /******************** TIMINGS *********************/
    #ifdef __DMRG_SUB_TIMINGS
        ierr = PetscTime(&rot_mat_time); CHKERRQ(ierr);
        rot_mat_time = rot_mat_time - rot_mat_time0;
        ierr = PetscPrintf(PETSC_COMM_WORLD, "%16s %-42s %.20f\n", "","RotMat Assembly:", rot_mat_time);
    #endif
    /**************************************************/

    /* Broadcast sum_sigma and new_sector_array */
    ierr = MPI_Bcast(&truncation_error, 1, MPIU_REAL, 0, PETSC_COMM_WORLD); CHKERRQ(ierr);
    ierr = MPI_Bcast(&new_sector_array[0], my_m, MPIU_SCALAR, 0, PETSC_COMM_WORLD); CHKERRQ(ierr);

    /* Replace block's sector_array */
    block.basis_sector_array = new_sector_array;

    /* Destroy temporary PETSc objects */
    if(!rank){
        for (auto svd: svd_list){ ierr = SVDDestroy(&svd); CHKERRQ(ierr); }
        for (auto vec: vec_list){ ierr = VecDestroy(&vec); CHKERRQ(ierr); }
    }

    DMRG_MPI_BARRIER("End of GetRotationMatrices subfunction");
    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "iDMRG::GetRotationMatrices"
PetscErrorCode iDMRG::GetRotationMatrices(PetscReal& truncerr_left, PetscReal& truncerr_right)
{
    PetscErrorCode  ierr = 0;
    DMRG_TIMINGS_START(__FUNCT__);
    DMRG_SUB_TIMINGS_START(__FUNCT__)

    #define __GET_SVD "    GetSVD"

    if (do_target_Sz) {

        /* Checkpoint */
        if(!dm_solved)
            SETERRQ(comm_, 1, "Reduced density matrices not yet solved.");

        if((do_svd_on_root && rank_ == 0) || !do_svd_on_root )
        {
            if(BlockLeft_.rho_block_dict.size()==0)
                SETERRQ(PETSC_COMM_SELF, 1, "No density matrices for left block.");
            if(BlockLeft_.rho_block_dict.size()==0)
                SETERRQ(PETSC_COMM_SELF, 1, "No density matrices for right block.");
        }

        /******************** TIMINGS *********************/
        DMRG_SUB_TIMINGS_START(__GET_SVD)
        /**************************************************/

        if(do_svd_on_root)
        {
            ierr = GetRotationMatrices_targetSz_root(mstates_, BlockLeft_, U_left_, truncerr_left); CHKERRQ(ierr);
            ierr = GetRotationMatrices_targetSz_root(mstates_, BlockRight_, U_right_, truncerr_right); CHKERRQ(ierr);
        }
        else
        {
            ierr = GetRotationMatrices_targetSz(mstates_, BlockLeft_, U_left_, truncerr_left); CHKERRQ(ierr);
            ierr = GetRotationMatrices_targetSz(mstates_, BlockRight_, U_right_, truncerr_right); CHKERRQ(ierr);
        }

        /******************** TIMINGS *********************/
        DMRG_SUB_TIMINGS_END(__GET_SVD)
        /**************************************************/

        if((do_svd_on_root && rank_ == 0) || !do_svd_on_root )
        {
            /* Clear rho_block_dict for both blocks */
            if(BlockLeft_.rho_block_dict.size())
                for (auto item: BlockLeft_.rho_block_dict)
                    MatDestroy(&item.second);

            if(BlockRight_.rho_block_dict.size())
                for (auto item: BlockRight_.rho_block_dict)
                    MatDestroy(&item.second);

            BlockLeft_.rho_block_dict.clear();
            BlockRight_.rho_block_dict.clear();
        }



    } else {

        if(!(dm_left && dm_right && dm_solved))
            SETERRQ(comm_, 1, "Reduced density matrices not yet solved.");

        FILE *fp_left = nullptr, *fp_right = nullptr;

        #ifdef __TESTING
            char filename[PETSC_MAX_PATH_LEN];
            sprintf(filename,"data/dm_left_singularvalues_%06d.dat",iter());
            ierr = PetscFOpen(PETSC_COMM_WORLD, filename, "w", &fp_left); CHKERRQ(ierr);
            sprintf(filename,"data/dm_right_singularvalues_%06d.dat",iter());
            ierr = PetscFOpen(PETSC_COMM_WORLD, filename, "w", &fp_right); CHKERRQ(ierr);
        #endif

        PetscInt M_left, M_right;
        ierr = MatGetSize(dm_left, &M_left, nullptr); CHKERRQ(ierr);
        ierr = MatGetSize(dm_right, &M_right, nullptr); CHKERRQ(ierr);
        M_left = std::min(M_left, mstates_);
        M_right = std::min(M_right, mstates_);

        /******************** TIMINGS *********************/
        DMRG_SUB_TIMINGS_START(__GET_SVD)
        /**************************************************/

        /* Do SVD on subcommunicator */
        if(do_svd_commsplit)
        {
            Mat dm_left_red, dm_right_red; //, U_left_red, U_right_red;
            ierr = MatCreateRedundantMatrix(dm_left, svd_nsubcomm, MPI_COMM_NULL, MAT_INITIAL_MATRIX, &dm_left_red); CHKERRQ(ierr);
            ierr = MatCreateRedundantMatrix(dm_right, svd_nsubcomm, MPI_COMM_NULL, MAT_INITIAL_MATRIX, &dm_right_red); CHKERRQ(ierr);

            ierr = SVDLargestStates_split(dm_left_red, M_left, truncerr_left, U_left_, nullptr); CHKERRQ(ierr);
            ierr = SVDLargestStates_split(dm_right_red, M_right, truncerr_right, U_right_, nullptr); CHKERRQ(ierr);

            /* Reconstruct global rotation matrices */

            ierr = MatDestroy(&dm_left_red);
            ierr = MatDestroy(&dm_right_red);

        } else {
            ierr = SVDLargestStates(dm_left, M_left, truncerr_left, U_left_,fp_left); CHKERRQ(ierr);
            ierr = SVDLargestStates(dm_right, M_right, truncerr_right, U_right_,fp_right); CHKERRQ(ierr);
        }

        /******************** TIMINGS *********************/
        DMRG_SUB_TIMINGS_END(__GET_SVD)
        /**************************************************/

        #ifdef __TESTING
            ierr = PetscFClose(PETSC_COMM_WORLD, fp_left); CHKERRQ(ierr);
            ierr = PetscFClose(PETSC_COMM_WORLD, fp_right); CHKERRQ(ierr);
        #endif

        if (dm_left)   {ierr = MatDestroy(&dm_left); CHKERRQ(ierr);}
        if (dm_right)  {ierr = MatDestroy(&dm_right); CHKERRQ(ierr);}

    }

    #ifdef __PRINT_TRUNCATION_ERROR
        ierr = PetscPrintf(comm_,
            "%12sTruncation error (left):  %12e\n",
            " ", truncerr_left);

        ierr = PetscPrintf(comm_,
            "%12sTruncation error (right): %12e\n",
            " ", truncerr_right); CHKERRQ(ierr);
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

    DMRG_MPI_BARRIER("End of GetRotationMatrices");
    DMRG_SUB_TIMINGS_END(__FUNCT__)
    DMRG_TIMINGS_END(__FUNCT__);
    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "iDMRG::TruncateOperators"
PetscErrorCode iDMRG::TruncateOperators()
{
    PetscErrorCode ierr = 0;
    DMRG_TIMINGS_START(__FUNCT__);
    DMRG_SUB_TIMINGS_START(__FUNCT__);
    DMRG_MPI_BARRIER("Start of TruncateOperators");

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

    if(!(dm_svd && U_left_))
        SETERRQ(comm_, 1, "SVD of (LEFT) reduced density matrices not yet solved.");

    ierr = MatHermitianTranspose(U_left_, MAT_INITIAL_MATRIX, &U_hc); CHKERRQ(ierr);
    ierr = MatAssemblyBegin(U_hc, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(U_hc, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    ierr = MatMatMatMult(U_hc, BlockLeft_.H(), U_left_, MAT_INITIAL_MATRIX, PETSC_DECIDE, &mat_temp); CHKERRQ(ierr);
    ierr = BlockLeft_.update_H(mat_temp); CHKERRQ(ierr);

    ierr = MatMatMatMult(U_hc, BlockLeft_.Sz(), U_left_, MAT_INITIAL_MATRIX, PETSC_DECIDE, &mat_temp); CHKERRQ(ierr);
    ierr = BlockLeft_.update_Sz(mat_temp); CHKERRQ(ierr);

    ierr = MatMatMatMult(U_hc, BlockLeft_.Sp(), U_left_, MAT_INITIAL_MATRIX, PETSC_DECIDE, &mat_temp); CHKERRQ(ierr);
    ierr = BlockLeft_.update_Sp(mat_temp); CHKERRQ(ierr);

    ierr = MatDestroy(&U_hc); CHKERRQ(ierr);


    if(!(dm_svd && U_right_))
        SETERRQ(comm_, 1, "SVD of (RIGHT) reduced density matrices not yet solved.");

    ierr = MatHermitianTranspose(U_right_, MAT_INITIAL_MATRIX, &U_hc); CHKERRQ(ierr);
    ierr = MatAssemblyBegin(U_hc, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(U_hc, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

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

    ntruncations_ += 1;

    /* Save operator state after rotation */

    #ifdef __CHECK_ROTATION
        sprintf(filename,"data/H_left_post_%06d.dat",iter());
        ierr = MatWrite(BlockLeft_.H(), filename); CHKERRQ(ierr);

        sprintf(filename,"data/Sz_left_post_%06d.dat",iter());
        ierr = MatWrite(BlockLeft_.Sz(), filename); CHKERRQ(ierr);

        sprintf(filename,"data/Sp_left_post_%06d.dat",iter());
        ierr = MatWrite(BlockLeft_.Sp(), filename); CHKERRQ(ierr);

        sprintf(filename,"data/H_right_post_%06d.dat",iter());
        ierr = MatWrite(BlockRight_.H(), filename); CHKERRQ(ierr);

        sprintf(filename,"data/Sz_right_post_%06d.dat",iter());
        ierr = MatWrite(BlockRight_.Sz(), filename); CHKERRQ(ierr);

        sprintf(filename,"data/Sp_right_post_%06d.dat",iter());
        ierr = MatWrite(BlockRight_.Sp(), filename); CHKERRQ(ierr);
    #endif // __CHECK_ROTATION
    #undef __CHECK_ROTATION

    DMRG_MPI_BARRIER("End of TruncateOperators");
    DMRG_SUB_TIMINGS_END(__FUNCT__)
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
