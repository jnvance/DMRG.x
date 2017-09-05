#include "idmrg_1d_heisenberg.hpp"

/* Implementation of the Heisenberg Hamiltonian */

/** TODO:
 *  Implement coupling J and anisotropy Jz (DONE)
 *  Transfer SVDLargest to iDMRG as private member
 *  Change values and observe degeneracies
 *
 *  Implement eye as iDMRG object that gets renewed only when necessary
 *  (separate for block and superblock cases)
 *
 *  For operators implement reallocation only when needed, and implement a separate
 *  matrix object for enlarged operator
 *      H_pre, H
 *      H, H_enl
 */

#define PRINT_VEC(stdvectorpetscscalar,msg) \
        printf("%s\n",msg);\
        for (std::vector<PetscScalar>::const_iterator i = stdvectorpetscscalar.begin(); \
            i != stdvectorpetscscalar.end(); ++i) printf("%f\n",PetscRealPart(*i)); \
        printf("\n");

#undef __FUNCT__
#define __FUNCT__ "iDMRG_Heisenberg::SetParameters"
PetscErrorCode iDMRG_Heisenberg::SetParameters(PetscScalar J_in, PetscScalar Jz_in, PetscReal Mz_in)
{
    PetscErrorCode  ierr = 0;

    if(parameters_set == PETSC_TRUE)
    {
        SETERRQ(comm_, 1, "Parameters already set.");
    }
    else
    {
        J = J_in;
        Jz = Jz_in;
        Mz = Mz_in;

        parameters_set = PETSC_TRUE;
    }

    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "iDMRG_Heisenberg::BuildBlockLeft"
PetscErrorCode iDMRG_Heisenberg::BuildBlockLeft()
{
    PetscErrorCode  ierr = 0;
    DMRG_TIMINGS_START(__FUNCT__);
    DMRG_SUB_TIMINGS_START(__FUNCT__);

    ierr = CheckSetParameters(); CHKERRQ(ierr);

    PetscBool assembled;
    /*
        Declare aliases and auxiliary matrices
    */
    Mat H_L = BlockLeft_.H();
    Mat Sz_L = BlockLeft_.Sz();
    Mat Sp_L = BlockLeft_.Sp();
    Mat eye_L = nullptr;
    Mat Sm_L = nullptr;
    Mat Mat_temp;
    /*
        Determine the basis size of the block
    */
    PetscInt M_L, N_L;
    ierr = MatGetSize(H_L, &M_L, &N_L); CHKERRQ(ierr);
    /*
        Fill in auxiliary matrices with values
    */
    ierr = MatEyeCreate(comm_, eye_L, M_L); CHKERRQ(ierr);

    LINALG_TOOLS__MATASSEMBLY_FINAL(Sp_L);
    ierr = MatHermitianTranspose(Sp_L, MAT_INITIAL_MATRIX, &Sm_L); CHKERRQ(ierr);
    /*
        Update the block Hamiltonian
    */
    std::vector<PetscScalar>    a = {1.0,   Jz,  0.5*J,  0.5*J};
    std::vector<Mat>            A = {H_L,   Sz_L, Sp_L, Sm_L};
    std::vector<Mat>            B = {eye1_, Sz1_, Sm1_, Sp1_};
    ierr = MatKronProdSum(a, A, B, Mat_temp, PETSC_TRUE); CHKERRQ(ierr);
    ierr = BlockLeft_.update_H(Mat_temp);
    Mat_temp = nullptr;
    /*
        Update the Sz operator
    */
    ierr = MatKronProd(1.0, eye_L, Sz1_, Mat_temp); CHKERRQ(ierr);
    ierr = BlockLeft_.update_Sz(Mat_temp); CHKERRQ(ierr);
    Mat_temp = nullptr;
    /*
        Update the Sp operator
    */
    ierr = MatKronProd(1.0, eye_L, Sp1_, Mat_temp); CHKERRQ(ierr);
    ierr = BlockLeft_.update_Sp(Mat_temp); CHKERRQ(ierr);
    Mat_temp = nullptr;
    /*
        Update the basis_sectors
    */
    BlockLeft_.basis_sector_array = OuterSumFlatten(BlockLeft_.basis_sector_array, single_site_sectors);
    /*
        Update block length
    */
    BlockLeft_.length(BlockLeft_.length() + 1);
    /*
        Verify block validity
    */
    if(!BlockLeft_.is_valid()) SETERRQ(comm_, 1, "Invalid left block");
    #ifdef __PRINT_SIZES
        PetscPrintf(comm_, "%12sLeft       basis size: %-5d nsites: %-5d \n", "", BlockLeft_.basis_size(), BlockLeft_.length());
    #endif
    /*
        Destroy auxiliary objects
    */
    ierr = MatDestroy(&Sm_L); CHKERRQ(ierr);
    ierr = MatDestroy(&eye_L); CHKERRQ(ierr);

    DMRG_SUB_TIMINGS_END(__FUNCT__);
    DMRG_TIMINGS_END(__FUNCT__);
    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "iDMRG_Heisenberg::BuildBlockRight"
PetscErrorCode iDMRG_Heisenberg::BuildBlockRight()
{
    PetscErrorCode  ierr = 0;
    DMRG_TIMINGS_START(__FUNCT__);
    DMRG_SUB_TIMINGS_START(__FUNCT__);

    ierr = CheckSetParameters(); CHKERRQ(ierr);

    PetscBool assembled;
    /*
        Declare aliases and auxiliary matrices
    */
    Mat H_R = BlockRight_.H();
    Mat Sz_R = BlockRight_.Sz();
    Mat Sp_R = BlockRight_.Sp();
    Mat eye_R = nullptr;
    Mat Sm_R = nullptr;
    Mat Mat_temp;
    /*
        Determine the basis size of the block
    */
    PetscInt M_R, N_R;
    ierr = MatGetSize(H_R, &M_R, &N_R); CHKERRQ(ierr);
    /*
        Fill in auxiliary matrices with values
    */
    ierr = MatEyeCreate(comm_, eye_R, M_R); CHKERRQ(ierr);

    LINALG_TOOLS__MATASSEMBLY_FINAL(Sp_R);
    ierr = MatHermitianTranspose(Sp_R, MAT_INITIAL_MATRIX, &Sm_R);CHKERRQ(ierr);
    /*
        Update the block Hamiltonian
    */
    std::vector<PetscScalar>    a = {1.0,   Jz,  0.5*J,  0.5*J};
    std::vector<Mat>            A = {eye1_, Sz1_, Sm1_, Sp1_};
    std::vector<Mat>            B = {H_R,   Sz_R, Sp_R, Sm_R};
    ierr = MatKronProdSum(a, A, B, Mat_temp, PETSC_TRUE); CHKERRQ(ierr);
    ierr = BlockRight_.update_H(Mat_temp);
    Mat_temp = nullptr;
    /*
        Update the Sz operator
    */
    ierr = MatKronProd(1.0, Sz1_, eye_R, Mat_temp); CHKERRQ(ierr);
    ierr = BlockRight_.update_Sz(Mat_temp); CHKERRQ(ierr);
    Mat_temp = nullptr;
    /*
        Update the Sp operator
    */
    ierr = MatKronProd(1.0, Sp1_, eye_R, Mat_temp); CHKERRQ(ierr);
    ierr = BlockRight_.update_Sp(Mat_temp); CHKERRQ(ierr);
    Mat_temp = nullptr;
    /*
        Update the basis_sectors
    */
    BlockRight_.basis_sector_array = OuterSumFlatten(single_site_sectors,BlockRight_.basis_sector_array);
    /*
        Update block length
    */
    BlockRight_.length(BlockRight_.length() + 1);
    /*
        Verify block validity
    */
    if(!BlockRight_.is_valid()) SETERRQ(comm_, 1, "Invalid right block");
    #ifdef __PRINT_SIZES
        PetscPrintf(comm_, "%12sRight      basis size: %-5d nsites: %-5d \n", "", BlockRight_.basis_size(), BlockRight_.length());
    #endif
    /*
        Destroy auxiliary objects
    */
    ierr = MatDestroy(&Sm_R); CHKERRQ(ierr);
    ierr = MatDestroy(&eye_R); CHKERRQ(ierr);

    DMRG_SUB_TIMINGS_END(__FUNCT__);
    DMRG_TIMINGS_END(__FUNCT__);
    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "iDMRG_Heisenberg::BuildSuperBlock"

#if SUPERBLOCK_OPTIMIZATION == 4 || !defined(SUPERBLOCK_OPTIMIZATION)
PetscErrorCode iDMRG_Heisenberg::BuildSuperBlock()
{
    PetscErrorCode  ierr = 0;
    DMRG_TIMINGS_START(__FUNCT__);
    DMRG_SUB_TIMINGS_START(__FUNCT__);

    ierr = CheckSetParameters(); CHKERRQ(ierr);

    PetscBool assembled;
    /*
        Declare aliases and auxiliary matrices
    */
    Mat H_L = BlockLeft_.H();
    Mat H_R = BlockRight_.H();
    Mat Sz_L = BlockLeft_.Sz();
    Mat Sz_R = BlockRight_.Sz();
    Mat Sp_L = BlockLeft_.Sp();
    Mat Sp_R = BlockRight_.Sp();
    Mat eye_L = nullptr;
    Mat eye_R = nullptr;
    Mat Sm_L = nullptr;
    Mat Sm_R = nullptr;
    /*
        Build a restricted basis of states

    */
#if 1
    /* Return type: std::map<PetscScalar,std::vector<PetscInt>> */
    auto sys_enl_basis_by_sector = IndexMap(BlockLeft_.basis_sector_array);
    auto env_enl_basis_by_sector = IndexMap(BlockRight_.basis_sector_array);

    // auto M_sys_enl = BlockLeft_.basis_size();
    auto M_env_enl = BlockRight_.basis_size();

    PetscScalar target_Sz = Mz * (BlockLeft_.length()+BlockRight_.length() + 2);

    if(sector_indices.size() > 0) sector_indices.clear();
    std::vector<PetscInt> restricted_basis_indices = {};

    for (auto elem: sys_enl_basis_by_sector)
    {
        auto& sys_enl_Sz = elem.first;
        auto& sys_enl_basis_states = elem.second;
        auto  env_enl_Sz = target_Sz - sys_enl_Sz;

        sector_indices[sys_enl_Sz].reserve(
            sys_enl_basis_states.size()*env_enl_basis_by_sector[env_enl_Sz].size());

        std::cout << "sys_enl_Sz:  " << sys_enl_Sz << std::endl;
        std::cout << "env_enl_Sz:  " << env_enl_Sz << std::endl;

        if (env_enl_basis_by_sector.find(env_enl_Sz) == env_enl_basis_by_sector.end()){
        } else {
            /* found */
            for (auto i : sys_enl_basis_states)
            {
                auto i_offset = M_env_enl * i;
                for (auto j: env_enl_basis_by_sector[env_enl_Sz])
                {
                    auto current_index = (PetscInt)(restricted_basis_indices.size());
                    sector_indices[sys_enl_Sz].push_back(current_index);
                    restricted_basis_indices.push_back(i_offset + j);
                }
            }
        }
    }

    std::cout << "restricted_basis_indices:  ";
    for (auto elem: restricted_basis_indices) std::cout << "  " << elem;
    std::cout << std::endl;

#endif

    /*
        Determine the basis sizes of enlarged block
    */
    PetscInt M_L, N_L, M_R, N_R, M_H, N_H;
    ierr = MatGetSize(H_L, &M_L, &N_L); CHKERRQ(ierr);
    ierr = MatGetSize(H_R, &M_R, &N_R); CHKERRQ(ierr);
    /*
        Verify that the matrix is square
    */
    M_H = M_L * M_R;
    N_H = N_L * N_R;
    if (M_H != N_H)
        SETERRQ(comm_, 1, "Hamiltonian should be square."
            "Check block operators from previous step.");
    /*
        Fill in auxiliary matrices with values
    */

    ierr = MatEyeCreate(comm_, eye_L, M_L); CHKERRQ(ierr);
    ierr = MatEyeCreate(comm_, eye_R, M_R); CHKERRQ(ierr);

    LINALG_TOOLS__MATASSEMBLY_FINAL(BlockLeft_.Sp());
    MatHermitianTranspose(Sp_L, MAT_INITIAL_MATRIX, &Sm_L);

    LINALG_TOOLS__MATASSEMBLY_FINAL(BlockRight_.Sp());
    MatHermitianTranspose(Sp_R, MAT_INITIAL_MATRIX, &Sm_R);
    /*
        Decide whether to preallocate
        Conditions to reallocate the Hamiltonian matrix:
            1.  size mismatch
            2.  after performing the first truncation
                where there is a sudden change in sparsity
    */
    PetscBool prealloc = PETSC_TRUE;

    if(superblock_set_==PETSC_TRUE || superblock_H_){
        PetscInt M_H_curr, N_H_curr;
        ierr = MatGetSize(superblock_H_, &M_H_curr, &N_H_curr); CHKERRQ(ierr);
        if ((M_H == M_H_curr) && (N_H == N_H_curr) && ntruncations_ != 1)
            prealloc = PETSC_FALSE;
    }
    /*
        Do preallocation at each iteration
    */
    prealloc = PETSC_TRUE;
    if(prealloc && superblock_H_)
    {
        ierr = MatDestroy(&superblock_H_); CHKERRQ(ierr);
        superblock_H_ = nullptr;
        superblock_set_=PETSC_FALSE;
        PetscPrintf(comm_, "Prealloc H\n");
    }
    /*
        Construct the Hamiltonian matrix
    */
    #define SUPERBLOCK_CONSTRUCTION "    Superblock Construction with MatKronProdSum"
    DMRG_SUB_TIMINGS_START(SUPERBLOCK_CONSTRUCTION)

        std::vector<PetscScalar>    a = {1.0,   1.0,   Jz,  0.5*J,  0.5*J};
        std::vector<Mat>            A = {H_L,   eye_L, Sz_L, Sp_L, Sm_L};
        std::vector<Mat>            B = {eye_R, H_R,   Sz_R, Sm_R, Sp_R};
        ierr = MatKronProdSum(a, A, B, superblock_H_, prealloc); CHKERRQ(ierr);

    DMRG_SUB_TIMINGS_END(SUPERBLOCK_CONSTRUCTION)
    #undef SUPERBLOCK_CONSTRUCTION

    LINALG_TOOLS__MATDESTROY(eye_L);
    LINALG_TOOLS__MATDESTROY(eye_R);
    LINALG_TOOLS__MATDESTROY(Sm_L);
    LINALG_TOOLS__MATDESTROY(Sm_R);
    /*
        Checkpoint
    */
    superblock_set_ = PETSC_TRUE;
    PetscInt M_superblock;
    ierr = MatGetSize(superblock_H_, &M_superblock, nullptr); CHKERRQ(ierr);
    if(M_superblock != TotalBasisSize()) SETERRQ(comm_, 1, "Basis size mismatch.\n");
    /*
        Final Assembly
    */
    #define SUPERBLOCK_ASSEMBLY "    Superblock Assembly"
    DMRG_SUB_TIMINGS_START(SUPERBLOCK_ASSEMBLY)

        LINALG_TOOLS__MATASSEMBLY_FINAL(superblock_H_);

    DMRG_SUB_TIMINGS_END(SUPERBLOCK_ASSEMBLY)
    #undef SUPERBLOCK_ASSEMBLY

    DMRG_SUB_TIMINGS_END(__FUNCT__);
    DMRG_TIMINGS_END(__FUNCT__);
    return ierr;
}

#endif
