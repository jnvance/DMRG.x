#include "idmrg_1d_heisenberg.hpp"

/* Implementation of the Heisenberg Hamiltonian */

/** TODO:
 *  Implement coupling J
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

#undef __FUNCT__
#define __FUNCT__ "iDMRG_Heisenberg::BuildBlockLeft"
PetscErrorCode iDMRG_Heisenberg::BuildBlockLeft()
{
    PetscErrorCode  ierr = 0;

    LINALG_TOOLS__MATASSEMBLY_INIT();

    DMRG_TIMINGS_START(__FUNCT__);

    /*
        Prepare Sm as explicit Hermitian conjugate of Sp
        TODO: Implement as part of Kronecker product
    */
    Mat BlockLeft_Sm;
    LINALG_TOOLS__MATASSEMBLY_FINAL(BlockLeft_.Sp());
    ierr = MatHermitianTranspose(BlockLeft_.Sp(), MAT_INITIAL_MATRIX, &BlockLeft_Sm); CHKERRQ(ierr);

    /* Prepare the identity */
    Mat eye_L;
    PetscInt dim;
    ierr = MatGetSize(BlockLeft_.H(), &dim, nullptr);
    ierr = MatEyeCreate(comm_, eye_L, dim);

    /*
        Update the Hamiltonian
    */
    Mat Mat_temp;
    ierr = MatKron(BlockLeft_.H(), eye1_, Mat_temp, comm_); CHKERRQ(ierr);
    ierr = MatKronAdd(BlockLeft_.Sz(), Sz1_, Mat_temp, comm_); CHKERRQ(ierr);
    ierr = MatKronScaleAdd(0.5, BlockLeft_.Sp(), Sm1_, Mat_temp, comm_); CHKERRQ(ierr);
    ierr = MatKronScaleAdd(0.5, BlockLeft_Sm, Sp1_, Mat_temp, comm_); CHKERRQ(ierr);
    ierr = BlockLeft_.update_H(Mat_temp); /* H_temp is destroyed here */ CHKERRQ(ierr);
    Mat_temp = nullptr;

    /*
        Update the Sz operator
    */
    // ierr = MatKron(eye1_, BlockLeft_.Sz(), Mat_temp, comm_); CHKERRQ(ierr);
    ierr = MatKron(eye_L, Sz1_, Mat_temp, comm_); CHKERRQ(ierr);
    ierr = BlockLeft_.update_Sz(Mat_temp); CHKERRQ(ierr);
    Mat_temp = nullptr;

    /*
        Update the Sp operator
    */
    // ierr = MatKron(eye1_, BlockLeft_.Sp(), Mat_temp, comm_); CHKERRQ(ierr);
    ierr = MatKron(eye_L, Sp1_, Mat_temp, comm_); CHKERRQ(ierr);
    ierr = BlockLeft_.update_Sp(Mat_temp); CHKERRQ(ierr);
    Mat_temp = nullptr;

    BlockLeft_.length(BlockLeft_.length() + 1);
    if(!BlockLeft_.is_valid()) SETERRQ(comm_, 1, "Invalid left block");
    #ifdef __PRINT_SIZES
        PetscPrintf(comm_, "%12sLeft       basis size: %-5d nsites: %-5d \n", "", BlockLeft_.basis_size(), BlockLeft_.length());
    #endif
    ierr = MatDestroy(&BlockLeft_Sm); CHKERRQ(ierr);
    ierr = MatDestroy(&eye_L); CHKERRQ(ierr);

    DMRG_TIMINGS_END(__FUNCT__);
    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "iDMRG_Heisenberg::BuildBlockRight"
PetscErrorCode iDMRG_Heisenberg::BuildBlockRight()
{
    PetscErrorCode  ierr = 0;

    LINALG_TOOLS__MATASSEMBLY_INIT();

    DMRG_TIMINGS_START(__FUNCT__);

    /*
        Prepare Sm as explicit Hermitian conjugate of Sp
        TODO: Implement as part of Kronecker product
    */
    Mat BlockRight_Sm;
    LINALG_TOOLS__MATASSEMBLY_FINAL(BlockRight_.Sp());
    ierr = MatHermitianTranspose(BlockRight_.Sp(), MAT_INITIAL_MATRIX, &BlockRight_Sm); CHKERRQ(ierr);

    /* Prepare the identity */
    Mat eye_R;
    PetscInt dim;
    ierr = MatGetSize(BlockRight_.H(), &dim, nullptr);
    ierr = MatEyeCreate(comm_, eye_R, dim);

    /*
        Update the Hamiltonian
    */
    Mat Mat_temp;
    ierr = MatKron(eye1_, BlockRight_.H(), Mat_temp, comm_); CHKERRQ(ierr);
    ierr = MatKronAdd(Sz1_, BlockRight_.Sz(), Mat_temp, comm_); CHKERRQ(ierr);
    ierr = MatKronScaleAdd(0.5, Sm1_, BlockRight_.Sp(), Mat_temp, comm_); CHKERRQ(ierr);
    ierr = MatKronScaleAdd(0.5, Sp1_, BlockRight_Sm, Mat_temp, comm_); CHKERRQ(ierr);
    ierr = BlockRight_.update_H(Mat_temp); /* H_temp is destroyed here */ CHKERRQ(ierr);
    ierr = MatDestroy(&BlockRight_Sm); CHKERRQ(ierr);
    Mat_temp = nullptr;

    /*
        Update the Sz operator
    */
    // ierr = MatKron(BlockRight_.Sz(), eye1_, Mat_temp, comm_); CHKERRQ(ierr);
    ierr = MatKron(Sz1_, eye_R, Mat_temp, comm_); CHKERRQ(ierr);
    ierr = BlockRight_.update_Sz(Mat_temp); CHKERRQ(ierr);
    Mat_temp = nullptr;

    /*
        Update the Sp operator
    */
    // ierr = MatKron(BlockRight_.Sp(), eye1_, Mat_temp, comm_); CHKERRQ(ierr);
    ierr = MatKron(Sp1_, eye_R, Mat_temp, comm_); CHKERRQ(ierr);
    ierr = BlockRight_.update_Sp(Mat_temp); CHKERRQ(ierr);
    Mat_temp = nullptr;

    BlockRight_.length(BlockRight_.length() + 1);
    if(!BlockRight_.is_valid()) SETERRQ(comm_, 1, "Invalid right block");
    #ifdef __PRINT_SIZES
        PetscPrintf(comm_, "%12sRight      basis size: %-5d nsites: %-5d \n", "", BlockRight_.basis_size(), BlockRight_.length());
    #endif
    ierr = MatDestroy(&BlockRight_Sm); CHKERRQ(ierr);
    ierr = MatDestroy(&eye_R); CHKERRQ(ierr);

    DMRG_TIMINGS_END(__FUNCT__);
    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "iDMRG_Heisenberg::BuildSuperBlock"
PetscErrorCode iDMRG_Heisenberg::BuildSuperBlock()
{
    PetscErrorCode  ierr = 0;
    DMRG_TIMINGS_START(__FUNCT__);

    Mat             mat_temp;
    PetscInt        M_left, M_right, M_superblock;

    /*
        Impose a checkpoint on the correctness of blocks
    */
    if(!BlockLeft_.is_valid()) SETERRQ(comm_, 1, "Invalid left block");
    if(!BlockRight_.is_valid()) SETERRQ(comm_, 1, "Invalid right block");


    /*
        Generic macro to destroy superblock and assign a null pointer
    */
    #define DESTROYSUPERBLOCKH \
        ierr = MatDestroy(&superblock_H_); CHKERRQ(ierr); \
        superblock_H_ = nullptr; \
        superblock_set_=PETSC_FALSE; \
        PetscPrintf(comm_, "Destroyed H\n");

    PetscInt M_A, N_A, M_B, N_B, M_C_req, N_C_req;
    ierr = MatGetSize(BlockLeft_.H(), &M_A, &N_A); CHKERRQ(ierr);
    ierr = MatGetSize(BlockRight_.H(), &M_B, &N_B); CHKERRQ(ierr);
    M_C_req = M_A * M_B;
    N_C_req = N_A * N_B;

    if (M_C_req != N_C_req)
        SETERRQ(comm_, 1, "Hamiltonian should be square. Check block operators from previous step.");

    /*
        OPTIMIZATION

        Check size of superblock
        If the new superblock size does not match the current superblock, delete and reallocate
        Otherwise, reuse the superblock_H_
        Also, preallocate the superblock with full size by
            determining the sizes of submatrices and
            using MPIAIJSetPreallocation
    */

    #if !defined(SUPERBLOCK_OPTIMIZATION)
        #define __OPTIMIZATION02 // default
    #elif SUPERBLOCK_OPTIMIZATION == 1
        #define __OPTIMIZATION01
    #elif SUPERBLOCK_OPTIMIZATION == 2
        #define __OPTIMIZATION02
    #elif SUPERBLOCK_OPTIMIZATION == 0

    #endif


    #if defined(__OPTIMIZATION01)

        #define SETUPSUPERBLOCKH \
            ierr = MatCreate(PETSC_COMM_WORLD, &superblock_H_); CHKERRQ(ierr); \
            ierr = MatSetSizes(superblock_H_, PETSC_DECIDE, PETSC_DECIDE, M_C_req, N_C_req); CHKERRQ(ierr); \
            ierr = MatSetFromOptions(superblock_H_); CHKERRQ(ierr); \
            ierr = MatSetUp(superblock_H_); CHKERRQ(ierr); \
            ierr = MatSetOption(superblock_H_, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE); \
            ierr = MatSetOption(superblock_H_, MAT_IGNORE_OFF_PROC_ENTRIES, PETSC_TRUE); \
            ierr = MatSetOption(superblock_H_, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE);

    #elif defined(__OPTIMIZATION02)

        #undef __OPTIMIZATION01 // defaults to OPTIMIZATION02

        PetscMPIInt     nprocs, rank;
        MPI_Comm_size(comm_, &nprocs);
        MPI_Comm_rank(comm_, &rank);

        PetscInt Istart, Iend, Irows, remrows, locrows, startrow;
        /* Guess layout where remrows are distributed to first few rows */
        remrows = M_C_req % nprocs;
        locrows = M_C_req / nprocs;
        startrow = locrows*rank;
        if (remrows > 0){
            if (rank < remrows){
                locrows += 1;
                startrow += rank;
            } else {
                startrow += remrows;
            }
        }

        #define SETUPSUPERBLOCKH \
            ierr = MatCreate(PETSC_COMM_WORLD, &superblock_H_); CHKERRQ(ierr); \
            ierr = MatSetSizes(superblock_H_, PETSC_DECIDE, PETSC_DECIDE, M_C_req, N_C_req); CHKERRQ(ierr); \
            ierr = MatSetFromOptions(superblock_H_); CHKERRQ(ierr); \
            ierr = MatMPIAIJSetPreallocation(superblock_H_, locrows+1, NULL, M_C_req - locrows+1, NULL); CHKERRQ(ierr); \
            ierr = MatSetOption(superblock_H_, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE); \
            ierr = MatSetOption(superblock_H_, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE); \
            ierr = MatSetOption(superblock_H_, MAT_IGNORE_OFF_PROC_ENTRIES, PETSC_TRUE); \
            ierr = MatSetOption(superblock_H_, MAT_KEEP_NONZERO_PATTERN, PETSC_TRUE); \
            ierr = MatGetOwnershipRange(superblock_H_, &Istart, &Iend); \
            Irows = Iend - Istart; \
            if(Irows != locrows) { SETERRQ(comm_, 1, "WRONG GUESS\n");}

    #endif // __OPTIMIZATION01 & __OPTIMIZATION02

    #if defined(__OPTIMIZATION01) || defined(__OPTIMIZATION02)

        if(superblock_set_==PETSC_TRUE || superblock_H_){
            PetscInt M_C, N_C;
            ierr = MatGetSize(superblock_H_, &M_C, &N_C); CHKERRQ(ierr);
            /*
                Conditions to reallocate the Hamiltonian matrix:
                    1.  size mismatch
                    2.  after performing the first truncation
                            where there is a sudden change in sparsity
            */
            if ( (M_C_req!=M_C) || (N_C_req!=N_C) || ntruncations_ == 1 ) {
                DESTROYSUPERBLOCKH
                SETUPSUPERBLOCKH
            } else{
                ierr = MatZeroEntries(superblock_H_); CHKERRQ(ierr);
            }
        } else {
            SETUPSUPERBLOCKH
        }

    #else // !defined(__OPTIMIZATION01) && !defined(__OPTIMIZATION02)

        if(superblock_set_==PETSC_TRUE || superblock_H_){
            DESTROYSUPERBLOCKH
        }

    #endif

    /*
        Update the Hamiltonian

        First term:  H_{L,i+1} \otimes 1_{DR×2}    ???? DRx2 ????

        Prepare mat_temp = Identity corresponding to right block
    */
    ierr = MatGetSize(BlockRight_.H(), &M_right, NULL); CHKERRQ(ierr);
    ierr = MatEyeCreate(comm_, mat_temp, M_right); CHKERRQ(ierr);

    #ifdef __KRON_TIMINGS
        PetscPrintf(PETSC_COMM_WORLD, "%40s %s\nSize: %10d x %-10d\n",
            __FUNCT__,"MatKron(BlockLeft_.H(), mat_temp, superblock_H_, comm_)",
            M_right*M_right,M_right*M_right);
    #endif

    ierr = MatKronAdd(BlockLeft_.H(), mat_temp, superblock_H_, comm_); CHKERRQ(ierr);

    #undef SETUPSUPERBLOCKH
    #undef DESTROYSUPERBLOCKH
    #undef __OPTIMIZATION01
    #undef __OPTIMIZATION02

    /*
        If the left and right sizes are the same, re-use the identity.
        Otherwise, create a new identity matrix with the correct size.
    */
    ierr = MatGetSize(BlockLeft_.H(), &M_left, NULL); CHKERRQ(ierr);
    if(M_left != M_right){
        ierr = MatDestroy(&mat_temp); CHKERRQ(ierr);
        ierr = MatEyeCreate(comm_, mat_temp, M_left); CHKERRQ(ierr);
    }
    /*
        Second term: 1_{DL×2} \otimes H_{R,i+2}
    */
    #ifdef __KRON_TIMINGS
        PetscPrintf(PETSC_COMM_WORLD, "%40s %s\nSize: %10d x %-10d\n", __FUNCT__,"MatKronAdd(mat_temp, BlockRight_.H(), superblock_H_, comm_)",M_left*M_left,M_left*M_left);
    #endif
    ierr = MatKronAdd(mat_temp, BlockRight_.H(), superblock_H_, comm_); CHKERRQ(ierr);
    /*
        Third term: S^z_{L,i+1} \otimes S^z_{R,i+2}
    */
    #ifdef __KRON_TIMINGS
        PetscPrintf(PETSC_COMM_WORLD, "%40s %s\nSize: %10d x %-10d\n", __FUNCT__,"MatKronAdd(BlockLeft_.Sz(), BlockRight_.Sz(), superblock_H_, comm_)",M_left*M_left,M_left*M_left);
    #endif
    ierr = MatKronAdd(BlockLeft_.Sz(), BlockRight_.Sz(), superblock_H_, comm_); CHKERRQ(ierr);
    /*
        Fourth term: 1/2 S^+_{L,i+1} \otimes S^-_{R,i+2}

        Prepare mat_temp = BlockRight_.Sm
    */
    ierr = MatDestroy(&mat_temp); CHKERRQ(ierr);
    ierr = MatAssemblyBegin(BlockRight_.Sp(), MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(BlockRight_.Sp(), MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatTranspose(BlockRight_.Sp(), MAT_INITIAL_MATRIX, &mat_temp); CHKERRQ(ierr);
    ierr = MatConjugate(mat_temp); CHKERRQ(ierr);
    ierr = MatKronScaleAdd(0.5, BlockLeft_.Sp(), mat_temp, superblock_H_, comm_); CHKERRQ(ierr);
    /*
        Fifth term: 1/2 S^-_{L,i+1} \otimes S^+_{R,i+2}

        Prepare mat_temp = BlockLeft_.Sm
    */
    ierr = MatDestroy(&mat_temp); CHKERRQ(ierr);
    ierr = MatAssemblyBegin(BlockLeft_.Sp(), MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(BlockLeft_.Sp(), MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatTranspose(BlockLeft_.Sp(), MAT_INITIAL_MATRIX, &mat_temp); CHKERRQ(ierr);
    ierr = MatConjugate(mat_temp); CHKERRQ(ierr);
    ierr = MatKronScaleAdd(0.5, mat_temp, BlockRight_.Sp(), superblock_H_, comm_); CHKERRQ(ierr);
    /*
        Clear temporary matrix
     */
    ierr = MatDestroy(&mat_temp); CHKERRQ(ierr);
    /*
        Checkpoint
    */
    superblock_set_ = PETSC_TRUE;
    ierr = MatGetSize(superblock_H_, &M_superblock, nullptr); CHKERRQ(ierr);
    if(M_superblock != TotalBasisSize()) SETERRQ(comm_, 1, "Basis size mismatch.\n");
    #ifdef __PRINT_SIZES
        PetscPrintf(comm_, "%12sSuperblock basis size: %-5d nsites: %-5d \n", "", M_superblock, BlockLeft_.length() + BlockRight_.length());
    #endif

    DMRG_TIMINGS_END(__FUNCT__);
    return ierr;
}
