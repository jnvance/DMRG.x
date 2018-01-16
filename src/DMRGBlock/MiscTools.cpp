#include <petscsys.h>
#include <slepceps.h>

/* Obtained from: https://gist.github.com/orlp/3551590 */
PETSC_EXTERN int64_t ipow(int64_t base, uint8_t exp) {
    static const uint8_t highest_bit_set[] = {
        0, 1, 2, 2, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4,
        5, 5, 5, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 255, // anything past 63 is a guaranteed overflow with base > 1
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
    };

    uint64_t result = 1;

    switch (highest_bit_set[exp]) {
    case 255: // we use 255 as an overflow marker and return 0 on overflow/underflow
        if (base == 1) {
            return 1;
        }

        if (base == -1) {
            return 1 - 2 * (exp & 1);
        }

        return 0;
    case 6:
        if (exp & 1) result *= base;
        exp >>= 1;
        base *= base;
    case 5:
        if (exp & 1) result *= base;
        exp >>= 1;
        base *= base;
    case 4:
        if (exp & 1) result *= base;
        exp >>= 1;
        base *= base;
    case 3:
        if (exp & 1) result *= base;
        exp >>= 1;
        base *= base;
    case 2:
        if (exp & 1) result *= base;
        exp >>= 1;
        base *= base;
    case 1:
        if (exp & 1) result *= base;
    default:
        return result;
    }
}


PETSC_EXTERN PetscErrorCode PreSplitOwnership(const MPI_Comm comm, const PetscInt N, PetscInt& locrows, PetscInt& Istart)
{
    PetscErrorCode ierr = 0;

    PetscInt Nsize = N;
    PetscInt Lrows = PETSC_DECIDE;
    ierr = PetscSplitOwnership(comm, &Lrows, &Nsize); CHKERRQ(ierr);

    Istart = 0;
    ierr = MPI_Exscan(&Lrows, &Istart, 1, MPIU_INT, MPI_SUM, comm); CHKERRQ(ierr);
    locrows = Lrows;

    return ierr;
}


PETSC_EXTERN PetscErrorCode InitSingleSiteOperator(const MPI_Comm& comm, const PetscInt dim, Mat* mat)
{
    PetscErrorCode ierr = 0;

    if(*mat) SETERRQ(comm, 1, "Matrix was previously initialized. First, destroy the matrix and set to NULL.");

    ierr = MatCreate(comm, mat); CHKERRQ(ierr);
    ierr = MatSetSizes(*mat, PETSC_DECIDE, PETSC_DECIDE, dim, dim); CHKERRQ(ierr);
    ierr = MatSetFromOptions(*mat); CHKERRQ(ierr);
    ierr = MatSetUp(*mat); CHKERRQ(ierr);

    ierr = MatSetOption(*mat, MAT_NO_OFF_PROC_ENTRIES          , PETSC_TRUE);
    ierr = MatSetOption(*mat, MAT_NO_OFF_PROC_ZERO_ROWS        , PETSC_TRUE);
    ierr = MatSetOption(*mat, MAT_IGNORE_OFF_PROC_ENTRIES      , PETSC_TRUE);
    ierr = MatSetOption(*mat, MAT_IGNORE_ZERO_ENTRIES          , PETSC_TRUE);

    return ierr;
}


/*----- Spin-1/2 functions -----*/

#undef __FUNCT__
#define __FUNCT__ "MatSzCreate"
PETSC_EXTERN PetscErrorCode MatSpinOneHalfSzCreate(const MPI_Comm& comm, Mat& Sz)
{
    PetscErrorCode  ierr = 0;

    PetscInt loc_dim = 2;
    ierr = InitSingleSiteOperator(comm, loc_dim, &Sz); CHKERRQ(ierr);

    PetscInt locrows, Istart;
    ierr = PreSplitOwnership(comm, loc_dim, locrows, Istart); CHKERRQ(ierr);
    PetscInt Iend = Istart + locrows;

    if (Istart <= 0 && 0 < Iend){
        ierr = MatSetValue(Sz, 0, 0, +0.5, INSERT_VALUES); CHKERRQ(ierr);
    }
    if (Istart <= 1 && 1 < Iend){
        ierr = MatSetValue(Sz, 1, 1, -0.5, INSERT_VALUES); CHKERRQ(ierr);
    }

    ierr = MatAssemblyBegin(Sz, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Sz, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "MatSpCreate"
PETSC_EXTERN PetscErrorCode MatSpinOneHalfSpCreate(const MPI_Comm& comm, Mat& Sp)
{
    PetscErrorCode  ierr = 0;

    PetscInt loc_dim = 2;
    ierr = InitSingleSiteOperator(comm, loc_dim, &Sp); CHKERRQ(ierr);

    PetscInt locrows, Istart;
    ierr = PreSplitOwnership(comm, loc_dim, locrows, Istart); CHKERRQ(ierr);
    PetscInt Iend = Istart + locrows;

    if (Istart <= 0 && 0 < Iend){
        ierr = MatSetValue(Sp, 0, 1, +1.0, INSERT_VALUES); CHKERRQ(ierr);
    }

    ierr = MatAssemblyBegin(Sp, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Sp, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    return ierr;
}


PETSC_EXTERN PetscErrorCode MatEnsureAssembled(const Mat& matin)
{
    PetscErrorCode ierr = 0;

    PetscBool assembled;
    ierr = MatAssembled(matin, &assembled); CHKERRQ(ierr);
    if(!assembled)
    {
        ierr = MatAssemblyBegin(matin, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(matin, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }

    return ierr;
}
