#include <vector>
#include "petscmat.h"

PETSC_EXTERN const char hborder[] = "------------------------------------------------------------"
                                     "------------------------------------------------------------";

PETSC_EXTERN PetscErrorCode SetRow(const Mat& A, const PetscInt& row, const std::vector<PetscInt>& idxn)
{
    PetscErrorCode ierr = 0;
    /*  Check row ownership range */
    PetscInt rstart, rend;
    ierr = MatGetOwnershipRange(A, &rstart, &rend); CHKERRQ(ierr);
    if(rstart <= row && row < rend)
    {
        std::vector<PetscInt> idxm = {row};
        std::vector<PetscReal> v;
        for(const PetscInt i: idxn) v.push_back(PetscReal(i));
        ierr = MatSetValues(A, 1, &idxm.front(), idxn.size(), &idxn.front(), &v.front(), INSERT_VALUES); CHKERRQ(ierr);
    }
    return ierr;
}

PETSC_EXTERN PetscErrorCode CheckRow(const Mat& A, const char* label, const PetscInt& row, const std::vector<PetscInt>& idxn,
    const std::vector<PetscScalar>& v, const PetscBool& CheckZeros = PETSC_FALSE)
{
    PetscErrorCode ierr = 0;

    /*  Check row ownership range */
    PetscInt rstart, rend;
    ierr = MatGetOwnershipRange(A, &rstart, &rend); CHKERRQ(ierr);

    if(rstart <= row && row < rend)
    {
        if(idxn.size()!=v.size())
            SETERRQ3(PETSC_COMM_SELF, 1, "On matrix %s: Input size mismatch: idxn (%d) != v (%d).", label, idxn.size(), v.size());

        std::vector<PetscInt> idxm = {row};
        PetscInt ncols;
        const PetscInt *cols;
        const PetscScalar *vals;
        ierr = MatGetRow(A, row, &ncols, &cols, &vals); CHKERRQ(ierr);

        /* Compare cols and vals with idxn and v */
        if (size_t(ncols) != idxn.size())
            SETERRQ4(PETSC_COMM_SELF, 1, "On matrix %s: Error at row %d: ncols (%d) != idxn.size (%d).", label, row, ncols, idxn.size());

        /* Compare column indices */
        for(PetscInt i = 0; i < ncols; ++i)
        {
            if(cols[i] != idxn[i])
                SETERRQ5(PETSC_COMM_SELF, 1, "On matrix %s: Error at row %d idx %d: cols (%d) != idxn (%d).",
                    label, row, i, cols[i], idxn[i] );
        }

        /* Compare values */
        for(PetscInt i = 0; i < ncols; ++i)
        {
            if(!PetscEqualScalar(vals[i], v[i]))
                SETERRQ5(PETSC_COMM_SELF, 1, "On matrix %s: Error at row %d idx %d: vals (%g) != v (%g).",
                    label, row, i, vals[i], v[i] );
        }

        ierr = MatRestoreRow(A, row, &ncols, &cols, &vals); CHKERRQ(ierr);
    }
    return ierr;
}

PETSC_EXTERN PetscErrorCode CatchErrorCode(const MPI_Comm& comm, const PetscInt& ierr_in, const PetscInt& ierr_exp)
{
    PetscErrorCode ierr = 0;

    if(ierr_in != ierr_exp)
    {
        SETERRQ2(comm, 1, "Failed test. Expected error code %d. Got %d.", ierr_exp, ierr_in);
    } else {
        PetscPrintf(comm, "\n    Exception caught. :)\n\n");
    }

    return ierr;
}

PETSC_EXTERN PetscErrorCode SetSz0(const Mat& Sz)
{
    PetscErrorCode ierr = 0;

    ierr = SetRow(Sz, 0, {0,1}); CHKERRQ(ierr);
    ierr = SetRow(Sz, 1, {1}); CHKERRQ(ierr);
    ierr = SetRow(Sz, 2, {2,3,4}); CHKERRQ(ierr);
    ierr = SetRow(Sz, 3, {3}); CHKERRQ(ierr);
    ierr = SetRow(Sz, 4, {2,4}); CHKERRQ(ierr);
    ierr = SetRow(Sz, 5, {5,6}); CHKERRQ(ierr);
    ierr = SetRow(Sz, 7, {7}); CHKERRQ(ierr);

    return ierr;
}

PETSC_EXTERN PetscErrorCode SetSp0(const Mat& Sp)
{
    PetscErrorCode ierr = 0;

    ierr = SetRow(Sp, 0, {2,3,4}); CHKERRQ(ierr);
    ierr = SetRow(Sp, 1, {2,4}); CHKERRQ(ierr);
    ierr = SetRow(Sp, 2, {5,6}); CHKERRQ(ierr);
    ierr = SetRow(Sp, 3, {5}); CHKERRQ(ierr);
    ierr = SetRow(Sp, 4, {6}); CHKERRQ(ierr);
    ierr = SetRow(Sp, 6, {7}); CHKERRQ(ierr);

    return ierr;
}

PETSC_EXTERN PetscErrorCode SetSz1(const Mat& Sz)
{
    PetscErrorCode ierr = 0;

    ierr = SetRow(Sz, 0, {0,1}); CHKERRQ(ierr);
    ierr = SetRow(Sz, 1, {1}); CHKERRQ(ierr);
    ierr = SetRow(Sz, 3, {3}); CHKERRQ(ierr);
    ierr = SetRow(Sz, 4, {2,4}); CHKERRQ(ierr);
    ierr = SetRow(Sz, 5, {5,6}); CHKERRQ(ierr);
    ierr = SetRow(Sz, 7, {7}); CHKERRQ(ierr);

    return ierr;
}

PETSC_EXTERN PetscErrorCode SetSp1(const Mat& Sp)
{
    PetscErrorCode ierr = 0;

    ierr = SetRow(Sp, 0, {2,3,4}); CHKERRQ(ierr);
    ierr = SetRow(Sp, 1, {2,4}); CHKERRQ(ierr);
    ierr = SetRow(Sp, 2, {5,6}); CHKERRQ(ierr);
    ierr = SetRow(Sp, 4, {6}); CHKERRQ(ierr);
    ierr = SetRow(Sp, 6, {7}); CHKERRQ(ierr);

    return ierr;
}
