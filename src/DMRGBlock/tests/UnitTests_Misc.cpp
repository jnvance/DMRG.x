#include <vector>
#include "petscmat.h"

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
