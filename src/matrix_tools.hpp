#ifndef __MATRIX_TOOLS_HPP__
#define __MATRIX_TOOLS_HPP__



#undef __FUNCT__
#define __FUNCT__ "MatEyeCreate"
PetscErrorCode MatEyeCreate(MPI_Comm comm, Mat& eye, PetscInt dim)
{
    PetscErrorCode  ierr = 0;

    MatCreate(comm, &eye);
    MatSetSizes(eye, PETSC_DECIDE, PETSC_DECIDE, dim, dim);
    MatSetFromOptions(eye);
    MatSetUp(eye);
    MatZeroEntries(eye);

    ierr = MatAssemblyBegin(eye, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(eye, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    MatShift(eye, 1.00);

    return ierr;
}


#undef __FUNCT__
#define __FUNCT__ "MatPeek"
PetscErrorCode MatPeek(MPI_Comm comm, Mat mat, const char* label)
{
    PetscErrorCode  ierr = 0;

    // Peek into values
    PetscViewer fd = nullptr;
    ierr = MatAssemblyBegin(mat, MAT_FLUSH_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    PetscPrintf(comm, "\n%s\n", label);
    ierr = MatView(mat, fd); CHKERRQ(ierr);

    return ierr;
}

#endif // __MATRIX_TOOLS_HPP__
