static char help[] =
    "Test code for DMRGBlock module\n";

#include <slepceps.h>
#include "DMRGBlock.hpp"
#include "linalg_tools.hpp"
#include <iostream>

PETSC_EXTERN PetscErrorCode SetRow(const Mat& A, const PetscInt& row, const std::vector<PetscInt>& idxn);
PETSC_EXTERN PetscErrorCode Makedir(const std::string& dir_name);
PETSC_EXTERN PetscErrorCode SetSz0(const Mat& Sz);
PETSC_EXTERN PetscErrorCode SetSp0(const Mat& Sp);
PETSC_EXTERN PetscErrorCode SetSz1(const Mat& Sz);
PETSC_EXTERN PetscErrorCode SetSp1(const Mat& Sp);

static char hborder[] = //"************************************************************"
                        //"************************************************************";
                        "------------------------------------------------------------"
                        "------------------------------------------------------------";

#define PrintHeader(COMM,TEXT) \
    ierr = PetscPrintf((COMM), "%s\n%s\n%s\n", hborder, (TEXT), hborder); CHKERRQ(ierr);


/** Tests initialization and ownership transfer*/
PetscErrorCode Test_InitAndCopy()
{
    PetscErrorCode ierr = 0;

    Block::SpinBase blk;
    ierr = blk.Initialize(PETSC_COMM_WORLD, 2, {1.5,0.5,-0.5,-1.5}, {2,3,2,1});CHKERRQ(ierr);
    ierr = blk.CheckSectors(); CHKERRQ(ierr);

    {
        /*  Set the entries of Sz(0) following the correct sectors */
        ierr = SetSz0(blk.Sz(0)); CHKERRQ(ierr);
        ierr = MatPeek(blk.Sz(0), "blk.Sz(0)"); CHKERRQ(ierr);
        ierr = blk.MatOpCheckOperatorBlocks(OpSz, 0); CHKERRQ(ierr);

        /*  Set the entries of Sp(0) following the correct sectors */
        ierr = SetSp0(blk.Sp(0)); CHKERRQ(ierr);
        ierr = MatPeek(blk.Sp(0), "blk.Sp(0)"); CHKERRQ(ierr);
        ierr = blk.MatOpCheckOperatorBlocks(OpSp, 0); CHKERRQ(ierr);

        /*  Set the entries of Sz(1) following the correct sectors */
        ierr = SetSz1(blk.Sz(1)); CHKERRQ(ierr);
        ierr = MatPeek(blk.Sz(1), "blk.Sz(1)"); CHKERRQ(ierr);
        ierr = blk.MatOpCheckOperatorBlocks(OpSz, 1); CHKERRQ(ierr);

        /*  Set the entries of Sp(1) following the correct sectors */
        ierr = SetSp1(blk.Sp(1)); CHKERRQ(ierr);
        ierr = MatPeek(blk.Sp(1), "blk.Sp(1)"); CHKERRQ(ierr);
        ierr = blk.MatOpCheckOperatorBlocks(OpSp, 1); CHKERRQ(ierr);
    }

    /* Copy to blk2 */
    Block::SpinBase blk2 = blk;

    ierr = MatPeek(blk2.Sz(0), "blk2.Sz(0)"); CHKERRQ(ierr);
    ierr = MatPeek(blk2.Sp(0), "blk2.Sp(0)"); CHKERRQ(ierr);
    ierr = MatPeek(blk2.Sz(1), "blk2.Sz(1)"); CHKERRQ(ierr);
    ierr = MatPeek(blk2.Sp(1), "blk2.Sp(1)"); CHKERRQ(ierr);

    ierr = blk2.MatOpCheckOperatorBlocks(OpSz, 0); CHKERRQ(ierr);
    ierr = blk2.MatOpCheckOperatorBlocks(OpSp, 0); CHKERRQ(ierr);
    ierr = blk2.MatOpCheckOperatorBlocks(OpSz, 1); CHKERRQ(ierr);
    ierr = blk2.MatOpCheckOperatorBlocks(OpSp, 1); CHKERRQ(ierr);

    /* Call Destroy only on blk2 which should also destroy matrices in blk */
    ierr = blk2.Destroy(); CHKERRQ(ierr);

    return ierr;
}

/** Tests the CheckOperatorBlocks method */
PetscErrorCode Test_MatOpCheckOperatorBlocks()
{
    PetscErrorCode ierr = 0;

    Block::SpinBase blk;
    ierr = blk.Initialize(PETSC_COMM_WORLD, 2, {1.5,0.5,-0.5,-1.5}, {2,3,2,1});CHKERRQ(ierr);
    ierr = blk.CheckSectors(); CHKERRQ(ierr);

    /*  Set the entries of Sz(0) following the correct sectors */
    ierr = SetRow(blk.Sz(0), 0, {0,1}); CHKERRQ(ierr);
    ierr = SetRow(blk.Sz(0), 1, {1}); CHKERRQ(ierr);
    ierr = SetRow(blk.Sz(0), 2, {2,3,4}); CHKERRQ(ierr);
    ierr = SetRow(blk.Sz(0), 3, {3}); CHKERRQ(ierr);
    ierr = SetRow(blk.Sz(0), 4, {2,4}); CHKERRQ(ierr);
    ierr = SetRow(blk.Sz(0), 5, {5,6}); CHKERRQ(ierr);
    ierr = SetRow(blk.Sz(0), 7, {7}); CHKERRQ(ierr);
    ierr = MatPeek(blk.Sz(0), "blk.Sz(0)"); CHKERRQ(ierr);
    ierr = blk.MatOpCheckOperatorBlocks(OpSz, 0); CHKERRQ(ierr);

    /*  Set the entries of Sp(0) following the correct sectors */
    ierr = SetRow(blk.Sp(0), 0, {2,3,4}); CHKERRQ(ierr);
    ierr = SetRow(blk.Sp(0), 1, {2,4}); CHKERRQ(ierr);
    ierr = SetRow(blk.Sp(0), 2, {5,6}); CHKERRQ(ierr);
    ierr = SetRow(blk.Sp(0), 3, {5}); CHKERRQ(ierr);
    ierr = SetRow(blk.Sp(0), 4, {6}); CHKERRQ(ierr);
    ierr = SetRow(blk.Sp(0), 6, {7}); CHKERRQ(ierr);
    ierr = MatPeek(blk.Sp(0), "blk.Sp(0)"); CHKERRQ(ierr);
    ierr = blk.MatOpCheckOperatorBlocks(OpSp, 0); CHKERRQ(ierr);

    /*  Set the entries of Sz(1) following the INCORRECT sectors */
    ierr = SetRow(blk.Sz(1), 0, {0,1}); CHKERRQ(ierr);
    ierr = SetRow(blk.Sz(1), 1, {1}); CHKERRQ(ierr);
    ierr = SetRow(blk.Sz(1), 2, {2,3,4}); CHKERRQ(ierr);
    ierr = SetRow(blk.Sz(1), 3, {3}); CHKERRQ(ierr);
    ierr = SetRow(blk.Sz(1), 4, {2,4}); CHKERRQ(ierr);
    ierr = SetRow(blk.Sz(1), 5, {5,6}); CHKERRQ(ierr);
    ierr = SetRow(blk.Sz(1), 7, {1,7}); CHKERRQ(ierr); /* Error here */
    ierr = MatPeek(blk.Sz(1), "blk.Sz(1)"); CHKERRQ(ierr);
    ierr = blk.MatOpCheckOperatorBlocks(OpSz, 1);
    /*  Verify that error has been caught */
    {
        PetscInt rstart, rend;
        MatGetOwnershipRange(blk.Sz(1), &rstart, &rend);
        if(rstart <= 7 && 7 < rend){
            if(ierr!=PETSC_ERR_ARG_OUTOFRANGE){
                SETERRQ(PETSC_COMM_SELF, 1, "Failed test");
            } else {
                printf("%s\n",hborder);
                printf("%s   Exception caught.\n",__FUNCTION__);
                printf("%s\n",hborder);
            }
        }
    }
    ierr = blk.Destroy(); CHKERRQ(ierr);
    return ierr;
}

PetscErrorCode Test_SavingBlocks()
{
    PetscErrorCode ierr = 0;

    ierr = Makedir("trash_block_test/"); CHKERRQ(ierr);

    Block::SpinBase blk;
    ierr = blk.Initialize(PETSC_COMM_WORLD, 2, {1.5,0.5,-0.5,-1.5}, {2,3,2,1});CHKERRQ(ierr);
    ierr = blk.InitializeSave("trash_block_test/"); CHKERRQ(ierr);
    ierr = blk.CheckSectors(); CHKERRQ(ierr);
    {
        /*  Set the entries of Sz(0) following the correct sectors */
        ierr = SetSz0(blk.Sz(0)); CHKERRQ(ierr);
        ierr = blk.MatOpCheckOperatorBlocks(OpSz, 0); CHKERRQ(ierr);

        /*  Set the entries of Sp(0) following the correct sectors */
        ierr = SetSp0(blk.Sp(0)); CHKERRQ(ierr);
        ierr = blk.MatOpCheckOperatorBlocks(OpSp, 0); CHKERRQ(ierr);

        /*  Set the entries of Sz(1) following the correct sectors */
        ierr = SetSz1(blk.Sz(1)); CHKERRQ(ierr);
        ierr = blk.MatOpCheckOperatorBlocks(OpSz, 1); CHKERRQ(ierr);

        /*  Set the entries of Sp(1) following the correct sectors */
        ierr = SetSp1(blk.Sp(1)); CHKERRQ(ierr);
        ierr = blk.MatOpCheckOperatorBlocks(OpSp, 1); CHKERRQ(ierr);
    }

    ierr = blk.SaveAndDestroy(); CHKERRQ(ierr);

    /* Ensure that matrices are null */
    for(PetscInt isite = 0; isite < blk.NumSites(); ++isite){
        if(blk.Sz(isite)!=NULL) SETERRQ1(PETSC_COMM_SELF,1,"Matrix Sz(%d) must be null.", isite);
        if(blk.Sp(isite)!=NULL) SETERRQ1(PETSC_COMM_SELF,1,"Matrix Sp(%d) must be null.", isite);
    }
    if(blk.H!=NULL) SETERRQ(PETSC_COMM_SELF,1,"Matrix H must be null.");

    ierr = blk.Retrieve(); CHKERRQ(ierr);
    /* Ensure that retrieved matrices have the same entries */
    {
        Block::SpinBase blk2;
        ierr = blk2.Initialize(PETSC_COMM_WORLD, 2, {1.5,0.5,-0.5,-1.5}, {2,3,2,1});CHKERRQ(ierr);
        ierr = blk2.InitializeSave("trash_block_test/"); CHKERRQ(ierr);
        ierr = blk2.CheckSectors(); CHKERRQ(ierr);
        PetscBool flg;
        ierr = SetSz0(blk2.Sz(0)); CHKERRQ(ierr);
        ierr = SetSp0(blk2.Sp(0)); CHKERRQ(ierr);
        ierr = SetSz1(blk2.Sz(1)); CHKERRQ(ierr);
        ierr = SetSp1(blk2.Sp(1)); CHKERRQ(ierr);
        ierr = blk2.AssembleOperators(); CHKERRQ(ierr);

        ierr = MatEqual(blk2.Sz(0), blk.Sz(0), &flg); CHKERRQ(ierr);
        if(!flg) SETERRQ(PETSC_COMM_WORLD,1,"Wrong retrieval of matrix Sz0");
        ierr = MatEqual(blk2.Sp(0), blk.Sp(0), &flg); CHKERRQ(ierr);
        if(!flg) SETERRQ(PETSC_COMM_WORLD,1,"Wrong retrieval of matrix Sp0");
        ierr = MatEqual(blk2.Sz(1), blk.Sz(1), &flg); CHKERRQ(ierr);
        if(!flg) SETERRQ(PETSC_COMM_WORLD,1,"Wrong retrieval of matrix Sz1");
        ierr = MatEqual(blk2.Sp(1), blk.Sp(1), &flg); CHKERRQ(ierr);
        if(!flg) SETERRQ(PETSC_COMM_WORLD,1,"Wrong retrieval of matrix Sp1");

        ierr = blk2.Destroy(); CHKERRQ(ierr);
    }

    ierr = blk.Destroy(); CHKERRQ(ierr);

    return(0);
}


int main(int argc, char **argv)
{
    PetscErrorCode  ierr = 0;
    PetscMPIInt     nprocs, rank;
    MPI_Comm&       comm = PETSC_COMM_WORLD;

    /*  Initialize MPI  */
    ierr = SlepcInitialize(&argc, &argv, (char*)0, help); CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm, &nprocs); CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);

    PrintHeader(comm, "Test 01: Test_InitAndCopy");
    ierr = Test_InitAndCopy(); CHKERRQ(ierr);
    PrintHeader(comm, "Test 02: Test_SavingBlocks");
    ierr = Test_SavingBlocks(); CHKERRQ(ierr);
    PrintHeader(comm, "Test 03: Test_MatOpCheckOperatorBlocks");
    ierr = Test_MatOpCheckOperatorBlocks(); CHKERRQ(ierr);

    ierr = SlepcFinalize(); CHKERRQ(ierr);
    return ierr;
}
