static char help[] =
    "Test code for DMRGKron routines\n";

#include <slepceps.h>
#include "linalg_tools.hpp"
#include "DMRGKron.hpp"
#include <iostream>

/* MiscTools.cpp */
PETSC_EXTERN PetscErrorCode InitSingleSiteOperator(const MPI_Comm& comm, const PetscInt dim, Mat* mat);

/* UnitTests_Misc.cpp */
PETSC_EXTERN PetscErrorCode SetRow(const Mat& A, const PetscInt& row, const std::vector<PetscInt>& idxn);
PETSC_EXTERN PetscErrorCode CheckRow(const Mat& A, const char* label, const PetscInt& row, const std::vector<PetscInt>& idxn,
    const std::vector<PetscScalar>& v);
PETSC_EXTERN const char hborder[];

#define PrintHeader(COMM,TEXT)  PetscPrintf((COMM), "%s\n%s\n%s\n", hborder, (TEXT), hborder)

static PetscBool verbose = PETSC_FALSE;

/** Combines two aritificially-created blocks and tests their contents */
PetscErrorCode TestKron01();

/** Combines small two single-site blocks together */
PetscErrorCode TestKron02();

int main(int argc, char **argv)
{
    PetscErrorCode  ierr = 0;
    ierr = SlepcInitialize(&argc, &argv, (char*)0, help); CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(NULL,NULL,"-verbose",&verbose,NULL); CHKERRQ(ierr);
    // ierr = TestKron01(); CHKERRQ(ierr);
    ierr = TestKron02(); CHKERRQ(ierr);
    ierr = SlepcFinalize(); CHKERRQ(ierr);
    return(0);
}

PetscErrorCode TestKron01()
{
    PetscErrorCode  ierr = 0;
    MPI_Comm&       comm = PETSC_COMM_WORLD;
    PetscMPIInt     nprocs, rank;
    ierr = MPI_Comm_size(comm, &nprocs); CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);

    Block::SpinBase RightBlock, LeftBlock, BlockOut;

    ierr = LeftBlock.Initialize(PETSC_COMM_WORLD, 3, {+0.5,-0.5}, {2,1}); CHKERRQ(ierr);

    SetRow( LeftBlock.Sz(0),0, {0, 1    });
    SetRow( LeftBlock.Sz(0),1, {   1    });
    SetRow( LeftBlock.Sz(0),2, {      2 });

    SetRow( LeftBlock.Sz(1),0, {0       });
    SetRow( LeftBlock.Sz(1),1, {   1    });
    SetRow( LeftBlock.Sz(1),2, {      2 });

    SetRow( LeftBlock.Sz(2),0, {   1    });
    SetRow( LeftBlock.Sz(2),1, {        });
    SetRow( LeftBlock.Sz(2),2, {      2 });

    SetRow( LeftBlock.Sp(0),0, {      2 });
    SetRow( LeftBlock.Sp(0),1, {        });
    SetRow( LeftBlock.Sp(0),2, {        });

    SetRow( LeftBlock.Sp(1),0, {      2 });
    SetRow( LeftBlock.Sp(1),1, {      2 });
    SetRow( LeftBlock.Sp(1),2, {        });

    SetRow( LeftBlock.Sp(2),0, {        });
    SetRow( LeftBlock.Sp(2),1, {      2 });
    SetRow( LeftBlock.Sp(2),2, {        });

    ierr = RightBlock.Initialize(PETSC_COMM_WORLD, 2, {+1.,0.,-1.}, {1,2,1}); CHKERRQ(ierr);

    SetRow(RightBlock.Sz(0),0, {0         });
    SetRow(RightBlock.Sz(0),1, {   1      });
    SetRow(RightBlock.Sz(0),2, {   1, 2   });
    SetRow(RightBlock.Sz(0),3, {         3});

    SetRow(RightBlock.Sz(1),0, {0         });
    SetRow(RightBlock.Sz(1),1, {   1, 2   });
    SetRow(RightBlock.Sz(1),2, {      2   });
    SetRow(RightBlock.Sz(1),3, {          });

    SetRow(RightBlock.Sp(0),0, {   1      });
    SetRow(RightBlock.Sp(0),1, {         3});
    SetRow(RightBlock.Sp(0),2, {         3});
    SetRow(RightBlock.Sp(0),3, {          });

    SetRow(RightBlock.Sp(1),0, {   1, 2   });
    SetRow(RightBlock.Sp(1),1, {          });
    SetRow(RightBlock.Sp(1),2, {         3});
    SetRow(RightBlock.Sp(1),3, {          });

    /* Calculate the Kronecker product of the blocks */
    ierr = KronEye_Explicit(LeftBlock, RightBlock, {}, BlockOut); CHKERRQ(ierr);

    if(verbose){
        ierr = MatPeek(BlockOut.Sz(0), "BlockOut.Sz(0)"); CHKERRQ(ierr);
        ierr = MatPeek(BlockOut.Sz(1), "BlockOut.Sz(1)"); CHKERRQ(ierr);
        ierr = MatPeek(BlockOut.Sz(2), "BlockOut.Sz(2)"); CHKERRQ(ierr);
        ierr = MatPeek(BlockOut.Sz(3), "BlockOut.Sz(3)"); CHKERRQ(ierr);
        ierr = MatPeek(BlockOut.Sz(4), "BlockOut.Sz(4)"); CHKERRQ(ierr);

        ierr = MatPeek(BlockOut.Sp(0), "BlockOut.Sp(0)"); CHKERRQ(ierr);
        ierr = MatPeek(BlockOut.Sp(1), "BlockOut.Sp(1)"); CHKERRQ(ierr);
        ierr = MatPeek(BlockOut.Sp(2), "BlockOut.Sp(2)"); CHKERRQ(ierr);
        ierr = MatPeek(BlockOut.Sp(3), "BlockOut.Sp(3)"); CHKERRQ(ierr);
        ierr = MatPeek(BlockOut.Sp(4), "BlockOut.Sp(4)"); CHKERRQ(ierr);
    }
    /* Check all blocks */
    ierr = BlockOut.CheckOperatorBlocks(); CHKERRQ(ierr);

    /* Check all entries */
    ierr = CheckRow(BlockOut.Sz(0), "BlockOut.Sz(0)", 0, {0,1}, {0,1}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(0), "BlockOut.Sz(0)", 1, {1}, {1}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(0), "BlockOut.Sz(0)", 2, {2,4}, {0,1}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(0), "BlockOut.Sz(0)", 3, {3,5}, {0,1}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(0), "BlockOut.Sz(0)", 4, {  4}, {  1}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(0), "BlockOut.Sz(0)", 5, {  5}, {  1}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(0), "BlockOut.Sz(0)", 6, {6}, {2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(0), "BlockOut.Sz(0)", 7, {7,8}, {0,1}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(0), "BlockOut.Sz(0)", 8, {8}, {1}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(0), "BlockOut.Sz(0)", 9, {9}, {2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(0), "BlockOut.Sz(0)", 10, {10}, {2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(0), "BlockOut.Sz(0)", 11, {11}, {2}); CHKERRQ(ierr);

    ierr = CheckRow(BlockOut.Sz(1), "BlockOut.Sz(1)", 0, {0}, {0}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(1), "BlockOut.Sz(1)", 1, {1}, {1}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(1), "BlockOut.Sz(1)", 2, {2}, {0}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(1), "BlockOut.Sz(1)", 3, {3}, {0}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(1), "BlockOut.Sz(1)", 4, {4}, {1}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(1), "BlockOut.Sz(1)", 5, {5}, {1}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(1), "BlockOut.Sz(1)", 6, {6}, {2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(1), "BlockOut.Sz(1)", 7, {7}, {0}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(1), "BlockOut.Sz(1)", 8, {8}, {1}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(1), "BlockOut.Sz(1)", 9, {9}, {2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(1), "BlockOut.Sz(1)", 10, {10}, {2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(1), "BlockOut.Sz(1)", 11, {11}, {2}); CHKERRQ(ierr);

    ierr = CheckRow(BlockOut.Sz(2), "BlockOut.Sz(2)", 0, {1}, {1}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(2), "BlockOut.Sz(2)", 1, {}, {}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(2), "BlockOut.Sz(2)", 2, {4}, {1}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(2), "BlockOut.Sz(2)", 3, {5}, {1}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(2), "BlockOut.Sz(2)", 4, {}, {}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(2), "BlockOut.Sz(2)", 5, {}, {}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(2), "BlockOut.Sz(2)", 6, {6}, {2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(2), "BlockOut.Sz(2)", 7, {8}, {1}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(2), "BlockOut.Sz(2)", 8, {}, {}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(2), "BlockOut.Sz(2)", 9, {9}, {2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(2), "BlockOut.Sz(2)", 10, {10}, {2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(2), "BlockOut.Sz(2)", 11, {11}, {2}); CHKERRQ(ierr);

    ierr = CheckRow(BlockOut.Sp(0), "BlockOut.Sp(0)", 0, {6}, {2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(0), "BlockOut.Sp(0)", 1, {}, {}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(0), "BlockOut.Sp(0)", 2, {9}, {2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(0), "BlockOut.Sp(0)", 3, {10}, {2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(0), "BlockOut.Sp(0)", 4, {}, {}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(0), "BlockOut.Sp(0)", 5, {}, {}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(0), "BlockOut.Sp(0)", 6, {}, {}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(0), "BlockOut.Sp(0)", 7, {11}, {2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(0), "BlockOut.Sp(0)", 8, {}, {}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(0), "BlockOut.Sp(0)", 9, {}, {}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(0), "BlockOut.Sp(0)", 10, {}, {}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(0), "BlockOut.Sp(0)", 11, {}, {}); CHKERRQ(ierr);

    ierr = CheckRow(BlockOut.Sp(1), "BlockOut.Sp(1)", 0, {6}, {2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(1), "BlockOut.Sp(1)", 1, {6}, {2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(1), "BlockOut.Sp(1)", 2, {9}, {2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(1), "BlockOut.Sp(1)", 3, {10}, {2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(1), "BlockOut.Sp(1)", 4, {9}, {2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(1), "BlockOut.Sp(1)", 5, {10}, {2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(1), "BlockOut.Sp(1)", 6, {}, {}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(1), "BlockOut.Sp(1)", 7, {11}, {2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(1), "BlockOut.Sp(1)", 8, {11}, {2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(1), "BlockOut.Sp(1)", 9, {}, {}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(1), "BlockOut.Sp(1)", 10, {}, {}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(1), "BlockOut.Sp(1)", 11, {}, {}); CHKERRQ(ierr);

    ierr = CheckRow(BlockOut.Sp(2), "BlockOut.Sp(2)", 0, {}, {}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(2), "BlockOut.Sp(2)", 1, {6}, {2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(2), "BlockOut.Sp(2)", 2, {}, {}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(2), "BlockOut.Sp(2)", 3, {}, {}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(2), "BlockOut.Sp(2)", 4, {9}, {2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(2), "BlockOut.Sp(2)", 5, {10}, {2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(2), "BlockOut.Sp(2)", 6, {}, {}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(2), "BlockOut.Sp(2)", 7, {}, {}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(2), "BlockOut.Sp(2)", 8, {11}, {2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(2), "BlockOut.Sp(2)", 9, {}, {}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(2), "BlockOut.Sp(2)", 10, {}, {}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(2), "BlockOut.Sp(2)", 11, {}, {}); CHKERRQ(ierr);

    ierr = CheckRow(BlockOut.Sz(3), "BlockOut.Sz(3)", 0, {0}, {0}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(3), "BlockOut.Sz(3)", 1, {1}, {0}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(3), "BlockOut.Sz(3)", 2, {2}, {1}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(3), "BlockOut.Sz(3)", 3, {2, 3}, {1, 2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(3), "BlockOut.Sz(3)", 4, {4}, {1}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(3), "BlockOut.Sz(3)", 5, {4,5}, {1,2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(3), "BlockOut.Sz(3)", 6, {6}, {0}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(3), "BlockOut.Sz(3)", 7, {7}, {3}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(3), "BlockOut.Sz(3)", 8, {8}, {3}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(3), "BlockOut.Sz(3)", 9, {9}, {1}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(3), "BlockOut.Sz(3)", 10, {9,10}, {1,2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(3), "BlockOut.Sz(3)", 11, {11}, {3}); CHKERRQ(ierr);

    ierr = CheckRow(BlockOut.Sz(4), "BlockOut.Sz(4)", 0, {0}, {0}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(4), "BlockOut.Sz(4)", 1, {1}, {0}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(4), "BlockOut.Sz(4)", 2, {2,3}, {1,2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(4), "BlockOut.Sz(4)", 3, {3}, {2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(4), "BlockOut.Sz(4)", 4, {4,5}, {1,2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(4), "BlockOut.Sz(4)", 5, {5}, {2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(4), "BlockOut.Sz(4)", 6, {6}, {0}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(4), "BlockOut.Sz(4)", 7, {}, {}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(4), "BlockOut.Sz(4)", 8, {}, {}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(4), "BlockOut.Sz(4)", 9, {9,10}, {1,2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(4), "BlockOut.Sz(4)", 10, {10}, {2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sz(4), "BlockOut.Sz(4)", 11, {}, {}); CHKERRQ(ierr);

    ierr = CheckRow(BlockOut.Sp(3), "BlockOut.Sp(3)", 0, {2}, {1}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(3), "BlockOut.Sp(3)", 1, {4}, {1}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(3), "BlockOut.Sp(3)", 2, {7}, {3}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(3), "BlockOut.Sp(3)", 3, {7}, {3}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(3), "BlockOut.Sp(3)", 4, {8}, {3}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(3), "BlockOut.Sp(3)", 5, {8}, {3}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(3), "BlockOut.Sp(3)", 6, {9}, {1}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(3), "BlockOut.Sp(3)", 7, {}, {}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(3), "BlockOut.Sp(3)", 8, {}, {}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(3), "BlockOut.Sp(3)", 9, {11}, {3}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(3), "BlockOut.Sp(3)", 10, {11}, {3}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(3), "BlockOut.Sp(3)", 11, {}, {}); CHKERRQ(ierr);

    ierr = CheckRow(BlockOut.Sp(4), "BlockOut.Sp(4)", 0, {2,3}, {1,2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(4), "BlockOut.Sp(4)", 1, {4,5}, {1,2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(4), "BlockOut.Sp(4)", 2, {}, {}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(4), "BlockOut.Sp(4)", 3, {7}, {3}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(4), "BlockOut.Sp(4)", 4, {}, {}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(4), "BlockOut.Sp(4)", 5, {8}, {3}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(4), "BlockOut.Sp(4)", 6, {9,10}, {1,2}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(4), "BlockOut.Sp(4)", 7, {}, {}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(4), "BlockOut.Sp(4)", 8, {}, {}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(4), "BlockOut.Sp(4)", 9, {}, {}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(4), "BlockOut.Sp(4)", 10, {11}, {3}); CHKERRQ(ierr);
    ierr = CheckRow(BlockOut.Sp(4), "BlockOut.Sp(4)", 11, {}, {}); CHKERRQ(ierr);

    ierr = RightBlock.Destroy(); CHKERRQ(ierr);
    ierr = LeftBlock.Destroy(); CHKERRQ(ierr);
    ierr = BlockOut.Destroy(); CHKERRQ(ierr);

    return(0);
}


PetscErrorCode TestKron02()
{

    PetscErrorCode  ierr = 0;
    MPI_Comm&       comm = PETSC_COMM_WORLD;
    PetscMPIInt     nprocs, rank;
    ierr = MPI_Comm_size(comm, &nprocs); CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);

    Block::SpinBase RightBlock, LeftBlock, BlockOut;

    ierr = LeftBlock.Initialize(PETSC_COMM_WORLD, 1, PETSC_DEFAULT); CHKERRQ(ierr);
    ierr = RightBlock.Initialize(PETSC_COMM_WORLD, 1, PETSC_DEFAULT); CHKERRQ(ierr);

    ierr = KronEye_Explicit(LeftBlock, RightBlock, {}, BlockOut); CHKERRQ(ierr);

    ierr = RightBlock.Destroy(); CHKERRQ(ierr);
    ierr = LeftBlock.Destroy(); CHKERRQ(ierr);
    ierr = BlockOut.Destroy(); CHKERRQ(ierr);

    return(0);
}
