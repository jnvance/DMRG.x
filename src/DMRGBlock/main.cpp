static char help[] =
    "Test code for MATSHELL approach\n";

#include <slepceps.h>
#include "DMRGBlock.hpp"
#include "DMRGBlockContainer.hpp"
#include "linalg_tools.hpp"

#if 1
int main(int argc, char **argv)
{
    PetscErrorCode  ierr = 0;
    PetscMPIInt     nprocs, rank;
    MPI_Comm&       comm = PETSC_COMM_WORLD;

    /*  Initialize MPI  */
    ierr = SlepcInitialize(&argc, &argv, (char*)0, help); CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm, &nprocs); CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);

    HeisenbergSpinOneHalfLadder Ladder;

    ierr = Ladder.Initialize(); CHKERRQ(ierr);

    ierr = MatPeek( Ladder.SingleSite.Sz[0], "Ladder.SingleSite.Sz[0]" ); CHKERRQ(ierr);
    ierr = MatPeek( Ladder.SingleSite.Sp[0], "Ladder.SingleSite.Sp[0]" ); CHKERRQ(ierr);


    ierr = Ladder.Destroy(); CHKERRQ(ierr);

    ierr = SlepcFinalize(); CHKERRQ(ierr);
    return ierr;
}
#endif


#if 0
/* Testing of DMRGBlock */
int main(int argc, char **argv)
{
    PetscErrorCode  ierr = 0;
    PetscMPIInt     nprocs, rank;
    MPI_Comm&       comm = PETSC_COMM_WORLD;

    /*  Initialize MPI  */
    ierr = SlepcInitialize(&argc, &argv, (char*)0, help); CHKERRQ(ierr);

    ierr = MPI_Comm_size(comm, &nprocs); CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);

    Block_SpinOneHalf Block;
    ierr = Block.Initialize(comm, 2, 2); CHKERRQ(ierr);

    // ierr = Block.CheckOperators(); CHKERRQ(ierr);

    ierr = MatSzCreate(comm, Block.Sz[0]); CHKERRQ(ierr);
    ierr = MatSzCreate(comm, Block.Sz[1]); CHKERRQ(ierr);
    ierr = MatSpCreate(comm, Block.Sp[0]); CHKERRQ(ierr);
    ierr = MatSpCreate(comm, Block.Sp[1]); CHKERRQ(ierr);

    ierr = Block.CreateSm(); CHKERRQ(ierr);

    ierr = MatPeek(Block.Sp[1], "Sp[1]"); CHKERRQ(ierr);
    ierr = MatPeek(Block.Sm[1], "Sm[1]"); CHKERRQ(ierr);

    ierr = Block.DestroySm(); CHKERRQ(ierr);
    ierr = Block.Destroy(); CHKERRQ(ierr);


    ierr = SlepcFinalize(); CHKERRQ(ierr);
    return 0;
}
#endif
