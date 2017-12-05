#include "DMRGBlockContainer.hpp"


PetscErrorCode HeisenbergSpinOneHalfLadder::Initialize()
{
    PetscErrorCode ierr = 0;

    /* Initialize attributes */
    ierr = MPI_Comm_rank(mpi_comm, &mpi_rank); CHKERRQ(ierr);
    ierr = MPI_Comm_size(mpi_comm, &mpi_size); CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(NULL,NULL,"-verbose",&verbose,NULL); CHKERRQ(ierr);

    /* Get couplings and geometry from command line options */
    ierr = PetscOptionsGetReal(NULL,NULL,"-J1",&J1,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-J2",&J2,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-Lx",&Lx,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-Ly",&Ly,NULL); CHKERRQ(ierr);

    num_sites = Lx * Ly;
    num_blocks = num_sites - 1;

    /* Initialize a static block of single-site operators */
    ierr = SingleSite.Initialize(mpi_comm, 1, PETSC_DEFAULT); CHKERRQ(ierr);

    /* Allocate the system array of blocks with single sites */
    SysBlocks = new Block_SpinOneHalf[num_blocks];

    /* Initialize the first system block with one site */
    ierr = SysBlocks[0].Initialize(mpi_comm, 1, PETSC_DEFAULT); CHKERRQ(ierr);
    ++SysBlocks_num_init;
    ierr = SysBlocks[1].Initialize(mpi_comm, 1, PETSC_DEFAULT); CHKERRQ(ierr);
    ++SysBlocks_num_init;

    /* Initialize the environment block with one site */
    ierr = EnvBlock.Initialize(mpi_comm, 1, PETSC_DEFAULT); CHKERRQ(ierr);

    /* Print some info */
    if(verbose)
    {
        ierr = PetscPrintf(mpi_comm,
            "\n"
            "# HeisenbergSpinOneHalfLadder\n"
            "#   Coupling Constants:\n"
            "      J1 = %f\n"
            "      J2 = %f\n"
            "#   Geometry:\n"
            "      Lx = %d\n"
            "      Ly = %d\n",
            J1, J2, Lx, Ly); CHKERRQ(ierr);
    }

    return ierr;
}


PetscErrorCode HeisenbergSpinOneHalfLadder::Destroy()
{
    PetscErrorCode ierr = 0;

    ierr = SingleSite.Destroy(); CHKERRQ(ierr);

    /* Deallocate system blocks */

    for(PetscInt iblock = 0; iblock < SysBlocks_num_init; ++iblock){
        ierr = SysBlocks[iblock].Destroy(); CHKERRQ(ierr);
    }

    delete [] SysBlocks;
    SysBlocks = nullptr;

    ierr = EnvBlock.Destroy(); CHKERRQ(ierr);

    if(verbose){
        ierr = PetscPrintf(mpi_comm,"\n\n"); CHKERRQ(ierr);
    }

    return ierr;
}

