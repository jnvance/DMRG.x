#include "DMRGBlockContainer.hpp"
#include <iostream>

PETSC_EXTERN PetscErrorCode Kron_Explicit(
    const Block_SpinOneHalf& LeftBlock,
    const Block_SpinOneHalf& RightBlock,
    Block_SpinOneHalf& BlockOut,
    PetscBool BuildHamiltonian
    );



PetscErrorCode Heisenberg_SpinOneHalf_SquareLattice::Initialize()
{
    PetscErrorCode ierr = 0;

    /*  Initialize attributes  */
    ierr = MPI_Comm_rank(mpi_comm, &mpi_rank); CHKERRQ(ierr);
    ierr = MPI_Comm_size(mpi_comm, &mpi_size); CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(NULL,NULL,"-verbose",&verbose,NULL); CHKERRQ(ierr);

    /*  Get couplings and information on geometry from command line options  */
    ierr = PetscOptionsGetReal(NULL,NULL,"-J1",&J1,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(NULL,NULL,"-J2",&J2,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-Lx",&Lx,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-Ly",&Ly,NULL); CHKERRQ(ierr);
    ierr = PetscOptionsGetInt(NULL,NULL,"-mstates",&mstates,NULL); CHKERRQ(ierr);

    num_sites = Lx * Ly;
    num_blocks = num_sites - 1;

    /*  Require that the total number of sites be even */
    if(num_sites % 2) SETERRQ3(mpi_comm, 1, "Total number of sites must be even. "
        "Given: Lx=%d, Ly=%d, Lx*Ly=%d", Lx, Ly, num_sites);

    /*  Initialize a static block of single-site operators  */
    ierr = SingleSite.Initialize(mpi_comm, 1, PETSC_DEFAULT); CHKERRQ(ierr);

    /*  Allocate the array of system blocks */
    SysBlocks = new Block_SpinOneHalf[num_blocks];

    /*  Initialize the first system block with one site  */
    ierr = SysBlocks[0].Initialize(mpi_comm, 1, PETSC_DEFAULT); CHKERRQ(ierr);
    ++SysBlocks_num_init;

    /*  Initialize the environment block with one site  */
    ierr = EnvBlock.Initialize(mpi_comm, 1, PETSC_DEFAULT); CHKERRQ(ierr);

    /*  Print some info */
    if(verbose)
    {
        ierr = PetscPrintf(mpi_comm,
            "\n"
            "# Heisenberg_SpinOneHalf_SquareLattice\n"
            "#   Coupling Constants:\n"
            "      J1 = %f\n"
            "      J2 = %f\n"
            "#   Geometry:\n"
            "      Lx = %d\n"
            "      Ly = %d\n"
            "#   DMRG:\n"
            "      mstates = %d\n",
            J1, J2, Lx, Ly, mstates); CHKERRQ(ierr);
    }

    return ierr;
}


PetscErrorCode Heisenberg_SpinOneHalf_SquareLattice::Destroy()
{
    PetscErrorCode ierr = 0;

    ierr = SingleSite.Destroy(); CHKERRQ(ierr);

    /*  Deallocate system blocks  */
    for(PetscInt iblock = 0; iblock < SysBlocks_num_init; ++iblock){
        ierr = SysBlocks[iblock].Destroy(); CHKERRQ(ierr);
    }
    delete [] SysBlocks;
    SysBlocks = nullptr;

    /*  Deallocate environment block */
    ierr = EnvBlock.Destroy(); CHKERRQ(ierr);

    if(verbose){
        ierr = PetscPrintf(mpi_comm,"\n\n"); CHKERRQ(ierr);
    }

    return ierr;
}


PetscErrorCode Heisenberg_SpinOneHalf_SquareLattice::EnlargeBlock(
    const Block_SpinOneHalf& BlockIn,
    Block_SpinOneHalf& BlockOut)
{
    PetscErrorCode ierr = 0;

    /*  Check whether all operators and sectors are usable */
    ierr = BlockIn.CheckOperators(); CHKERRQ(ierr);
    ierr = BlockIn.CheckSectors(); CHKERRQ(ierr);

    #if 1
        std::cout << "AddSite qn_list:   ";
        for(auto i: AddSite.qn_list) std::cout << i << "   ";
        std::cout << std::endl;

        std::cout << "AddSite qn_size:   ";
        for(auto i: AddSite.qn_size) std::cout << i << "   ";
        std::cout << std::endl;

        std::cout << "AddSite qn_offset: ";
        for(auto i: AddSite.qn_offset) std::cout << i << "   ";
        std::cout << std::endl;

        std::cout << std::endl;

        std::cout << "BlockIn qn_list:   ";
        for(auto i: BlockIn.qn_list) std::cout << i << "   ";
        std::cout << std::endl;

        std::cout << "BlockIn qn_size:   ";
        for(auto i: BlockIn.qn_size) std::cout << i << "   ";
        std::cout << std::endl;

        std::cout << "BlockIn qn_offset: ";
        for(auto i: BlockIn.qn_offset) std::cout << i << "   ";
        std::cout << std::endl;
    #endif


    ierr = Kron_Explicit(BlockIn, AddSite, BlockOut, PETSC_FALSE); CHKERRQ(ierr);


    return ierr;
}


PetscErrorCode Heisenberg_SpinOneHalf_SquareLattice::SingleDMRGStep(
    const Block_SpinOneHalf& Sys,
    const Block_SpinOneHalf& Env,
    Block_SpinOneHalf& SysOut)
{
    PetscErrorCode ierr = 0;

    #if 0
    /* This part should be done inside EnlargeBlock */
    /* Check block operators */
    ierr = Sys.CheckOperators(); CHKERRQ(ierr);
    if(&Sys!=&Env){
        /* Check the environment block only if it is distinct from the system block */
        ierr = Env.CheckOperators(); CHKERRQ(ierr);
    }
    #endif

    return ierr;
}
