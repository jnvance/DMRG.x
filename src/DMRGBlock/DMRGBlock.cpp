#include "DMRGBlock.hpp"


PetscErrorCode Block_SpinOneHalf::Initialize(const MPI_Comm& comm_in, PetscInt num_sites_in, PetscInt num_states_in)
{
    PetscErrorCode ierr = 0;

    /* Check whether to do verbose logging */
    ierr = PetscOptionsGetBool(NULL,NULL,"-verbose",&verbose,NULL); CHKERRQ(ierr);

    /* Initialize attributes*/
    mpi_comm = comm_in;
    ierr = MPI_Comm_rank(mpi_comm, &mpi_rank); CPP_CHKERRQ(ierr);
    ierr = MPI_Comm_size(mpi_comm, &mpi_size); CPP_CHKERRQ(ierr);

    /* Initial number of sites and number of states */
    num_sites = num_sites_in;
    num_states = num_states_in;

    /* Initialize array of operator matrices */
    ierr = PetscCalloc3(num_sites, &Sz, num_sites, &Sp, num_sites, &Sm); CHKERRQ(ierr);

    /* Initialize switch */
    init = PETSC_TRUE;
    if(verbose && !mpi_rank) printf(">>> site::%s\n",__FUNCTION__);

    return ierr;
}


PetscErrorCode Block_SpinOneHalf::CheckOperatorArray(Mat *Op, const char* label)
{
    PetscErrorCode ierr = 0;

    /* Check the size of each matrix and make sure that it
     * matches the number of basis states */

    PetscInt M, N;
    for(PetscInt isite = 0; isite < num_sites; ++isite)
    {
        if(!Op[isite])
            SETERRQ2(mpi_comm, 1, "%s[%d] matrix not yet created.", label, isite);
        ierr = MatGetSize(Op[isite], &M, &N); CHKERRQ(ierr);
        if (M != N)
            SETERRQ2(mpi_comm, 1, "%s[%d] matrix not square.", label, isite);
        if (M != num_states)
            SETERRQ4(mpi_comm, 1, "%s[%d] matrix dimension does not match "
                "the number of states. Expected %d. Got %d.", label, isite, num_states, M);
    }

    return ierr;
}


PetscErrorCode Block_SpinOneHalf::CheckOperators()
{
    PetscErrorCode ierr = 0;

    if (!init) SETERRQ(mpi_comm, 1, "Block not yet initialized.");

    ierr = CheckOperatorArray(Sz, "Sz"); CHKERRQ(ierr);
    ierr = CheckOperatorArray(Sp, "Sp"); CHKERRQ(ierr);

    if (init_Sm){
        ierr = CheckOperatorArray(Sm, "Sm"); CHKERRQ(ierr);
    }

    return ierr;
}


PetscErrorCode Block_SpinOneHalf::CreateSm()
{
    PetscErrorCode ierr = 0;

    if(init_Sm) SETERRQ(mpi_comm, 1, "Sm was previously initialized. Call DestroySm() first.");

    for(PetscInt isite = 0; isite < num_sites; ++isite){
        ierr = MatHermitianTranspose(Sp[isite], MAT_INITIAL_MATRIX, &Sm[isite]); CHKERRQ(ierr);
    }
    init_Sm = PETSC_TRUE;

    return ierr;
}


PetscErrorCode Block_SpinOneHalf::DestroySm()
{
    PetscErrorCode ierr = 0;

    if(!init_Sm) SETERRQ(mpi_comm, 1, "Sm not initialized. Nothing to destroy.");

    ierr = MatDestroyMatrices(num_sites, &Sm); CHKERRQ(ierr);
    init_Sm = PETSC_FALSE;

    return ierr;
}


PetscErrorCode Block_SpinOneHalf::Destroy()
{
    PetscErrorCode ierr = 0;

    ierr = CheckOperators(); CHKERRQ(ierr);

    /* Destroy operator matrices */
    ierr = MatDestroyMatrices(num_sites, &Sz); CHKERRQ(ierr);
    ierr = MatDestroyMatrices(num_sites, &Sp); CHKERRQ(ierr);

    if (init_Sm){
        ierr = DestroySm(); CHKERRQ(ierr);
    }

    ierr = PetscFree3(Sz, Sp, Sm); CHKERRQ(ierr);
    init = PETSC_FALSE;

    if(verbose && !mpi_rank) printf(">>> site::%s\n",__FUNCTION__);

    return ierr;
}
