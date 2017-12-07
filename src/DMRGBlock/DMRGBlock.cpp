#include "DMRGBlock.hpp"

PETSC_EXTERN int64_t ipow(int64_t base, uint8_t exp);
PETSC_EXTERN PetscErrorCode MatSpinOneHalfSzCreate(const MPI_Comm& comm, Mat& Sz);
PETSC_EXTERN PetscErrorCode MatSpinOneHalfSpCreate(const MPI_Comm& comm, Mat& Sp);

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
    if(num_states_in == PETSC_DEFAULT){
        num_states = ipow(loc_dim, num_sites);
    } else{
        num_states = num_states_in;
    }

    /* Initialize array of operator matrices */
    ierr = PetscCalloc3(num_sites, &Sz, num_sites, &Sp, num_sites, &Sm); CHKERRQ(ierr);

    /* Initialize single-site operators */
    if (num_sites == 1)
    {
        ierr = MatSpinOneHalfSzCreate(mpi_comm, Sz[0]); CHKERRQ(ierr);
        ierr = MatSpinOneHalfSpCreate(mpi_comm, Sp[0]); CHKERRQ(ierr);
    }

    /* Initialize switch */
    init = PETSC_TRUE;

    return ierr;
}


PetscErrorCode Block_SpinOneHalf::CheckOperatorArray(Mat *Op, const char* label) const
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


PetscErrorCode Block_SpinOneHalf::CheckOperators() const
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

    ierr = CheckOperatorArray(Sp, "Sp"); CHKERRQ(ierr);
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

    for(PetscInt isite = 0; isite < num_sites; ++isite){
        ierr = MatDestroy(&Sm[isite]); CHKERRQ(ierr);
    }
    init_Sm = PETSC_FALSE;

    return ierr;
}


PetscErrorCode Block_SpinOneHalf::Destroy()
{
    PetscErrorCode ierr = 0;

    if (!init) SETERRQ(mpi_comm, 1, "Block not yet initialized.");

    /* Destroy operator matrices */
    for(PetscInt isite = 0; isite < num_sites; ++isite)
    {
        ierr = MatDestroy(&Sz[isite]); CHKERRQ(ierr);
        ierr = MatDestroy(&Sp[isite]); CHKERRQ(ierr);
    }

    if (init_Sm){
        ierr = DestroySm(); CHKERRQ(ierr);
    }

    ierr = PetscFree3(Sz, Sp, Sm); CHKERRQ(ierr);
    init = PETSC_FALSE;

    return ierr;
}
