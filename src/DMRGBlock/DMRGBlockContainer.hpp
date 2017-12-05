#ifndef __DMRG_BLOCK_HPP__
#define __DMRG_BLOCK_HPP__

#include <slepceps.h>
#include <petscmat.h>
#include <vector>
#include <map>

#include "DMRGBlock.hpp"
#include "linalg_tools.hpp"


class HeisenbergSpinOneHalfLadder
{

private:

    /*------ Backend ------*/

    /** MPI Communicator */
    MPI_Comm    mpi_comm = PETSC_COMM_WORLD;

    /** MPI rank in mpi_comm */
    PetscMPIInt mpi_rank;

    /** MPI size of mpi_comm */
    PetscMPIInt mpi_size;

    /** Tells whether to printout info during certain function calls */
    PetscBool   verbose = PETSC_FALSE;

    /*------ Coupling Constants ------*/

    /** Coupling strength for nearest-neighbor interactions */
    PetscScalar J1 = 1.0;

    /** Coupling strength for next-nearest-neighbor interactions */
    PetscScalar J2 = 1.0;

    /*------ Geometry ------*/

    /** Length along (growing) longitudinal direction */
    PetscInt    Lx = 4;

    /** Length along (fixed) transverse direction */
    PetscInt    Ly = 3;

    /** Total number of sites */
    PetscInt    num_sites;

    /*------ Blocks ------*/

    /** Number of system blocks to store, usually Lx*Ly-1 */
    PetscInt    num_blocks;

    /** Array of blocks */
    Block_SpinOneHalf *SysBlocks;

    /** Number of initialized blocks in SysBlocks */
    PetscInt    SysBlocks_num_init = 0;

    /** Environment block to be used only during warmup */
    Block_SpinOneHalf EnvBlock;

public:

    /** Static block containing single site operators for reference */
    Block_SpinOneHalf SingleSite;



    PetscErrorCode Initialize();

    PetscErrorCode Destroy();

};











#define false 0
#if false
typedef enum
{
    BlockSide_Left=0,
    BlockSide_Right=1
} BlockSide_t;


template <class SiteType>
class DMRGBlockContainer
{

private:

    MPI_Comm comm;
    PetscMPIInt rank;
    PetscBool verbose = PETSC_FALSE;
    BlockSide_t block_side;

    PetscInt nsites_target;
    PetscInt nsites = 0;
    PetscInt active_site_idx;
    PetscInt block_dim_exact;

    SiteType single_site;
    std::vector<SiteType> sites_list;

public:

    PetscErrorCode Initialize(
        const MPI_Comm& comm_in,
        const PetscInt& nsites_target_in,
        const BlockSide_t& block_side_in);
    PetscErrorCode Destroy();
    PetscErrorCode AdjustSiteOps(PetscInt isite);
    PetscErrorCode AddSite();
    PetscErrorCode CreateSm();
    PetscErrorCode DestroySm();
    PetscErrorCode CheckOperators();

    MPI_Comm Comm() const { return comm; }
    PetscInt Length() const { return nsites; }
    PetscInt TargetLength() const { return nsites_target; }
    BlockSide_t BlockSide() const { return block_side; }
    PetscInt ActiveSiteIdx() const { return active_site_idx; }

    SiteType& ActiveSite()
    {
        return sites_list[active_site_idx];
    }

    SiteType& Site(PetscInt isite)
    {
        return sites_list[isite];
    }

    SiteType& SingleSite() { return single_site; }

    /************************************************************
     * Members that depend on magnetization conservation
     ************************************************************/

    /** Keeps track of the Sz sectors */
    std::vector<PetscScalar> basis_sector_array;

    /** Keeps track of the basis in each Sz sector */
    std::unordered_map<PetscScalar,std::vector<PetscInt>> basis_by_sector;

    /** Keeps track of the basis in each Sz sector */
    std::unordered_map<PetscScalar, Mat> rho_block_dict;

};


template <class SiteType>
PetscErrorCode DMRGBlockContainer<SiteType>::Initialize(
        const MPI_Comm& comm_in,
        const PetscInt& nsites_target_in,
        const BlockSide_t& block_side_in)
{
    PetscErrorCode ierr = 0;

    comm = comm_in;
    nsites_target = nsites_target_in;
    block_side = block_side_in;
    MPI_Comm_rank(comm, &rank);

    sites_list.resize(nsites_target);

    /* Initialize single site template */
    ierr = single_site.Initialize(comm); CHKERRQ(ierr);
    ierr = single_site.CreateSm(); CHKERRQ(ierr);
    basis_sector_array = single_site.single_site_sectors;

    /* Initialize exact block dimension */
    block_dim_exact = single_site.LocDim();

    /* Check whether to do verbose logging */
    ierr = PetscOptionsGetBool(NULL,NULL,"-verbose",&verbose,NULL); CHKERRQ(ierr);

    /* Initialize a block of one site */
    ierr = sites_list[nsites].Initialize(comm); CHKERRQ(ierr);
    active_site_idx = 0;
    nsites = 1;

    if(verbose && !rank) printf("> block::%s\n",__FUNCTION__);
    if(verbose && !rank) printf("> block::length %d\n", Length());

    ierr = CheckOperators(); CHKERRQ(ierr);
    return ierr;
}


template <class SiteType>
PetscErrorCode DMRGBlockContainer<SiteType>::Destroy()
{
    PetscErrorCode ierr = 0;

    ierr = single_site.DestroySm(); CHKERRQ(ierr);
    ierr = single_site.Destroy(); CHKERRQ(ierr);
    for (size_t isite = 0; isite < nsites; ++isite)
    {
        if (sites_list[isite].Initialized())
        {
            ierr = sites_list[isite].Destroy(); CHKERRQ(ierr);
        }
    }

    return ierr;
}


template <class SiteType>
PetscErrorCode DMRGBlockContainer<SiteType>::AddSite()
{
    PetscErrorCode ierr = 0;

    /*  Check that the number of sites is within the target */
    if (nsites+1 > nsites_target)
        SETERRQ1(comm, 1, "Cannot add more sites. Block length will exceed the target (%d)", nsites_target);

    ierr = CheckOperators(); CHKERRQ(ierr);

    /*  Initialize a new site  */
    ierr = sites_list[nsites].Initialize(comm); CHKERRQ(ierr);

    active_site_idx = nsites;
    ++nsites;

    if(verbose && !rank) printf("> block::%s\n",__FUNCTION__);
    return ierr;
}


template <class SiteType>
PetscErrorCode DMRGBlockContainer<SiteType>::AdjustSiteOps(PetscInt isite)
{
    PetscErrorCode ierr = 0;

    /*
        Change its single operator representation by appending an identity representing the combined
        Hilbert space of previously added sites ...
     */
    if(verbose && !rank) printf("> isite %d\n", isite);

    // ierr = BuildBlockHamiltonian(&this, isite); CHKERRQ(ierr);
    // ierr = PetscPrintf(this->Comm(), "H_adds: %p\n", this->Site(isite-1).H());
    // if(!sites_list[isite].H()) SETERRQ(comm, 1, "Matrix is null.");
    // ierr = MatPeek(sites_list[isite].H(),"sites_list"); CHKERRQ(ierr);

    /* TODO: Think about generalizing this */
    PetscInt block_dim = sites_list[isite-1].MatOpDim();
    if(block_side==BlockSide_Left){
        /* ... on the left side of the new operator on the left block */
        ierr = sites_list[isite].EyeKronOp(block_dim); CHKERRQ(ierr);
    } else {
        /* ... on the right side of the new operator on the right block */
        ierr = sites_list[isite].OpKronEye(block_dim); CHKERRQ(ierr);
    }

    if(verbose && !rank) printf("> block::%s\n",__FUNCTION__);
    return ierr;
}


template <class SiteType>
PetscErrorCode DMRGBlockContainer<SiteType>::CreateSm()
{
    PetscErrorCode ierr = 0;

    for (size_t isite = 0; isite < nsites; ++isite)
    {
        if (sites_list[isite].Initialized())
        {
            ierr = sites_list[isite].CreateSm(); CHKERRQ(ierr);
        }
    }

    return ierr;
}


template <class SiteType>
PetscErrorCode DMRGBlockContainer<SiteType>::DestroySm()
{
    PetscErrorCode ierr = 0;

    for (size_t isite = 0; isite < nsites; ++isite)
    {
        if (sites_list[isite].Initialized())
        {
            ierr = sites_list[isite].DestroySm(); CHKERRQ(ierr);
        }
    }

    return ierr;
}


template <class SiteType>
PetscErrorCode DMRGBlockContainer<SiteType>::CheckOperators()
{
    PetscErrorCode ierr = 0;

    /* TODO: Check only active and initialized sites */
    // block_dim_exact = Site(0).MatOpDim();
    // ierr = Site(0).CheckOperators(); CHKERRQ(ierr);
    // for (size_t isite = 1; isite < nsites; ++isite)
    // {
    //     ierr = Site(isite).CheckOperators(); CHKERRQ(ierr);
    //     PetscInt block_dim_exact_site = Site(isite).MatOpDim();
    //     if(block_dim_exact_site!=block_dim_exact) SETERRQ3(comm, 1, "Site %d has different operator dimension. "
    //         "Expected %d. Got %d.", isite, block_dim_exact, block_dim_exact_site);
    // }

    return ierr;
}
#endif

#endif // __DMRG_BLOCK_HPP__
