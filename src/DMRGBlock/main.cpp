static char help[] =
    "Test code for MATSHELL approach\n";

#include <slepceps.h>
#include "DMRGBlock.hpp"
#include "DMRGBlockContainer.hpp"

/*
    Templated block Hamiltonian builder
    Updates the Hamiltonian representation by considering the previous block and the addition of a new site
    TODO: Implement as part of a DMRGSystem class templated with the BlockType and the Hamiltonian
    TODO: Generalize to the case independent of spin
    TODO: Create a Hamiltonian base class and let Hamiltonian_Heisenberg inherit
    TODO: For more efficient Kron, wrap matrix operators in a class that handles special matrices like identities and spin operators without explicitly writing the matrix to memory
 */


template <class BlockType>
class Hamiltonian_Heisenberg
{
private:
    MPI_Comm comm = PETSC_COMM_WORLD;
    PetscBool parameters_set = PETSC_FALSE;
    PetscErrorCode GetParameters(); /* Declare as virtual in parent class */

    /* Overloading parameters */

    /** XX-coupling constant */
    PetscScalar J = 1.0;

    /** Z-coupling constant */
    PetscScalar Jz = 1.0;

public:

    /* Overload */
    PetscErrorCode HeisenbergHamiltonian(BlockType& L_Site, BlockType& R_Site, PetscBool matshell, Mat& H);

    /* Declare as virtual and overload with just blocks */
    PetscErrorCode MatHamiltonian(
        DMRGBlockContainer<BlockType>& L_Block,
        PetscInt L_isite,
        DMRGBlockContainer<BlockType>& R_Block,
        PetscInt R_isite,
        PetscBool matshell,
        Mat& H);

    PetscErrorCode MatHamiltonian(
        BlockType& L_Site,
        DMRGBlockContainer<BlockType>& R_Block,
        PetscInt R_isite,
        PetscBool matshell,
        Mat& H);

    /* Reimplement taking the entire block */

};


template <class BlockType>
PetscErrorCode Hamiltonian_Heisenberg<BlockType>::GetParameters()
{
    PetscErrorCode  ierr = 0;
    if(parameters_set) return 0;

    PetscScalar J_in = J;
    PetscScalar Jz_in = Jz;
    ierr = PetscOptionsGetReal(nullptr, nullptr, "-J", &J_in, nullptr); CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(nullptr, nullptr, "-Jz", &Jz_in, nullptr); CHKERRQ(ierr);
    J = J_in;
    Jz = Jz_in;
    parameters_set = PETSC_TRUE;

    return ierr;
}


template <class BlockType>
PetscErrorCode
Hamiltonian_Heisenberg<BlockType>::HeisenbergHamiltonian(
    BlockType& L_Site,
    BlockType& R_Site,
    PetscBool matshell,
    Mat& H)
{
    PetscErrorCode ierr = 0;

    if(!L_Site.Initialized()) SETERRQ(L_Site.Comm(), 1, "Left site not initialized.");
    if(!R_Site.Initialized()) SETERRQ(L_Site.Comm(), 1, "Right site not initialized.");

    Mat I_L, I_R;
    PetscInt dim_L = L_Site.MatOpDim(); CHKERRQ(ierr);
    PetscInt dim_R = R_Site.MatOpDim(); CHKERRQ(ierr);
    ierr = MatEyeCreate(L_Site.Comm(), I_L, dim_L); CHKERRQ(ierr);
    ierr = MatEyeCreate(R_Site.Comm(), I_R, dim_R); CHKERRQ(ierr);

    /* Aliases */
    const Mat& H_L  = L_Site.H();
    const Mat& Sz_L = L_Site.Sz();
    const Mat& Sp_L = L_Site.Sp();
    const Mat& Sm_L = L_Site.Sm();

    const Mat& H_R  = R_Site.H();
    const Mat& Sz_R = R_Site.Sz();
    const Mat& Sp_R = R_Site.Sp();
    const Mat& Sm_R = R_Site.Sm();

    ierr = GetParameters(); CHKERRQ(ierr);

    std::vector<PetscScalar>    a = { 1.0, 1.0,   Jz, 0.5*J, 0.5*J};
    std::vector<Mat>            A = { H_L, I_L, Sz_L,  Sp_L,  Sm_L};
    std::vector<Mat>            B = { I_R, H_R, Sz_R,  Sm_R,  Sp_R};

    if(matshell){
        ierr = MatKronProdSum_MATSHELL(a, A, B, H); CHKERRQ(ierr);
    } else {
        ierr = MatKronProdSum(a, A, B, H, PETSC_TRUE); CHKERRQ(ierr);
        ierr = MatAssemblyBegin(H, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(H, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }

    ierr = MatDestroy(&I_L); CHKERRQ(ierr);
    ierr = MatDestroy(&I_R); CHKERRQ(ierr);

    return 0;
}


/* TODO: Generalize this interface function for the block-site and block-block cases */
/*************************************/
template <class BlockType>
PetscErrorCode
Hamiltonian_Heisenberg<BlockType>::MatHamiltonian(
    DMRGBlockContainer<BlockType>& L_Block,
    PetscInt L_isite,
    DMRGBlockContainer<BlockType>& R_Block,
    PetscInt R_isite,
    PetscBool matshell,
    Mat& H)
{
    PetscErrorCode ierr = 0;

    ierr = HeisenbergHamiltonian(
        L_Block.Site(L_isite),
        R_Block.ActiveSite(R_isite),
        matshell, H); CHKERRQ(ierr);

    return ierr;
}


template <class BlockType>
PetscErrorCode
Hamiltonian_Heisenberg<BlockType>::MatHamiltonian(
    BlockType& L_Site,
    DMRGBlockContainer<BlockType>& R_Block,
    PetscInt R_isite,
    PetscBool matshell,
    Mat& H)
{
    PetscErrorCode ierr = 0;

    ierr = HeisenbergHamiltonian(
        L_Site,
        R_Block.ActiveSite(R_isite),
        matshell, H); CHKERRQ(ierr);

    return ierr;
}
/*************************************/

template <class BlockType, class HamiltonianType>
class DMRGBlockContainer_Hamiltonian: public DMRGBlockContainer<BlockType>
{
public:
    PetscErrorCode BuildBlockHamiltonian(HamiltonianType Hamiltonian, const PetscInt isite);
};


template <class BlockType, class HamiltonianType>
PetscErrorCode DMRGBlockContainer_Hamiltonian<BlockType, HamiltonianType>::
BuildBlockHamiltonian(HamiltonianType Hamiltonian, const PetscInt isite)
{
    PetscErrorCode ierr = 0;

    if(isite == 0) return 0;
    if(!this->Site(isite).Initialized()) SETERRQ1(this->Comm(), 1, "Requested site (%d) not yet initialized.", isite);
    /*
        Using the previous site and the added site, build the Hamiltonian of the Block comprised of all sites up to isite.
     */
    ierr = this->Site(isite-1).CreateSm(); CHKERRQ(ierr);

    PetscBool block_side_left = (PetscBool)(this->BlockSide()==BlockSide_Left);
    BlockType& L_Site = block_side_left ? this->Site(isite-1) : this->SingleSite();
    BlockType& R_Site = block_side_left ? this->SingleSite() : this->Site(isite-1);



    Mat H_temp;
    // ierr = Hamiltonian.MatHamiltonian()
    // ierr = Hamiltonian.MatHamiltonian(L_Site, R_Site, PETSC_FALSE, H_temp); CHKERRQ(ierr);
    ierr = Hamiltonian.HeisenbergHamiltonian(L_Site, R_Site, PETSC_FALSE, H_temp); CHKERRQ(ierr);
    ierr = this->Site(isite).UpdateH(H_temp); CHKERRQ(ierr);

    ierr = this->Site(isite-1).DestroySm(); CHKERRQ(ierr);
    ierr = MatDestroy(&H_temp); CHKERRQ(ierr);

    return 0;
}


template <class BlockType, class HamiltonianType>
PetscErrorCode BuildSuperblockHamiltonian(
    HamiltonianType Hamiltonian,
    DMRGBlockContainer<BlockType>& BlockSys,
    DMRGBlockContainer<BlockType>& BlockEnv,
    PetscInt isite,
    PetscBool matshell,
    Mat& H_super
    )
{
    PetscErrorCode ierr = 0;
    const MPI_Comm& comm = BlockSys.Comm();

    /* Check that both blocks have the same target length */
    if(BlockSys.TargetLength() != BlockEnv.TargetLength())
        SETERRQ2(comm, 1, "Both blocks must have the same target lengths. %d != %d.",
            BlockSys.TargetLength(), BlockEnv.TargetLength());

    PetscInt sys_site = isite;
    PetscInt env_site = BlockSys.TargetLength() - isite - 2;

    printf("\nsys: %-6d env: %-6d\n\n", sys_site, env_site);

    /* Build the superblock Hamiltonian explicitly */
    BlockType& L_Site = BlockSys.Site(sys_site);
    BlockType& R_Site = BlockEnv.Site(env_site);

    ierr = L_Site.CreateSm(); CHKERRQ(ierr);
    if(L_Site.Sm() != R_Site.Sm()){
        ierr = R_Site.CreateSm(); CHKERRQ(ierr);
    }

    // ierr = Hamiltonian.MatHamiltonian(L_Site, R_Site, matshell, H_super); CHKERRQ(ierr);
    ierr = Hamiltonian.HeisenbergHamiltonian(L_Site, R_Site, matshell, H_super); CHKERRQ(ierr);

    ierr = L_Site.DestroySm(); CHKERRQ(ierr);
    ierr = R_Site.DestroySm(); CHKERRQ(ierr);

    return ierr;
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

    /* Prepare a (left) block of sites and warm-up half of the sites */
    PetscInt target_sites = 10;
    ierr = PetscOptionsGetInt(nullptr, nullptr, "-nsites", &target_sites, nullptr); CHKERRQ(ierr);

    DMRGBlockContainer_Hamiltonian<Block_SpinOneHalf,Hamiltonian_Heisenberg<Block_SpinOneHalf>> BlockSys;
    Hamiltonian_Heisenberg<Block_SpinOneHalf> Hamiltonian;

    ierr = BlockSys.Initialize(comm, target_sites, BlockSide_Left); CHKERRQ(ierr);

    for(PetscInt isite = 1; isite < target_sites/2; ++isite)
    {
        ierr = BlockSys.AddSite(); CHKERRQ(ierr);
        ierr = BlockSys.BuildBlockHamiltonian(Hamiltonian, isite); CHKERRQ(ierr);
        ierr = BlockSys.AdjustSiteOps(isite); CHKERRQ(ierr);
        /* Destroy previous site only when doing iDMRG only */
        ierr = BlockSys.Site(isite-1).Destroy(); CHKERRQ(ierr);
    }

    PetscBool use_matshell = PETSC_FALSE;
    ierr = PetscOptionsGetBool(nullptr, nullptr, "-shell", &use_matshell, nullptr); CHKERRQ(ierr);

    /* Form the superblock Hamiltonian explicitly */
    Mat H;
    ierr = BuildSuperblockHamiltonian(Hamiltonian, BlockSys, BlockSys, target_sites/2 - 1, use_matshell, H); CHKERRQ(ierr);

    /* Iterative diagonalization to get the groundstate */
    EPS eps;
    ierr = EPSCreate(comm, &eps); CHKERRQ(ierr);
    ierr = EPSSetOperators(eps, H, nullptr); CHKERRQ(ierr);
    ierr = EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL); CHKERRQ(ierr);
    ierr = EPSSetProblemType(eps, EPS_HEP); CHKERRQ(ierr);
    ierr = EPSSetType(eps, EPSKRYLOVSCHUR); CHKERRQ(ierr);
    ierr = EPSSetFromOptions(eps); CHKERRQ(ierr);
    ierr = EPSSolve(eps); CHKERRQ(ierr);

    PetscScalar gse_r=0, gse_i=0;
    ierr = EPSGetEigenvalue(eps, 0, &gse_r, &gse_i); CHKERRQ(ierr);
    if(gse_i) SETERRQ1(comm, 1, "Ground state energy must be real. Obtained a non-zero value (%g).",gse_i);

    ierr = PetscPrintf(comm, "GroundStateEnergy/Site: %f\n\n", (double)gse_r/((double)target_sites));

    ierr = EPSDestroy(&eps); CHKERRQ(ierr);


    ierr = MatDestroy(&H); CHKERRQ(ierr);
    ierr = BlockSys.Destroy(); CHKERRQ(ierr);
    ierr = SlepcFinalize(); CHKERRQ(ierr);
    return 0;
}
