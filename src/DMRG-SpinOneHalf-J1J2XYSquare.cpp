static char help[] =
    "DMRG executable for the Spin-1/2 J1-J2 XY Model on a two-dimensional square lattice.\n";

#include "DMRGBlock.hpp"
#include "Hamiltonians.hpp"
#include "DMRGBlockContainer.hpp"

#define MAX_SWEEPS 100

int main(int argc, char **argv)
{
    PetscErrorCode  ierr;
    PetscMPIInt     nprocs, rank;
    ierr = SlepcInitialize(&argc, &argv, (char*)0, help); CHKERRQ(ierr);
    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &nprocs); CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank); CHKERRQ(ierr);
    {
        DMRGBlockContainer<Block::SpinOneHalf, Hamiltonians::J1J2XYModel_SquareLattice> DMRG(PETSC_COMM_WORLD);

        /* Default behavior: Use the same number of states for warmup and sweeps */
        PetscInt nsweeps = 1, mstates = 8;
        ierr = PetscOptionsGetInt(NULL,NULL,"-mstates",&mstates,NULL); CHKERRQ(ierr);
        ierr = PetscOptionsGetInt(NULL,NULL,"-nsweeps",&nsweeps,NULL); CHKERRQ(ierr);

        /* Optional behavior: Specify the number of states for warmup, and the number of states for successive sweeps */
        PetscBool use_msweeps = PETSC_FALSE;
        PetscInt num_msweeps = MAX_SWEEPS;
        std::vector<PetscInt> msweeps(MAX_SWEEPS);
        ierr = PetscOptionsGetIntArray(NULL,NULL,"-msweeps",&msweeps[0],&num_msweeps,&use_msweeps); CHKERRQ(ierr);
        msweeps.resize(num_msweeps);

        /* Print some info */
        if(!rank){
            std::cout
                << "WARMUP\n"
                << "  NumStates to keep:       " << mstates << "\n"
                << "SWEEPS\n"
                << "  Use msweeps array:       " << (use_msweeps?"yes":"no") << "\n"
                << "  Number of sweeps:        " << (use_msweeps?num_msweeps:nsweeps) << "\n"
                << "  NumStates to keep:      ";
            if(use_msweeps) for(const PetscInt& m: msweeps) std::cout << " " << m;
            else std::cout << " " << mstates;
            std::cout << std::endl;
        }

        {
            if(!rank) std::cout << "MEASUREMENTS" << std::endl;
            /*  Explicitly give a list of operators. */
            PetscInt Lx = DMRG.HamiltonianRef().Lx();
            PetscInt Ly = DMRG.HamiltonianRef().Ly();
            PetscInt NumSitesSys = Lx*Ly/2;

            /*  The magnetization at each site */
            for(PetscInt idx=0; idx < NumSitesSys; ++idx)
            {
                std::vector< Op > OpList;
                OpList.push_back({OpSz,idx});

                PetscInt ix, jy;
                std::string desc;
                ierr = DMRG.HamiltonianRef().To2D(idx,ix,jy); CHKERRQ(ierr);
                desc += "< ";
                desc += "Sz_{"+ std::to_string(ix) + "," + std::to_string(jy) + "} ";
                desc += ">";
                std::string name = "Magnetization(" + std::to_string(idx) + ")";
                ierr = DMRG.SetUpCorrelation(OpList, name, desc); CHKERRQ(ierr);
            }

            /*  The pair-wise correlation of nearest neighbors (Sz-Sz, Sp-Sm, Sm-Sp) for calculating bond energies
                for the corresponding Heisenberg model */
            {
                const std::vector< std::vector < PetscInt > > nnp = DMRG.HamiltonianRef().NeighborPairs();
                for(const std::vector< PetscInt > pair : nnp)
                {
                    if(pair.size()!=2) SETERRQ1(PETSC_COMM_WORLD,1,"Invalid 2-point correlator. Got %lu operators instead.", pair.size());
                    std::vector< std::vector< Op_t > > OpTypesList = { {OpSz, OpSz}, {OpSp, OpSm}, {OpSm, OpSp} };
                    for (const std::vector< Op_t >& OpTypes : OpTypesList)
                    {
                        std::vector< Op > OpList;
                        std::string desc = "< ";
                        std::string name = std::string("NearestNeighbor") + OpToStr(OpTypes[0]) + OpToStr(OpTypes[1]) + "( ";
                        for(PetscInt i=0; i < 2; ++i)
                        {
                            const PetscInt idx = pair.at(i);
                            const Op_t OpType = OpTypes.at(i);
                            OpList.push_back({OpType,idx});
                            PetscInt ix, jy;
                            ierr = DMRG.HamiltonianRef().To2D(idx,ix,jy); CHKERRQ(ierr);
                            desc += OpToStr(OpType) + "_{" + std::to_string(ix) + "," + std::to_string(jy) + "} ";
                            name += std::to_string(idx) + " ";
                        }
                        desc += ">";
                        name += ")";
                        ierr = DMRG.SetUpCorrelation(OpList, name, desc); CHKERRQ(ierr);
                    }
                }
            }

            /*  The 1st row from the top <Sz_{0,0} Sz_{1,0} ... Sz_{Lx-1,0}> */
            {
                std::vector< Op > OpList;
                std::string desc;
                desc += "< ";
                for(PetscInt i = 0; i < Lx; ++i){
                    const PetscInt ix  = i;
                    const PetscInt jy  = 1;
                    const PetscInt idx = DMRG.HamiltonianRef().To1D(ix,jy);
                    OpList.push_back({OpSz,idx});
                    desc += "Sz_{"+ std::to_string(ix) + "," + std::to_string(jy) + "} ";
                }
                desc += ">";
                ierr = DMRG.SetUpCorrelation(OpList, "MagnetizationRowX1", desc); CHKERRQ(ierr);
            }

            /*  The second left-most column <Sz_{1,0} Sz_{1,1} ... Sz_{1,Ly-1}>. Equivalent to a Polyakov
                loop when BCx is periodic */
            {
                std::vector< Op > OpList;
                std::string desc;
                desc += "< ";
                for(PetscInt j = 0; j < Ly; ++j){
                    const PetscInt ix  = 1;
                    const PetscInt jy  = j;
                    const PetscInt idx = DMRG.HamiltonianRef().To1D(ix,jy);
                    OpList.push_back({OpSz,idx});
                    desc += "Sz_{"+ std::to_string(ix) + "," + std::to_string(jy) + "} ";
                }
                desc += ">";
                ierr = DMRG.SetUpCorrelation(OpList, "Polyakov", desc); CHKERRQ(ierr);
            }

            /*  The second to the last left-most column <Sz_{Lx-2,0} Sz_{Lx-2,1} ... Sz_{Lx-2,Ly-1}>. Equivalent to a Polyakov
                loop when BCx is periodic */
            {
                std::vector< Op > OpList;
                std::string desc;
                desc += "< ";
                for(PetscInt j = 0; j < Ly; ++j){
                    const PetscInt ix  = Lx-2;
                    const PetscInt jy  = j;
                    const PetscInt idx = DMRG.HamiltonianRef().To1D(ix,jy);
                    OpList.push_back({OpSz,idx});
                    desc += "Sz_{"+ std::to_string(ix) + "," + std::to_string(jy) + "} ";
                }
                desc += ">";
                ierr = DMRG.SetUpCorrelation(OpList, "Polyakov2", desc); CHKERRQ(ierr);
            }

            /*  We can also measure a Wilson loop of size (Lx-2)*(Ly-2) on the interior. */
            {
                std::vector< Op > OpList;
                std::string desc;
                desc += "< ";
                for(PetscInt j = 1; j < Ly-2; ++j){
                    const PetscInt ix  = 1;
                    const PetscInt jy  = j;
                    const PetscInt idx = DMRG.HamiltonianRef().To1D(ix,jy);
                    OpList.push_back({OpSz,idx});
                    desc += "Sz_{"+ std::to_string(ix) + "," + std::to_string(jy) + "} ";
                }
                for(PetscInt i = 1; i < Lx-2; ++i){
                    const PetscInt ix  = i;
                    const PetscInt jy  = Ly-2;
                    const PetscInt idx = DMRG.HamiltonianRef().To1D(ix,jy);
                    OpList.push_back({OpSz,idx});
                    desc += "Sz_{"+ std::to_string(ix) + "," + std::to_string(jy) + "} ";
                }
                for(PetscInt j = Ly-2; j > 1; --j){
                    const PetscInt ix  = Lx-2;
                    const PetscInt jy  = j;
                    const PetscInt idx = DMRG.HamiltonianRef().To1D(ix,jy);
                    OpList.push_back({OpSz,idx});
                    desc += "Sz_{"+ std::to_string(ix) + "," + std::to_string(jy) + "} ";
                }
                for(PetscInt i = Lx-2; i > 1; --i){
                    const PetscInt ix  = i;
                    const PetscInt jy  = 1;
                    const PetscInt idx = DMRG.HamiltonianRef().To1D(ix,jy);
                    OpList.push_back({OpSz,idx});
                    desc += "Sz_{"+ std::to_string(ix) + "," + std::to_string(jy) + "} ";
                }
                desc += ">";
                ierr = DMRG.SetUpCorrelation(OpList, "Wilson", desc); CHKERRQ(ierr);
            }
        }

        if(!rank){
            std::cout << "=========================================" << std::endl;
        }

        /* Perform DMRG steps */
        ierr = DMRG.Warmup(mstates); CHKERRQ(ierr);
        if(use_msweeps){
            for(const PetscInt& mstates: msweeps){
                ierr = DMRG.Sweep(mstates); CHKERRQ(ierr);
            }
        } else {
            for(PetscInt isweep = 0; isweep < nsweeps; ++isweep){
                ierr = DMRG.Sweep(mstates); CHKERRQ(ierr);
            }
        }

        ierr = DMRG.Destroy(); CHKERRQ(ierr);
    }
    ierr = SlepcFinalize(); CHKERRQ(ierr);
    return(0);
}
