#include <iostream>
#include <petscsys.h>
#include <slepceps.h>
#include "DMRGBlock.hpp"

#define DMRG_KRON_TESTING 1

#if DMRG_KRON_TESTING
    #ifndef PRINT_RANK_BEGIN
    #define PRINT_RANK_BEGIN() \
        for(PetscMPIInt irank = 0; irank < mpi_size; ++irank){\
            if(irank==mpi_rank){std::cout << "[" << mpi_rank << "]<<" << std::endl;
    #endif

    #ifndef PRINT_RANK_END
    #define PRINT_RANK_END() \
            std::cout << ">>[" << mpi_rank << "]" << std::endl;}\
        ierr = MPI_Barrier(mpi_comm); CHKERRQ(ierr);}
    #endif
#else
    #ifndef PRINT_RANK_BEGIN
    #define PRINT_RANK_BEGIN()
    #endif

    #ifndef PRINT_RANK_END
    #define PRINT_RANK_END()
    #endif
#endif


/** Storage for information on resulting blocks of quantum numbers
    0th entry:  Quantum number
    1st entry:  Left block index
    2nd entry:  Right block index
    3rd entry:  Number of states in the block */
typedef std::tuple<PetscReal, PetscInt, PetscInt, PetscInt> KronBlock_t;


/** Comparison function to sort KronBlocks in descending order of quantum numbers */
bool compare_descending_qn(KronBlock_t a, KronBlock_t b)
{
    return (std::get<0>(a)) > (std::get<0>(b));
}


/** Calculates the new block combining two spin-1/2 blocks */
PETSC_EXTERN PetscErrorCode Kron_Explicit(
    const Block_SpinOneHalf& LeftBlock,
    const Block_SpinOneHalf& RightBlock,
    Block_SpinOneHalf& BlockOut,
    PetscBool BuildHamiltonian
    )
{
    PetscErrorCode ierr = 0;

    /*  Extract MPI Information through allocated matrices
     *  NOTE: This call assumes that the Sz operator at the 0th site
     *  of the left block has been allocated.
     *  TODO: Change the way we obtain mpi_comm
     */
    MPI_Comm mpi_comm = LeftBlock.MPIComm();
    PetscMPIInt mpi_rank, mpi_size;
    // ierr = PetscObjectGetComm((PetscObject)LeftBlock.Sz[0], &mpi_comm); CHKERRQ(ierr);
    ierr = MPI_Comm_rank(mpi_comm, &mpi_rank); CHKERRQ(ierr);
    ierr = MPI_Comm_size(mpi_comm, &mpi_size); CHKERRQ(ierr);

    /*  For checking the accuracy of the routine
        TODO: Remove later */
    #if DMRG_KRON_TESTING
        PRINT_RANK_BEGIN()
        std::cout << "***** Kron_Explicit *****" << std::endl;
        std::cout << "LeftBlock  qn_list:   ";
        for(auto i: LeftBlock.Magnetization.List()) std::cout << i << "   ";
        std::cout << std::endl;

        std::cout << "LeftBlock  qn_size:   ";
        for(auto i: LeftBlock.Magnetization.Sizes()) std::cout << i << "   ";
        std::cout << std::endl;

        std::cout << "LeftBlock  qn_offset: ";
        for(auto i: LeftBlock.Magnetization.Offsets()) std::cout << i << "   ";
        std::cout << std::endl;

        std::cout << std::endl;

        std::cout << "RightBlock qn_list:   ";
        for(auto i: RightBlock.Magnetization.List()) std::cout << i << "   ";
        std::cout << std::endl;

        std::cout << "RightBlock qn_size:   ";
        for(auto i: RightBlock.Magnetization.Sizes()) std::cout << i << "   ";
        std::cout << std::endl;

        std::cout << "RightBlock qn_offset: ";
        for(auto i: RightBlock.Magnetization.Offsets()) std::cout << i << "   ";
        std::cout << std::endl;
        PRINT_RANK_END()
    #endif

    /* Check the validity of input blocks */
    ierr = LeftBlock.CheckOperators(); CHKERRQ(ierr);
    ierr = LeftBlock.CheckSectors(); CHKERRQ(ierr);
    ierr = LeftBlock.CheckOperatorBlocks(); CHKERRQ(ierr); /* NOTE: Possibly costly operation */
    ierr = RightBlock.CheckOperators(); CHKERRQ(ierr);
    ierr = RightBlock.CheckSectors(); CHKERRQ(ierr);
    ierr = RightBlock.CheckOperatorBlocks(); CHKERRQ(ierr); /* NOTE: Possibly costly operation */

    /*  Create a list of tuples of quantum numbers following the kronecker product structure */
    std::vector<KronBlock_t> KronBlocks;
    // for (PetscInt IL = LeftBlock.qn_list.size()-1; IL >= 0; --IL) // Checks sorting
    for (size_t IL = 0; IL < LeftBlock.Magnetization.List().size(); ++IL){
        for (size_t IR = 0; IR < RightBlock.Magnetization.List().size(); ++IR){
            KronBlocks.push_back(std::make_tuple(
                LeftBlock.Magnetization.List()[IL] + RightBlock.Magnetization.List()[IR], IL, IR,
                LeftBlock.Magnetization.Sizes()[IL] * RightBlock.Magnetization.Sizes()[IR]));
        }
    }

    /*  Sort the list in descending order of quantum numbers */
    std::stable_sort(KronBlocks.begin(), KronBlocks.end(), compare_descending_qn);

    #if DMRG_KRON_TESTING
        PRINT_RANK_BEGIN()
        std::cout << "KronBlocks: \n";
        for(auto k: KronBlocks)
        {
            std::cout << "( "
                << std::get<0>(k) << ", "
                << std::get<1>(k) << ", "
                << std::get<2>(k) << ", "
                << std::get<3>(k) << " )\n";
        }
        std::cout << "*************************" << std::endl;
        PRINT_RANK_END()
    #endif

    /*  Count the input and output number of sites */
    PetscInt nsites_left  = LeftBlock.NumSites();
    PetscInt nsites_right = RightBlock.NumSites();
    PetscInt nsites_out   = nsites_left + nsites_right;

    /*  Count the input and output number of sectors */
    PetscInt nsectors_left  = LeftBlock.Magnetization.NumSectors();
    PetscInt nsectors_right = RightBlock.Magnetization.NumSectors();
    PetscInt nsectors_out   = nsectors_left * nsectors_right;
    if(PetscUnlikely((size_t) KronBlocks.size() != nsectors_out ))
        SETERRQ2(mpi_comm, 1, "Mismatch in number of sectors. Expected %lu. Got %d.", KronBlocks.size(), nsectors_out);

    /*  Count the input and output number of states */
    PetscInt nstates_left  = LeftBlock.Magnetization.NumStates();
    PetscInt nstates_right = RightBlock.Magnetization.NumStates();
    PetscInt nstates_out   = nstates_left * nstates_right;
    PetscInt KronBlocks_nstates = 0;
    for (auto tup: KronBlocks) KronBlocks_nstates += std::get<3>(tup);
    if(PetscUnlikely(KronBlocks_nstates != nstates_out))
        SETERRQ2(mpi_comm, 1, "Mismatch in number of states. Expected %lu. Got %d.", KronBlocks_nstates, nstates_out);

    /*  Some quantum numbers that appear multiple times need to be grouped into a single quantum number block
        NOTE: This assumes that KronBlocks has been sorted */
    std::vector<PetscReal>  QN_List;
    std::vector<PetscInt>   QN_Size;
    PetscReal QN_last = 0;
    for (auto tup: KronBlocks){
        const PetscReal& qn   = std::get<0>(tup);
        const PetscInt&  size = std::get<3>(tup);
        if(qn < QN_last || QN_List.size()==0){
            QN_List.push_back(qn);
            QN_Size.push_back(size);
        } else {
            QN_Size.back() += size;
        }
        QN_last = qn;
    }

    PetscInt QN_Size_total = 0;
    for(PetscInt size: QN_Size) QN_Size_total += size;
    if(PetscUnlikely(nstates_out != QN_Size_total))
        SETERRQ2(mpi_comm, 1, "Mismatch in number of states. Expected %d. Got %d.", nstates_out, QN_Size_total);

    #if DMRG_KRON_TESTING
        PRINT_RANK_BEGIN()
        std::cout << "QN_List: "; for(auto q: QN_List) std::cout << q << " "; std::cout << std::endl;
        std::cout << "Total Sites: " << nsites_out << std::endl;
        std::cout << "Total Sectors: " << nsectors_out << std::endl;
        std::cout << "Total States: " << nstates_out << std::endl;
        PRINT_RANK_END()
    #endif

    /*  Initialize the new block using the quantum number blocks */
    ierr = BlockOut.Initialize(mpi_comm, nsites_out, QN_List, QN_Size); CHKERRQ(ierr);

    #if DMRG_KRON_TESTING
        PRINT_RANK_BEGIN()
        std::cout << "Mag: QN_List: "; for(auto q: BlockOut.Magnetization.List()) std::cout << q << " "; std::cout << std::endl;
        std::cout << "Mag: QN_Size: "; for(auto q: BlockOut.Magnetization.Sizes()) std::cout << q << " "; std::cout << std::endl;
        PRINT_RANK_END()
    #endif

    /*  Combine sites from the old blocks to form the new block */
    /*  Expand the left-block states explicitly by padding identities to the right */
    for(PetscInt isite = 0; isite < nsites_left; ++isite)
    {
        /* TODO: Call routine that fills in the matrices */
        if(!mpi_rank) printf("isite: %d\n",isite);
    }
    /*  Expand the left-block states explicitly by padding identities to the right */
    for(PetscInt isite = nsites_left; isite < nsites_left + nsites_right; ++isite)
    {
        /* TODO: Call routine that fills in the matrices */
        if(!mpi_rank) printf("isite: %d\n",isite);
    }

    return ierr;
}
