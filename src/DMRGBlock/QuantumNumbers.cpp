#include "QuantumNumbers.hpp"


PetscErrorCode QuantumNumbers::Initialize(
    MPI_Comm mpi_comm_in,
    std::vector<PetscReal> qn_list_in,
    std::vector<PetscInt> qn_size_in
    )
{
    mpi_comm = mpi_comm_in;
    if(qn_list_in.size()==0) SETERRQ(mpi_comm,1,"Initialization error: Empty input list.");
    if(qn_list_in.size()!=qn_size_in.size()) SETERRQ(mpi_comm,1,"Initialization error: Input list sizes mismatch.");
    num_sectors = (PetscInt) qn_list_in.size();
    qn_list = qn_list_in;
    qn_size = qn_size_in;

    qn_offset.resize(num_sectors+1);
    qn_offset[0] = 0;
    for(PetscInt i = 1; i < num_sectors+1; ++i)
        qn_offset[i] = qn_offset[i-1] + qn_size[i-1];

    num_states = qn_offset.back();
    initialized = PETSC_TRUE;

    return 0;
}
