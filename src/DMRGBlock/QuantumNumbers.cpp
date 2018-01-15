#include "QuantumNumbers.hpp"


PetscErrorCode QuantumNumbers::Initialize(
    const MPI_Comm& mpi_comm_in,
    const std::vector<PetscReal>& qn_list_in,
    const std::vector<PetscInt>& qn_size_in
    )
{
    mpi_comm = mpi_comm_in;

    if(qn_list_in.size()==0)
        SETERRQ(mpi_comm,1,"Initialization error: Empty input list.");
    if(qn_list_in.size()!=qn_size_in.size())
        SETERRQ(mpi_comm,1,"Initialization error: Input list sizes mismatch.");

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


PetscErrorCode QuantumNumbers::BlockIdxToGlobalRange(
    const PetscInt& BlockIdx,
    PetscInt& GlobIdxStart,
    PetscInt& GlobIdxEnd
    ) const
{
    if(PetscUnlikely(!initialized))
        SETERRQ(mpi_comm, 1, "Object not initialized. Call Initialize() first.");
    if(PetscUnlikely((BlockIdx < 0) || (BlockIdx >= num_sectors)))
        SETERRQ2(mpi_comm, 1, "Given BlockIdx (%d) out of bounds [0,%d).", BlockIdx, num_sectors);

    GlobIdxStart = qn_offset[BlockIdx];
    GlobIdxEnd   = qn_offset[BlockIdx + 1];

    return 0;
}


PetscErrorCode QuantumNumbers::QNToGlobalRange(
    const PetscReal& QNValue,
    PetscInt& GlobIdxStart,
    PetscInt& GlobIdxEnd
    ) const
{
    if(PetscUnlikely(!initialized))
        SETERRQ(mpi_comm, 1, "Object not initialized. Call Initialize() first.");

    /*  Search QNValue from qn_list and get the index */
    PetscInt BlockIdx = std::find(qn_list.begin(), qn_list.end(), QNValue) - qn_list.begin();

    if(PetscUnlikely(BlockIdx==num_sectors))
        SETERRQ1(mpi_comm, 1, "Given QNValue (%g) not found.", QNValue);

    GlobIdxStart = qn_offset[BlockIdx];
    GlobIdxEnd   = qn_offset[BlockIdx + 1];

    return 0;
}


PetscErrorCode QuantumNumbers::GlobalIdxToBlockIdx(
    const PetscInt& GlobIdx,
    PetscInt& BlockIdx
    ) const
{
    PetscInt ierr = 0;

    if(PetscUnlikely(!initialized))
        SETERRQ(mpi_comm, 1, "Object not initialized. Call Initialize() first.");

    if(PetscUnlikely(GlobIdx < 0 || GlobIdx >= num_states))
        SETERRQ2(PETSC_COMM_SELF, 1, "Given GlobIdx (%d) out of bounds [0,%d).", GlobIdx, num_states);

    BlockIdx = -1;
    while(GlobIdx >= qn_offset[BlockIdx+1]) ++BlockIdx;

    return ierr;
}


PetscErrorCode QuantumNumbers::GlobalIdxToQN(
    const PetscInt& GlobIdx,
    PetscReal& QNValue
    ) const
{
    PetscInt ierr = 0;

    PetscInt BlockIdx;
    ierr = GlobalIdxToBlockIdx(GlobIdx, BlockIdx); CHKERRQ(ierr);

    return ierr;
}
