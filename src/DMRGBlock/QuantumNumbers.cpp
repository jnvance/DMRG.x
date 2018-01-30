#include "QuantumNumbers.hpp"
#include <algorithm>

/* Internal macro for checking the initialization state of the qn object */
#define CheckInit(func) \
    if(PetscUnlikely(!initialized)) \
        SETERRQ(mpi_comm, PETSC_ERR_ARG_WRONGSTATE, "Object not initialized. Call Initialize() first.");

PetscErrorCode QuantumNumbers::Initialize(
    const MPI_Comm& mpi_comm_in,
    const std::vector<PetscReal>& qn_list_in,
    const std::vector<PetscInt>& qn_size_in
    )
{
    mpi_comm = mpi_comm_in;

    /** @throw PETSC_ERR_ARG_WRONG Empty input list */
    if(qn_list_in.size()==0)
        SETERRQ(mpi_comm, PETSC_ERR_ARG_WRONG, "Initialization error: Empty input list.");
    /** @throw PETSC_ERR_ARG_WRONG Input list sizes mismatch */
    if(qn_list_in.size()!=qn_size_in.size())
        SETERRQ(mpi_comm, PETSC_ERR_ARG_WRONG, "Initialization error: Input list sizes mismatch.");

    /** @remarks __TODO:__
            - Ensure that inputs are the same across entire communicator
            - Ensure that input list is complete and in descending order
            - Think about how to handle zero-sized quantum number blocks esp in the context of truncation
    */
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
    CheckInit(__FUNCTION__);
    /** @throw PETSC_ERR_ARG_WRONGSTATE Object not initialized. Call Initialize() first.*/

    if(PetscUnlikely((BlockIdx < 0) || (BlockIdx >= num_sectors)))
        /** @throw PETSC_ERR_ARG_OUTOFRANGE Given BlockIdx out of bounds [0, num_sectors) */
        SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Given BlockIdx (%d) out of bounds [0, %d).", BlockIdx, num_sectors);

    GlobIdxStart = qn_offset[BlockIdx];
    GlobIdxEnd   = qn_offset[BlockIdx + 1];
    return 0;
}


PetscErrorCode QuantumNumbers::OpBlockToGlobalRange(
    const PetscInt& BlockIdx,
    const PetscInt& BlockShift,
    PetscInt& GlobIdxStart,
    PetscInt& GlobIdxEnd,
    PetscBool& flg
    ) const
{
    CheckInit(__FUNCTION__);
    /** @throw PETSC_ERR_ARG_WRONGSTATE Object not initialized. Call Initialize() first.*/

    if(PetscUnlikely((BlockIdx < 0) || (BlockIdx >= num_sectors)))
        /** @throw PETSC_ERR_ARG_OUTOFRANGE Given BlockIdx is out of bounds [0, num_sectors).*/
        SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Given BlockIdx (%d) out of bounds [0, %d).", BlockIdx, num_sectors);

    PetscInt BlockIdx_out = BlockIdx + BlockShift;
    if(BlockIdx_out < 0 || BlockIdx_out >= num_states){
        flg = PETSC_FALSE;
        return 0;
    }
    flg = PETSC_TRUE;
    GlobIdxStart = qn_offset[BlockIdx_out];
    GlobIdxEnd   = qn_offset[BlockIdx_out + 1];
    return 0;
}


PetscErrorCode QuantumNumbers::QNToGlobalRange(
    const PetscReal& QNValue,
    PetscInt& GlobIdxStart,
    PetscInt& GlobIdxEnd
    ) const
{
    /** @throw PETSC_ERR_ARG_WRONGSTATE Object not initialized. Call Initialize() first.*/
    CheckInit(__FUNCTION__);

    /*  Search QNValue from qn_list and get the index */
    PetscInt BlockIdx = std::find(qn_list.begin(), qn_list.end(), QNValue) - qn_list.begin();

    /** @throw PETSC_ERR_ARG_OUTOFRANGE Given quantum number was not found. */
    if(PetscUnlikely(BlockIdx==num_sectors))
        SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Given QNValue (%g) not found.", QNValue);

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

    /** @throw PETSC_ERR_ARG_WRONGSTATE Object not initialized. Call Initialize() first.*/
    CheckInit(__FUNCTION__);
    /** @throw PETSC_ERR_ARG_OUTOFRANGE Given GlobIdx is out of bounds [0, num_states).*/
    if(PetscUnlikely(GlobIdx < 0 || GlobIdx >= num_states))
        SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Given GlobIdx (%d) out of bounds [0, %d).", GlobIdx, num_states);

    BlockIdx = -1;
    while(GlobIdx >= qn_offset[BlockIdx+1]) ++BlockIdx;

    return ierr;
}


PetscErrorCode QuantumNumbers::GlobalIdxToBlockIdx(
    const PetscInt& GlobIdx,
    PetscInt& BlockIdx,
    PetscInt& LocIdx
    ) const
{
    PetscInt ierr;

    ierr = GlobalIdxToBlockIdx(GlobIdx, BlockIdx); CHKERRQ(ierr);
    LocIdx = GlobIdx - qn_offset[BlockIdx];

    return(0);
}



PetscErrorCode QuantumNumbers::GlobalIdxToQN(
    const PetscInt& GlobIdx,
    PetscReal& QNValue
    ) const
{
    PetscInt ierr = 0;
    /** @throw PETSC_ERR_ARG_WRONGSTATE Object not initialized. Call Initialize() first.*/
    CheckInit(__FUNCTION__);

    PetscInt BlockIdx;
    /** @throw PETSC_ERR_ARG_OUTOFRANGE Given GlobIdx is out of bounds [0, num_states).*/
    ierr = GlobalIdxToBlockIdx(GlobIdx, BlockIdx); CHKERRQ(ierr);

    return ierr;
}
