#ifndef __DMRG_KRON_HPP
#define __DMRG_KRON_HPP

/**
    @defgroup   DMRGKron   DMRGKron
    @brief      Implementation of the Block_SpinOneHalf class which contains the data and methods
                for a block of spin sites
    @addtogroup DMRGKron
    @{ */

#include <petscsys.h>
#include <slepceps.h>

#include "DMRGBlock.hpp"

/** Storage for information on resulting blocks of quantum numbers stored as a tuple for quick sorting.
    - 0th entry:  PetscReal - Quantum number
    - 1st entry:  PetscInt - Left block index
    - 2nd entry:  PetscInt - Right block index
    - 3rd entry:  PetscInt - Number of states in the block */
typedef std::tuple<PetscReal, PetscInt, PetscInt, PetscInt> KronBlock_t;

/** A container of ordered KronBlock_t objects representing a Kronecker product structure */
class KronBlocks_t
{
public:

    KronBlocks_t(
        const Block_SpinOneHalf& LeftBlock,
        const Block_SpinOneHalf& RightBlock
        )
    {
        /** Generate the array of KronBlocks */
        for (size_t IL = 0; IL < LeftBlock.Magnetization.List().size(); ++IL){
            for (size_t IR = 0; IR < RightBlock.Magnetization.List().size(); ++IR){
                KronBlocks.push_back(std::make_tuple(
                    LeftBlock.Magnetization.List()[IL] + RightBlock.Magnetization.List()[IR], IL, IR,
                    LeftBlock.Magnetization.Sizes()[IL] * RightBlock.Magnetization.Sizes()[IR]));
            }
        }

        /*  Sort by descending quantum numbers */
        std::stable_sort(KronBlocks.begin(), KronBlocks.end(), DescendingQN);

        /* Fill-in size and offset data from the sorted blocks */
        num_blocks = KronBlocks.size();
        kb_list.reserve(num_blocks);
        kb_size.reserve(num_blocks);
        kb_offset.reserve(num_blocks+1);

        /* Unload the data into separate vectors */
        for(KronBlock_t kb: KronBlocks) kb_list.push_back(std::get<0>(kb));
        for(KronBlock_t kb: KronBlocks) kb_size.push_back(std::get<3>(kb));

        PetscInt idx = 0;
        for(KronBlock_t kb: KronBlocks)
            kb_map[std::make_tuple( std::get<1>(kb), std::get<2>(kb) )] = idx++;

        PetscInt sum = 0;
        for(KronBlock_t kb: KronBlocks)
        {
            kb_offset.push_back(sum);
            sum += std::get<3>(kb);
        }
        kb_offset.push_back(sum);
        num_states = sum;

    }

    size_t size() const { return KronBlocks.size(); }
    const std::vector<KronBlock_t>& data() const { return KronBlocks; }
    KronBlock_t data(size_t idx) const { return KronBlocks[idx]; }
    KronBlock_t operator[](size_t idx) const { return KronBlocks[idx]; }

    /** Returns the list of quantum numbers */
    std::vector<PetscReal> List() const { return kb_list; }

    /** Returns the offsets for each quantum number block */
    std::vector<PetscInt> Offsets() const { return kb_offset; }

    /** Returns the offsets for the KronBlock corresponding to a pair of left and right block indices */
    PetscInt Offsets(const PetscInt& lidx, const PetscInt& ridx) const {
        PetscInt idx = Map(lidx, ridx);
        if(idx >= 0) return kb_offset[idx];
        else return -1;
    }

    /** Returns the number of basis states in each quantum number block */
    std::vector<PetscInt> Sizes() const{ return kb_size; }

    /** Returns the position index of the KronBlock corresponding to a pair of left and right block indices;
        if the index is not found it returns -1 */
    PetscInt Map(const PetscInt& lidx, const PetscInt& ridx) const
    {
        auto tup = std::make_tuple(lidx,ridx);
        auto kb_find = kb_map.find(tup);
        if(kb_find != kb_map.end())
        {
            return kb_find->second;
        } else
        {
            return -1;
        }
    }

    /** Returns the total number of states */
    PetscInt NumStates() const { return num_states; }

private:

    /** Storage for kronblocks */
    std::vector<KronBlock_t> KronBlocks;

    std::vector<PetscReal> kb_list;

    std::vector<PetscInt> kb_size;

    std::vector<PetscInt> kb_offset;

    std::map< std::tuple<PetscInt,PetscInt>, PetscInt > kb_map;

    PetscInt num_blocks = 0;

    PetscInt num_states = 0;

    /** Comparison function to sort KronBlocks in descending order of quantum numbers */
    static bool DescendingQN(const KronBlock_t& a, const KronBlock_t& b)
    {
        return (std::get<0>(a)) > (std::get<0>(b));
    }

};


/** Calculates a new block combining two spin-1/2 blocks */
PetscErrorCode KronEye_Explicit(
    const Block_SpinOneHalf& LeftBlock,
    const Block_SpinOneHalf& RightBlock,
    Block_SpinOneHalf& BlockOut,
    PetscBool BuildHamiltonian
    );

/** Iterates through a range of basis states represented in the KronBlocks object */
class KronBlocksIterator
{
public:

    typedef KronBlocksIterator Self_t;

    /** Initialize an iterator through a range of indices */
    KronBlocksIterator(
        const KronBlocks_t& KronBlocks,
        const PetscInt& GlobIdxStart,
        const PetscInt& GlobIdxEnd
        ):
        KronBlocks(KronBlocks),
        istart_(GlobIdxStart),
        iend_(GlobIdxEnd),
        idx_(istart_)
    {
        if(istart_==iend_) {}
        else
        {
            kb_size.reserve(KronBlocks.size());
            kb_offset.reserve(KronBlocks.size()+1);

            for(KronBlock_t kb: KronBlocks.data()) kb_size.push_back(std::get<3>(kb));

            PetscInt sum = 0;
            for(KronBlock_t kb: KronBlocks.data())
            {
                kb_offset.push_back(sum);
                sum += std::get<3>(kb);
            }
            kb_offset.push_back(sum);
            num_states = sum;
            assert(istart_ < sum);

            while(idx_ >= kb_offset[blockidx_+1]) ++blockidx_;
        }
    }

    /** Gets the first quantum number block index */
    PetscInt IdxStart() const {return istart_;}

    /** Gets the first quantum number block index */
    PetscInt IdxEnd() const {return iend_;}

    /** Gets the current state index */
    PetscInt Idx() const {return idx_;}

    /** Gets the current quantum number block index */
    PetscInt BlockIdx() const {return blockidx_;}

    /** Gets the current local index in the quantum number block */
    PetscInt LocIdx() const {return idx_ - kb_offset[blockidx_];}

    /** Gets the column index of the starting block with a shift */
    PetscInt BlockStartIdx(
        const PetscInt& BlockShift  /**< [in]  Shift in quantum number associated to the operator */
        ) const
    {
        PetscInt BlockIdx_out = blockidx_ + BlockShift;
        if(BlockIdx_out < 0 || BlockIdx_out >= num_states){
            return -1;
        }
        return kb_offset[BlockIdx_out];
    }

    /** Determines whether the end of the range has not yet been reached */
    bool Loop() const {return idx_ < iend_;}

    /** Gets the number of steps incremented from the starting index */
    PetscInt Steps() const {return idx_-istart_;}

    /** Overloading the ++ increment */
    Self_t operator++()
    {
        ++idx_;
        if(idx_ >= kb_offset[blockidx_+1]){
            ++blockidx_;
            updated_block = PETSC_TRUE;
        } else {
            updated_block = PETSC_FALSE;
        }
        return *this;
    }

    /** Gets the block index for the left side */
    PetscInt BlockIdxLeft() const {return std::get<1>(KronBlocks.data()[blockidx_]);}

    /** Gets the block index for the right side */
    PetscInt BlockIdxRight() const {return std::get<2>(KronBlocks.data()[blockidx_]);}

    /** Gets the update state of the block index from the previous increment */
    PetscBool UpdatedBlock() const {return updated_block; }

private:

    /** Reference to the KronBlocks object on which this iterator is based on */
    const KronBlocks_t& KronBlocks;

    /** Starting index in the range [Istart, Iend) */
    PetscInt istart_ = 0;

    /** The final excluded index of the range [Istart, Iend) */
    PetscInt iend_ = 0;

    /** Stores the value of the current index */
    PetscInt idx_ = 0;

    /** The block index associated with Idx */
    PetscInt blockidx_ = -1;

    /** The local index in the block associated with Idx */
    PetscInt locidx_ = 0;

    /** Number of states in each quantum number block */
    std::vector<PetscInt> kb_size;

    /** Offset for each quantum number block */
    std::vector<PetscInt> kb_offset;

    /** Total number of states */
    PetscInt num_states = 0;

    /** Whether block was updated during the previous increment */
    PetscBool updated_block = PETSC_TRUE;
};


/**
    @}
 */

#endif // __DMRG_KRON_HPP
