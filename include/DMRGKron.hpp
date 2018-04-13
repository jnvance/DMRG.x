#ifndef __DMRG_KRON_HPP
#define __DMRG_KRON_HPP

/**
    @defgroup   DMRGKron   DMRGKron
    @brief      Implementation of the Kronecker product routines
    @addtogroup DMRGKron
    @{ */

#include <petscsys.h>
#include <slepceps.h>

#include "DMRGBlock.hpp"
#include "Hamiltonians.hpp"

/** Storage for information on resulting blocks of quantum numbers stored as a tuple for quick sorting.
    - 0th entry:  PetscReal - Quantum number
    - 1st entry:  PetscInt - Left block index
    - 2nd entry:  PetscInt - Right block index
    - 3rd entry:  PetscInt - Number of states in the block */
typedef std::tuple<PetscReal, PetscInt, PetscInt, PetscInt> KronBlock_t;

/** A single term of the KronSum.
    If one matrix is set to null, then that matrix is interpreted as an identity */
struct KronSumTerm {
    PetscScalar a;
    Op_t OpTypeA;
    Mat A;
    Op_t OpTypeB;
    Mat B;
};

struct KronSumCtx {
    PetscInt rstart=0;  /**< Starting index of local rows */
    PetscInt rend=0;    /**< Index after last of local rows, exclusive */
    PetscInt lrows=0;   /**< Number of local rows */
    PetscInt cstart=0;  /**< Starting index of local columns */
    PetscInt cend=0;    /**< Index after last of local columns, exclusive */
    PetscInt lcols=0;   /**< Number of local columns */
    PetscInt Nrows=0;   /**< Total number of rows */
    PetscInt Ncols=0;   /**< Total number of columns */

    /** Number of required rows of the left and right blocks */
    PetscInt NReqRowsL, NReqRowsR;

    /** List of required rows of the left and right blocks */
    std::vector< PetscInt > ReqRowsL, ReqRowsR;

    /** Maps the global indices of the rows of L and R to their local indices in the corresponding submatrices */
    std::unordered_map< PetscInt, PetscInt > MapRowsL, MapRowsR;

    /** Lists down all the terms of the KronSum with Mat entries filled with local submatrices */
    std::vector< KronSumTerm > Terms;

    /** List of unique submatrices to be destroyed later */
    std::vector< Mat* > LocalSubMats;

    /** Preallocation data of the output matrix for local diagonal rows */
    PetscInt *Dnnz;

    /** Preallocation data of the output matrix for local off-diagonal diagonal rows */
    PetscInt *Onnz;

    /** Smallest non-zero index in the current set of local rows */
    PetscInt MinIdx=0;

    /** Largest non-zero index in the current set of local rows */
    PetscInt MaxIdx=0;

    /** Predicted maximum number of elements on each local row */
    std::vector< PetscInt > Maxnnz;

    PetscInt Nfiltered=0;

    PetscInt Nnz=0;

    PetscInt Nelts=0;
};


struct KronSumTermRow {
    PetscBool   skip;
    PetscInt    nz_L, nz_R, bks_L, bks_R, col_NStatesR, fws_O;
    PetscInt    *idx_L, *idx_R;
    PetscScalar *v_L, *v_R;
};

/** Context for the shell matrix object */
struct KronSumShellCtx {

    /** Contains the usual ctx object for explicit matrices. This must be allocated and deallocated
        with `new` and `delete`, respectively. */
    KronSumCtx      *p_ctx;

    /** Contains the KronSumTermRow's needed for calculating the terms of the full matrix */
    KronSumTermRow  *kstr;
    PetscInt        Nterms;
    PetscInt        Nrowterms;

    PetscInt        *Rows_L;
    PetscInt        *Rows_R;
    PetscScalar     *term_a;

    PetscScalar     one;

    /* Mat-Vec multiplication */
    VecScatter      vsctx;
    Vec             x_seq;
};

PetscErrorCode MatDestroy_KronSumShell(Mat *p_mat);

/** A container of ordered KronBlock_t objects representing a Kronecker product structure */
class KronBlocks_t
{

friend class KronBlocksIterator;

public:

    KronBlocks_t(
        Block::SpinOneHalf& LeftBlock,
        Block::SpinOneHalf& RightBlock,
        const std::vector<PetscReal>& QNSectors, /**< [in] list of quantum number sectors for keeping selected states */
        FILE *fp_prealloc,
        const PetscInt& GlobIdx
        ):
        GlobIdx(GlobIdx),
        LeftBlock(LeftBlock),
        RightBlock(RightBlock),
        fp_prealloc(fp_prealloc)
    {
        /* Require blocks to be initialized */
        if(!LeftBlock.Initialized()) throw std::runtime_error("Left input block not initialized.");
        if(!RightBlock.Initialized()) throw std::runtime_error("Right input block not initialized.");
        /* Fill in mpi information */
        mpi_comm = LeftBlock.MPIComm();
        if(mpi_comm != RightBlock.MPIComm())
            throw std::runtime_error("Left and right blocks must have the same communicator.");
        if(MPI_Comm_rank(mpi_comm, &mpi_rank)) throw std::runtime_error("MPI Error.");
        if(MPI_Comm_size(mpi_comm, &mpi_size)) throw std::runtime_error("MPI Error.");

        /** Generate the array of KronBlocks keeping all QNs */
        if(QNSectors.size() == 0)
        {
            for (size_t IL = 0; IL < LeftBlock.Magnetization.List().size(); ++IL){
                for (size_t IR = 0; IR < RightBlock.Magnetization.List().size(); ++IR){
                    KronBlocks.push_back(std::make_tuple(
                        LeftBlock.Magnetization.List()[IL] + RightBlock.Magnetization.List()[IR], IL, IR,
                        LeftBlock.Magnetization.Sizes()[IL] * RightBlock.Magnetization.Sizes()[IR]));
                }
            }
            /*  Sort by descending quantum numbers */
            std::stable_sort(KronBlocks.begin(), KronBlocks.end(), DescendingQN);
        }
        /** Generate the array of KronBlocks keeping only specified QN (single value) */
        else if (QNSectors.size() == 1)
        {
            for (size_t IL = 0; IL < LeftBlock.Magnetization.List().size(); ++IL){
                for (size_t IR = 0; IR < RightBlock.Magnetization.List().size(); ++IR){
                    PetscReal QN = LeftBlock.Magnetization.List()[IL] + RightBlock.Magnetization.List()[IR];
                    if(QN == QNSectors[0])
                        KronBlocks.push_back(std::make_tuple(
                            QN, IL, IR,
                            LeftBlock.Magnetization.Sizes()[IL] * RightBlock.Magnetization.Sizes()[IR]));
                }
            }
        }
        /** Generate the array of KronBlocks keeping only specified QNs */
        else
        {
            /* Convert QNSectors into a set */
            std::set< PetscReal > QNSectorsSet(QNSectors.begin(), QNSectors.end());
            for (size_t IL = 0; IL < LeftBlock.Magnetization.List().size(); ++IL){
                for (size_t IR = 0; IR < RightBlock.Magnetization.List().size(); ++IR){
                    PetscReal QN = LeftBlock.Magnetization.List()[IL] + RightBlock.Magnetization.List()[IR];
                    if(QNSectorsSet.find(QN) != QNSectorsSet.end())
                        KronBlocks.push_back(std::make_tuple(
                            QN, IL, IR,
                            LeftBlock.Magnetization.Sizes()[IL] * RightBlock.Magnetization.Sizes()[IR]));
                }
            }
        }

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

        do_saveprealloc = PetscBool(fp_prealloc!=NULL);
        MPIU_Allreduce(MPI_IN_PLACE, &do_saveprealloc, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD);
    };

    /** Returns the total number blocks */
    size_t size() const { return KronBlocks.size(); }

    /** Returns a const reference to the KronBlocks object */
    const std::vector<KronBlock_t>& data() const { return KronBlocks; }

    /** Returns the KronBlock for specified index */
    KronBlock_t data(size_t idx) const { return KronBlocks[idx]; }

    /** Returns the KronBlock for specified index */
    KronBlock_t operator[](size_t idx) const { return KronBlocks[idx]; }

    /** Returns the list of quantum numbers */
    std::vector<PetscReal> List() const { return kb_list; }

    /** Returns the offsets for each quantum number block */
    std::vector<PetscInt> Offsets() const { return kb_offset; }

    /** Returns the offsets for a given index */
    PetscInt Offsets(const PetscInt& idx ) const {
        assert(idx >= 0 && idx < num_blocks + 1);
        return kb_offset[idx];
    }

    /** Returns the left quantum number index (KronBlocks 1st endtry) for a given KronBlock index */
    PetscReal QN(const PetscInt& idx) const {
        return std::get<0>(KronBlocks[idx]);
    }

    /** Returns the left quantum number index (KronBlocks 1st endtry) for a given KronBlock index */
    PetscInt LeftIdx(const PetscInt& idx) const {
        return std::get<1>(KronBlocks[idx]);
    }

    /** Returns the left quantum number index (KronBlocks 1st endtry) for a given KronBlock index */
    PetscInt RightIdx(const PetscInt& idx) const {
        return std::get<2>(KronBlocks[idx]);
    }

    /** Returns the number of basis states for a given KronBlock index */
    PetscInt Sizes(const PetscInt& idx) const {
        return std::get<3>(KronBlocks[idx]);
    }

    /** Returns a const reference to the left block object */
    const Block::SpinOneHalf& LeftBlockRef() const { return LeftBlock; }

    /** Returns a const reference to the right block object */
    const Block::SpinOneHalf& RightBlockRef() const { return RightBlock; }

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

    /** Constructs the explicit sum of Kronecker products of matrices from the blocks */
    PetscErrorCode KronSumConstruct(
        const std::vector< Hamiltonians::Term >& Terms, /**< [in]   indicates the Kronecker product terms to be constructed */
        Mat& MatOut                                     /**< [out]  resultant matrix */
        );

    /** Decide whether to create an implicit MATSHELL matrix */
    PetscErrorCode KronSumSetShellMatrix(const PetscBool& do_shell_in)
    {
        do_shell = do_shell_in;
        return(0);
    }

    PetscErrorCode KronSumSetRedistribute(
        const PetscBool& do_redistribute_in = PETSC_TRUE
        )
    {
        do_redistribute = do_redistribute_in;
        return(0);
    }

    PetscErrorCode KronSumSetToleranceFromOptions()
    {
        PetscErrorCode ierr;
        ierr = PetscOptionsGetReal(NULL,NULL,"-ks_tol",&ks_tol,NULL); CHKERRQ(ierr);
        return(0);
    }

private:

    MPI_Comm mpi_comm = PETSC_COMM_SELF;
    PetscMPIInt mpi_rank, mpi_size;

    const PetscInt GlobIdx;

    /** Storage for kronblocks */
    std::vector<KronBlock_t> KronBlocks;

    /** (Redundant) storage for quantum numbers */
    std::vector<PetscReal> kb_list;

    /** (Redundant) storage for sizes of each quantum number block */
    std::vector<PetscInt> kb_size;

    /** Storage for offsets or starting elements of each block */
    std::vector<PetscInt> kb_offset;

    /** Kronecker product mapping from (L,R) block to the corresponding index */
    std::map< std::tuple<PetscInt,PetscInt>, PetscInt > kb_map;

    /** The number of blocks stored */
    PetscInt num_blocks = 0;

    /** The total number of states stored */
    PetscInt num_states = 0;

    /** Reference to the left block object */
    Block::SpinOneHalf& LeftBlock;

    /** Reference to the right block object */
    Block::SpinOneHalf& RightBlock;

    /** File to store preallocation data for each processor */
    FILE *fp_prealloc;

    /** Whether to store preallocation data for each processor */
    PetscBool do_saveprealloc = PETSC_FALSE;

    /** Whether to redistribute the resulting KronSum */
    PetscBool do_redistribute = PETSC_FALSE;

    /** Whether to create an implicit MATSHELL matrix */
    PetscBool do_shell = PETSC_FALSE;

    /** Tolerance */
    #if defined(PETSC_USE_REAL_DOUBLE)
    PetscReal ks_tol = 1.0e-16;
    #else
    #error Only double precision real numbers supported.
    #endif

    /** Comparison function to sort KronBlocks in descending order of quantum numbers */
    static bool DescendingQN(const KronBlock_t& a, const KronBlock_t& b)
    {
        return (std::get<0>(a)) > (std::get<0>(b));
    }

    /** Calculates the Kronecker product of a set of matrices with the identity and adds it to */
    PetscErrorCode MatKronEyeAdd(
        const std::vector< Mat >& Matrices,
        const Side_t& SideType,
        Mat& MatOut
        );

    /** Verifies that the assumtpion of intra-block terms to follow Sz form is valid.
        Model-specific check: Since all possible terms of the Hamiltonian take the form of
        Sz-Sz or Sp-Sm with operators on the same block, then the resulting product should be of Sz-type. */
    PetscErrorCode VerifySzAssumption(
        const std::vector< Mat >& Matrices,
        const Side_t& SideType
        );

    PetscErrorCode KronSumConstructExplicit(
        const std::vector< Hamiltonians::Term >& TermsLR,
        Mat& MatOut
        );

    PetscErrorCode KronSumConstructShell(
        const std::vector< Hamiltonians::Term >& TermsLR,
        Mat& MatOut
        );

    PetscErrorCode KronSumSetUpShellTerms(
        KronSumShellCtx *shellctx
        );

    PetscErrorCode KronSumGetSubmatrices(
        const Mat& OpProdSumLL,
        const Mat& OpProdSumRR,
        const std::vector< Hamiltonians::Term >& TermsLR,
        KronSumCtx& SubMat
        );

    PetscErrorCode KronSumCalcPreallocation(
        KronSumCtx& ctx
        );

    PetscErrorCode KronSumRedistribute(
        KronSumCtx& ctx,
        PetscBool& flg
        );

    PetscErrorCode KronSumPreallocate(
        KronSumCtx& ctx,
        Mat& MatOut
        );

    PetscErrorCode KronSumFillMatrix(
        KronSumCtx& ctx,
        Mat& MatOut
        );

    PetscErrorCode SavePreallocData(const KronSumCtx& ctx);
};


/** Calculates a new block combining two spin-1/2 blocks */
PetscErrorCode KronEye_Explicit(
    Block::SpinOneHalf& LeftBlock,          /**< [in]   left block of sites */
    Block::SpinOneHalf& RightBlock,         /**< [in]   right block of sites */
    const std::vector< Hamiltonians::Term >& Terms, /**< [in]   Hamiltonina terms */
    Block::SpinOneHalf& BlockOut            /**< [out]  combined block */
    );

/** Calculates the sum of the Kronecker product of operators on two blocks following the terms of a Hamiltonian.

    This routine treats the right block in inverted order. For example, given a left block of 5 sites and a right block
    of 3 sites, the interpretation of the output block is as follows:

        L0 L1 L2 L3 L4 R2 R1 R0

    */

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

    /** Gets the size of the block with a shift */
    PetscInt BlockSize(
        const PetscInt& BlockShift  /**< [in]  Shift in quantum number associated to the operator */
        ) const
    {
        PetscInt BlockIdx_out = blockidx_ + BlockShift;
        if(BlockIdx_out < 0 || BlockIdx_out >= num_states){
            return -1;
        }
        return kb_size[BlockIdx_out];
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

    /** Gets the local index for the left side */
    PetscInt LocIdxLeft() const { return LocIdx() / NumStatesRight(); }

    /** Gets the number of states for the right block */
    PetscInt NumStatesRight() const { return KronBlocks.RightBlock.Magnetization.Sizes()[BlockIdxRight()]; }

    /** Gets the block index for the right side */
    PetscInt BlockIdxRight() const {return std::get<2>(KronBlocks.data()[blockidx_]);}

    /** Gets the local index for the right side */
    PetscInt LocIdxRight() const { return LocIdx() % NumStatesRight(); }

    PetscInt GlobalIdxLeft() const {
        return KronBlocks.LeftBlock.Magnetization.BlockIdxToGlobalIdx(BlockIdxLeft(), LocIdxLeft());
    }

    PetscInt GlobalIdxRight() const {
        return KronBlocks.RightBlock.Magnetization.BlockIdxToGlobalIdx(BlockIdxRight(), LocIdxRight());
    }

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
