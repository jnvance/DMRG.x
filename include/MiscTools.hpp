#ifndef __DMRG_MISCTOOLS_HPP__
#define __DMRG_MISCTOOLS_HPP__

#define DMRG_KRON_TIMINGS

#if defined(DMRG_KRON_TIMINGS)
    #include <petsctime.h>
    #define TIMINGS_NEWLINE() \
        if(!mpi_rank) printf("\n");
    #define FUNCTION_TIMINGS_BEGIN() \
        PetscLogDouble tstart = 0.0, tend = 0.0; \
        if(!mpi_rank) PetscTime(&tstart);
    #define FUNCTION_TIMINGS_END() \
        if(!mpi_rank){ \
            PetscTime(&tend); \
            printf("    %-28s   %-12.6f s\n", __FUNCTION__, tend - tstart); \
        }
    #define FUNCTION_TIMINGS_PRINT_SPACE() if(!mpi_rank) printf("\n");
    #define INTERVAL_TIMINGS_SETUP() PetscLogDouble itstart = 0.0, itend = 0.0;
    #define INTERVAL_TIMINGS_BEGIN() if(!mpi_rank) PetscTime(&itstart);
    #define INTERVAL_TIMINGS_END(LABEL) \
        if(!mpi_rank){ \
            PetscTime(&itend); \
            printf("      %-28s %-12.6f s\n", LABEL, itend - itstart); \
        }
    #define ACCUM_TIMINGS_SETUP(LABEL)  PetscLogDouble ts_##LABEL = 0.0, te_##LABEL = 0.0, tot_##LABEL = 0.0;
    #define ACCUM_TIMINGS_BEGIN(LABEL)  if(!mpi_rank){ PetscTime(&ts_##LABEL); }
    #define ACCUM_TIMINGS_END(LABEL)    if(!mpi_rank){ PetscTime(&te_##LABEL); \
        tot_##LABEL += (te_##LABEL - ts_##LABEL); }
    #define ACCUM_TIMINGS_PRINT(LABEL, TEXT)  \
        if(!mpi_rank){ \
            printf("      %-28s %-12.6f s\n", TEXT, tot_##LABEL); \
        }
#else
    #define TIMINGS_NEWLINE()
    #define FUNCTION_TIMINGS_BEGIN()
    #define FUNCTION_TIMINGS_END()
    #define FUNCTION_TIMINGS_PRINT_SPACE()
    #define INTERVAL_TIMINGS_SETUP()
    #define INTERVAL_TIMINGS_BEGIN()
    #define INTERVAL_TIMINGS_END(LABEL)
    #define ACCUM_TIMINGS_SETUP(LABEL)
    #define ACCUM_TIMINGS_BEGIN(LABEL)
    #define ACCUM_TIMINGS_END(LABEL)
    #define ACCUM_TIMINGS_PRINT(LABEL, TEXT)
#endif

PETSC_EXTERN PetscErrorCode PreSplitOwnership(const MPI_Comm comm, const PetscInt N, PetscInt& locrows, PetscInt& Istart);

#endif // __DMRG_MISCTOOLS_HPP__
