#ifndef __DMRG_MISCTOOLS_HPP__
#define __DMRG_MISCTOOLS_HPP__

#include <petscsys.h>
#include <slepceps.h>

#include <map>
#include <string>
#include <fstream>
#include <iostream>

/**
    @defgroup   MiscTools   MiscTools
    @brief      Miscellaneous tools and other utilities commonly used by other modules.
*/

// #define DMRG_KRON_TIMINGS

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

/** @addtogroup MiscTools
    @{
 */

PetscErrorCode PreSplitOwnership(const MPI_Comm comm, const PetscInt N, PetscInt& locrows, PetscInt& Istart);

PetscErrorCode SplitOwnership(
    const PetscMPIInt& rank,
    const PetscMPIInt& nprocs ,
    const PetscInt N,
    PetscInt& locrows,
    PetscInt& Istart);

/** Utility to retrieve the contents of a key-value data file
    where the keys are strings and the values are of a given type @p T.
    @tparam     T           type which the value will be casted into
*/
template< typename T >
PetscErrorCode RetrieveInfoFile(
    const MPI_Comm& mpi_comm,           /**< [in] communicator */
    const std::string& filename,        /**< [in] path to input file */
    std::map< std::string, T >& infomap /**< [out] stores the key-value pairs */
    )
{
    PetscErrorCode ierr;
    PetscMPIInt mpi_rank;
    ierr = MPI_Comm_rank(mpi_comm, &mpi_rank); CHKERRQ(ierr);

    /* Read the file and build a key-value pair */
    PetscBool flg;
    std::string opt_string;
    PetscInt opt_string_length;

    if(!mpi_rank) {
        ierr = PetscTestFile(filename.c_str(), 'r', &flg); CHKERRQ(ierr);
        if(!flg) SETERRQ1(PETSC_COMM_SELF,1,"File %s unaccessible.", filename.c_str());
        std::ifstream infofile(filename.c_str());
        opt_string = std::string(
            (std::istreambuf_iterator<char>(infofile)),
            std::istreambuf_iterator<char>());
        infofile.close();
        opt_string_length = opt_string.size();
    }

    ierr = MPI_Bcast(&opt_string_length, 1, MPIU_INT, 0, mpi_comm); CHKERRQ(ierr);
    opt_string.resize(opt_string_length);
    ierr = MPI_Bcast(&opt_string[0], opt_string_length, MPI_CHAR, 0, mpi_comm); CHKERRQ(ierr);

    std::stringstream ss(opt_string);
    std::string line;
    infomap.clear();
    while(std::getline(ss, line)) {
        std::string key;
        T val;
        std::istringstream line_ss(line);
        line_ss >> key >> val;
        infomap[key] = val;
    }

    return(0);
}

/** Reads a file containing keys and values and sets them as command line arguments.
    The contents of the file must take the form:

        -key1   <value1>
        -key2   <value2>
        ...

 */
PetscErrorCode SetOptionsFromFile(
    MPI_Comm& mpi_comm,         /**< [in] communicator */
    const std::string& filename /**< [in] path to input file */
    );

/**
    @}
 */

#endif // __DMRG_MISCTOOLS_HPP__
