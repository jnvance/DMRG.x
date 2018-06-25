#include <petscsys.h>
#include <slepceps.h>

#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <map>

#include <sys/stat.h>
#include <sys/types.h>
#include <dirent.h>
#include "MiscTools.hpp"


/* Obtained from: https://gist.github.com/orlp/3551590 */
PETSC_EXTERN int64_t ipow(int64_t base, uint8_t exp) {
    static const uint8_t highest_bit_set[] = {
        0, 1, 2, 2, 3, 3, 3, 3,
        4, 4, 4, 4, 4, 4, 4, 4,
        5, 5, 5, 5, 5, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 6,
        6, 6, 6, 6, 6, 6, 6, 255, // anything past 63 is a guaranteed overflow with base > 1
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
        255, 255, 255, 255, 255, 255, 255, 255,
    };

    uint64_t result = 1;

    switch (highest_bit_set[exp]) {
    case 255: // we use 255 as an overflow marker and return 0 on overflow/underflow
        if (base == 1) {
            return 1;
        }

        if (base == -1) {
            return 1 - 2 * (exp & 1);
        }

        return 0;
    case 6:
        if (exp & 1) result *= base;
        exp >>= 1;
        base *= base;
    case 5:
        if (exp & 1) result *= base;
        exp >>= 1;
        base *= base;
    case 4:
        if (exp & 1) result *= base;
        exp >>= 1;
        base *= base;
    case 3:
        if (exp & 1) result *= base;
        exp >>= 1;
        base *= base;
    case 2:
        if (exp & 1) result *= base;
        exp >>= 1;
        base *= base;
    case 1:
        if (exp & 1) result *= base;
    default:
        return result;
    }
}


PetscErrorCode PreSplitOwnership(const MPI_Comm comm, const PetscInt N, PetscInt& locrows, PetscInt& Istart)
{
    PetscErrorCode ierr = 0;

#if 0
    /* The petsc way */
    PetscInt Nsize = N;
    PetscInt Lrows = PETSC_DECIDE;
    ierr = PetscSplitOwnership(comm, &Lrows, &Nsize); CHKERRQ(ierr);
    Istart = 0;
    ierr = MPI_Exscan(&Lrows, &Istart, 1, MPIU_INT, MPI_SUM, comm); CHKERRQ(ierr);
    locrows = Lrows;
#else
    /* Calculate predictively */
    PetscMPIInt nprocs,rank;
    ierr = MPI_Comm_size(comm, &nprocs); CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);
    const PetscInt remrows = N % nprocs;
    locrows = N / nprocs + PetscInt(rank < remrows);
    Istart =  N / nprocs * rank + (rank < remrows ? rank : remrows);
#endif

    return ierr;
}


PetscErrorCode SplitOwnership(
    const PetscMPIInt& rank,
    const PetscMPIInt& nprocs ,
    const PetscInt N,
    PetscInt& locrows,
    PetscInt& Istart)
{
    const PetscInt remrows = N % nprocs;
    locrows = N / nprocs + PetscInt(rank < remrows);
    Istart =  N / nprocs * rank + (rank < remrows ? rank : remrows);
    return 0;
}


PETSC_EXTERN PetscErrorCode InitSingleSiteOperator(const MPI_Comm& comm, const PetscInt dim, Mat* mat)
{
    PetscErrorCode ierr = 0;

    if(*mat) SETERRQ(comm, 1, "Matrix was previously initialized. First, destroy the matrix and set to NULL.");

    ierr = MatCreate(comm, mat); CHKERRQ(ierr);
    ierr = MatSetSizes(*mat, PETSC_DECIDE, PETSC_DECIDE, dim, dim); CHKERRQ(ierr);
    ierr = MatSetType(*mat, MATMPIAIJ); CHKERRQ(ierr);
    ierr = MatSetFromOptions(*mat); CHKERRQ(ierr);
    ierr = MatSetUp(*mat); CHKERRQ(ierr);

    ierr = MatSetOption(*mat, MAT_NO_OFF_PROC_ENTRIES          , PETSC_TRUE);
    ierr = MatSetOption(*mat, MAT_NO_OFF_PROC_ZERO_ROWS        , PETSC_TRUE);
    ierr = MatSetOption(*mat, MAT_IGNORE_OFF_PROC_ENTRIES      , PETSC_TRUE);
    ierr = MatSetOption(*mat, MAT_IGNORE_ZERO_ENTRIES          , PETSC_TRUE);

    return ierr;
}


PETSC_EXTERN PetscErrorCode MatSetOption_MultipleMats(
    const std::vector<Mat>& matrices,
    const std::vector<MatOption>& options,
    const std::vector<PetscBool>& flgs)
{
    PetscErrorCode ierr = 0;

    if(flgs.size() == 1)
    {
        for(const Mat& mat: matrices){
            for(const MatOption& op: options){
                ierr = MatSetOption(mat, op, flgs[0]); CHKERRQ(ierr);
            }
        }
    }
    else if(flgs.size() == options.size())
    {
        for(const Mat& mat: matrices){
            size_t i = 0;
            for(const MatOption& op: options){
                ierr = MatSetOption(mat, op, flgs[i++]); CHKERRQ(ierr);
            }
        }
    }
    else
    {
        SETERRQ(PETSC_COMM_WORLD, 1, "Incorrect input. Either flgs.size() == 1 or flgs.size() == options.size()");
    }

    return ierr;
}


PETSC_EXTERN PetscErrorCode MatSetOption_MultipleMatGroups(
    const std::vector<std::vector<Mat>>& matgroups,
    const std::vector<MatOption>& options,
    const std::vector<PetscBool>& flgs)
{
    PetscErrorCode ierr = 0;

    for(const std::vector<Mat>& matrices: matgroups){
        ierr = MatSetOption_MultipleMats(matrices, options, flgs); CHKERRQ(ierr);
    }

    return ierr;
}


/*----- Spin-1/2 functions -----*/

PETSC_EXTERN PetscErrorCode MatSpinOneHalfSzCreate(const MPI_Comm& comm, Mat& Sz)
{
    PetscErrorCode  ierr = 0;

    PetscInt loc_dim = 2;
    ierr = InitSingleSiteOperator(comm, loc_dim, &Sz); CHKERRQ(ierr);

    PetscInt locrows = 0, Istart = 0;
    ierr = PreSplitOwnership(comm, loc_dim, locrows, Istart); CHKERRQ(ierr);
    PetscInt Iend = Istart + locrows;

    if (Istart <= 0 && 0 < Iend){
        ierr = MatSetValue(Sz, 0, 0, +0.5, INSERT_VALUES); CHKERRQ(ierr);
    }
    if (Istart <= 1 && 1 < Iend){
        ierr = MatSetValue(Sz, 1, 1, -0.5, INSERT_VALUES); CHKERRQ(ierr);
    }

    ierr = MatAssemblyBegin(Sz, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Sz, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    return ierr;
}


PETSC_EXTERN PetscErrorCode MatSpinOneHalfSpCreate(const MPI_Comm& comm, Mat& Sp)
{
    PetscErrorCode  ierr = 0;

    PetscInt loc_dim = 2;
    ierr = InitSingleSiteOperator(comm, loc_dim, &Sp); CHKERRQ(ierr);

    PetscInt locrows = 0, Istart = 0;
    ierr = PreSplitOwnership(comm, loc_dim, locrows, Istart); CHKERRQ(ierr);
    PetscInt Iend = Istart + locrows;

    if (Istart <= 0 && 0 < Iend){
        ierr = MatSetValue(Sp, 0, 1, +1.0, INSERT_VALUES); CHKERRQ(ierr);
    }

    ierr = MatAssemblyBegin(Sp, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Sp, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    return ierr;
}


PETSC_EXTERN PetscErrorCode MatEnsureAssembled(const Mat& matin)
{
    PetscErrorCode ierr = 0;

    PetscBool assembled;
    ierr = MatAssembled(matin, &assembled); CHKERRQ(ierr);
    if(!assembled)
    {
        ierr = MatAssemblyBegin(matin, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(matin, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    }

    return ierr;
}

PETSC_EXTERN PetscErrorCode MatEnsureAssembled_MultipleMats(const std::vector<Mat>& matrices)
{
    PetscErrorCode ierr = 0;

    for(const Mat& mat: matrices){
        ierr = MatEnsureAssembled(mat); CHKERRQ(ierr);
    }

    return ierr;
}


PETSC_EXTERN PetscErrorCode MatEnsureAssembled_MultipleMatGroups(const std::vector<std::vector<Mat>>& matgroups)
{
    PetscErrorCode ierr = 0;

    for(const std::vector<Mat>& matrices: matgroups){
        ierr = MatEnsureAssembled_MultipleMats(matrices); CHKERRQ(ierr);
    }

    return ierr;
}


PETSC_EXTERN PetscErrorCode MatEyeCreate(const MPI_Comm& comm, const PetscInt& dim, Mat& eye)
{
    PetscErrorCode ierr = 0;

    ierr = InitSingleSiteOperator(comm, dim, &eye); CHKERRQ(ierr);
    ierr = MatEnsureAssembled(eye); CHKERRQ(ierr);
    ierr = MatShift(eye, 1.00); CHKERRQ(ierr);
    ierr = MatEnsureAssembled(eye); CHKERRQ(ierr);

    return ierr;
}


/*----- Utlities -----*/

PETSC_EXTERN PetscErrorCode Makedir(const std::string& dir_name)
{
    PetscErrorCode ierr;
    DIR *dir = opendir(dir_name.c_str());
    if(!dir){
        /* Info on mode_t: https://jameshfisher.com/2017/02/24/what-is-mode_t.html */
        ierr = mkdir(dir_name.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
        if(ierr){
            DIR *dir = opendir(dir_name.c_str());
            if(!dir){
                if(ierr) PetscPrintf(PETSC_COMM_SELF,"mkdir error code: %d for dir: %s\n",
                    ierr, dir_name.c_str());
                CHKERRQ(ierr);
            }
            closedir(dir);
        }
    } else {
        closedir(dir);
    }
    return(0);
}


PetscErrorCode SetOptionsFromFile(
    MPI_Comm& mpi_comm,
    const std::string& filename
    )
{
    PetscErrorCode ierr;
    PetscMPIInt mpi_rank;
    ierr = MPI_Comm_rank(mpi_comm, &mpi_rank); CHKERRQ(ierr);
    std::map< std::string, std::string > infomap;
    ierr = RetrieveInfoFile<std::string>(mpi_comm,filename,infomap); CHKERRQ(ierr);
    for(auto it: infomap) {
        ierr = PetscOptionsSetValue(NULL,(it.first).c_str(),(it.second).c_str()); CHKERRQ(ierr);
    }
    if(!mpi_rank) {
        std::cout << "========================================="  << std::endl;
        std::cout << "WARNING:\n"
        "The following directives from " << filename << "\n" <<
        "will override command-line arguments:" << std::endl;
        for(auto it: infomap)
            std::cout << "  " << it.first << "  " << it.second << std::endl;
    }
    return(0);
}
