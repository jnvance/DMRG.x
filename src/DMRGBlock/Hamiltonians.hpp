#ifndef __HAMILTONIANS
#define __HAMILTONIANS

#include "petscsys.h"
#include "DMRGBlock.hpp"
#include <vector>

struct HamiltonianTerm
{
    PetscScalar a;
    Op_t        Iop;
    PetscInt    Isite;
    Op_t        Jop;
    PetscInt    Jsite;
};

class J1J2XYModel_SquareLattice
{

public:

    J1J2XYModel_SquareLattice()
    {
        /* Get values from command line options */
        PetscErrorCode ierr;
        ierr = PetscOptionsGetReal(NULL,NULL,"-J1",&J1,NULL); assert(!ierr);
        ierr = PetscOptionsGetReal(NULL,NULL,"-J2",&J2,NULL); assert(!ierr);
        ierr = PetscOptionsGetInt(NULL,NULL,"-Lx",&Lx,NULL); assert(!ierr);
        ierr = PetscOptionsGetInt(NULL,NULL,"-Ly",&Ly,NULL); assert(!ierr);
        ierr = PetscOptionsGetBool(NULL,NULL,"-verbose",&verbose,NULL); assert(!ierr);
    }

    /** Returns the number of sites in the square lattice */
    PetscInt NumSites() const { return Lx*Ly; }

    /**/
    std::vector<PetscInt> Hamiltonian() const;

private:
    /** Coupling strength for nearest-neighbor interactions */
    PetscScalar J1 = 1.0;

    /** Coupling strength for next-nearest-neighbor interactions */
    PetscScalar J2 = 1.0;

    /** Length along (growing) longitudinal direction */
    PetscInt Lx = 4;

    /** Length along (fixed) transverse direction */
    PetscInt Ly = 3;

    /** Flag that tells whether to print some information */
    PetscBool verbose = PETSC_FALSE;

    PetscErrorCode PrintInfo() const
    {
        // PetscErrorCode ierr;


        return(0);
    }

};

#endif // __HAMILTONIANS
