#ifndef __HAMILTONIANS
#define __HAMILTONIANS

#include "petscsys.h"
#include "DMRGBlock.hpp"
#include <vector>

/** @defgroup   Hamiltonians   Hamiltonians
    @brief      Implementation of the Hamiltonians and geometries for spin lattices
    @addtogroup Hamiltonians
    @{
 */

/** Implementation of Hamiltonian classes */
namespace Hamiltonian
{

    /** A single interaction term on a one-dimensional lattice */
    struct HamiltonianTerm
    {
        PetscScalar a;      /**< Constant coefficient */
        Op_t        Iop;    /**< The operator type for the left side */
        PetscInt    Isite;  /**< The index of the site for the left side */
        Op_t        Jop;    /**< The operator type for the right side */
        PetscInt    Jsite;  /**< The index of the site for the right side */
    };

    /** Implements the Hamiltonian for the J1-J2 XY model on the square lattice */
    class J1J2XYModel_SquareLattice
    {

    public:

        /** Constructor that takes in command line options */
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

        /** Returns the Hamiltonian */
        std::vector< HamiltonianTerm > Hamiltonian() const;

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

        /** Prints some information related to the parameters of the Hamiltonian */
        PetscErrorCode PrintInfo() const
        {
            SETERRQ1(PETSC_COMM_SELF,1,"Function %s not implemented.", __FUNCTION__);
        }
    };
}

/** @} */

#endif // __HAMILTONIANS
