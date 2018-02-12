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
namespace Hamiltonians
{
    /** A single interaction term on a one-dimensional lattice */
    struct Term
    {
        PetscScalar a;      /**< Constant coefficient */
        Op_t        Iop;    /**< The operator type for the left side */
        PetscInt    Isite;  /**< The index of the site for the left side */
        Op_t        Jop;    /**< The operator type for the right side */
        PetscInt    Jsite;  /**< The index of the site for the right side */
    };

    /** Identifies the possible boundary conditions at the edges of the lattice */
    typedef enum
    {
        OpenBC=0,
        PeriodicBC=1
    } BC_t;

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

            /* TODO: Also get boundary conditions from command line */

        }

        /** Returns the number of sites in the square lattice */
        PetscInt NumSites() const { return Lx*Ly; }

        /** Returns the object used to construct the Hamiltonian using sites counted from the left side of the lattice */
        std::vector< Term > H(
            const PetscInt& nsites  /**< Number of sites involved in the interaction */
            ) const;

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

        /** Boundary conditions for the longitudinal direction (defaults to cylindrical BC) */
        BC_t BCx = OpenBC;

        /** Boundary conditions for the transverse direction (defaults to cylindrical BC) */
        BC_t BCy = PeriodicBC;

        /** Prints some information related to the parameters of the Hamiltonian */
        PetscErrorCode PrintInfo() const
        {
            SETERRQ1(PETSC_COMM_SELF,1,"Function %s not implemented.", __FUNCTION__);
        }

        /** Given the 2d lattice coordinates, this function retrieves the nearest neighbors' 1D coordinates */
        std::vector<PetscInt> GetNearestNeighbors(
            const PetscInt& ix, const PetscInt& jy, const PetscInt& nsites_in
            ) const;

        /** Given the 2d lattice coordinates, this function retrieves the next-nearest neighbors' 1D coordinates */
        std::vector<PetscInt> GetNextNearestNeighbors(
            const PetscInt& ix, const PetscInt& jy, const PetscInt& nsites_in
            ) const;

    };
}

/** @} */

#endif // __HAMILTONIANS
