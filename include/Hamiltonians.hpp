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

        /** Constructor */
        J1J2XYModel_SquareLattice()
        {

        }

        PetscErrorCode SetFromOptions()
        {
            PetscErrorCode ierr;
            /* Get values from command line options */
            ierr = PetscOptionsGetReal(NULL,NULL,"-J1",&J1,NULL); CHKERRQ(ierr);
            ierr = PetscOptionsGetReal(NULL,NULL,"-J2",&J2,NULL); CHKERRQ(ierr);
            ierr = PetscOptionsGetReal(NULL,NULL,"-Jz1",&Jz1,NULL); CHKERRQ(ierr);
            ierr = PetscOptionsGetReal(NULL,NULL,"-Jz2",&Jz2,NULL); CHKERRQ(ierr);
            ierr = PetscOptionsGetInt(NULL,NULL,"-Lx",&Lx,NULL); CHKERRQ(ierr);
            ierr = PetscOptionsGetInt(NULL,NULL,"-Ly",&Ly,NULL); CHKERRQ(ierr);
            ierr = PetscOptionsGetBool(NULL,NULL,"-verbose",&verbose,NULL); CHKERRQ(ierr);
            /* TODO: Also get boundary conditions from command line */
            PetscBool BCopen = PETSC_FALSE;
            ierr = PetscOptionsGetBool(NULL,NULL,"-BCopen",&BCopen,NULL); CHKERRQ(ierr);
            if(BCopen) { BCx=OpenBC; BCy=OpenBC; }
            PetscBool BCperiodic = PETSC_FALSE;
            ierr = PetscOptionsGetBool(NULL,NULL,"-BCperiodic",&BCperiodic,NULL); CHKERRQ(ierr);
            if(BCperiodic) { BCx=PeriodicBC; BCy=PeriodicBC; }
            set_from_options = PETSC_TRUE;
            return(0);
        }

        /** Returns the number of sites in the square lattice */
        PetscInt NumSites() const { return Lx*Ly; }

        /** Returns the number of sites in a single cluster/column of the environment block as suggested by Liang and
            Pang, 1994, for the square lattice */
        PetscInt NumEnvSites() const { return Ly; }

        /** Returns the object used to construct the Hamiltonian using sites counted from the left side of the lattice */
        std::vector< Term > H(
            const PetscInt& nsites  /**< Number of sites involved in the interaction */
            );

        /** Prints out some information to stdout */
        void PrintOut() const {
            printf( "HAMILTONIAN: J1J2XYModel_SquareLattice\n");
            printf( "  Lx  : %d\n", Lx);
            printf( "  Ly  : %d\n", Ly);
            printf( "  J1  : %g\n", J1);
            printf( "  Jz1 : %g\n", Jz1);
            printf( "  J2  : %g\n", J2);
            printf( "  Jz2 : %g\n", Jz2);
            printf( "  BCx : %s\n", BCx?"Periodic":"Open");
            printf( "  BCy : %s\n", BCy?"Periodic":"Open");
        }

        /** Writes out some JSON information to stdout */
        void SaveOut(FILE *fp) const {
            fprintf(fp, "  \"Hamiltonian\": {\n");
            fprintf(fp, "    \"label\":\"J1J2XYModel_SquareLattice\",\n");
            fprintf(fp, "    \"parameters\": {\n");
            fprintf(fp, "      \"Lx\"  : %d,\n", Lx);
            fprintf(fp, "      \"Ly\"  : %d,\n", Ly);
            fprintf(fp, "      \"J1\"  : %g,\n", J1);
            fprintf(fp, "      \"Jz1\" : %g,\n", Jz1);
            fprintf(fp, "      \"J2\"  : %g,\n", J2);
            fprintf(fp, "      \"Jz2\" : %g,\n", Jz2);
            fprintf(fp, "      \"BCx\" : \"%s\",\n", BCx?"Periodic":"Open");
            fprintf(fp, "      \"BCy\" : \"%s\"\n", BCy?"Periodic":"Open");
            fprintf(fp, "    }\n");
            fprintf(fp, "  }");
            fflush(fp);
        }

    private:

        /** Set from options */
        PetscBool set_from_options = PETSC_FALSE;

        /** Transverse anisotropy for nearest-neighbor interactions */
        PetscScalar Jz1 = 0.0;

        /** Transverse anisotropy for next-nearest-neighbor interactions */
        PetscScalar Jz2 = 0.0;

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

        /** Contains the Hamiltonian involving the entire lattice */
        std::vector< Term > H_full;

        /** Tells whether H_full has been constructed */
        PetscBool H_full_filled = PETSC_FALSE;
    };
}

/** @} */

#endif // __HAMILTONIANS
