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
            ierr = PetscOptionsGetReal(NULL,NULL,"-J1",&_J1,NULL); CHKERRQ(ierr);
            ierr = PetscOptionsGetReal(NULL,NULL,"-J2",&_J2,NULL); CHKERRQ(ierr);
            ierr = PetscOptionsGetReal(NULL,NULL,"-Jz1",&_Jz1,NULL); CHKERRQ(ierr);
            ierr = PetscOptionsGetReal(NULL,NULL,"-Jz2",&_Jz2,NULL); CHKERRQ(ierr);
            ierr = PetscOptionsGetInt(NULL,NULL,"-Lx",&_Lx,NULL); CHKERRQ(ierr);
            ierr = PetscOptionsGetInt(NULL,NULL,"-Ly",&_Ly,NULL); CHKERRQ(ierr);
            ierr = PetscOptionsGetBool(NULL,NULL,"-verbose",&verbose,NULL); CHKERRQ(ierr);
            /* TODO: Also get boundary conditions from command line */
            PetscBool BCopen = PETSC_FALSE;
            ierr = PetscOptionsGetBool(NULL,NULL,"-BCopen",&BCopen,NULL); CHKERRQ(ierr);
            if(BCopen) { _BCx=OpenBC; _BCy=OpenBC; }
            PetscBool BCperiodic = PETSC_FALSE;
            ierr = PetscOptionsGetBool(NULL,NULL,"-BCperiodic",&BCperiodic,NULL); CHKERRQ(ierr);
            if(BCperiodic) { _BCx=PeriodicBC; _BCy=PeriodicBC; }
            set_from_options = PETSC_TRUE;
            return(0);
        }

        /** Returns the number of sites in the square lattice */
        PetscInt NumSites() const { return _Lx*_Ly; }

        /** Returns the number of sites in a single cluster/column of the environment block as suggested by Liang and
            Pang, 1994, for the square lattice */
        PetscInt NumEnvSites() const { return _Ly; }

        /** Returns the object used to construct the Hamiltonian using sites counted from the left side of the lattice */
        std::vector< Term > H(
            const PetscInt& nsites  /**< Number of sites involved in the interaction */
            );

        /** Prints out some information to stdout */
        void PrintOut() const {
            printf( "HAMILTONIAN: J1J2XYModel_SquareLattice\n");
            printf( "  Lx  : %lld\n", LLD(_Lx));
            printf( "  Ly  : %lld\n", LLD(_Ly));
            printf( "  J1  : %g\n", _J1);
            printf( "  Jz1 : %g\n", _Jz1);
            printf( "  J2  : %g\n", _J2);
            printf( "  Jz2 : %g\n", _Jz2);
            printf( "  BCx : %s\n", _BCx?"Periodic":"Open");
            printf( "  BCy : %s\n", _BCy?"Periodic":"Open");
        }

        /** Writes out some JSON information to stdout */
        void SaveOut(FILE *fp) const {
            fprintf(fp, "  \"Hamiltonian\": {\n");
            fprintf(fp, "    \"label\":\"J1J2XYModel_SquareLattice\",\n");
            fprintf(fp, "    \"parameters\": {\n");
            fprintf(fp, "      \"Lx\"  : %lld,\n", LLD(_Lx));
            fprintf(fp, "      \"Ly\"  : %lld,\n", LLD(_Ly));
            fprintf(fp, "      \"J1\"  : %g,\n", _J1);
            fprintf(fp, "      \"Jz1\" : %g,\n", _Jz1);
            fprintf(fp, "      \"J2\"  : %g,\n", _J2);
            fprintf(fp, "      \"Jz2\" : %g,\n", _Jz2);
            fprintf(fp, "      \"BCx\" : \"%s\",\n", _BCx?"Periodic":"Open");
            fprintf(fp, "      \"BCy\" : \"%s\"\n", _BCy?"Periodic":"Open");
            fprintf(fp, "    }\n");
            fprintf(fp, "  }");
            fflush(fp);
        }

        PetscInt Lx() const { return _Lx; }

        PetscInt Ly() const { return _Ly; }

        PetscInt To1D(
            const PetscInt ix,
            const PetscInt jy
            ) const;

        PetscErrorCode To2D(
            const PetscInt idx,
            PetscInt& ix,
            PetscInt& jy
            ) const;

        /** Returns all pairs of nearest-neighbors in the full square lattice */
        std::vector< std::vector< PetscInt > > NeighborPairs(const PetscInt d=1) const;

    private:

        /** Set from options */
        PetscBool set_from_options = PETSC_FALSE;

        /** Transverse anisotropy for nearest-neighbor interactions */
        PetscScalar _Jz1 = 0.0;

        /** Transverse anisotropy for next-nearest-neighbor interactions */
        PetscScalar _Jz2 = 0.0;

        /** Coupling strength for nearest-neighbor interactions */
        PetscScalar _J1 = 1.0;

        /** Coupling strength for next-nearest-neighbor interactions */
        PetscScalar _J2 = 1.0;

        /** Length along (growing) longitudinal direction */
        PetscInt _Lx = 4;

        /** Length along (fixed) transverse direction */
        PetscInt _Ly = 4;

        /** Flag that tells whether to print some information */
        PetscBool verbose = PETSC_FALSE;

        /** Boundary conditions for the longitudinal direction (defaults to cylindrical BC) */
        BC_t _BCx = OpenBC;

        /** Boundary conditions for the transverse direction (defaults to cylindrical BC) */
        BC_t _BCy = PeriodicBC;

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
