#ifndef __HAMILTONIANS__
#define __HAMILTONIANS__

#include "petscsys.h"
#include "DMRGBlock.hpp"
#include <vector>
#include <iostream>

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

    /** Implements the Hamiltonian for the J1-J2 XY model on the square lattice

    <table>
    <caption id="multi_row">Options Database</caption>
    <tr>
        <th style="min-width:260px!important" > Command line argument
        <th> Description
    <tr>
        <td>`-Lx [int]`
        <td>lattice dimension in the longitudinal direction (growing) [def: 4]
    <tr>
        <td>`-Ly [int]`
        <td>lattice dimension in the transverse direction [def: 4]
    <tr>
        <td>`-J1 [float]`
        <td>coupling constant for the nearest neighbor interaction
    <tr>
        <td>`-Jz1 [float]`
        <td>anisotropy in the z-direction for the nearest neighbor interaction
    <tr>
        <td>`-J2 [float]`
        <td>coupling constant for the next-nearest neighbor interaction
    <tr>
        <td>`-Jz2 [float]`
        <td>anisotropy in the z-direction for the next-nearest neighbor interaction
    <tr>
        <td>`-heisenberg [float]`
        <td>Overrides other coupling constants and reverts to an XXZ Heisenberg model where the given
            parameter is the coupling constant along the z-direction
    <tr>
        <td>`-BCopen`
        <td rowspan=2>Either for open boundary conditions or for periodic (toroidal) boundary conditions.
            Only at most one of them can be true. If none of them are set, the default cylindrical
            boundary conditions are used.
    <tr>
        <td>`-BCperiodic`
    </table>

    @sa
    Options regarding the DMRG runtime can be found at the DMRGBlockContainer documentation.

     */
    class J1J2XXZModel_SquareLattice
    {

    public:

        /** Constructor */
        J1J2XXZModel_SquareLattice()
        {

        }

        /** Set attributes from command line arguments */
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

            /* One may also specify whether to do the NN Heisenberg model and specify only the anisotropy */
            ierr = PetscOptionsGetReal(NULL,NULL,"-heisenberg",&_Jz1,&heisenberg); CHKERRQ(ierr);
            if(heisenberg)
            {
                _J1 = 0.50;
                _J2 = 0.0;
                _Jz2= 0.0;
            }

            PetscBool BCopen = PETSC_FALSE;
            ierr = PetscOptionsGetBool(NULL,NULL,"-BCopen",&BCopen,NULL); CHKERRQ(ierr);
            if(BCopen) { _BCx=OpenBC; _BCy=OpenBC; }
            PetscBool BCperiodic = PETSC_FALSE;
            ierr = PetscOptionsGetBool(NULL,NULL,"-BCperiodic",&BCperiodic,NULL); CHKERRQ(ierr);
            if(BCperiodic) { _BCx=PeriodicBC; _BCy=PeriodicBC; }
            set_from_options = PETSC_TRUE;
            return(0);
        }

        /** Dumps options keys and values to file.
            @note Not collective, must be called only on one process. */
        PetscErrorCode SaveAsOptions(const std::string& filename)
        {
            PetscErrorCode ierr;
            char val[4096];
            PetscBool set;
            std::vector< std::string > keys = {
                "-J1",
                "-J2",
                "-Jz1",
                "-Jz2",
                "-Lx",
                "-Ly",
                "-heisenberg",
                "-BCopen",
                "-BCperiodic"
            };

            std::ostringstream oss;
            for(std::string& key: keys) {
                ierr = PetscOptionsGetString(NULL,NULL,key.c_str(),val,4096,&set); CHKERRQ(ierr);
                if(set) {
                    std::string val_str(val);
                    if(val_str.empty()) val_str = "yes";
                    oss << key << " " << val_str << std::endl;
                }
            }

            FILE *fp;
            ierr = PetscFOpen(PETSC_COMM_SELF,filename.c_str(),"w", &fp); CHKERRQ(ierr);
            ierr = PetscFPrintf(PETSC_COMM_SELF,fp,"%s",oss.str().c_str()); CHKERRQ(ierr);
            ierr = PetscFClose(PETSC_COMM_SELF,fp); CHKERRQ(ierr);

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
            if(heisenberg)
            {
                printf( "HAMILTONIAN: HeisenbergModel_SquareLattice\n");
            }
            else
            {
                printf( "HAMILTONIAN: J1J2XXZModel_SquareLattice\n");
            }
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
            if(heisenberg)
            {
                fprintf(fp, "    \"label\":\"HeisenbergModel_SquareLattice\",\n");
            }
            else
            {
                fprintf(fp, "    \"label\":\"J1J2XXZModel_SquareLattice\",\n");
            }
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

        /** Whether to use the Heisenberg model */
        PetscBool heisenberg = PETSC_FALSE;

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

#endif // __HAMILTONIANS__
