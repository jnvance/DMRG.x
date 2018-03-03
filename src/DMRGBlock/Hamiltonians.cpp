#include "Hamiltonians.hpp"

/** Encodes the 2d coordinate on the square lattice to the 1d coordinate on the s-shaped snake */
#define SSNAKE_2D_1D(ix,jy,Lx,Ly) (((ix)*(Ly)+jy)*(1-((ix)%2)) + ((ix+1)*(Ly) - (jy+1))*((ix)%2))

std::vector<PetscInt> Hamiltonians::J1J2XYModel_SquareLattice::GetNearestNeighbors(
    const PetscInt& ix, const PetscInt& jy, const PetscInt& nsites_in
    ) const
{
    std::vector<PetscInt> nn(0);
    /* Above */
    if(((0 <= jy) && (jy < (Ly-1))) || ((jy == (Ly-1)) && (BCy == PeriodicBC)))
    {
        const PetscInt jy_above = (jy+1)%Ly;
        const PetscInt nn1d = SSNAKE_2D_1D(ix,jy_above,Lx,Ly);
        if(nn1d < nsites_in && jy_above != jy) nn.push_back(nn1d);
    }
    /* Right */
    if(((0 <= ix) && (ix < (Lx-1))) || ((ix == (Lx-1)) && (BCx == PeriodicBC)))
    {
        const PetscInt ix_right = (ix+1)%Lx;
        const PetscInt nn1d = SSNAKE_2D_1D(ix_right,jy,Lx,Ly);
        if(nn1d < nsites_in && ix_right != ix) nn.push_back(nn1d);
    }
    return nn;
}

std::vector<PetscInt> Hamiltonians::J1J2XYModel_SquareLattice::GetNextNearestNeighbors(
    const PetscInt& ix, const PetscInt& jy, const PetscInt& nsites_in
    ) const
{
    std::vector<PetscInt> nnn(0);
    /* Upper-left */
    if( (((1 <= ix) && (ix < Lx  )) || ((ix == 0)      && (BCx == PeriodicBC))) &&
        (((0 <= jy) && (jy < Ly-1)) || ((jy == (Ly-1)) && (BCy == PeriodicBC))) )
    {
        const PetscInt nnn1d = SSNAKE_2D_1D((ix+Lx-1)%Lx,(jy+1)%Ly,Lx,Ly);
        if(nnn1d < nsites_in) nnn.push_back(nnn1d);
    }
    /* Upper-right */
    if( (((0 <= ix) && (ix < Lx-1)) || ((ix == (Lx-1)) && (BCx == PeriodicBC))) &&
        (((0 <= jy) && (jy < Ly-1)) || ((jy == (Ly-1)) && (BCy == PeriodicBC))) )
    {
        const PetscInt nnn1d = SSNAKE_2D_1D((ix+1)%Lx,(jy+1)%Ly,Lx,Ly);
        if(nnn1d < nsites_in) nnn.push_back(nnn1d);
    }
    return nnn;
}

std::vector< Hamiltonians::Term > Hamiltonians::J1J2XYModel_SquareLattice::H(const PetscInt& nsites_in) const
{
    PetscInt ns = (nsites_in == PETSC_DEFAULT) ? Lx*Ly : nsites_in;
    std::vector< Hamiltonians::Term > Terms(0);
    Terms.reserve(ns*4*2); /* Assume the maximum when all sites have all 4 interactions and 2 terms each */
    for (PetscInt is = 0; is < ns; ++is)
    {
        const PetscInt ix = is / Ly;
        const PetscInt jy = (is % Ly)*(1 - 2 * (ix % 2)) + (Ly - 1)*(ix % 2);
        /* Get nearest neighbors */
        if(J1 != 0.0 || Jz1 != 0.0)
        {
            const std::vector<PetscInt> nn = GetNearestNeighbors(ix,jy,ns);
            for(const PetscInt& in: nn)
            {
                /* Ensure that 1d block indices are ordered */
                PetscInt ia = (in < is) ? in : is;
                PetscInt ib = (in > is) ? in : is;
                /* Append the terms into the Hamiltonian */
                if(J1 != 0.0) Terms.push_back({ J1, OpSp, ia, OpSm, ib }); /* J1 S^+_ia S^-_ib */
                if(J1 != 0.0) Terms.push_back({ J1, OpSm, ia, OpSp, ib }); /* J1 S^-_ia S^+_ib */
                if(Jz1 != 0.0) Terms.push_back({ Jz1, OpSz, ia, OpSz, ib }); /* Jz1 S^z_ia S^z_ib */
            }
        }
        /* Get next-nearest neighbors */
        /* FIXME: Verify that on a one-dimensional chain, there are no next-nearest neighbors */
        /* When J2==0 do not generate any terms */
        if((J2 != 0.0 && Jz2 != 0.0) && Lx > 1 && Ly > 1)
        {
            const std::vector<PetscInt> nnn = GetNextNearestNeighbors(ix,jy,ns);
            for(const PetscInt& in: nnn)
            {
                /* Ensure that 1d block indices are ordered */
                PetscInt il = (in < is) ? in : is;
                PetscInt ir = (in > is) ? in : is;
                /* Append the terms into the Hamiltonian */
                if(J2 != 0.0) Terms.push_back({ J2, OpSp, il, OpSm, ir }); /* J2 S^+_ia S^-_ib */
                if(J2 != 0.0) Terms.push_back({ J2, OpSm, il, OpSp, ir }); /* J2 S^-_ia S^+_ib */
                if(Jz2 != 0.0) Terms.push_back({ Jz2, OpSz, il, OpSz, ir }); /* J2 S^+_ia S^-_ib */
            }
        }
    }
    return Terms;
}
