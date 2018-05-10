#include "Hamiltonians.hpp"

/** Encodes the 2d coordinate on the square lattice to the 1d coordinate on the s-shaped snake */
#define SSNAKE_2D_1D(ix,jy,_Lx,_Ly) (((ix)*(_Ly)+jy)*(1-((ix)%2)) + ((ix+1)*(_Ly) - (jy+1))*((ix)%2))

PetscInt Hamiltonians::J1J2XXZModel_SquareLattice::To1D(
    const PetscInt ix,
    const PetscInt jy
    ) const
{
    return SSNAKE_2D_1D(ix,jy,_Lx,_Ly);
}

PetscErrorCode Hamiltonians::J1J2XXZModel_SquareLattice::To2D(
    const PetscInt idx,
    PetscInt& ix,
    PetscInt& jy
    ) const
{
    ix = idx / _Ly;
    const PetscInt t1 = ix % 2;
    jy = (idx % _Ly) * (1 - 2*t1) + (_Ly - 1)*t1;
    return(0);
}

std::vector<PetscInt> Hamiltonians::J1J2XXZModel_SquareLattice::GetNearestNeighbors(
    const PetscInt& ix, const PetscInt& jy, const PetscInt& nsites_in
    ) const
{
    std::vector<PetscInt> nn(0);
    /* Above */
    if(((0 <= jy) && (jy < (_Ly-1))) || ((jy == (_Ly-1)) && (_BCy == PeriodicBC)))
    {
        const PetscInt jy_above = (jy+1)%_Ly;
        const PetscInt nn1d = SSNAKE_2D_1D(ix,jy_above,_Lx,_Ly);
        if(nn1d < nsites_in && jy_above != jy) nn.push_back(nn1d);
    }
    /* Right */
    if(((0 <= ix) && (ix < (_Lx-1))) || ((ix == (_Lx-1)) && (_BCx == PeriodicBC)))
    {
        const PetscInt ix_right = (ix+1)%_Lx;
        const PetscInt nn1d = SSNAKE_2D_1D(ix_right,jy,_Lx,_Ly);
        if(nn1d < nsites_in && ix_right != ix) nn.push_back(nn1d);
    }
    return nn;
}

std::vector<PetscInt> Hamiltonians::J1J2XXZModel_SquareLattice::GetNextNearestNeighbors(
    const PetscInt& ix, const PetscInt& jy, const PetscInt& nsites_in
    ) const
{
    std::vector<PetscInt> nnn(0);
    /* Upper-left */
    if( (((1 <= ix) && (ix < _Lx  )) || ((ix == 0)      && (_BCx == PeriodicBC))) &&
        (((0 <= jy) && (jy < _Ly-1)) || ((jy == (_Ly-1)) && (_BCy == PeriodicBC))) )
    {
        const PetscInt nnn1d = SSNAKE_2D_1D((ix+_Lx-1)%_Lx,(jy+1)%_Ly,_Lx,_Ly);
        if(nnn1d < nsites_in) nnn.push_back(nnn1d);
    }
    /* Upper-right */
    if( (((0 <= ix) && (ix < _Lx-1)) || ((ix == (_Lx-1)) && (_BCx == PeriodicBC))) &&
        (((0 <= jy) && (jy < _Ly-1)) || ((jy == (_Ly-1)) && (_BCy == PeriodicBC))) )
    {
        const PetscInt nnn1d = SSNAKE_2D_1D((ix+1)%_Lx,(jy+1)%_Ly,_Lx,_Ly);
        if(nnn1d < nsites_in) nnn.push_back(nnn1d);
    }
    return nnn;
}

std::vector< Hamiltonians::Term > Hamiltonians::J1J2XXZModel_SquareLattice::H(const PetscInt& nsites_in)
{
    PetscInt ns = (nsites_in == PETSC_DEFAULT) ? _Lx*_Ly : nsites_in;
    PetscBool full_lattice = PetscBool(nsites_in == _Lx*_Ly);
    if(full_lattice && H_full_filled){
        return H_full;
    };
    std::vector< Hamiltonians::Term > Terms(0);
    Terms.reserve(ns*4*2); /* Assume the maximum when all sites have all 4 interactions and 2 terms each */
    for (PetscInt is = 0; is < ns; ++is)
    {
        const PetscInt ix = is / _Ly;
        const PetscInt jy = (is % _Ly)*(1 - 2 * (ix % 2)) + (_Ly - 1)*(ix % 2);
        /* Get nearest neighbors */
        if(_J1 != 0.0 || _Jz1 != 0.0)
        {
            const std::vector<PetscInt> nn = GetNearestNeighbors(ix,jy,ns);
            for(const PetscInt& in: nn)
            {
                /* Ensure that 1d block indices are ordered */
                PetscInt ia = (in < is) ? in : is;
                PetscInt ib = (in > is) ? in : is;
                /* Append the terms into the Hamiltonian */
                if(_J1 != 0.0) Terms.push_back({ _J1, OpSp, ia, OpSm, ib }); /* J1 S^+_ia S^-_ib */
                if(_J1 != 0.0) Terms.push_back({ _J1, OpSm, ia, OpSp, ib }); /* J1 S^-_ia S^+_ib */
                if(_Jz1 != 0.0) Terms.push_back({ _Jz1, OpSz, ia, OpSz, ib }); /* Jz1 S^z_ia S^z_ib */
            }
        }
        /* Get next-nearest neighbors */
        /* FIXME: Verify that on a one-dimensional chain, there are no next-nearest neighbors */
        /* When _J2==0 do not generate any terms */
        if((_J2 != 0.0 && _Jz2 != 0.0) && _Lx > 1 && _Ly > 1)
        {
            const std::vector<PetscInt> nnn = GetNextNearestNeighbors(ix,jy,ns);
            for(const PetscInt& in: nnn)
            {
                /* Ensure that 1d block indices are ordered */
                PetscInt il = (in < is) ? in : is;
                PetscInt ir = (in > is) ? in : is;
                /* Append the terms into the Hamiltonian */
                if(_J2 != 0.0) Terms.push_back({ _J2, OpSp, il, OpSm, ir }); /* J2 S^+_ia S^-_ib */
                if(_J2 != 0.0) Terms.push_back({ _J2, OpSm, il, OpSp, ir }); /* J2 S^-_ia S^+_ib */
                if(_Jz2 != 0.0) Terms.push_back({ _Jz2, OpSz, il, OpSz, ir }); /* Jz2 S^z_ia S^z_ib */
            }
        }
    }
    if(full_lattice && !H_full_filled){
        H_full = Terms;
        H_full_filled = PETSC_TRUE;
        return H_full;
    }
    return Terms;
}

std::vector< std::vector< PetscInt > > Hamiltonians::J1J2XXZModel_SquareLattice::NeighborPairs(
    const PetscInt d
    ) const
{
    if(d!=1) CPP_CHKERRQ_MSG(1, "Only d=1 supported.");
    std::vector< std::vector< PetscInt > > nnp;
    PetscInt ns = _Lx*_Ly;
    for (PetscInt is = 0; is < ns; ++is)
    {
        const PetscInt ix = is / _Ly;
        const PetscInt jy = (is % _Ly)*(1 - 2 * (ix % 2)) + (_Ly - 1)*(ix % 2);
        /* Get nearest neighbors */
        const std::vector<PetscInt> nn = GetNearestNeighbors(ix,jy,ns);
        for(const PetscInt& in: nn)
        {
            /* Ensure that 1d block indices are ordered */
            PetscInt ia = (in < is) ? in : is;
            PetscInt ib = (in > is) ? in : is;
            /* Append the pair */
            nnp.push_back({ia,ib});
        }
    }
    return nnp;
}
