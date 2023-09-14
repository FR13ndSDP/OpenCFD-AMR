#ifndef EBR_PROB_H_
#define EBR_PROB_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include "IndexDefines.H"
#include "Constants.H"
#include "prob_parm.H"
#include <AMReX_EBCellFlag.H>
#include "LiDryer.H"

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
ebr_initdata (int i, int j, int k, amrex::Array4<amrex::Real> const& state,
              amrex::GeometryData const& geomdata, amrex::Array4<amrex::Real const> const& /*vfrac*/,
              amrex::Array4<amrex::EBCellFlag const> const& /*flag*/,
              Parm const& parm, ProbParm const& prob_parm)
{
    using amrex::Real;

    Real Pt = prob_parm.p;
    Real temp = prob_parm.T;
    Real rho, et;

    const Real* prob_lo = geomdata.ProbLo();
    const Real* dx      = geomdata.CellSize();
    Real Xt[NSPECS] = {0.0}, Yt[NSPECS], rhoi[NSPECS];
    Xt[0] = 0.667;
    Xt[1] = 0.333;

    Real x = prob_lo[0] + (i+Real(0.5))*dx[0];
    if (x < 0.0004) {
        Pt = prob_parm.p0;
        temp = prob_parm.T0;
    }

    CKXTY(Xt, Yt, parm);

    CKRHOY(Pt, temp, Yt, rho, parm);

    for (int n=0; n<NSPECS; ++n) {
        rhoi[n] = rho * Yt[n];
    }

    CKUBMS(temp, rhoi, et, parm);

    state(i,j,k,URHO ) = rho;
    state(i,j,k,UMX  ) = Real(0.0);
    state(i,j,k,UMY  ) = Real(0.0);
    state(i,j,k,UMZ  ) = Real(0.0);
    state(i,j,k,UEDEN) = et;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
ebr_initspec (int i, int j, int k, 
              amrex::Array4<amrex::Real const> const& state,
              amrex::Array4<amrex::Real> const& spec,
              amrex::GeometryData const& /*geomdata*/,
              Parm const& parm)
{
    using amrex::Real;

    Real rho;
    // const Real* prob_lo = geomdata.ProbLo();
    // const Real* dx      = geomdata.CellSize();
    Real Xt[NSPECS] = {0.0}, Yt[NSPECS], rhoi[NSPECS];
    Xt[0] = 0.667;
    Xt[1] = 0.333;

    CKXTY(Xt, Yt, parm);

    rho = state(i,j,k,URHO);
    for (int n=0; n<NSPECS; ++n) {
        rhoi[n] = rho * Yt[n];
    }

    for (int n=0; n<NSPECS; ++n) {
        spec(i,j,k,n) = rhoi[n];
    }
}

#endif