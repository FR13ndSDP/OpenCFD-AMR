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
              Parm const& parm, ProbParm const& prob_parm) noexcept
{
    using amrex::Real;

    Real Pt, temp, ux, rho, et;

    const Real* prob_lo = geomdata.ProbLo();
    const Real* dx      = geomdata.CellSize();
    Real Xt[NSPECS] = {0.0}, Yt[NSPECS], rhoi[NSPECS];

    Real y = prob_lo[1] + (j+Real(0.5))*dx[1];
    Real z = prob_lo[2] + (k+Real(0.5))*dx[2];
    Real r = std::sqrt(y*y+z*z);

    if (r <= 0.0003) {
        Xt[0] = 0.15;
        Xt[8] = 0.85;
        ux = prob_parm.u_jet;
        temp = prob_parm.T_jet;
        Pt = prob_parm.p_jet;
    } else {
        Xt[1] = 0.21;
        Xt[8] = 0.79;
        ux = prob_parm.u_ambient;
        temp = prob_parm.T_ambient;
        Pt = prob_parm.p_ambient;
    }

    CKXTY(Xt, Yt, parm);

    CKRHOY(Pt, temp, Yt, rho, parm);

    for (int n=0; n<NSPECS; ++n) {
        rhoi[n] = rho * Yt[n];
    }

    CKUBMS(temp, rhoi, et, parm);

    state(i,j,k,URHO ) = rho;
    state(i,j,k,UMX  ) = rho * ux;
    state(i,j,k,UMY  ) = Real(0.0);
    state(i,j,k,UMZ  ) = Real(0.0);
    state(i,j,k,UEDEN) = et + 0.5 * rho * ux * ux;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
ebr_initspec (int i, int j, int k, 
              amrex::Array4<amrex::Real const> const& state,
              amrex::Array4<amrex::Real> const& spec,
              amrex::GeometryData const& geomdata,
              Parm const& parm) noexcept
{
    using amrex::Real;

    Real rho;
    const Real* prob_lo = geomdata.ProbLo();
    const Real* dx      = geomdata.CellSize();
    Real Xt[NSPECS] = {0.0}, Yt[NSPECS], rhoi[NSPECS];

    Real y = prob_lo[1] + (j+Real(0.5))*dx[1];
    Real z = prob_lo[2] + (k+Real(0.5))*dx[2];
    Real r = std::sqrt(y*y+z*z);

    if (r <= 0.0003) {
        Xt[0] = 0.15;
        Xt[8] = 0.85;
    } else {
        Xt[1] = 0.21;
        Xt[8] = 0.79;
    }

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