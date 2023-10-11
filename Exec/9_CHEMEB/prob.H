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
    const Real* prob_lo = geomdata.ProbLo();
    const Real* dx      = geomdata.CellSize();
    Real Xt[NSPECS] = {0.0}, Yt[NSPECS], rhoi[NSPECS];

    Real x = prob_lo[0] + (i+Real(0.5))*dx[0];
    Real y = prob_lo[1] + (j+Real(0.5))*dx[1];
    Real z = prob_lo[2] + (k+Real(0.5))*dx[2];
    Real r = std::sqrt(x*x+y*y+z*z);

    Real Pt, rhot;
    if (r < Real(0.25)) {
        Xt[0] = 1;
        Pt = prob_parm.p_l;
        rhot = prob_parm.rho_l;
    } else {
        Xt[1] = 1;
        Pt = prob_parm.p_r;
        rhot = prob_parm.rho_r;
    }
    CKXTY(Xt, Yt, parm);
    Real YOW = 0;
    amrex::GpuArray<amrex::Real, NSPECS> tmp;

    for (int i = 0; i < NSPECS; ++i) {
        tmp[i] = Yt[i] / parm.mw[i];
    }
    for (int i = 0; i<NSPECS; ++i) {
        YOW += tmp[i];
    }
    for (int n=0; n<NSPECS; ++n) {
        rhoi[n] = rhot * Yt[n];
    }

    Real T, et;
    T = Pt / (parm.Ru * rhot * YOW);
    CKUBMS(T, rhoi, et, parm);

    state(i,j,k,URHO ) = rhot;
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
              amrex::GeometryData const& geomdata,
              Parm const& parm) noexcept
{
    using amrex::Real;

    Real rho;
    const Real* prob_lo = geomdata.ProbLo();
    const Real* dx      = geomdata.CellSize();
    Real Xt[NSPECS] = {0.0}, Yt[NSPECS], rhoi[NSPECS];

    Real x = prob_lo[0] + (i+Real(0.5))*dx[0];
    Real y = prob_lo[1] + (j+Real(0.5))*dx[1];
    Real z = prob_lo[2] + (k+Real(0.5))*dx[2];
    Real r = std::sqrt(x*x+y*y+z*z);

    if (r < Real(0.25)) {
        Xt[0] = 1;
    } else {
        Xt[1] = 1;
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