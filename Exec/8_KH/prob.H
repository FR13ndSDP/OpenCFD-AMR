#ifndef EBR_PROB_H_
#define EBR_PROB_H_

#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include "IndexDefines.H"
#include "Constants.H"
#include "prob_parm.H"
#include <AMReX_EBCellFlag.H>

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

    Real y = prob_lo[1] + (j+Real(0.5))*dx[1];
    Real Pt, rhot, ux, uy;
    if (amrex::Math::abs(y) < Real(0.25)) {
        Pt = prob_parm.p;
        rhot = prob_parm.rho_in;
        ux = 0.5 + (amrex::Random() - 0.5)/50;
        uy = (amrex::Random() - 0.5) / 50;
    } else {
        Pt = prob_parm.p;
        rhot = prob_parm.rho_out;
        ux = -0.5 + (amrex::Random() - 0.5)/50;
        uy = (amrex::Random() - 0.5)/50;
    }
    state(i,j,k,URHO ) = rhot;
    state(i,j,k,UMX  ) = rhot * ux;
    state(i,j,k,UMY  ) = rhot * uy;
    state(i,j,k,UMZ  ) = Real(0.0);
    Real et = Pt/(parm.eos_gamma-Real(1.0)) + 0.5*rhot*(ux*ux+uy*uy);
    state(i,j,k,UEDEN) = et;
}

#endif