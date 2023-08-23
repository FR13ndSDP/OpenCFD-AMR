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

    Real x = prob_lo[0] + (i+Real(0.5))*dx[0];
    Real y = prob_lo[1] + (j+Real(0.5))*dx[1];

    Real x0 = Real(1.0)/Real(6.0);
    Real Pt, rhot, uxt, vxt;
    if (x < x0+std::sqrt(1/3.0)*y) {
        Pt = prob_parm.p_l;
        rhot = prob_parm.rho_l;
        uxt = prob_parm.u_l;
        vxt = prob_parm.v_l;
    } else {
        Pt = prob_parm.p_r;
        rhot = prob_parm.rho_r;
        uxt = prob_parm.u_r;
        vxt = prob_parm.v_r;
    }
    state(i,j,k,URHO ) = rhot;
    state(i,j,k,UMX  ) = rhot*uxt;
    state(i,j,k,UMY  ) = rhot*vxt;
    state(i,j,k,UMZ  ) = Real(0.0);
    Real et = Pt/(parm.eos_gamma-Real(1.0));
    state(i,j,k,UEDEN) = et + Real(0.5)*rhot*(uxt*uxt+vxt*vxt);
}

#endif