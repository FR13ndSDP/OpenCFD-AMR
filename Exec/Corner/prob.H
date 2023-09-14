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
              amrex::Array4<amrex::EBCellFlag const> const& flag,
              Parm const& parm, ProbParm const& prob_parm)
{
    using amrex::Real;

    Real Pt, rhot, uxt;
    Pt = prob_parm.p_l;
    rhot = prob_parm.rho_l;
    uxt = prob_parm.u_l;
    if (flag(i,j,k).isCovered()) {
        state(i,j,k,URHO ) = rhot;
        state(i,j,k,UMX  ) = Real(0.0);
        state(i,j,k,UMY  ) = Real(0.0);
        state(i,j,k,UMZ  ) = Real(0.0);
        Real et = Pt/(parm.eos_gamma-Real(1.0));
        state(i,j,k,UEDEN) = et;
    } else {
        state(i,j,k,URHO ) = rhot;
        state(i,j,k,UMX  ) = rhot*uxt;
        state(i,j,k,UMY  ) = Real(0.0);
        state(i,j,k,UMZ  ) = Real(0.0);
        Real et = Pt/(parm.eos_gamma-Real(1.0));
        state(i,j,k,UEDEN) = et + Real(0.5)*rhot*uxt*uxt;
    }
}

#endif