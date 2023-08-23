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

    Real Xt[NSPECS] = {0.0}, Yt[NSPECS];
    Xt[0] = 0.005;
    Xt[1] = 0.005;
    Xt[8] = 0.99;

    CKXTY(Xt, Yt);

    CKRHOY(Pt, temp, Yt, rho);

    CKUBMS(temp, Yt, et);

    state(i,j,k,URHO ) = rho;
    state(i,j,k,UMX  ) = Real(0.0);
    state(i,j,k,UMY  ) = Real(0.0);
    state(i,j,k,UMZ  ) = Real(0.0);
    state(i,j,k,UEDEN) = et*rho;
    state(i,j,k,UTemp) = temp;
}

#endif