#ifndef KERNELS_H
#define KERNELS_H

#ifdef CHEM
#include "LiDryer.H"
#endif
#include <IndexDefines.H>
#include <Constants.H>
#include <AMReX_FArrayBox.H>
#include <algorithm>
#include <cmath>

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
c2prim (int i, int j, int k,
             amrex::Array4<amrex::Real const> const& u,
             amrex::Array4<amrex::Real> const& q,
             Parm const& parm) noexcept
{
    using amrex::Real;

    Real rho = amrex::max(u(i,j,k,URHO),parm.smallr);
    Real rhoinv = Real(1.0)/rho;
    Real ux = u(i,j,k,UMX)*rhoinv;
    Real uy = u(i,j,k,UMY)*rhoinv;
    Real uz = u(i,j,k,UMZ)*rhoinv;
    Real kineng = Real(0.5)*(ux*ux+uy*uy+uz*uz);
    Real ei = u(i,j,k,UEDEN) - rho * kineng;
    Real p = amrex::max((parm.eos_gamma-Real(1.0))*ei,parm.smallp);
    Real T = p/(rho * parm.Rg);

    q(i,j,k,QRHO) = rho;
    q(i,j,k,QU) = ux;
    q(i,j,k,QV) = uy;
    q(i,j,k,QW) = uz;
    q(i,j,k,QPRES) = p;
    q(i,j,k,QT) = T;
    q(i,j,k,QC) = std::sqrt(parm.eos_gamma*parm.Rg*T);
}

#ifdef CHEM
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
c2prim_rgas (int i, int j, int k,
             amrex::Array4<amrex::Real const> const& u,
             amrex::Array4<amrex::Real const> const& spec,
             amrex::Array4<amrex::Real> const& q,
             Parm const& parm) noexcept
{
    using amrex::Real;

    Real rho = amrex::max(u(i,j,k,URHO),parm.smallr);
    Real rhoinv = Real(1.0)/rho;
    Real ux = u(i,j,k,UMX)*rhoinv;
    Real uy = u(i,j,k,UMY)*rhoinv;
    Real uz = u(i,j,k,UMZ)*rhoinv;
    Real kineng = Real(0.5)*(ux*ux+uy*uy+uz*uz);
    Real ei = u(i,j,k,UEDEN) - rho * kineng;
    Real T, p, rhoi[NSPECS], gama;

    q(i,j,k,QRHO) = rho;
    q(i,j,k,QU) = ux;
    q(i,j,k,QV) = uy;
    q(i,j,k,QW) = uz;
    
    for (int n=0; n<NSPECS; ++n) {
        rhoi[n] = spec(i,j,k,n);
    }
    GET_T_GIVEN_EY(ei, rhoi, T, parm);
    CKPY(rhoi, T, p, parm);
    q(i,j,k,QT) = T;
    q(i,j,k,QPRES) = p;

    // approximate
    gama = p/ei + 1.0;
    // CKGAMMA(T, rhoi, gama, parm);
    q(i,j,k,QGAMA) = gama;
    q(i,j,k,QC) = std::sqrt(gama*p/rho);
}
#endif

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
divop (int i, int j, int k, int n,
             amrex::Array4<amrex::Real> const& dudt,
             amrex::Array4<amrex::Real const> const& fx,
             amrex::Array4<amrex::Real const> const& fy,
             amrex::Array4<amrex::Real const> const& fz,
amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv) noexcept
{
    dudt(i,j,k,n) = dxinv[0] * (fx(i,j,k,n) - fx(i+1,j,k,n))
        +           dxinv[1] * (fy(i,j,k,n) - fy(i,j+1,k,n))
        +           dxinv[2] * (fz(i,j,k,n) - fz(i,j,k+1,n));
}

#endif