#ifndef DIFFUSION_H
#define DIFFUSION_H

#include <Constants.H>
#include <IndexDefines.H>

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
compute_visc_x (int i, int j, int k,
             amrex::Array4<amrex::Real const> const& q,
             amrex::Array4<amrex::Real> const& fx,
amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv,
             Parm const& parm) noexcept
{
    using amrex::Real;
    // first compute temperature
    Real Tr = q(i,j,k,QT);
    Real Tl = q(i-1,j,k,QT);
    // derive mu
    Real mutl = parm.C_s*Tl*sqrt(Tl)/(Tl+parm.T_s);
    Real mutr = parm.C_s*Tr*sqrt(Tr)/(Tr+parm.T_s);
    // derive k
    Real ktl = mutl * parm.kOverMu;
    Real ktr = mutr * parm.kOverMu;

    Real dTdx = (Tr-Tl)*dxinv[0];
    Real dudx = (q(i,j,k,QU)-q(i-1,j,k,QU))*dxinv[0];
    Real dvdx = (q(i,j,k,QV)-q(i-1,j,k,QV))*dxinv[0];
    Real dwdx = (q(i,j,k,QW)-q(i-1,j,k,QW))*dxinv[0];
    Real dudy = (q(i,j+1,k,QU)+q(i-1,j+1,k,QU)-q(i,j-1,k,QU)-q(i-1,j-1,k,QU))*(Real(0.25)*dxinv[1]);
    Real dvdy = (q(i,j+1,k,QV)+q(i-1,j+1,k,QV)-q(i,j-1,k,QV)-q(i-1,j-1,k,QV))*(Real(0.25)*dxinv[1]);
    Real dudz = (q(i,j,k+1,QU)+q(i-1,j,k+1,QU)-q(i,j,k-1,QU)-q(i-1,j,k-1,QU))*(Real(0.25)*dxinv[2]);
    Real dwdz = (q(i,j,k+1,QW)+q(i-1,j,k+1,QW)-q(i,j,k-1,QW)-q(i-1,j,k-1,QW))*(Real(0.25)*dxinv[2]);
    Real divu = dudx + dvdy + dwdz;
    Real muf = Real(0.5)*(mutl+mutr);
    Real tauxx = muf*(Real(2.0)*dudx-parm.twoThirds*divu);
    Real tauxy = muf*(dudy+dvdx);
    Real tauxz = muf*(dudz+dwdx);
    fx(i,j,k,UMX)   += -tauxx;
    fx(i,j,k,UMY)   += -tauxy;
    fx(i,j,k,UMZ)   += -tauxz;
    fx(i,j,k,UEDEN) += -Real(0.5)*((q(i,j,k,QU)+q(i-1,j,k,QU))*tauxx \
                                  +(q(i,j,k,QV)+q(i-1,j,k,QV))*tauxy \
                                  +(q(i,j,k,QW)+q(i-1,j,k,QW))*tauxz \
                                  +(ktl+ktr)*dTdx);
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
compute_visc_y (int i, int j, int k,
             amrex::Array4<amrex::Real const> const& q,
             amrex::Array4<amrex::Real> const& fy,
amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv,
             Parm const& parm) noexcept
{
    using amrex::Real;
    // first compute temperature
    Real Tr = q(i,j,k,QT);
    Real Tl = q(i,j-1,k,QT);
    // derive mu
    Real mutl = parm.C_s*Tl*sqrt(Tl)/(Tl+parm.T_s);
    Real mutr = parm.C_s*Tr*sqrt(Tr)/(Tr+parm.T_s);
    // derive k
    Real ktl = mutl * parm.kOverMu;
    Real ktr = mutr * parm.kOverMu;

    Real dTdy = (Tr-Tl)*dxinv[1];
    Real dudy = (q(i,j,k,QU)-q(i,j-1,k,QU))*dxinv[1];
    Real dvdy = (q(i,j,k,QV)-q(i,j-1,k,QV))*dxinv[1];
    Real dwdy = (q(i,j,k,QW)-q(i,j-1,k,QW))*dxinv[1];
    Real dudx = (q(i+1,j,k,QU)+q(i+1,j-1,k,QU)-q(i-1,j,k,QU)-q(i-1,j-1,k,QU))*(Real(0.25)*dxinv[0]);
    Real dvdx = (q(i+1,j,k,QV)+q(i+1,j-1,k,QV)-q(i-1,j,k,QV)-q(i-1,j-1,k,QV))*(Real(0.25)*dxinv[0]);
    Real dvdz = (q(i,j,k+1,QV)+q(i,j-1,k+1,QV)-q(i,j,k-1,QV)-q(i,j-1,k-1,QV))*(Real(0.25)*dxinv[2]);
    Real dwdz = (q(i,j,k+1,QW)+q(i,j-1,k+1,QW)-q(i,j,k-1,QW)-q(i,j-1,k-1,QW))*(Real(0.25)*dxinv[2]);
    Real divu = dudx + dvdy + dwdz;
    Real muf = Real(0.5)*(mutl+mutr);
    Real tauyy = muf*(Real(2.0)*dvdy-parm.twoThirds*divu);
    Real tauxy = muf*(dudy+dvdx);
    Real tauyz = muf*(dwdy+dvdz);
    fy(i,j,k,UMX)   += -tauxy;
    fy(i,j,k,UMY)   += -tauyy;
    fy(i,j,k,UMZ)   += -tauyz;
    fy(i,j,k,UEDEN) += -Real(0.5)*((q(i,j,k,QU)+q(i,j-1,k,QU))*tauxy \
                                  +(q(i,j,k,QV)+q(i,j-1,k,QV))*tauyy \
                                  +(q(i,j,k,QW)+q(i,j-1,k,QW))*tauyz \
                                  +(ktl+ktr)*dTdy);
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
compute_visc_z (int i, int j, int k,
             amrex::Array4<amrex::Real const> const& q,
             amrex::Array4<amrex::Real> const& fz,
amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv,
             Parm const& parm) noexcept
{
    using amrex::Real;
    // first compute temperature
    Real Tr = q(i,j,k,QT);
    Real Tl = q(i,j,k-1,QT);
    // derive mu
    Real mutl = parm.C_s*Tl*sqrt(Tl)/(Tl+parm.T_s);
    Real mutr = parm.C_s*Tr*sqrt(Tr)/(Tr+parm.T_s);
    // derive k
    Real ktl = mutl * parm.kOverMu;
    Real ktr = mutr * parm.kOverMu;

    Real dTdz = (Tr-Tl)*dxinv[2];
    Real dudz = (q(i,j,k,QU)-q(i,j,k-1,QU))*dxinv[2];
    Real dvdz = (q(i,j,k,QV)-q(i,j,k-1,QV))*dxinv[2];
    Real dwdz = (q(i,j,k,QW)-q(i,j,k-1,QW))*dxinv[2];
    Real dudx = (q(i+1,j,k,QU)+q(i+1,j,k-1,QU)-q(i-1,j,k,QU)-q(i-1,j,k-1,QU))*(Real(0.25)*dxinv[0]);
    Real dwdx = (q(i+1,j,k,QW)+q(i+1,j,k-1,QW)-q(i-1,j,k,QW)-q(i-1,j,k-1,QW))*(Real(0.25)*dxinv[0]);
    Real dvdy = (q(i,j+1,k,QV)+q(i,j+1,k-1,QV)-q(i,j-1,k,QV)-q(i,j-1,k-1,QV))*(Real(0.25)*dxinv[1]);
    Real dwdy = (q(i,j+1,k,QW)+q(i,j+1,k-1,QW)-q(i,j-1,k,QW)-q(i,j-1,k-1,QW))*(Real(0.25)*dxinv[1]);
    Real divu = dudx + dvdy + dwdz;
    Real muf = Real(0.5)*(mutl+mutr);
    Real tauxz = muf*(dudz+dwdx);
    Real tauyz = muf*(dvdz+dwdy);
    Real tauzz = muf*(Real(2.0)*dwdz-parm.twoThirds*divu);
    fz(i,j,k,UMX)   += -tauxz;
    fz(i,j,k,UMY)   += -tauyz;
    fz(i,j,k,UMZ)   += -tauzz;
    fz(i,j,k,UEDEN) += -Real(0.5)*((q(i,j,k,QU)+q(i,j,k-1,QU))*tauxz \
                                  +(q(i,j,k,QV)+q(i,j,k-1,QV))*tauyz \
                                  +(q(i,j,k,QW)+q(i,j,k-1,QW))*tauzz \
                                  +(ktl+ktr)*dTdz);
}

#endif