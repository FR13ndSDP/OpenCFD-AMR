#ifndef EBDIFFUSION_H
#define EBDIFFUSION_H

#include <Constants.H>
#include <IndexDefines.H>
#include <AMReX_EBCellFlag.H>

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
eb_compute_visc_x (int i, int j, int k,
             amrex::Array4<amrex::Real const> const& q,
             amrex::Array4<amrex::Real> const& fx,
             amrex::Array4<amrex::EBCellFlag const> const& flag,
amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv,
             amrex::GpuArray<amrex::Real,3> const& weights,
             Parm const& parm) noexcept
{
    using amrex::Real;
    // first compute temperature
    Real Tl = q(i,j,k,QT);
    Real Tr = q(i-1,j,k,QT);
    // derive mu
    Real mutl = parm.C_s*Tl*sqrt(Tl)/(Tl+parm.T_s);
    Real mutr = parm.C_s*Tr*sqrt(Tr)/(Tr+parm.T_s);
    Real muf = Real(0.5)*(mutl+mutr);
    // derive k
    Real ktl = mutl * parm.kOverMu;
    Real ktr = mutr * parm.kOverMu;

    Real  dTdx = (Tl-Tr)*dxinv[0];
    Real  dudx = (q(i,j,k,QU)-q(i-1,j,k,QU))*dxinv[0];
    Real  dvdx = (q(i,j,k,QV)-q(i-1,j,k,QV))*dxinv[0];
    Real  dwdx = (q(i,j,k,QW)-q(i-1,j,k,QW))*dxinv[0];

    int  jhip = j + (flag(i  ,j,k).isConnected(0, 1,0) ? 1 : 0 );
    int  jhim = j - (flag(i  ,j,k).isConnected(0,-1,0) ? 1 : 0 );
    int  jlop = j + (flag(i-1,j,k).isConnected(0, 1,0) ? 1 : 0 );
    int  jlom = j - (flag(i-1,j,k).isConnected(0,-1,0) ? 1 : 0 );

    Real  whi = weights[jhip-jhim];
    Real  wlo = weights[jlop-jlom];
    Real  dudy = (0.5*dxinv[1]) * ((q(i  ,jhip,k,QU)-q(i  ,jhim,k,QU))*whi
                                  +(q(i-1,jlop,k,QU)-q(i-1,jlom,k,QU))*wlo);
    Real  dvdy = (0.5*dxinv[1]) * ((q(i  ,jhip,k,QV)-q(i  ,jhim,k,QV))*whi
                                  +(q(i-1,jlop,k,QV)-q(i-1,jlom,k,QV))*wlo);

    int  khip = k + (flag(i  ,j,k).isConnected(0,0, 1) ? 1 : 0 );
    int  khim = k - (flag(i  ,j,k).isConnected(0,0,-1) ? 1 : 0 );
    int  klop = k + (flag(i-1,j,k).isConnected(0,0, 1) ? 1 : 0 );
    int  klom = k - (flag(i-1,j,k).isConnected(0,0,-1) ? 1 : 0 );

    whi = weights[khip-khim];
    wlo = weights[klop-klom];
    Real  dudz = (0.5*dxinv[2]) * ((q(i  ,j,khip,QU)-q(i  ,j,khim,QU))*whi
                                  +(q(i-1,j,klop,QU)-q(i-1,j,klom,QU))*wlo);
    Real  dwdz = (0.5*dxinv[2]) * ((q(i  ,j,khip,QW)-q(i  ,j,khim,QW))*whi
                                  +(q(i-1,j,klop,QW)-q(i-1,j,klom,QW))*wlo);
    Real  divu = dudx + dvdy + dwdz;
    Real  tauxx = muf*(2.0*dudx-parm.twoThirds*divu);
    Real  tauxy = muf*(dudy+dvdx);
    Real  tauxz = muf*(dudz+dwdx);

    fx(i,j,k,UMX)   += -tauxx;
    fx(i,j,k,UMY)   += -tauxy;
    fx(i,j,k,UMZ)   += -tauxz;
    fx(i,j,k,UEDEN) += -0.5*( (q(i,j,k,QU)+q(i-1,j,k,QU))*tauxx+
                              (q(i,j,k,QV)+q(i-1,j,k,QV))*tauxy+
                              (q(i,j,k,QW)+q(i-1,j,k,QW))*tauxz+
                              (ktl+ktr)*dTdx);
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
eb_compute_visc_y (int i, int j, int k,
             amrex::Array4<amrex::Real const> const& q,
             amrex::Array4<amrex::Real> const& fy,
             amrex::Array4<amrex::EBCellFlag const> const& flag,
amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv,
             amrex::GpuArray<amrex::Real,3> const& weights,
             Parm const& parm) noexcept
{
    using amrex::Real;
    // first compute temperature
    Real Tl = q(i,j,k,QT);
    Real Tr = q(i,j-1,k,QT);
    // derive mu
    Real mutl = parm.C_s*Tl*sqrt(Tl)/(Tl+parm.T_s);
    Real mutr = parm.C_s*Tr*sqrt(Tr)/(Tr+parm.T_s);
    Real muf = Real(0.5)*(mutl+mutr);
    // derive k
    Real ktl = mutl * parm.kOverMu;
    Real ktr = mutr * parm.kOverMu;

    Real  dTdy = (Tl-Tr)*dxinv[1];
    Real  dudy = (q(i,j,k,QU)-q(i,j-1,k,QU))*dxinv[1];
    Real  dvdy = (q(i,j,k,QV)-q(i,j-1,k,QV))*dxinv[1];
    Real  dwdy = (q(i,j,k,QW)-q(i,j-1,k,QW))*dxinv[1];

    int  ihip = i + (flag(i,j  ,k).isConnected( 1,0,0) ? 1 : 0 );
    int  ihim = i - (flag(i,j  ,k).isConnected(-1,0,0) ? 1 : 0 );
    int  ilop = i + (flag(i,j-1,k).isConnected( 1,0,0) ? 1 : 0 );
    int  ilom = i - (flag(i,j-1,k).isConnected(-1,0,0) ? 1 : 0 );

    Real  whi = weights[ihip-ihim];
    Real  wlo = weights[ilop-ilom];
    Real  dudx = (0.5*dxinv[0]) * ((q(ihip,j  ,k,QU)-q(ihim,j  ,k,QU))*whi
                                  +(q(ilop,j-1,k,QU)-q(ilom,j-1,k,QU))*wlo);
    Real  dvdx = (0.5*dxinv[0]) * ((q(ihip,j  ,k,QV)-q(ihim,j  ,k,QV))*whi
                                  +(q(ilop,j-1,k,QV)-q(ilom,j-1,k,QV))*wlo);

    int  khip = k + (flag(i,j  ,k).isConnected(0,0, 1) ? 1 : 0 );
    int  khim = k - (flag(i,j  ,k).isConnected(0,0,-1) ? 1 : 0 );
    int  klop = k + (flag(i,j-1,k).isConnected(0,0, 1) ? 1 : 0 );
    int  klom = k - (flag(i,j-1,k).isConnected(0,0,-1) ? 1 : 0 );

    whi = weights[khip-khim];
    wlo = weights[klop-klom];
    Real  dvdz = (0.5*dxinv[2]) * ((q(i,j  ,khip,QV)-q(i,j  ,khim,QV))*whi
                                  +(q(i,j-1,klop,QV)-q(i,j-1,klom,QV))*wlo);
    Real  dwdz = (0.5*dxinv[2]) * ((q(i,j  ,khip,QW)-q(i,j  ,khim,QW))*whi
                                  +(q(i,j-1,klop,QW)-q(i,j-1,klom,QW))*wlo);
    Real  divu = dudx + dvdy + dwdz;
    Real  tauyy = muf*(2.0*dvdy-parm.twoThirds*divu);
    Real  tauxy = muf*(dudy+dvdx);
    Real  tauyz = muf*(dwdy+dvdz);

    fy(i,j,k,UMX)   += -tauxy;
    fy(i,j,k,UMY)   += -tauyy;
    fy(i,j,k,UMZ)   += -tauyz;
    fy(i,j,k,UEDEN) += -0.5*( (q(i,j,k,QU)+q(i,j-1,k,QU))*tauxy
                             +(q(i,j,k,QV)+q(i,j-1,k,QV))*tauyy
                             +(q(i,j,k,QW)+q(i,j-1,k,QW))*tauyz
                             +(ktl+ktr)*dTdy );
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
eb_compute_visc_z (int i, int j, int k,
             amrex::Array4<amrex::Real const> const& q,
             amrex::Array4<amrex::Real> const& fz,
             amrex::Array4<amrex::EBCellFlag const> const& flag,
amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv,
             amrex::GpuArray<amrex::Real,3> const& weights,
             Parm const& parm) noexcept
{
    using amrex::Real;
    // first compute temperature
    Real Tl = q(i,j,k,QT);
    Real Tr = q(i,j,k-1,QT);
    // derive mu
    Real mutl = parm.C_s*Tl*sqrt(Tl)/(Tl+parm.T_s);
    Real mutr = parm.C_s*Tr*sqrt(Tr)/(Tr+parm.T_s);
    Real muf = Real(0.5)*(mutl+mutr);
    // derive k
    Real ktl = mutl * parm.kOverMu;
    Real ktr = mutr * parm.kOverMu;

    Real  dTdz = (Tl-Tr)*dxinv[2];
    Real  dudz = (q(i,j,k,QU)-q(i,j,k-1,QU))*dxinv[2];
    Real  dvdz = (q(i,j,k,QV)-q(i,j,k-1,QV))*dxinv[2];
    Real  dwdz = (q(i,j,k,QW)-q(i,j,k-1,QW))*dxinv[2];

    int  ihip = i + (flag(i,j,k  ).isConnected( 1,0,0) ? 1 : 0 );
    int  ihim = i - (flag(i,j,k  ).isConnected(-1,0,0) ? 1 : 0 );
    int  ilop = i + (flag(i,j,k-1).isConnected( 1,0,0) ? 1 : 0 );
    int  ilom = i - (flag(i,j,k-1).isConnected(-1,0,0) ? 1 : 0 );

    Real   whi = weights[ihip-ihim];
    Real   wlo = weights[ilop-ilom];
    Real   dudx = (0.5*dxinv[0]) * ((q(ihip,j,k  ,QU)-q(ihim,j,k  ,QU))*whi
                                   +(q(ilop,j,k-1,QU)-q(ilom,j,k-1,QU))*wlo);
     Real  dwdx = (0.5*dxinv[0]) * ((q(ihip,j,k  ,QW)-q(ihim,j,k  ,QW))*whi
                                   +(q(ilop,j,k-1,QW)-q(ilom,j,k-1,QW))*wlo);


    int  jhip = j + (flag(i,j,k  ).isConnected(0, 1,0) ? 1 : 0 );
    int  jhim = j - (flag(i,j,k  ).isConnected(0,-1,0) ? 1 : 0 );
    int  jlop = j + (flag(i,j,k-1).isConnected(0, 1,0) ? 1 : 0 );
    int  jlom = j - (flag(i,j,k-1).isConnected(0,-1,0) ? 1 : 0 );

    whi = weights[jhip-jhim];
    wlo = weights[jlop-jlom];
    Real   dvdy = (0.5*dxinv[1]) * ((q(i,jhip,k  ,QV)-q(i,jhim,k  ,QV))*whi
                                   +(q(i,jlop,k-1,QV)-q(i,jlom,k-1,QV))*wlo);
    Real  dwdy = (0.5*dxinv[1]) *  ((q(i,jhip,k  ,QW)-q(i,jhim,k  ,QW))*whi
                                   +(q(i,jlop,k-1,QW)-q(i,jlom,k-1,QW))*wlo);

    Real  divu = dudx + dvdy + dwdz;
    Real  tauxz = muf*(dudz+dwdx);
    Real  tauyz = muf*(dvdz+dwdy);
    Real  tauzz = muf*(2.0*dwdz-parm.twoThirds*divu);

    fz(i,j,k,UMX)   += -tauxz;
    fz(i,j,k,UMY)   += -tauyz;
    fz(i,j,k,UMZ)   += -tauzz;
    fz(i,j,k,UEDEN) += -0.5*( (q(i,j,k,QU)+q(i,j,k-1,QU))*tauxz
                             +(q(i,j,k,QV)+q(i,j,k-1,QV))*tauyz
                             +(q(i,j,k,QW)+q(i,j,k-1,QW))*tauzz
                             +(ktl+ktr)*dTdz );
}

#endif