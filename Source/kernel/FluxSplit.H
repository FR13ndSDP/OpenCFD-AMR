#ifndef FLUXSPLIT_H
#define FLUXSPLIT_H


#include "AMReX_GpuPrint.H"
#include "AMReX_GpuQualifiers.H"
#include <Constants.H>
#include <IndexDefines.H>
#include <cstdio>

// Steger-Warming
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
compute_flux_x (int i, int j, int k,
             amrex::Array4<amrex::Real const> const& ql,
             amrex::Array4<amrex::Real const> const& qr,
             amrex::Array4<amrex::Real> const& fx,
             Parm const& parm) noexcept
{
    using amrex::Real;
    amrex::GpuArray<Real, NCONS> fm;
    amrex::GpuArray<Real, NCONS> fp;
    Real tmp = Real(1.0)/(Real(2.0)*parm.eos_gamma);

    Real dl= ql(i,j,k,QRHO);
    Real uul=ql(i,j,k,QU);
    Real vvl=ql(i,j,k,QV);
    Real wwl=ql(i,j,k,QW);
    Real pl=amrex::max(ql(i,j,k,QPRES),parm.smallp);
    Real dr= qr(i,j,k,QRHO);
    Real uur=qr(i,j,k,QU);
    Real vvr=qr(i,j,k,QV);
    Real wwr=qr(i,j,k,QW);
    Real pr=amrex::max(qr(i,j,k,QPRES),parm.smallp);
    Real al=std::sqrt(parm.eos_gamma*pl/dl);
    Real ar=std::sqrt(parm.eos_gamma*pr/dr);

    Real tmp1=Real(2.0)*(parm.eos_gamma-Real(1.0));
    Real tmp3=(Real(3.0)-parm.eos_gamma)/(Real(2.0)*(parm.eos_gamma-Real(1.0)));
    // eigenvalues---------
    Real E1P=(uul+amrex::Math::abs(uul))*Real(0.5);
    Real E2P=(uul-al+amrex::Math::abs(uul-al))*Real(0.5);
    Real E3P=(uul+al+amrex::Math::abs(uul+al))*Real(0.5);
    Real tmp0=dl*tmp;
    fp[0]=tmp0*(tmp1*E1P+E2P+E3P);
    fp[1]=tmp0*(tmp1*E1P*uul+E2P*(uul-al)+E3P*(uul+al));
    fp[2]=tmp0*(tmp1*E1P*vvl+E2P*vvl+E3P*vvl);
    fp[3]=tmp0*(tmp1*E1P*wwl+E2P*wwl+E3P*wwl);
    fp[4]=tmp0*(E1P*(parm.eos_gamma-Real(1.0))*(uul*uul+vvl*vvl+wwl*wwl)+E2P*((uul-al)*(uul-al)+vvl*vvl+wwl*wwl)*Real(0.5) \
      +E3P*((uul+al)*(uul+al)+vvl*vvl+wwl*wwl)*Real(0.5)+tmp3*al*al*(E2P+E3P));

    Real E1M=(uur-amrex::Math::abs(uur))*Real(0.5);
    Real E2M=(uur-ar-amrex::Math::abs(uur-ar))*Real(0.5);
    Real E3M=(uur+ar-amrex::Math::abs(uur+ar))*Real(0.5);
    tmp0=dr*tmp;
    fm[0]=tmp0*(tmp1*E1M+E2M+E3M);
    fm[1]=tmp0*(tmp1*E1M*uur+E2M*(uur-ar)+E3M*(uur+ar));
    fm[2]=tmp0*(tmp1*E1M*vvr+E2M*vvr+E3M*vvr);
    fm[3]=tmp0*(tmp1*E1M*wwr+E2M*wwr+E3M*wwr);
    fm[4]=tmp0*(E1M*(parm.eos_gamma-Real(1.0))*(uur*uur+vvr*vvr+wwr*wwr)+E2M*((uur-ar)*(uur-ar)+vvr*vvr+wwr*wwr)*Real(0.5) \
      +E3M*((uur+ar)*(uur+ar)+vvr*vvr+wwr*wwr)*Real(0.5)+tmp3*ar*ar*(E2M+E3M));

    for (int n=0; n<NCONS; ++n) {
        fx(i,j,k,n) = fp[n]+fm[n];
    }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
compute_flux_y (int i, int j, int k,
             amrex::Array4<amrex::Real const> const& ql,
             amrex::Array4<amrex::Real const> const& qr,
             amrex::Array4<amrex::Real> const& fy,
             Parm const& parm) noexcept
{
    using amrex::Real;
    amrex::GpuArray<Real, NCONS> fp;
    amrex::GpuArray<Real, NCONS> fm;
    Real tmp = Real(1.0)/(Real(2.0)*parm.eos_gamma);

    Real dl=ql(i,j,k,QRHO);
    Real uul=ql(i,j,k,QV);
    Real vvl=-ql(i,j,k,QU);
    Real wwl=ql(i,j,k,QW);
    Real pl=amrex::max(ql(i,j,k,QPRES),parm.smallp);
    Real dr=qr(i,j,k,QRHO);
    Real uur=qr(i,j,k,QV);
    Real vvr=-qr(i,j,k,QU);
    Real wwr=qr(i,j,k,QW);
    Real pr=amrex::max(qr(i,j,k,QPRES),parm.smallp);
    Real al=std::sqrt(parm.eos_gamma*pl/dl);
    Real ar=std::sqrt(parm.eos_gamma*pr/dr);

    Real tmp1=Real(2.0)*(parm.eos_gamma-Real(1.0));
    Real tmp3=(Real(3.0)-parm.eos_gamma)/(Real(2.0)*(parm.eos_gamma-Real(1.0)));
    // eigenvalues---------
    Real E1P=(uul+amrex::Math::abs(uul))*Real(0.5);
    Real E2P=(uul-al+amrex::Math::abs(uul-al))*Real(0.5);
    Real E3P=(uul+al+amrex::Math::abs(uul+al))*Real(0.5);
    Real tmp0=dl*tmp;
    fp[0]=tmp0*(tmp1*E1P+E2P+E3P);
    fp[1]=tmp0*(tmp1*E1P*uul+E2P*(uul-al)+E3P*(uul+al));
    fp[2]=tmp0*(tmp1*E1P*vvl+E2P*vvl+E3P*vvl);
    fp[3]=tmp0*(tmp1*E1P*wwl+E2P*wwl+E3P*wwl);
    fp[4]=tmp0*(E1P*(parm.eos_gamma-Real(1.0))*(uul*uul+vvl*vvl+wwl*wwl)+E2P*((uul-al)*(uul-al)+vvl*vvl+wwl*wwl)*Real(0.5) \
      +E3P*((uul+al)*(uul+al)+vvl*vvl+wwl*wwl)*Real(0.5)+tmp3*al*al*(E2P+E3P));

    Real E1M=(uur-amrex::Math::abs(uur))*Real(0.5);
    Real E2M=(uur-ar-amrex::Math::abs(uur-ar))*Real(0.5);
    Real E3M=(uur+ar-amrex::Math::abs(uur+ar))*Real(0.5);
    tmp0=dr*tmp;
    fm[0]=tmp0*(tmp1*E1M+E2M+E3M);
    fm[1]=tmp0*(tmp1*E1M*uur+E2M*(uur-ar)+E3M*(uur+ar));
    fm[2]=tmp0*(tmp1*E1M*vvr+E2M*vvr+E3M*vvr);
    fm[3]=tmp0*(tmp1*E1M*wwr+E2M*wwr+E3M*wwr);
    fm[4]=tmp0*(E1M*(parm.eos_gamma-Real(1.0))*(uur*uur+vvr*vvr+wwr*wwr)+E2M*((uur-ar)*(uur-ar)+vvr*vvr+wwr*wwr)*Real(0.5) \
      +E3M*((uur+ar)*(uur+ar)+vvr*vvr+wwr*wwr)*Real(0.5)+tmp3*ar*ar*(E2M+E3M));

    fy(i,j,k,URHO)  =   fp[URHO]+fm[URHO];
    fy(i,j,k,UMX)   = -(fp[UMY]+fm[UMY]);
    fy(i,j,k,UMY)   =   fp[UMX]+fm[UMX];
    fy(i,j,k,UMZ)   =   fp[UMZ]+fm[UMZ];
    fy(i,j,k,UEDEN) =   fp[UEDEN]+fm[UEDEN];
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
compute_flux_z (int i, int j, int k,
             amrex::Array4<amrex::Real const> const& ql,
             amrex::Array4<amrex::Real const> const& qr,
             amrex::Array4<amrex::Real> const& fz,
             Parm const& parm) noexcept
{
    using amrex::Real;
    amrex::GpuArray<Real, NCONS> fp;
    amrex::GpuArray<Real, NCONS> fm;
    Real tmp = Real(1.0)/(Real(2.0)*parm.eos_gamma);

    Real dl=ql(i,j,k,QRHO);
    Real uul=ql(i,j,k,QW);
    Real vvl=ql(i,j,k,QV);
    Real wwl=-ql(i,j,k,QU);
    Real pl=amrex::max(ql(i,j,k,QPRES),parm.smallp);
    Real dr=qr(i,j,k,QRHO);
    Real uur=qr(i,j,k,QW);
    Real vvr=qr(i,j,k,QV);
    Real wwr=-qr(i,j,k,QU);
    Real pr=amrex::max(qr(i,j,k,QPRES),parm.smallp);
    Real al=std::sqrt(parm.eos_gamma*pl/dl);
    Real ar=std::sqrt(parm.eos_gamma*pr/dr);

    Real tmp1=Real(2.0)*(parm.eos_gamma-Real(1.0));
    Real tmp3=(Real(3.0)-parm.eos_gamma)/(Real(2.0)*(parm.eos_gamma-Real(1.0)));
    // eigenvalues---------
    Real E1P=(uul+amrex::Math::abs(uul))*Real(0.5);
    Real E2P=(uul-al+amrex::Math::abs(uul-al))*Real(0.5);
    Real E3P=(uul+al+amrex::Math::abs(uul+al))*Real(0.5);
    Real tmp0=dl*tmp;
    fp[0]=tmp0*(tmp1*E1P+E2P+E3P);
    fp[1]=tmp0*(tmp1*E1P*uul+E2P*(uul-al)+E3P*(uul+al));
    fp[2]=tmp0*(tmp1*E1P*vvl+E2P*vvl+E3P*vvl);
    fp[3]=tmp0*(tmp1*E1P*wwl+E2P*wwl+E3P*wwl);
    fp[4]=tmp0*(E1P*(parm.eos_gamma-Real(1.0))*(uul*uul+vvl*vvl+wwl*wwl)+E2P*((uul-al)*(uul-al)+vvl*vvl+wwl*wwl)*Real(0.5) \
      +E3P*((uul+al)*(uul+al)+vvl*vvl+wwl*wwl)*Real(0.5)+tmp3*al*al*(E2P+E3P));

    Real E1M=(uur-amrex::Math::abs(uur))*Real(0.5);
    Real E2M=(uur-ar-amrex::Math::abs(uur-ar))*Real(0.5);
    Real E3M=(uur+ar-amrex::Math::abs(uur+ar))*Real(0.5);
    tmp0=dr*tmp;
    fm[0]=tmp0*(tmp1*E1M+E2M+E3M);
    fm[1]=tmp0*(tmp1*E1M*uur+E2M*(uur-ar)+E3M*(uur+ar));
    fm[2]=tmp0*(tmp1*E1M*vvr+E2M*vvr+E3M*vvr);
    fm[3]=tmp0*(tmp1*E1M*wwr+E2M*wwr+E3M*wwr);
    fm[4]=tmp0*(E1M*(parm.eos_gamma-Real(1.0))*(uur*uur+vvr*vvr+wwr*wwr)+E2M*((uur-ar)*(uur-ar)+vvr*vvr+wwr*wwr)*Real(0.5) \
      +E3M*((uur+ar)*(uur+ar)+vvr*vvr+wwr*wwr)*Real(0.5)+tmp3*ar*ar*(E2M+E3M));

    fz(i,j,k,URHO)  =   fp[URHO]+fm[URHO];
    fz(i,j,k,UMX)   = -(fp[UMZ]+fm[UMZ]);
    fz(i,j,k,UMY)   =   fp[UMY]+fm[UMY];
    fz(i,j,k,UMZ)   =   fp[UMX]+fm[UMX];
    fz(i,j,k,UEDEN) =   fp[UEDEN]+fm[UEDEN];

// #if AMREX_USE_GPU
//     AMREX_DEVICE_PRINTF("flux_z : %f, %f, %f, %f, %f\n", fz(i,j,k,URHO), fz(i,j,k,UMX), fz(i,j,k,UMY), fz(i,j,k,UMZ), fz(i,j,k,UEDEN));
// #else
//   std::printf("flux_z : %f, %f, %f, %f, %f\n", fz(i,j,k,URHO), fz(i,j,k,UMX), fz(i,j,k,UMY), fz(i,j,k,UMZ), fz(i,j,k,UEDEN));
// #endif
}

#endif