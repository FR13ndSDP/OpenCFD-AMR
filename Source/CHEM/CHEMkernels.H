#ifndef CHEMKERNELS_H
#define CHEMKERNELS_H

#include "EBR.H"
#include "Constants.H"

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE 
void conv_flux_x(int i, int j, int k,
                amrex::Array4<amrex::Real> const &ql,
                amrex::Array4<amrex::Real> const &qr,
                amrex::Array4<amrex::Real const> const &q,
                amrex::Array4<amrex::Real      > const &fx,
                Parm const& parm) noexcept
{
    using amrex::Real;
    amrex::GpuArray<Real, NSPECS> fm;
    amrex::GpuArray<Real, NSPECS> fp;
    Real tmp = Real(1.0)/(Real(2.0)*parm.eos_gamma);

    Real dl = q(i-1,j,k,QRHO);
    Real uul=q(i-1,j,k,QU);
    Real pl=q(i-1,j,k,QPRES);
    Real dr =q(i,j,k,QRHO);
    Real uur=q(i,j,k,QU);
    Real pr=q(i,j,k,QPRES);

    Real al=std::sqrt(parm.eos_gamma*pl/dl);
    Real ar=std::sqrt(parm.eos_gamma*pr/dr);

    Real tmp1=Real(2.0)*(parm.eos_gamma-Real(1.0));
    // eigenvalues---------
    Real E1P=(uul+amrex::Math::abs(uul))*Real(0.5);
    Real E2P=(uul-al+amrex::Math::abs(uul-al))*Real(0.5);
    Real E3P=(uul+al+amrex::Math::abs(uul+al))*Real(0.5);
    
    Real E1M=(uur-amrex::Math::abs(uur))*Real(0.5);
    Real E2M=(uur-ar-amrex::Math::abs(uur-ar))*Real(0.5);
    Real E3M=(uur+ar-amrex::Math::abs(uur+ar))*Real(0.5);

    for (int n=0; n<NSPECS; ++n) {
    fp[n]=tmp*(tmp1*E1P+E2P+E3P)*ql(i,j,k,n);
    fm[n]=tmp*(tmp1*E1M+E2M+E3M)*qr(i,j,k,n);
        fx(i,j,k,n) = fp[n]+fm[n];
    }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE 
void conv_flux_y(int i, int j, int k,
                amrex::Array4<amrex::Real> const &ql,
                amrex::Array4<amrex::Real> const &qr,
                amrex::Array4<amrex::Real const> const &q,
                amrex::Array4<amrex::Real      > const &fy,
                Parm const& parm) noexcept
{
    using amrex::Real;
    amrex::GpuArray<Real, NSPECS> fm;
    amrex::GpuArray<Real, NSPECS> fp;
    Real tmp = Real(1.0)/(Real(2.0)*parm.eos_gamma);

    Real dl = q(i-1,j,k,QRHO);
    Real uul=q(i-1,j,k,QV);
    Real pl=q(i-1,j,k,QPRES);
    Real dr =q(i,j,k,QRHO);
    Real uur=q(i,j,k,QV);
    Real pr=q(i,j,k,QPRES);

    Real al=std::sqrt(parm.eos_gamma*pl/dl);
    Real ar=std::sqrt(parm.eos_gamma*pr/dr);

    Real tmp1=Real(2.0)*(parm.eos_gamma-Real(1.0));
    // eigenvalues---------
    Real E1P=(uul+amrex::Math::abs(uul))*Real(0.5);
    Real E2P=(uul-al+amrex::Math::abs(uul-al))*Real(0.5);
    Real E3P=(uul+al+amrex::Math::abs(uul+al))*Real(0.5);
    
    Real E1M=(uur-amrex::Math::abs(uur))*Real(0.5);
    Real E2M=(uur-ar-amrex::Math::abs(uur-ar))*Real(0.5);
    Real E3M=(uur+ar-amrex::Math::abs(uur+ar))*Real(0.5);

    for (int n=0; n<NSPECS; ++n) {
    fp[n]=tmp*(tmp1*E1P+E2P+E3P)*ql(i,j,k,n);
    fm[n]=tmp*(tmp1*E1M+E2M+E3M)*qr(i,j,k,n);
        fy(i,j,k,n) = fp[n]+fm[n];
    }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE 
void conv_flux_z(int i, int j, int k,
                amrex::Array4<amrex::Real> const &ql,
                amrex::Array4<amrex::Real> const &qr,
                amrex::Array4<amrex::Real const> const &q,
                amrex::Array4<amrex::Real      > const &fz,
                Parm const& parm) noexcept
{
    using amrex::Real;
    amrex::GpuArray<Real, NSPECS> fm;
    amrex::GpuArray<Real, NSPECS> fp;
    Real tmp = Real(1.0)/(Real(2.0)*parm.eos_gamma);

    Real dl = q(i-1,j,k,QRHO);
    Real uul=q(i-1,j,k,QW);
    Real pl=q(i-1,j,k,QPRES);
    Real dr =q(i,j,k,QRHO);
    Real uur=q(i,j,k,QW);
    Real pr=q(i,j,k,QPRES);

    Real al=std::sqrt(parm.eos_gamma*pl/dl);
    Real ar=std::sqrt(parm.eos_gamma*pr/dr);

    Real tmp1=Real(2.0)*(parm.eos_gamma-Real(1.0));
    // eigenvalues---------
    Real E1P=(uul+amrex::Math::abs(uul))*Real(0.5);
    Real E2P=(uul-al+amrex::Math::abs(uul-al))*Real(0.5);
    Real E3P=(uul+al+amrex::Math::abs(uul+al))*Real(0.5);
    
    Real E1M=(uur-amrex::Math::abs(uur))*Real(0.5);
    Real E2M=(uur-ar-amrex::Math::abs(uur-ar))*Real(0.5);
    Real E3M=(uur+ar-amrex::Math::abs(uur+ar))*Real(0.5);

    for (int n=0; n<NSPECS; ++n) {
    fp[n]=tmp*(tmp1*E1P+E2P+E3P)*ql(i,j,k,n);
    fm[n]=tmp*(tmp1*E1M+E2M+E3M)*qr(i,j,k,n);
        fz(i,j,k,n) = fp[n]+fm[n];
    }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE 
void diffuse_flux_x(int i, int j, int k,
                amrex::Array4<amrex::Real const> const &spec,
                amrex::Array4<amrex::Real const> const &state,
                amrex::Array4<amrex::Real> const &fx,
amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv) noexcept
{
    using amrex::Real;
    std::array<Real, 4> Di{0.001, 0.002, 0.001, 0.001};

    for (int n=0; n<NSPECS; ++n) {
        Real dcdx = (spec(i,j,k,n) - spec(i-1,j,k,n))* dxinv[0];
        fx(i,j,k,n) += -Di[n]*dcdx;
    }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE 
void diffuse_flux_y(int i, int j, int k,
                amrex::Array4<amrex::Real const> const &spec,
                amrex::Array4<amrex::Real const> const &state,
                amrex::Array4<amrex::Real> const &fy,
amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv) noexcept
{
    using amrex::Real;
    std::array<Real, 4> Di{0.001, 0.002, 0.001, 0.001};

    for (int n=0; n<NSPECS; ++n) {
        Real dcdy = (spec(i,j,k,n) - spec(i,j-1,k,n))* dxinv[1];
        fy(i,j,k,n) += -Di[n]*dcdy;
    }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE 
void diffuse_flux_z(int i, int j, int k,
                amrex::Array4<amrex::Real const> const &spec,
                amrex::Array4<amrex::Real const> const &state,
                amrex::Array4<amrex::Real> const &fz,
amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv) noexcept
{
    using amrex::Real;
    std::array<Real, 4> Di{0.001, 0.002, 0.001, 0.001};

    for (int n=0; n<NSPECS; ++n) {
        Real dcdz = (spec(i,j,k,n) - spec(i,j,k-1,n))* dxinv[2];
        fz(i,j,k,n) += -Di[n]*dcdz;
    }
}

#endif