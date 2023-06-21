#ifndef EBR_KERNEL_H_
#define EBR_KERNEL_H_

#include <EBR_index_macros.H>
#include <EBR_parm.H>
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
    Real kineng = Real(0.5)*rho*(ux*ux+uy*uy+uz*uz);
    Real ei = u(i,j,k,UEDEN) - kineng;
    Real p = amrex::max((parm.eos_gamma-Real(1.0))*ei,parm.smallp);

    q(i,j,k,QRHO) = rho;
    q(i,j,k,QU) = ux;
    q(i,j,k,QV) = uy;
    q(i,j,k,QW) = uz;
    q(i,j,k,QPRES) = p;
}

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

// return will possibly affect performance
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
amrex::Real
minmod(amrex::Real const a, amrex::Real const b)
{
    if (a*b > 0) {
        return amrex::Math::abs(a)>amrex::Math::abs(b)? b:a;
    } else
    {
        return amrex::Real(0);
    }
}

// // 2nd order NND
// AMREX_GPU_DEVICE
// AMREX_FORCE_INLINE
// void
// reconstruction_x (int i, int j, int k, int n,
//              amrex::Array4<amrex::Real> const& ql,
//              amrex::Array4<amrex::Real> const& qr,
//              amrex::Array4<amrex::Real const> const& q,
//              Parm const& parm) noexcept
// {
//     ql(i,j,k,n) = q(i-1,j,k,n) + amrex::Real(0.5)*minmod(q(i,j,k,n)-q(i-1,j,k,n), q(i-1,j,k,n)-q(i-2,j,k,n));
//     qr(i,j,k,n) = q(i,j,k,n) - amrex::Real(0.5)*minmod(q(i,j,k,n)-q(i-1,j,k,n), q(i+1,j,k,n)-q(i,j,k,n));
// }

// AMREX_GPU_DEVICE
// AMREX_FORCE_INLINE
// void
// reconstruction_y (int i, int j, int k, int n,
//              amrex::Array4<amrex::Real> const& ql,
//              amrex::Array4<amrex::Real> const& qr,
//              amrex::Array4<amrex::Real const> const& q,
//              Parm const& parm) noexcept
// {
//     ql(i,j,k,n) = q(i,j-1,k,n) + amrex::Real(0.5)*minmod(q(i,j,k,n)-q(i,j-1,k,n), q(i,j-1,k,n)-q(i,j-2,k,n));
//     qr(i,j,k,n) = q(i,j,k,n) - amrex::Real(0.5)*minmod(q(i,j,k,n)-q(i,j-1,k,n), q(i,j+1,k,n)-q(i,j,k,n));
// }

// AMREX_GPU_DEVICE
// AMREX_FORCE_INLINE
// void
// reconstruction_z (int i, int j, int k, int n,
//              amrex::Array4<amrex::Real> const& ql,
//              amrex::Array4<amrex::Real> const& qr,
//              amrex::Array4<amrex::Real const> const& q,
//              Parm const& parm) noexcept
// {
//     ql(i,j,k,n) = q(i,j,k-1,n) + amrex::Real(0.5)*minmod(q(i,j,k,n)-q(i,j,k-1,n), q(i,j,k-1,n)-q(i,j,k-2,n));
//     qr(i,j,k,n) = q(i,j,k,n) - amrex::Real(0.5)*minmod(q(i,j,k,n)-q(i,j,k-1,n), q(i,j,k+1,n)-q(i,j,k,n));
// }

// 5th order TENO
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
reconstruction_x (amrex::Box const& bx,
             amrex::Array4<amrex::Real> const& ql,
             amrex::Array4<amrex::Real> const& qr,
             amrex::Array4<amrex::Real const> const& q,
             Parm const& parm) noexcept
{
    using amrex::Real;

    const auto lo = amrex::lbound(bx);
    const auto hi = amrex::ubound(bx);

    for (int n = 0; n < NPRIM; ++n) {
        for (int k = lo.z; k<= hi.z; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    Real V1 = q(i-3,j,k,n);
                    Real V2 = q(i-2,j,k,n);
                    Real V3 = q(i-1,j,k,n);
                    Real V4 = q(i,j,k,n);
                    Real V5 = q(i+1,j,k,n);
                    Real V6 = q(i+2,j,k,n);

                    // qL
                    Real s11 = 13.0*(V1-2.0*V2+V3)*(V1-2.0*V2+V3) + 3.0*(V1-4.0*V2+3.0*V3)*(V1-4.0*V2+3.0*V3);
                    Real s22 = 13.0*(V2-2.0*V3+V4)*(V2-2.0*V3+V4) + 3.0*(V2-V4)*(V2-V4);
                    Real s33 = 13.0*(V3-2.0*V4+V5)*(V3-2.0*V4+V5) + 3.0*(3.0*V3-4.0*V4+V5)*(3.0*V3-4.0*V4+V5);

                    Real s55 = amrex::Math::abs(s11-s33);

                    // get rid of pow here, calculate (1+x)^6
                    Real a1 = 1.0+s55/(s11+parm.eps);
                    Real a2 = 1.0+s55/(s22+parm.eps);
                    Real a3 = 1.0+s55/(s33+parm.eps);
                    a1 = a1*a1*a1*a1*a1*a1;
                    a2 = a2*a2*a2*a2*a2*a2;
                    a3 = a3*a3*a3*a3*a3*a3;

                    Real invsum = 1.0/(a1+a2+a3);
                    Real b1 = a1*invsum;
                    Real b2 = a2*invsum;
                    Real b3 = a3*invsum;


                    a1 = 0.1*(b1<1.e-5? 0.0:1.0);
                    a2 = 0.6*(b2<1.e-5? 0.0:1.0);
                    a3 = 0.3*(b3<1.e-5? 0.0:1.0);

                    Real v1 = parm.oneSix*(2.0*V1-7.0*V2+5.0*V3);
                    Real v2 = parm.oneSix*(-V2-V3+2.0*V4);
                    Real v3 = parm.oneSix*(-4.0*V3+5.0*V4-V5);

                    invsum = 1.0/(a1+a2+a3);
                    Real w1 = a1*invsum;
                    Real w2 = a2*invsum;
                    Real w3 = a3*invsum;

                    ql(i,j,k,n) = V3+w1*v1+w2*v2+w3*v3;

                    // qR
                    s11 = 13.0*(V6-2.0*V5+V4)*(V6-2.0*V5+V4) + 3.0*(V6-4.0*V5+3.0*V4)*(V6-4.0*V5+3.0*V4);
                    s22 = 13.0*(V3-2.0*V4+V5)*(V3-2.0*V4+V5) + 3.0*(V5-V3)*(V5-V3);
                    s33 = 13.0*(V4-2.0*V3+V2)*(V4-2.0*V3+V2) + 3.0*(3.0*V4-4.0*V3+V2)*(3.0*V4-4.0*V3+V2);

                    s55 = amrex::Math::abs(s11-s33);

                    // get rid of pow here, calculate (1+x)^6
                    a1 = 1.0+s55/(s11+parm.eps);
                    a2 = 1.0+s55/(s22+parm.eps);
                    a3 = 1.0+s55/(s33+parm.eps);
                    a1 = a1*a1*a1*a1*a1*a1;
                    a2 = a2*a2*a2*a2*a2*a2;
                    a3 = a3*a3*a3*a3*a3*a3;

                    invsum = 1.0/(a1+a2+a3);
                    b1 = a1*invsum;
                    b2 = a2*invsum;
                    b3 = a3*invsum;

                    a1 = 0.1*(b1<1.e-5? 0.0:1.0);
                    a2 = 0.6*(b2<1.e-5? 0.0:1.0);
                    a3 = 0.3*(b3<1.e-5? 0.0:1.0);

                    v1 = parm.oneSix*(2.0*V6-7.0*V5+5.0*V4);
                    v2 = parm.oneSix*(-V5-V4+2.0*V3);
                    v3 = parm.oneSix*(-4.0*V4+5.0*V3-V2);

                    invsum = 1.0/(a1+a2+a3);
                    w1 = a1*invsum;
                    w2 = a2*invsum;
                    w3 = a3*invsum;

                    qr(i,j,k,n) = V4+w1*v1+w2*v2+w3*v3;
                }
            }
        }
    }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
reconstruction_y (amrex::Box const& bx,
             amrex::Array4<amrex::Real> const& ql,
             amrex::Array4<amrex::Real> const& qr,
             amrex::Array4<amrex::Real const> const& q,
             Parm const& parm) noexcept
{
    using amrex::Real;
    const auto lo = amrex::lbound(bx);
    const auto hi = amrex::ubound(bx);

    for (int n = 0; n < NPRIM; ++n) {
        for (int k = lo.z; k<= hi.z; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    Real V1 = q(i,j-3,k,n);
                    Real V2 = q(i,j-2,k,n);
                    Real V3 = q(i,j-1,k,n);
                    Real V4 = q(i,j,k,n);
                    Real V5 = q(i,j+1,k,n);
                    Real V6 = q(i,j+2,k,n);

                    // qL
                    Real s11 = 13.0*(V1-2.0*V2+V3)*(V1-2.0*V2+V3) + 3.0*(V1-4.0*V2+3.0*V3)*(V1-4.0*V2+3.0*V3);
                    Real s22 = 13.0*(V2-2.0*V3+V4)*(V2-2.0*V3+V4) + 3.0*(V2-V4)*(V2-V4);
                    Real s33 = 13.0*(V3-2.0*V4+V5)*(V3-2.0*V4+V5) + 3.0*(3.0*V3-4.0*V4+V5)*(3.0*V3-4.0*V4+V5);

                    Real s55 = amrex::Math::abs(s11-s33);

                    // get rid of pow here, calculate (1+x)^6
                    Real a1 = 1.0+s55/(s11+parm.eps);
                    Real a2 = 1.0+s55/(s22+parm.eps);
                    Real a3 = 1.0+s55/(s33+parm.eps);
                    a1 = a1*a1*a1*a1*a1*a1;
                    a2 = a2*a2*a2*a2*a2*a2;
                    a3 = a3*a3*a3*a3*a3*a3;

                    Real invsum = 1.0/(a1+a2+a3);
                    Real b1 = a1*invsum;
                    Real b2 = a2*invsum;
                    Real b3 = a3*invsum;

                    b1 = amrex::max(b1-1.e-5, 0.0)/(b1-1.e-5);
                    b2 = amrex::max(b2-1.e-5, 0.0)/(b2-1.e-5);
                    b3 = amrex::max(b3-1.e-5, 0.0)/(b3-1.e-5);

                    Real v1 = parm.oneSix*(2.0*V1-7.0*V2+5.0*V3);
                    Real v2 = parm.oneSix*(-V2-V3+2.0*V4);
                    Real v3 = parm.oneSix*(-4.0*V3+5.0*V4-V5);

                    a1 = 0.10*b1;
                    a2 = 0.60*b2;
                    a3 = 0.30*b3;

                    invsum = 1.0/(a1+a2+a3);
                    Real w1 = a1*invsum;
                    Real w2 = a2*invsum;
                    Real w3 = a3*invsum;

                    ql(i,j,k,n) = V3+w1*v1+w2*v2+w3*v3;

                    // qR
                    s11 = 13.0*(V6-2.0*V5+V4)*(V6-2.0*V5+V4) + 3.0*(V6-4.0*V5+3.0*V4)*(V6-4.0*V5+3.0*V4);
                    s22 = 13.0*(V3-2.0*V4+V5)*(V3-2.0*V4+V5) + 3.0*(V5-V3)*(V5-V3);
                    s33 = 13.0*(V4-2.0*V3+V2)*(V4-2.0*V3+V2) + 3.0*(3.0*V4-4.0*V3+V2)*(3.0*V4-4.0*V3+V2);

                    s55 = amrex::Math::abs(s11-s33);

                    // get rid of pow here, calculate (1+x)^6
                    a1 = 1.0+s55/(s11+parm.eps);
                    a2 = 1.0+s55/(s22+parm.eps);
                    a3 = 1.0+s55/(s33+parm.eps);
                    a1 = a1*a1*a1*a1*a1*a1;
                    a2 = a2*a2*a2*a2*a2*a2;
                    a3 = a3*a3*a3*a3*a3*a3;
                    
                    invsum = 1.0/(a1+a2+a3);
                    b1 = a1*invsum;
                    b2 = a2*invsum;
                    b3 = a3*invsum;

                    b1 = amrex::max(b1-1.e-5, 0.0)/(b1-1.e-5);
                    b2 = amrex::max(b2-1.e-5, 0.0)/(b2-1.e-5);
                    b3 = amrex::max(b3-1.e-5, 0.0)/(b3-1.e-5);

                    v1 = parm.oneSix*(2.0*V6-7.0*V5+5.0*V4);
                    v2 = parm.oneSix*(-V5-V4+2.0*V3);
                    v3 = parm.oneSix*(-4.0*V4+5.0*V3-V2);

                    a1 = 0.10*b1;
                    a2 = 0.60*b2;
                    a3 = 0.30*b3;

                    invsum = 1.0/(a1+a2+a3);
                    w1 = a1*invsum;
                    w2 = a2*invsum;
                    w3 = a3*invsum;

                    qr(i,j,k,n) = V4+w1*v1+w2*v2+w3*v3;
                }
            }
        }
    }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
reconstruction_z (amrex::Box const& bx,
             amrex::Array4<amrex::Real> const& ql,
             amrex::Array4<amrex::Real> const& qr,
             amrex::Array4<amrex::Real const> const& q,
             Parm const& parm) noexcept
{
    using amrex::Real;
    const auto lo = amrex::lbound(bx);
    const auto hi = amrex::ubound(bx);

    for (int n = 0; n < NPRIM; ++n) {
        for (int k = lo.z; k<= hi.z; ++k) {
            for (int j = lo.y; j <= hi.y; ++j) {
                for (int i = lo.x; i <= hi.x; ++i) {
                    Real V1 = q(i,j,k-3,n);
                    Real V2 = q(i,j,k-2,n);
                    Real V3 = q(i,j,k-1,n);
                    Real V4 = q(i,j,k,n);
                    Real V5 = q(i,j,k+1,n);
                    Real V6 = q(i,j,k+2,n);

                    // qL
                    Real s11 = 13.0*(V1-2.0*V2+V3)*(V1-2.0*V2+V3) + 3.0*(V1-4.0*V2+3.0*V3)*(V1-4.0*V2+3.0*V3);
                    Real s22 = 13.0*(V2-2.0*V3+V4)*(V2-2.0*V3+V4) + 3.0*(V2-V4)*(V2-V4);
                    Real s33 = 13.0*(V3-2.0*V4+V5)*(V3-2.0*V4+V5) + 3.0*(3.0*V3-4.0*V4+V5)*(3.0*V3-4.0*V4+V5);

                    Real s55 = amrex::Math::abs(s11-s33);

                    // get rid of pow here, calculate (1+x)^6
                    Real a1 = 1.0+s55/(s11+parm.eps);
                    Real a2 = 1.0+s55/(s22+parm.eps);
                    Real a3 = 1.0+s55/(s33+parm.eps);
                    a1 = a1*a1*a1*a1*a1*a1;
                    a2 = a2*a2*a2*a2*a2*a2;
                    a3 = a3*a3*a3*a3*a3*a3;
                    
                    Real invsum = 1.0/(a1+a2+a3);
                    Real b1 = a1*invsum;
                    Real b2 = a2*invsum;
                    Real b3 = a3*invsum;

                    b1 = amrex::max(b1-1.e-5, 0.0)/(b1-1.e-5);
                    b2 = amrex::max(b2-1.e-5, 0.0)/(b2-1.e-5);
                    b3 = amrex::max(b3-1.e-5, 0.0)/(b3-1.e-5);

                    Real v1 = parm.oneSix*(2.0*V1-7.0*V2+5.0*V3);
                    Real v2 = parm.oneSix*(-V2-V3+2.0*V4);
                    Real v3 = parm.oneSix*(-4.0*V3+5.0*V4-V5);

                    a1 = 0.10*b1;
                    a2 = 0.60*b2;
                    a3 = 0.30*b3;

                    invsum = 1.0/(a1+a2+a3);
                    Real w1 = a1*invsum;
                    Real w2 = a2*invsum;
                    Real w3 = a3*invsum;

                    ql(i,j,k,n) = V3+w1*v1+w2*v2+w3*v3;

                    // qR
                    s11 = 13.0*(V6-2.0*V5+V4)*(V6-2.0*V5+V4) + 3.0*(V6-4.0*V5+3.0*V4)*(V6-4.0*V5+3.0*V4);
                    s22 = 13.0*(V3-2.0*V4+V5)*(V3-2.0*V4+V5) + 3.0*(V5-V3)*(V5-V3);
                    s33 = 13.0*(V4-2.0*V3+V2)*(V4-2.0*V3+V2) + 3.0*(3.0*V4-4.0*V3+V2)*(3.0*V4-4.0*V3+V2);

                    s55 = amrex::Math::abs(s11-s33);

                    // get rid of pow here, calculate (1+x)^6
                    a1 = 1.0+s55/(s11+parm.eps);
                    a2 = 1.0+s55/(s22+parm.eps);
                    a3 = 1.0+s55/(s33+parm.eps);
                    a1 = a1*a1*a1*a1*a1*a1;
                    a2 = a2*a2*a2*a2*a2*a2;
                    a3 = a3*a3*a3*a3*a3*a3;

                    invsum = 1.0/(a1+a2+a3);
                    b1 = a1*invsum;
                    b2 = a2*invsum;
                    b3 = a3*invsum;

                    b1 = amrex::max(b1-1.e-5, 0.0)/(b1-1.e-5);
                    b2 = amrex::max(b2-1.e-5, 0.0)/(b2-1.e-5);
                    b3 = amrex::max(b3-1.e-5, 0.0)/(b3-1.e-5);

                    v1 = parm.oneSix*(2.0*V6-7.0*V5+5.0*V4);
                    v2 = parm.oneSix*(-V5-V4+2.0*V3);
                    v3 = parm.oneSix*(-4.0*V4+5.0*V3-V2);

                    a1 = 0.10*b1;
                    a2 = 0.60*b2;
                    a3 = 0.30*b3;

                    invsum = 1.0/(a1+a2+a3);
                    w1 = a1*invsum;
                    w2 = a2*invsum;
                    w3 = a3*invsum;

                    qr(i,j,k,n) = V4+w1*v1+w2*v2+w3*v3;
                }
            }
        }
    }
}


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

    Real dl= ql(i,j,k,0);
    Real uul=ql(i,j,k,1);
    Real vvl=ql(i,j,k,2);
    Real wwl=ql(i,j,k,3);
    Real pl=amrex::max(ql(i,j,k,4),parm.smallp);
    Real dr= qr(i,j,k,0);
    Real uur=qr(i,j,k,1);
    Real vvr=qr(i,j,k,2);
    Real wwr=qr(i,j,k,3);
    Real pr=amrex::max(qr(i,j,k,4),parm.smallp);
    Real al=std::sqrt(parm.eos_gamma*pl/dl);
    Real ar=std::sqrt(parm.eos_gamma*pr/dr);

    Real tmp1=Real(2.0)*(parm.eos_gamma-Real(1.0));
    Real tmp3=(Real(3.0)-parm.eos_gamma)/(Real(2.0)*(parm.eos_gamma-Real(1.0)));
    // eigenvalues---------
    Real E1P=(uul+amrex::Math::abs(uul))*Real(0.5);
    Real E2P=(uul-al+amrex::Math::abs(uul-al))*Real(0.5);
    Real E3P=(uul+al+amrex::Math::abs(uul+al))*Real(0.5);
    Real tmp0=dl/(Real(2.0)*parm.eos_gamma);
    fp[0]=tmp0*(tmp1*E1P+E2P+E3P);
    fp[1]=tmp0*(tmp1*E1P*uul+E2P*(uul-al)+E3P*(uul+al));
    fp[2]=tmp0*(tmp1*E1P*vvl+E2P*vvl+E3P*vvl);
    fp[3]=tmp0*(tmp1*E1P*wwl+E2P*wwl+E3P*wwl);
    fp[4]=tmp0*(E1P*(parm.eos_gamma-Real(1.0))*(uul*uul+vvl*vvl+wwl*wwl)+E2P*((uul-al)*(uul-al)+vvl*vvl+wwl*wwl)*Real(0.5) \
      +E3P*((uul+al)*(uul+al)+vvl*vvl+wwl*wwl)*Real(0.5)+tmp3*al*al*(E2P+E3P));

    Real E1M=(uur-amrex::Math::abs(uur))*Real(0.5);
    Real E2M=(uur-ar-amrex::Math::abs(uur-ar))*Real(0.5);
    Real E3M=(uur+ar-amrex::Math::abs(uur+ar))*Real(0.5);
    tmp0=dr/(Real(2.0)*parm.eos_gamma);
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

    Real dl=ql(i,j,k,0);
    Real uul=ql(i,j,k,2);
    Real vvl=-ql(i,j,k,1);
    Real wwl=ql(i,j,k,3);
    Real pl=amrex::max(ql(i,j,k,4),parm.smallp);
    Real dr=qr(i,j,k,0);
    Real uur=qr(i,j,k,2);
    Real vvr=-qr(i,j,k,1);
    Real wwr=qr(i,j,k,3);
    Real pr=amrex::max(qr(i,j,k,4),parm.smallp);
    Real al=std::sqrt(parm.eos_gamma*pl/dl);
    Real ar=std::sqrt(parm.eos_gamma*pr/dr);

    Real tmp1=Real(2.0)*(parm.eos_gamma-Real(1.0));
    Real tmp3=(Real(3.0)-parm.eos_gamma)/(Real(2.0)*(parm.eos_gamma-Real(1.0)));
    // eigenvalues---------
    Real E1P=(uul+amrex::Math::abs(uul))*Real(0.5);
    Real E2P=(uul-al+amrex::Math::abs(uul-al))*Real(0.5);
    Real E3P=(uul+al+amrex::Math::abs(uul+al))*Real(0.5);
    Real tmp0=dl/(Real(2.0)*parm.eos_gamma);
    fp[0]=tmp0*(tmp1*E1P+E2P+E3P);
    fp[1]=tmp0*(tmp1*E1P*uul+E2P*(uul-al)+E3P*(uul+al));
    fp[2]=tmp0*(tmp1*E1P*vvl+E2P*vvl+E3P*vvl);
    fp[3]=tmp0*(tmp1*E1P*wwl+E2P*wwl+E3P*wwl);
    fp[4]=tmp0*(E1P*(parm.eos_gamma-Real(1.0))*(uul*uul+vvl*vvl+wwl*wwl)+E2P*((uul-al)*(uul-al)+vvl*vvl+wwl*wwl)*Real(0.5) \
      +E3P*((uul+al)*(uul+al)+vvl*vvl+wwl*wwl)*Real(0.5)+tmp3*al*al*(E2P+E3P));

    Real E1M=(uur-amrex::Math::abs(uur))*Real(0.5);
    Real E2M=(uur-ar-amrex::Math::abs(uur-ar))*Real(0.5);
    Real E3M=(uur+ar-amrex::Math::abs(uur+ar))*Real(0.5);
    tmp0=dr/(Real(2.0)*parm.eos_gamma);
    fm[0]=tmp0*(tmp1*E1M+E2M+E3M);
    fm[1]=tmp0*(tmp1*E1M*uur+E2M*(uur-ar)+E3M*(uur+ar));
    fm[2]=tmp0*(tmp1*E1M*vvr+E2M*vvr+E3M*vvr);
    fm[3]=tmp0*(tmp1*E1M*wwr+E2M*wwr+E3M*wwr);
    fm[4]=tmp0*(E1M*(parm.eos_gamma-Real(1.0))*(uur*uur+vvr*vvr+wwr*wwr)+E2M*((uur-ar)*(uur-ar)+vvr*vvr+wwr*wwr)*Real(0.5) \
      +E3M*((uur+ar)*(uur+ar)+vvr*vvr+wwr*wwr)*Real(0.5)+tmp3*ar*ar*(E2M+E3M));

    fy(i,j,k,0) =   fp[0]+fm[0];
    fy(i,j,k,1) = -(fp[2]+fm[2]);
    fy(i,j,k,2) =   fp[1]+fm[1];
    fy(i,j,k,3) =   fp[3]+fm[3];
    fy(i,j,k,4) =   fp[4]+fm[4];
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

    Real dl=ql(i,j,k,0);
    Real uul=ql(i,j,k,3);
    Real vvl=ql(i,j,k,2);
    Real wwl=-ql(i,j,k,1);
    Real pl=amrex::max(ql(i,j,k,4),parm.smallp);
    Real dr=qr(i,j,k,0);
    Real uur=qr(i,j,k,3);
    Real vvr=qr(i,j,k,2);
    Real wwr=-qr(i,j,k,1);
    Real pr=amrex::max(qr(i,j,k,4),parm.smallp);
    Real al=std::sqrt(parm.eos_gamma*pl/dl);
    Real ar=std::sqrt(parm.eos_gamma*pr/dr);

    Real tmp1=Real(2.0)*(parm.eos_gamma-Real(1.0));
    Real tmp3=(Real(3.0)-parm.eos_gamma)/(Real(2.0)*(parm.eos_gamma-Real(1.0)));
    // eigenvalues---------
    Real E1P=(uul+amrex::Math::abs(uul))*Real(0.5);
    Real E2P=(uul-al+amrex::Math::abs(uul-al))*Real(0.5);
    Real E3P=(uul+al+amrex::Math::abs(uul+al))*Real(0.5);
    Real tmp0=dl/(Real(2.0)*parm.eos_gamma);
    fp[0]=tmp0*(tmp1*E1P+E2P+E3P);
    fp[1]=tmp0*(tmp1*E1P*uul+E2P*(uul-al)+E3P*(uul+al));
    fp[2]=tmp0*(tmp1*E1P*vvl+E2P*vvl+E3P*vvl);
    fp[3]=tmp0*(tmp1*E1P*wwl+E2P*wwl+E3P*wwl);
    fp[4]=tmp0*(E1P*(parm.eos_gamma-Real(1.0))*(uul*uul+vvl*vvl+wwl*wwl)+E2P*((uul-al)*(uul-al)+vvl*vvl+wwl*wwl)*Real(0.5) \
      +E3P*((uul+al)*(uul+al)+vvl*vvl+wwl*wwl)*Real(0.5)+tmp3*al*al*(E2P+E3P));

    Real E1M=(uur-amrex::Math::abs(uur))*Real(0.5);
    Real E2M=(uur-ar-amrex::Math::abs(uur-ar))*Real(0.5);
    Real E3M=(uur+ar-amrex::Math::abs(uur+ar))*Real(0.5);
    tmp0=dr/(Real(2.0)*parm.eos_gamma);
    fm[0]=tmp0*(tmp1*E1M+E2M+E3M);
    fm[1]=tmp0*(tmp1*E1M*uur+E2M*(uur-ar)+E3M*(uur+ar));
    fm[2]=tmp0*(tmp1*E1M*vvr+E2M*vvr+E3M*vvr);
    fm[3]=tmp0*(tmp1*E1M*wwr+E2M*wwr+E3M*wwr);
    fm[4]=tmp0*(E1M*(parm.eos_gamma-Real(1.0))*(uur*uur+vvr*vvr+wwr*wwr)+E2M*((uur-ar)*(uur-ar)+vvr*vvr+wwr*wwr)*Real(0.5) \
      +E3M*((uur+ar)*(uur+ar)+vvr*vvr+wwr*wwr)*Real(0.5)+tmp3*ar*ar*(E2M+E3M));

    fz(i,j,k,0) =   fp[0]+fm[0];
    fz(i,j,k,1) = -(fp[3]+fm[3]);
    fz(i,j,k,2) =   fp[2]+fm[2];
    fz(i,j,k,3) =   fp[1]+fm[1];
    fz(i,j,k,4) =   fp[4]+fm[4];
}

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
    Real Tl = q(i,j,k,QPRES)/(parm.Rg * q(i,j,k,QRHO));
    Real Tr = q(i-1,j,k,QPRES)/(parm.Rg * q(i-1,j,k,QRHO));
    // derive mu
    Real mutl = parm.C_s*Tl*sqrt(Tl)/(Tl+parm.T_s);
    Real mutr = parm.C_s*Tr*sqrt(Tr)/(Tr+parm.T_s);
    // derive k
    Real ktl = mutl * parm.cv * parm.eos_gamma * parm.Prinv;
    Real ktr = mutr * parm.cv * parm.eos_gamma * parm.Prinv;

    Real dTdx = (Tl-Tr)*dxinv[0];
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
    Real Tl = q(i,j,k,QPRES)/(parm.Rg * q(i,j,k,QRHO));
    Real Tr = q(i,j-1,k,QPRES)/(parm.Rg * q(i,j-1,k,QRHO));
    // derive mu
    Real mutl = parm.C_s*Tl*sqrt(Tl)/(Tl+parm.T_s);
    Real mutr = parm.C_s*Tr*sqrt(Tr)/(Tr+parm.T_s);
    // derive k
    Real ktl = mutl * parm.cv * parm.eos_gamma * parm.Prinv;
    Real ktr = mutr * parm.cv * parm.eos_gamma * parm.Prinv;

    Real dTdy = (Tl-Tr)*dxinv[1];
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
    Real Tl = q(i,j,k,QPRES)/(parm.Rg * q(i,j,k,QRHO));
    Real Tr = q(i,j,k-1,QPRES)/(parm.Rg * q(i,j,k-1,QRHO));
    // derive mu
    Real mutl = parm.C_s*Tl*sqrt(Tl)/(Tl+parm.T_s);
    Real mutr = parm.C_s*Tr*sqrt(Tr)/(Tr+parm.T_s);
    // derive k
    Real ktl = mutl * parm.cv * parm.eos_gamma * parm.Prinv;
    Real ktr = mutr * parm.cv * parm.eos_gamma * parm.Prinv;

    Real dTdz = (Tl-Tr)*dxinv[2];
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