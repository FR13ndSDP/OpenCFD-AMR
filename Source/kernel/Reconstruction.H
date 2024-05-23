#ifndef RECONSTRUCTION_H
#define RECONSTRUCTION_H

#include <EBR.H>
#include <Constants.H>

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

// 5th order WENO-Z
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
reconstruction_x (int i, int j, int k, int n,
             amrex::Array4<amrex::Real const> const& fp,
             amrex::Array4<amrex::Real const> const& fm,
             amrex::Array4<amrex::Real> const& fxfab,
             Parm const& parm) noexcept
{
    using amrex::Real;
    Real fpx,fmx;

    Real eps = 1e-16;

    Real V1 = fp(i-3,j,k,n);
    Real V2 = fp(i-2,j,k,n);
    Real V3 = fp(i-1,j,k,n);
    Real V4 = fp(i,j,k,n);
    Real V5 = fp(i+1,j,k,n);

    // qL
    Real s11 = (13.0/12.0)*(V1-2.0*V2+V3)*(V1-2.0*V2+V3) + 0.25*(V1-4.0*V2+3.0*V3)*(V1-4.0*V2+3.0*V3);
    Real s22 = (13.0/12.0)*(V2-2.0*V3+V4)*(V2-2.0*V3+V4) + 0.25*(V2-V4)*(V2-V4);
    Real s33 = (13.0/12.0)*(V3-2.0*V4+V5)*(V3-2.0*V4+V5) + 0.25*(3.0*V3-4.0*V4+V5)*(3.0*V3-4.0*V4+V5);

    Real s55 = amrex::Math::abs(s11-s33);

    s11 = 1.0 + (s55/(eps+s11))*(s55/(eps+s11));
    s22 = 1.0 + (s55/(eps+s22))*(s55/(eps+s22));
    s33 = 1.0 + (s55/(eps+s33))*(s55/(eps+s33));

    Real a1 = s11;
    Real a2 = 6.0*s22;
    Real a3 = 3.0*s33;
    Real invsum = 1.0/(a1+a2+a3);

    Real v1 = 2.0*V1-7.0*V2+11.0*V3;
    Real v2 = -V2+5.0*V3+2.0*V4;
    Real v3 = 2.0*V3+5.0*V4-V5;
    fpx = (1.0/6.0)*invsum*(a1*v1+a2*v2+a3*v3);

    V1 = fm(i-2,j,k,n);
    V2 = fm(i-1,j,k,n);
    V3 = fm(i,j,k,n);
    V4 = fm(i+1,j,k,n);
    V5 = fm(i+2,j,k,n);

    // qR
    s11 = (13.0/12.0)*(V5-2.0*V4+V3)*(V5-2.0*V4+V3) + 0.25*(V5-4.0*V4+3.0*V3)*(V5-4.0*V4+3.0*V3);
    s22 = (13.0/12.0)*(V2-2.0*V3+V4)*(V2-2.0*V3+V4) + 0.25*(V4-V2)*(V4-V2);
    s33 = (13.0/12.0)*(V3-2.0*V2+V1)*(V3-2.0*V2+V1) + 0.25*(3.0*V3-4.0*V2+V1)*(3.0*V3-4.0*V2+V1);

    s55 = amrex::Math::abs(s11-s33);

    s11 = 1.0 + (s55/(eps+s11))*(s55/(eps+s11));
    s22 = 1.0 + (s55/(eps+s22))*(s55/(eps+s22));
    s33 = 1.0 + (s55/(eps+s33))*(s55/(eps+s33));

    a1 = s11;
    a2 = 6.0*s22;
    a3 = 3.0*s33;
    invsum = 1.0/(a1+a2+a3);

    v1 = 11.0*V3-7.0*V4+2.0*V5;
    v2 = -V4+5.0*V3+2.0*V2;
    v3 = 2.0*V3+5.0*V2-V1;
    fmx = (1.0/6.0)*invsum*(a1*v1+a2*v2+a3*v3);
    fxfab(i,j,k,n) = fpx+fmx;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
reconstruction_y (int i, int j, int k, int n,
             amrex::Array4<amrex::Real const> const& fp,
             amrex::Array4<amrex::Real const> const& fm,
             amrex::Array4<amrex::Real> const& fyfab,
             Parm const& parm) noexcept
{
    using amrex::Real;
    Real fpy,fmy;

    Real eps = 1e-16;

    Real V1 = fp(i,j-3,k,n);
    Real V2 = fp(i,j-2,k,n);
    Real V3 = fp(i,j-1,k,n);
    Real V4 = fp(i,j,k,n);
    Real V5 = fp(i,j+1,k,n);

    // qL
    Real s11 = (13.0/12.0)*(V1-2.0*V2+V3)*(V1-2.0*V2+V3) + 0.25*(V1-4.0*V2+3.0*V3)*(V1-4.0*V2+3.0*V3);
    Real s22 = (13.0/12.0)*(V2-2.0*V3+V4)*(V2-2.0*V3+V4) + 0.25*(V2-V4)*(V2-V4);
    Real s33 = (13.0/12.0)*(V3-2.0*V4+V5)*(V3-2.0*V4+V5) + 0.25*(3.0*V3-4.0*V4+V5)*(3.0*V3-4.0*V4+V5);

    Real s55 = amrex::Math::abs(s11-s33);

    s11 = 1.0 + (s55/(eps+s11))*(s55/(eps+s11));
    s22 = 1.0 + (s55/(eps+s22))*(s55/(eps+s22));
    s33 = 1.0 + (s55/(eps+s33))*(s55/(eps+s33));

    Real a1 = s11;
    Real a2 = 6.0*s22;
    Real a3 = 3.0*s33;
    Real invsum = 1.0/(a1+a2+a3);

    Real v1 = 2.0*V1-7.0*V2+11.0*V3;
    Real v2 = -V2+5.0*V3+2.0*V4;
    Real v3 = 2.0*V3+5.0*V4-V5;
    fpy = (1.0/6.0)*invsum*(a1*v1+a2*v2+a3*v3);

    V1 = fm(i,j-2,k,n);
    V2 = fm(i,j-1,k,n);
    V3 = fm(i,j,k,n);
    V4 = fm(i,j+1,k,n);
    V5 = fm(i,j+2,k,n);

    // qR
    s11 = (13.0/12.0)*(V5-2.0*V4+V3)*(V5-2.0*V4+V3) + 0.25*(V5-4.0*V4+3.0*V3)*(V5-4.0*V4+3.0*V3);
    s22 = (13.0/12.0)*(V2-2.0*V3+V4)*(V2-2.0*V3+V4) + 0.25*(V4-V2)*(V4-V2);
    s33 = (13.0/12.0)*(V3-2.0*V2+V1)*(V3-2.0*V2+V1) + 0.25*(3.0*V3-4.0*V2+V1)*(3.0*V3-4.0*V2+V1);

    s55 = amrex::Math::abs(s11-s33);

    s11 = 1.0 + (s55/(eps+s11))*(s55/(eps+s11));
    s22 = 1.0 + (s55/(eps+s22))*(s55/(eps+s22));
    s33 = 1.0 + (s55/(eps+s33))*(s55/(eps+s33));

    a1 = s11;
    a2 = 6.0*s22;
    a3 = 3.0*s33;
    invsum = 1.0/(a1+a2+a3);

    v1 = 11.0*V3-7.0*V4+2.0*V5;
    v2 = -V4+5.0*V3+2.0*V2;
    v3 = 2.0*V3+5.0*V2-V1;
    fmy = (1.0/6.0)*invsum*(a1*v1+a2*v2+a3*v3);
    fyfab(i,j,k,n) = fpy+fmy;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
reconstruction_z (int i, int j, int k, int n,
             amrex::Array4<amrex::Real const> const& fp,
             amrex::Array4<amrex::Real const> const& fm,
             amrex::Array4<amrex::Real> const& fzfab,
             Parm const& parm) noexcept
{
    using amrex::Real;
    Real fpz,fmz;

    Real eps = 1e-16;

    Real V1 = fp(i,j,k-3,n);
    Real V2 = fp(i,j,k-2,n);
    Real V3 = fp(i,j,k-1,n);
    Real V4 = fp(i,j,k,n);
    Real V5 = fp(i,j,k+1,n);

    // qL
    Real s11 = (13.0/12.0)*(V1-2.0*V2+V3)*(V1-2.0*V2+V3) + 0.25*(V1-4.0*V2+3.0*V3)*(V1-4.0*V2+3.0*V3);
    Real s22 = (13.0/12.0)*(V2-2.0*V3+V4)*(V2-2.0*V3+V4) + 0.25*(V2-V4)*(V2-V4);
    Real s33 = (13.0/12.0)*(V3-2.0*V4+V5)*(V3-2.0*V4+V5) + 0.25*(3.0*V3-4.0*V4+V5)*(3.0*V3-4.0*V4+V5);

    Real s55 = amrex::Math::abs(s11-s33);

    s11 = 1.0 + (s55/(eps+s11))*(s55/(eps+s11));
    s22 = 1.0 + (s55/(eps+s22))*(s55/(eps+s22));
    s33 = 1.0 + (s55/(eps+s33))*(s55/(eps+s33));

    Real a1 = s11;
    Real a2 = 6.0*s22;
    Real a3 = 3.0*s33;
    Real invsum = 1.0/(a1+a2+a3);

    Real v1 = 2.0*V1-7.0*V2+11.0*V3;
    Real v2 = -V2+5.0*V3+2.0*V4;
    Real v3 = 2.0*V3+5.0*V4-V5;
    fpz = (1.0/6.0)*invsum*(a1*v1+a2*v2+a3*v3);

    V1 = fm(i,j,k-2,n);
    V2 = fm(i,j,k-1,n);
    V3 = fm(i,j,k,n);
    V4 = fm(i,j,k+1,n);
    V5 = fm(i,j,k+2,n);

    // qR
    s11 = (13.0/12.0)*(V5-2.0*V4+V3)*(V5-2.0*V4+V3) + 0.25*(V5-4.0*V4+3.0*V3)*(V5-4.0*V4+3.0*V3);
    s22 = (13.0/12.0)*(V2-2.0*V3+V4)*(V2-2.0*V3+V4) + 0.25*(V4-V2)*(V4-V2);
    s33 = (13.0/12.0)*(V3-2.0*V2+V1)*(V3-2.0*V2+V1) + 0.25*(3.0*V3-4.0*V2+V1)*(3.0*V3-4.0*V2+V1);

    s55 = amrex::Math::abs(s11-s33);

    s11 = 1.0 + (s55/(eps+s11))*(s55/(eps+s11));
    s22 = 1.0 + (s55/(eps+s22))*(s55/(eps+s22));
    s33 = 1.0 + (s55/(eps+s33))*(s55/(eps+s33));

    a1 = s11;
    a2 = 6.0*s22;
    a3 = 3.0*s33;
    invsum = 1.0/(a1+a2+a3);

    v1 = 11.0*V3-7.0*V4+2.0*V5;
    v2 = -V4+5.0*V3+2.0*V2;
    v3 = 2.0*V3+5.0*V2-V1;
    fmz = (1.0/6.0)*invsum*(a1*v1+a2*v2+a3*v3);
    fzfab(i,j,k,n) = fpz+fmz;
}
#endif