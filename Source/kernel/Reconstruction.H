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

// // 2nd order NND, we only need the left state, for qr(i) = ql(i+1)
// AMREX_GPU_DEVICE
// AMREX_FORCE_INLINE
// void
// reconstruction_x (int i, int j, int k, int n,
//              amrex::Array4<amrex::Real const> const& ql,
//              amrex::Array4<amrex::Real const> const& qr,
//              amrex::Array4<amrex::Real> const& fxfab,
//              Parm const& parm) noexcept
// {
//     Real fp, fm;

//     fp = ql(i-1,j,k,n) + amrex::Real(0.5)*minmod(ql(i,j,k,n)-ql(i-1,j,k,n), ql(i-1,j,k,n)-ql(i-2,j,k,n));
//     fm = qr(i,j,k,n) - amrex::Real(0.5)*minmod(qr(i+1,j,k,n)-qr(i,j,k,n), qr(i,j,k,n)-qr(i-1,j,k,n));
//     fxfab(i,j,k,n) = fp+fm;
// }

// AMREX_GPU_DEVICE
// AMREX_FORCE_INLINE
// void
// reconstruction_y (int i, int j, int k, int n,
//              amrex::Array4<amrex::Real const> const& ql,
//              amrex::Array4<amrex::Real const> const& qr,
//              amrex::Array4<amrex::Real> const& fyfab,
//              Parm const& parm) noexcept
// {
//     Real fp, fm;

//     fp = ql(i,j-1,k,n) + amrex::Real(0.5)*minmod(ql(i,j,k,n)-ql(i,j-1,k,n), ql(i,j-1,k,n)-ql(i,j-2,k,n));
//     fm = qr(i,j,k,n) - amrex::Real(0.5)*minmod(qr(i,j+1,k,n)-qr(i,j,k,n), qr(i,j,k,n)-qr(i,j-1,k,n));
//     fyfab(i,j,k,n) = fp+fm;
// }

// AMREX_GPU_DEVICE
// AMREX_FORCE_INLINE
// void
// reconstruction_z (int i, int j, int k, int n,
//              amrex::Array4<amrex::Real const> const& ql,
//              amrex::Array4<amrex::Real const> const& qr,
//              amrex::Array4<amrex::Real> const& fzfab,
//              Parm const& parm) noexcept
// {
//     Real fp, fm;

//     fp = ql(i,j,k-1,n) + amrex::Real(0.5)*minmod(ql(i,j,k,n)-ql(i,j,k-1,n), ql(i,j,k-1,n)-ql(i,j,k-2,n));
//     fm = qr(i,j,k,n) - amrex::Real(0.5)*minmod(qr(i,j,k+1,n)-qr(i,j,k,n), qr(i,j,k,n)-qr(i,j,k-1,n));
//     fzfab(i,j,k,n) = fp+fm;
// }

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

    Real eps = 1e-6;

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

    Real eps = 1e-6;

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

    Real eps = 1e-6;

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

// // 5th order TENO
// AMREX_GPU_DEVICE
// AMREX_FORCE_INLINE
// void
// reconstruction_x (int i, int j, int k, int n,
//              amrex::Array4<amrex::Real const> const& ql,
//              amrex::Array4<amrex::Real const> const& qr,
//              amrex::Array4<amrex::Real> const& fxfab,
//              Parm const& parm) noexcept
// {
//     using amrex::Real;
//     Real fp,fm;

//     Real V1 = ql(i-3,j,k,n);
//     Real V2 = ql(i-2,j,k,n);
//     Real V3 = ql(i-1,j,k,n);
//     Real V4 = ql(i,j,k,n);
//     Real V5 = ql(i+1,j,k,n);
//     Real V6 = ql(i+2,j,k,n);

//     // qL
//     Real s11 = 13.0*(V1-2.0*V2+V3)*(V1-2.0*V2+V3) + 3.0*(V1-4.0*V2+3.0*V3)*(V1-4.0*V2+3.0*V3);
//     Real s22 = 13.0*(V2-2.0*V3+V4)*(V2-2.0*V3+V4) + 3.0*(V2-V4)*(V2-V4);
//     Real s33 = 13.0*(V3-2.0*V4+V5)*(V3-2.0*V4+V5) + 3.0*(3.0*V3-4.0*V4+V5)*(3.0*V3-4.0*V4+V5);

//     Real s55 = amrex::Math::abs(s11-s33);

//     // get rid of pow here, calculate (1+x)^6
//     Real a1 = 1.0+s55/(s11+parm.eps);
//     Real a2 = 1.0+s55/(s22+parm.eps);
//     Real a3 = 1.0+s55/(s33+parm.eps);
//     a1 = a1*a1;
//     a1 = a1*a1*a1;
//     a2 = a2*a2;
//     a2 = a2*a2*a2;
//     a3 = a3*a3;
//     a3 = a3*a3*a3;

//     Real invsum = 1.0/(a1+a2+a3);
//     Real b1 = a1*invsum;
//     Real b2 = a2*invsum;
//     Real b3 = a3*invsum;


//     a1 = 0.1*(b1<1.e-5? 0.0:1.0);
//     a2 = 0.6*(b2<1.e-5? 0.0:1.0);
//     a3 = 0.3*(b3<1.e-5? 0.0:1.0);

//     Real v1 = parm.oneSix*(2.0*V1-7.0*V2+5.0*V3);
//     Real v2 = parm.oneSix*(-V2-V3+2.0*V4);
//     Real v3 = parm.oneSix*(-4.0*V3+5.0*V4-V5);

//     invsum = 1.0/(a1+a2+a3);
//     Real w1 = a1*invsum;
//     Real w2 = a2*invsum;
//     Real w3 = a3*invsum;

//     fp = V3+w1*v1+w2*v2+w3*v3;

//     V1 = qr(i-3,j,k,n);
//     V2 = qr(i-2,j,k,n);
//     V3 = qr(i-1,j,k,n);
//     V4 = qr(i,j,k,n);
//     V5 = qr(i+1,j,k,n);
//     V6 = qr(i+2,j,k,n);

//     // qR
//     s11 = 13.0*(V6-2.0*V5+V4)*(V6-2.0*V5+V4) + 3.0*(V6-4.0*V5+3.0*V4)*(V6-4.0*V5+3.0*V4);
//     s22 = 13.0*(V3-2.0*V4+V5)*(V3-2.0*V4+V5) + 3.0*(V5-V3)*(V5-V3);
//     s33 = 13.0*(V4-2.0*V3+V2)*(V4-2.0*V3+V2) + 3.0*(3.0*V4-4.0*V3+V2)*(3.0*V4-4.0*V3+V2);

//     s55 = amrex::Math::abs(s11-s33);

//     // get rid of pow here, calculate (1+x)^6
//     a1 = 1.0+s55/(s11+parm.eps);
//     a2 = 1.0+s55/(s22+parm.eps);
//     a3 = 1.0+s55/(s33+parm.eps);
//     a1 = a1*a1;
//     a1 = a1*a1*a1;
//     a2 = a2*a2;
//     a2 = a2*a2*a2;
//     a3 = a3*a3;
//     a3 = a3*a3*a3;

//     invsum = 1.0/(a1+a2+a3);
//     b1 = a1*invsum;
//     b2 = a2*invsum;
//     b3 = a3*invsum;

//     a1 = 0.1*(b1<1.e-5? 0.0:1.0);
//     a2 = 0.6*(b2<1.e-5? 0.0:1.0);
//     a3 = 0.3*(b3<1.e-5? 0.0:1.0);

//     v1 = parm.oneSix*(2.0*V6-7.0*V5+5.0*V4);
//     v2 = parm.oneSix*(-V5-V4+2.0*V3);
//     v3 = parm.oneSix*(-4.0*V4+5.0*V3-V2);

//     invsum = 1.0/(a1+a2+a3);
//     w1 = a1*invsum;
//     w2 = a2*invsum;
//     w3 = a3*invsum;

//     fm = V4+w1*v1+w2*v2+w3*v3;
//     fxfab(i,j,k,n) = fp+fm;

// }

// AMREX_GPU_DEVICE
// AMREX_FORCE_INLINE
// void
// reconstruction_y (int i, int j, int k, int n,
//              amrex::Array4<amrex::Real const> const& ql,
//              amrex::Array4<amrex::Real const> const& qr,
//              amrex::Array4<amrex::Real> const& fyfab,
//              Parm const& parm) noexcept
// {
//     using amrex::Real;
//     Real fp,fm;

//     Real V1 = ql(i,j-3,k,n);
//     Real V2 = ql(i,j-2,k,n);
//     Real V3 = ql(i,j-1,k,n);
//     Real V4 = ql(i,j,k,n);
//     Real V5 = ql(i,j+1,k,n);
//     Real V6 = ql(i,j+2,k,n);

//     // qL
//     Real s11 = 13.0*(V1-2.0*V2+V3)*(V1-2.0*V2+V3) + 3.0*(V1-4.0*V2+3.0*V3)*(V1-4.0*V2+3.0*V3);
//     Real s22 = 13.0*(V2-2.0*V3+V4)*(V2-2.0*V3+V4) + 3.0*(V2-V4)*(V2-V4);
//     Real s33 = 13.0*(V3-2.0*V4+V5)*(V3-2.0*V4+V5) + 3.0*(3.0*V3-4.0*V4+V5)*(3.0*V3-4.0*V4+V5);

//     Real s55 = amrex::Math::abs(s11-s33);

//     // get rid of pow here, calculate (1+x)^6
//     Real a1 = 1.0+s55/(s11+parm.eps);
//     Real a2 = 1.0+s55/(s22+parm.eps);
//     Real a3 = 1.0+s55/(s33+parm.eps);
//     a1 = a1*a1;
//     a1 = a1*a1*a1;
//     a2 = a2*a2;
//     a2 = a2*a2*a2;
//     a3 = a3*a3;
//     a3 = a3*a3*a3;

//     Real invsum = 1.0/(a1+a2+a3);
//     Real b1 = a1*invsum;
//     Real b2 = a2*invsum;
//     Real b3 = a3*invsum;


//     a1 = 0.1*(b1<1.e-5? 0.0:1.0);
//     a2 = 0.6*(b2<1.e-5? 0.0:1.0);
//     a3 = 0.3*(b3<1.e-5? 0.0:1.0);

//     Real v1 = parm.oneSix*(2.0*V1-7.0*V2+5.0*V3);
//     Real v2 = parm.oneSix*(-V2-V3+2.0*V4);
//     Real v3 = parm.oneSix*(-4.0*V3+5.0*V4-V5);

//     invsum = 1.0/(a1+a2+a3);
//     Real w1 = a1*invsum;
//     Real w2 = a2*invsum;
//     Real w3 = a3*invsum;

//     fp = V3+w1*v1+w2*v2+w3*v3;

//     V1 = qr(i,j-3,k,n);
//     V2 = qr(i,j-2,k,n);
//     V3 = qr(i,j-1,k,n);
//     V4 = qr(i,j,k,n);
//     V5 = qr(i,j+1,k,n);
//     V6 = qr(i,j+2,k,n);

//     // qR
//     s11 = 13.0*(V6-2.0*V5+V4)*(V6-2.0*V5+V4) + 3.0*(V6-4.0*V5+3.0*V4)*(V6-4.0*V5+3.0*V4);
//     s22 = 13.0*(V3-2.0*V4+V5)*(V3-2.0*V4+V5) + 3.0*(V5-V3)*(V5-V3);
//     s33 = 13.0*(V4-2.0*V3+V2)*(V4-2.0*V3+V2) + 3.0*(3.0*V4-4.0*V3+V2)*(3.0*V4-4.0*V3+V2);

//     s55 = amrex::Math::abs(s11-s33);

//     // get rid of pow here, calculate (1+x)^6
//     a1 = 1.0+s55/(s11+parm.eps);
//     a2 = 1.0+s55/(s22+parm.eps);
//     a3 = 1.0+s55/(s33+parm.eps);
//     a1 = a1*a1;
//     a1 = a1*a1*a1;
//     a2 = a2*a2;
//     a2 = a2*a2*a2;
//     a3 = a3*a3;
//     a3 = a3*a3*a3;

//     invsum = 1.0/(a1+a2+a3);
//     b1 = a1*invsum;
//     b2 = a2*invsum;
//     b3 = a3*invsum;

//     a1 = 0.1*(b1<1.e-5? 0.0:1.0);
//     a2 = 0.6*(b2<1.e-5? 0.0:1.0);
//     a3 = 0.3*(b3<1.e-5? 0.0:1.0);

//     v1 = parm.oneSix*(2.0*V6-7.0*V5+5.0*V4);
//     v2 = parm.oneSix*(-V5-V4+2.0*V3);
//     v3 = parm.oneSix*(-4.0*V4+5.0*V3-V2);

//     invsum = 1.0/(a1+a2+a3);
//     w1 = a1*invsum;
//     w2 = a2*invsum;
//     w3 = a3*invsum;

//     fm = V4+w1*v1+w2*v2+w3*v3;
//     fyfab(i,j,k,n) = fp+fm;
// }

// AMREX_GPU_DEVICE
// AMREX_FORCE_INLINE
// void
// reconstruction_z (int i, int j, int k, int n,
//              amrex::Array4<amrex::Real const> const& ql,
//              amrex::Array4<amrex::Real const> const& qr,
//              amrex::Array4<amrex::Real> const& fzfab,
//              Parm const& parm) noexcept
// {
//     using amrex::Real;
//     Real fp,fm;

//     Real V1 = ql(i,j,k-3,n);
//     Real V2 = ql(i,j,k-2,n);
//     Real V3 = ql(i,j,k-1,n);
//     Real V4 = ql(i,j,k,n);
//     Real V5 = ql(i,j,k+1,n);
//     Real V6 = ql(i,j,k+2,n);

//     // qL
//     Real s11 = 13.0*(V1-2.0*V2+V3)*(V1-2.0*V2+V3) + 3.0*(V1-4.0*V2+3.0*V3)*(V1-4.0*V2+3.0*V3);
//     Real s22 = 13.0*(V2-2.0*V3+V4)*(V2-2.0*V3+V4) + 3.0*(V2-V4)*(V2-V4);
//     Real s33 = 13.0*(V3-2.0*V4+V5)*(V3-2.0*V4+V5) + 3.0*(3.0*V3-4.0*V4+V5)*(3.0*V3-4.0*V4+V5);

//     Real s55 = amrex::Math::abs(s11-s33);

//     // get rid of pow here, calculate (1+x)^6
//     Real a1 = 1.0+s55/(s11+parm.eps);
//     Real a2 = 1.0+s55/(s22+parm.eps);
//     Real a3 = 1.0+s55/(s33+parm.eps);
//     a1 = a1*a1;
//     a1 = a1*a1*a1;
//     a2 = a2*a2;
//     a2 = a2*a2*a2;
//     a3 = a3*a3;
//     a3 = a3*a3*a3;

//     Real invsum = 1.0/(a1+a2+a3);
//     Real b1 = a1*invsum;
//     Real b2 = a2*invsum;
//     Real b3 = a3*invsum;


//     a1 = 0.1*(b1<1.e-5? 0.0:1.0);
//     a2 = 0.6*(b2<1.e-5? 0.0:1.0);
//     a3 = 0.3*(b3<1.e-5? 0.0:1.0);

//     Real v1 = parm.oneSix*(2.0*V1-7.0*V2+5.0*V3);
//     Real v2 = parm.oneSix*(-V2-V3+2.0*V4);
//     Real v3 = parm.oneSix*(-4.0*V3+5.0*V4-V5);

//     invsum = 1.0/(a1+a2+a3);
//     Real w1 = a1*invsum;
//     Real w2 = a2*invsum;
//     Real w3 = a3*invsum;

//     fp = V3+w1*v1+w2*v2+w3*v3;

//     V1 = qr(i,j,k-3,n);
//     V2 = qr(i,j,k-2,n);
//     V3 = qr(i,j,k-1,n);
//     V4 = qr(i,j,k,n);
//     V5 = qr(i,j,k+1,n);
//     V6 = qr(i,j,k+2,n);

//     // qR
//     s11 = 13.0*(V6-2.0*V5+V4)*(V6-2.0*V5+V4) + 3.0*(V6-4.0*V5+3.0*V4)*(V6-4.0*V5+3.0*V4);
//     s22 = 13.0*(V3-2.0*V4+V5)*(V3-2.0*V4+V5) + 3.0*(V5-V3)*(V5-V3);
//     s33 = 13.0*(V4-2.0*V3+V2)*(V4-2.0*V3+V2) + 3.0*(3.0*V4-4.0*V3+V2)*(3.0*V4-4.0*V3+V2);

//     s55 = amrex::Math::abs(s11-s33);

//     // get rid of pow here, calculate (1+x)^6
//     a1 = 1.0+s55/(s11+parm.eps);
//     a2 = 1.0+s55/(s22+parm.eps);
//     a3 = 1.0+s55/(s33+parm.eps);
//     a1 = a1*a1;
//     a1 = a1*a1*a1;
//     a2 = a2*a2;
//     a2 = a2*a2*a2;
//     a3 = a3*a3;
//     a3 = a3*a3*a3;

//     invsum = 1.0/(a1+a2+a3);
//     b1 = a1*invsum;
//     b2 = a2*invsum;
//     b3 = a3*invsum;

//     a1 = 0.1*(b1<1.e-5? 0.0:1.0);
//     a2 = 0.6*(b2<1.e-5? 0.0:1.0);
//     a3 = 0.3*(b3<1.e-5? 0.0:1.0);

//     v1 = parm.oneSix*(2.0*V6-7.0*V5+5.0*V4);
//     v2 = parm.oneSix*(-V5-V4+2.0*V3);
//     v3 = parm.oneSix*(-4.0*V4+5.0*V3-V2);

//     invsum = 1.0/(a1+a2+a3);
//     w1 = a1*invsum;
//     w2 = a2*invsum;
//     w3 = a3*invsum;

//     fm = V4+w1*v1+w2*v2+w3*v3;
//     fzfab(i,j,k,n) = fp+fm;
// }

#endif