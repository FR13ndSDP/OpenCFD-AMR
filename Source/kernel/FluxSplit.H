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
void flux_split_x(int i, int j, int k, 
                  amrex::Array4<amrex::Real> const &fp,
                  amrex::Array4<amrex::Real> const &fm,
                  amrex::Array4<amrex::Real const> const &q,
                  amrex::Array4<amrex::Real const> const &state,
                  Parm const &parm) noexcept {
  using amrex::Real;
  Real d = q(i, j, k, QRHO);
  Real u = q(i, j, k, QU);
  Real v = q(i, j, k, QV);
  Real w = q(i, j, k, QW);
  Real c = q(i, j, k, QC);
#ifndef CHEM
  Real gamma = parm.eos_gamma;
#else
  Real gamma = q(i,j,k,QGAMA);
#endif

  Real tmp1 = Real(1.0) / (Real(2.0) * gamma);
  Real tmp2 = Real(2.0) * (gamma - Real(1.0));
  Real tmp3 =
      (Real(3.0) - gamma) / (Real(2.0) * (gamma - Real(1.0)));


  Real uc1 = u - c;
  Real uc2 = u + c;
  Real vvc1 = 0.5 * (uc1 * uc1 + v * v + w * w);
  Real vvc2 = 0.5 * (uc2 * uc2 + v * v + w * w);
  Real v2 = (gamma - 1) * (u * u + v * v + w * w);
  Real w2 = tmp3 * c * c;

  // eigenvalues---------
  Real E1 = u;
  Real E2 = u - c;
  Real E3 = u + c;
  Real E1P = (E1 + amrex::Math::abs(E1)) * Real(0.5);
  Real E2P = (E2 + amrex::Math::abs(E2)) * Real(0.5);
  Real E3P = (E3 + amrex::Math::abs(E3)) * Real(0.5);

  Real E1M = E1 - E1P;
  Real E2M = E2 - E2P;
  Real E3M = E3 - E3P;

  Real tmp0 = d * tmp1;

  fp(i, j, k, URHO) = tmp0 * (tmp2 * E1P + E2P + E3P);
  fp(i, j, k, UMX) = tmp0 * (tmp2 * E1P * u + E2P * uc1 + E3P * uc2);
  fp(i, j, k, UMY) = tmp0 * v * (tmp2 * E1P + E2P + E3P);
  fp(i, j, k, UMZ) = tmp0 * w * (tmp2 * E1P + E2P + E3P);
  fp(i, j, k, UEDEN) =
      tmp0 * (E1P * v2 + E2P * vvc1 + E3P * vvc2 + w2 * (E2P + E3P));

  fm(i, j, k, URHO) = tmp0 * (tmp2 * E1M + E2M + E3M);
  fm(i, j, k, UMX) = tmp0 * (tmp2 * E1M * u + E2M * uc1 + E3M * uc2);
  fm(i, j, k, UMY) = tmp0 * v * (tmp2 * E1M + E2M + E3M);
  fm(i, j, k, UMZ) = tmp0 * w * (tmp2 * E1M + E2M + E3M);
  fm(i, j, k, UEDEN) =
      tmp0 * (E1M * v2 + E2M * vvc1 + E3M * vvc2 + w2 * (E2M + E3M));
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void flux_split_y(int i, int j, int k, 
                  amrex::Array4<amrex::Real> const &fp,
                  amrex::Array4<amrex::Real> const &fm,
                  amrex::Array4<amrex::Real const> const &q,
                  amrex::Array4<amrex::Real const> const &state,
                  Parm const &parm) noexcept {
  using amrex::Real;
  Real d = q(i, j, k, QRHO);
  Real u = q(i, j, k, QU);
  Real v = q(i, j, k, QV);
  Real w = q(i, j, k, QW);
  Real c = q(i, j, k, QC);
#ifndef CHEM
  Real gamma = parm.eos_gamma;
#else
  Real gamma = q(i,j,k,QGAMA);
#endif

  Real tmp1 = Real(1.0) / (Real(2.0) * gamma);
  Real tmp2 = Real(2.0) * (gamma - Real(1.0));
  Real tmp3 =
      (Real(3.0) - gamma) / (Real(2.0) * (gamma - Real(1.0)));

  Real vc1 = v - c;
  Real vc2 = v + c;
  Real vvc1 = 0.5 * (u * u + vc1 * vc1 + w * w);
  Real vvc2 = 0.5 * (u * u + vc2 * vc2 + w * w);
  Real v2 = (gamma - 1) * (u * u + v * v + w * w);
  Real w2 = tmp3 * c * c;

  // eigenvalues---------
  Real E1 = v;
  Real E2 = v - c;
  Real E3 = v + c;
  Real E1P = (E1 + amrex::Math::abs(E1)) * Real(0.5);
  Real E2P = (E2 + amrex::Math::abs(E2)) * Real(0.5);
  Real E3P = (E3 + amrex::Math::abs(E3)) * Real(0.5);

  Real E1M = E1 - E1P;
  Real E2M = E2 - E2P;
  Real E3M = E3 - E3P;

  Real tmp0 = d * tmp1;

  fp(i, j, k, URHO) = tmp0 * (tmp2 * E1P + E2P + E3P);
  fp(i, j, k, UMX) = tmp0 * u * (tmp2 * E1P + E2P + E3P);
  fp(i, j, k, UMY) = tmp0 * (tmp2 * E1P * v + E2P * vc1 + E3P * vc2);
  fp(i, j, k, UMZ) = tmp0 * w * (tmp2 * E1P + E2P + E3P);
  fp(i, j, k, UEDEN) =
      tmp0 * (E1P * v2 + E2P * vvc1 + E3P * vvc2 + w2 * (E2P + E3P));

  fm(i, j, k, URHO) = tmp0 * (tmp2 * E1M + E2M + E3M);
  fm(i, j, k, UMX) = tmp0 * u * (tmp2 * E1M + E2M + E3M);
  fm(i, j, k, UMY) = tmp0 * (tmp2 * E1M * v + E2M * vc1 + E3M * vc2);
  fm(i, j, k, UMZ) = tmp0 * w * (tmp2 * E1M + E2M + E3M);
  fm(i, j, k, UEDEN) =
      tmp0 * (E1M * v2 + E2M * vvc1 + E3M * vvc2 + w2 * (E2M + E3M));
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void flux_split_z(int i, int j, int k, 
                  amrex::Array4<amrex::Real> const &fp,
                  amrex::Array4<amrex::Real> const &fm,
                  amrex::Array4<amrex::Real const> const &q,
                  amrex::Array4<amrex::Real const> const &state,
                  Parm const &parm) noexcept {
  using amrex::Real;
  Real d = q(i, j, k, QRHO);
  Real u = q(i, j, k, QU);
  Real v = q(i, j, k, QV);
  Real w = q(i, j, k, QW);
  Real c = q(i, j, k, QC);
#ifndef CHEM
  Real gamma = parm.eos_gamma;
#else
  Real gamma = q(i,j,k,QGAMA);
#endif

  Real tmp1 = Real(1.0) / (Real(2.0) * gamma);
  Real tmp2 = Real(2.0) * (gamma - Real(1.0));
  Real tmp3 =
      (Real(3.0) - gamma) / (Real(2.0) * (gamma - Real(1.0)));

  Real wc1 = w - c;
  Real wc2 = w + c;
  Real vvc1 = 0.5 * (u * u + v * v + wc1 * wc1);
  Real vvc2 = 0.5 * (u * u + v * v + wc2 * wc2);
  Real v2 = (gamma - 1) * (u * u + v * v + w * w);
  Real w2 = tmp3 * c * c;

  // eigenvalues---------
  Real E1 = w;
  Real E2 = w - c;
  Real E3 = w + c;
  Real E1P = (E1 + amrex::Math::abs(E1)) * Real(0.5);
  Real E2P = (E2 + amrex::Math::abs(E2)) * Real(0.5);
  Real E3P = (E3 + amrex::Math::abs(E3)) * Real(0.5);

  Real E1M = E1 - E1P;
  Real E2M = E2 - E2P;
  Real E3M = E3 - E3P;

  Real tmp0 = d * tmp1;

  fp(i, j, k, URHO) = tmp0 * (tmp2 * E1P + E2P + E3P);
  fp(i, j, k, UMX) = tmp0 * u * (tmp2 * E1P + E2P + E3P);
  fp(i, j, k, UMY) = tmp0 * v * (tmp2 * E1P + E2P + E3P);
  fp(i, j, k, UMZ) = tmp0 * (tmp2 * E1P * w + E2P * wc1 + E3P * wc2);
  fp(i, j, k, UEDEN) =
      tmp0 * (E1P * v2 + E2P * vvc1 + E3P * vvc2 + w2 * (E2P + E3P));

  fm(i, j, k, URHO) = tmp0 * (tmp2 * E1M + E2M + E3M);
  fm(i, j, k, UMX) = tmp0 * u * (tmp2 * E1M + E2M + E3M);
  fm(i, j, k, UMY) = tmp0 * v * (tmp2 * E1M + E2M + E3M);
  fm(i, j, k, UMZ) = tmp0 * (tmp2 * E1M * w + E2M * wc1 + E3M * wc2);
  fm(i, j, k, UEDEN) =
      tmp0 * (E1M * v2 + E2M * vvc1 + E3M * vvc2 + w2 * (E2M + E3M));
}

/*
// Local Lax Friedrichs
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void flux_split_x(int i, int j, int k, 
                  amrex::Array4<amrex::Real> const &fp,
                  amrex::Array4<amrex::Real> const &fm,
                  amrex::Array4<amrex::Real const> const &q,
                  amrex::Array4<amrex::Real const> const &state,
                  Parm const &parm) noexcept {
  using amrex::Real;
  Real d = q(i, j, k, QRHO);
  Real u = q(i, j, k, QU);
  Real v = q(i, j, k, QV);
  Real w = q(i, j, k, QW);
  Real c = q(i, j, k, QC);
  Real p = q(i, j, k, QPRES);

  Real un = u;
  Real lambda0 = amrex::Math::abs(un) + c;
  Real E0 = state(i,j,k,UEDEN);
  
  fp(i, j, k, URHO) = Real(0.5)*(d*un + lambda0*d);
  fp(i, j, k, UMX) = Real(0.5)*(d*u*un + lambda0*d*u + p);
  fp(i, j, k, UMY) = Real(0.5)*(d*v*un + lambda0*d*v);
  fp(i, j, k, UMZ) = Real(0.5)*(d*w*un + lambda0*d*w);
  fp(i, j, k, UEDEN) = Real(0.5)*((E0+p)*un + lambda0*E0);

  fm(i, j, k, URHO) = Real(0.5)*(d*un - lambda0*d);
  fm(i, j, k, UMX) = Real(0.5)*(d*u*un - lambda0*d*u + p);
  fm(i, j, k, UMY) = Real(0.5)*(d*v*un - lambda0*d*v);
  fm(i, j, k, UMZ) = Real(0.5)*(d*w*un - lambda0*d*w);
  fm(i, j, k, UEDEN) = Real(0.5)*((E0+p)*un - lambda0*E0);
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void flux_split_y(int i, int j, int k, 
                  amrex::Array4<amrex::Real> const &fp,
                  amrex::Array4<amrex::Real> const &fm,
                  amrex::Array4<amrex::Real const> const &q,
                  amrex::Array4<amrex::Real const> const &state,
                  Parm const &parm) noexcept {
  using amrex::Real;
  Real d = q(i, j, k, QRHO);
  Real u = q(i, j, k, QU);
  Real v = q(i, j, k, QV);
  Real w = q(i, j, k, QW);
  Real c = q(i, j, k, QC);
  Real p = q(i, j, k, QPRES);

  Real un = v;
  Real lambda0 = amrex::Math::abs(un) + c;
  Real E0 = state(i,j,k,UEDEN);

  fp(i, j, k, URHO) = Real(0.5)*(d*un + lambda0*d);
  fp(i, j, k, UMX) = Real(0.5)*(d*u*un + lambda0*d*u);
  fp(i, j, k, UMY) = Real(0.5)*(d*v*un + lambda0*d*v + p);
  fp(i, j, k, UMZ) = Real(0.5)*(d*w*un + lambda0*d*w);
  fp(i, j, k, UEDEN) = Real(0.5)*((E0+p)*un + lambda0*E0);

  fm(i, j, k, URHO) = Real(0.5)*(d*un - lambda0*d);
  fm(i, j, k, UMX) = Real(0.5)*(d*u*un - lambda0*d*u);
  fm(i, j, k, UMY) = Real(0.5)*(d*v*un - lambda0*d*v + p);
  fm(i, j, k, UMZ) = Real(0.5)*(d*w*un - lambda0*d*w);
  fm(i, j, k, UEDEN) = Real(0.5)*((E0+p)*un - lambda0*E0);
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void flux_split_z(int i, int j, int k, 
                  amrex::Array4<amrex::Real> const &fp,
                  amrex::Array4<amrex::Real> const &fm,
                  amrex::Array4<amrex::Real const> const &q,
                  amrex::Array4<amrex::Real const> const &state,
                  Parm const &parm) noexcept {
  using amrex::Real;
  Real d = q(i, j, k, QRHO);
  Real u = q(i, j, k, QU);
  Real v = q(i, j, k, QV);
  Real w = q(i, j, k, QW);
  Real c = q(i, j, k, QC);
  Real p = q(i, j, k, QPRES);

  Real un = w;
  Real lambda0 = amrex::Math::abs(un) + c;
  Real E0 = state(i,j,k,UEDEN);

  fp(i, j, k, URHO) = Real(0.5)*(d*un + lambda0*d);
  fp(i, j, k, UMX) = Real(0.5)*(d*u*un + lambda0*d*u);
  fp(i, j, k, UMY) = Real(0.5)*(d*v*un + lambda0*d*v);
  fp(i, j, k, UMZ) = Real(0.5)*(d*w*un + lambda0*d*w + p);
  fp(i, j, k, UEDEN) = Real(0.5)*((E0+p)*un + lambda0*E0);

  fm(i, j, k, URHO) = Real(0.5)*(d*un - lambda0*d);
  fm(i, j, k, UMX) = Real(0.5)*(d*u*un - lambda0*d*u);
  fm(i, j, k, UMY) = Real(0.5)*(d*v*un - lambda0*d*v);
  fm(i, j, k, UMZ) = Real(0.5)*(d*w*un - lambda0*d*w + p);
  fm(i, j, k, UEDEN) = Real(0.5)*((E0+p)*un - lambda0*E0);
}
*/
#endif