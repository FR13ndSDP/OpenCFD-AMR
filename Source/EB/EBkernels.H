#ifndef EBKERNELS_H
#define EBKERNELS_H

#include "IndexDefines.H"
#include "Reconstruction.H"
#include <Constants.H>
#include <EBR.H>

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void eb_recon_x(int i, int j, int k, amrex::Array4<amrex::Real> const &ql,
              amrex::Array4<amrex::Real> const &qr,
              amrex::Array4<amrex::Real const> const &q,
              amrex::Array4<amrex::EBCellFlag const> const &flag,
              Parm const &parm) noexcept {
  if (flag(i,j,k).isConnected(-1,0,0)) {
    for (int n = 0; n < NPRIM; ++n) {
      ql(i, j, k, n) = q(i - 1, j, k, n);
      qr(i, j, k, n) = q(i, j, k, n);
      if (flag(i, j, k).isConnected(-2, 0, 0)) {
          ql(i,j,k,n) = q(i-1,j,k,n) + amrex::Real(0.5)*minmod(q(i,j,k,n)-q(i-1,j,k,n), q(i-1,j,k,n)-q(i-2,j,k,n));
          if (flag(i,j,k).isConnected(1,0,0)) {
            qr(i,j,k,n) = q(i,j,k,n) + amrex::Real(0.5)*minmod(q(i+1,j,k,n)-q(i,j,k,n), q(i,j,k,n)-q(i-1,j,k,n));
          }
      }
    }
  }
} 
  
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void eb_recon_y(int i, int j, int k, amrex::Array4<amrex::Real> const &ql,
              amrex::Array4<amrex::Real> const &qr,
              amrex::Array4<amrex::Real const> const &q,
              amrex::Array4<amrex::EBCellFlag const> const &flag,
              Parm const &parm) noexcept {
  if (flag(i,j,k).isConnected(0,-1,0)) {
    for (int n = 0; n < NPRIM; ++n) {
      ql(i, j, k, n) = q(i, j-1, k, n);
      qr(i, j, k, n) = q(i, j, k, n);
      if (flag(i, j, k).isConnected(0, -2, 0)) {
          ql(i,j,k,n) = q(i,j-1,k,n) + amrex::Real(0.5)*minmod(q(i,j,k,n)-q(i,j-1,k,n), q(i,j-1,k,n)-q(i,j-2,k,n));
          if (flag(i,j,k).isConnected(0,1,0)) {
            qr(i,j,k,n) = q(i,j,k,n) + amrex::Real(0.5)*minmod(q(i,j+1,k,n)-q(i,j,k,n), q(i,j,k,n)-q(i,j-1,k,n));
          }
      }
    }
  }
} 
  
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void eb_recon_z(int i, int j, int k, amrex::Array4<amrex::Real> const &ql,
              amrex::Array4<amrex::Real> const &qr,
              amrex::Array4<amrex::Real const> const &q,
              amrex::Array4<amrex::EBCellFlag const> const &flag,
              Parm const &parm) noexcept {
  if (flag(i,j,k).isConnected(0,0,-1)) {
    for (int n = 0; n < NPRIM; ++n) {
      ql(i, j, k, n) = q(i, j, k-1, n);
      qr(i, j, k, n) = q(i, j, k, n);
      if (flag(i, j, k).isConnected(0, 0, -2)) {
          ql(i,j,k,n) = q(i,j,k-1,n) + amrex::Real(0.5)*minmod(q(i,j,k,n)-q(i,j,k-1,n), q(i,j,k-1,n)-q(i,j,k-2,n));
          if (flag(i,j,k).isConnected(0,0,1)) {
            qr(i,j,k,n) = q(i,j,k,n) + amrex::Real(0.5)*minmod(q(i,j,k+1,n)-q(i,j,k,n), q(i,j,k,n)-q(i,j,k-1,n));
          }
      }
    }
  }
}

// only account for conservative update
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE 
void eb_compute_div(
    int i, int j, int k, int n,
    Array4<Real> const &q, Array4<Real> const& dsdt_arr,
    Array4<Real> const &fx, Array4<Real> const &fy, Array4<Real> const &fz,
    Array4<EBCellFlag const> const &flag, Array4<Real const> const &vfrc,
    Array4<Real const> const &bcent,
    Array4<Real const> const &apx, Array4<Real const> const &apy, Array4<Real const> const &apz,
    Array4<Real const> const &fcx, Array4<Real const> const &fcy, Array4<Real const> const &fcz,
    GpuArray<Real, AMREX_SPACEDIM> const &dxinv, Parm const &parm) {

  Real tmp = dxinv[0]/vfrc(i,j,k);

  if (flag(i, j, k).isCovered()) {
    dsdt_arr(i, j, k, n) = Real(0.0);
  } else if (flag(i,j,k).isRegular()) {
    dsdt_arr(i, j, k, n) = dxinv[0] * (fx(i + 1, j, k, n) - fx(i, j, k, n)) +
                       dxinv[1] * (fy(i, j + 1, k, n) - fy(i, j, k, n)) +
                       dxinv[2] * (fz(i, j, k + 1, n) - fz(i, j, k, n));
  } else {
    dsdt_arr(i, j, k, n) = tmp *
        (apx(i + 1, j, k) * fx(i+1,j,k,n) - apx(i, j, k) * fx(i,j,k,n) +
         apy(i, j + 1, k) * fy(i,j+1,k,n) - apy(i, j, k) * fy(i,j,k,n) +
         apz(i, j, k + 1) * fz(i,j,k+1,n) - apz(i, j, k) * fz(i,j,k,n));

    GpuArray<Real, NCONS> flux_wall;

    // the slip wall flux
    flux_wall[n] = Real(0.0);

    flux_wall[UMX] = (apx(i,j,k)-apx(i+1,j,k))*q(i,j,k,QPRES);
    flux_wall[UMY] = (apy(i,j,k)-apy(i,j+1,k))*q(i,j,k,QPRES);
    flux_wall[UMZ] = (apz(i,j,k)-apz(i,j,k+1))*q(i,j,k,QPRES);

    // Here we assume dx == dy == dz
    dsdt_arr(i, j, k, n) += flux_wall[n] * tmp;

// TODO: GPU precision problem
// if (dsdt_arr(i,j,k,3) != 0.0) {
// #if AMREX_USE_GPU
//     AMREX_DEVICE_PRINTF("flux = %.12f, %.12f, %.12f -- \n", fx(i,j,k,3), fy(i,j,k,3), fz(i,j,k,3));
//     AMREX_DEVICE_PRINTF("flux+1 = %.12f, %.12f, %.12f -- \n", fx(i+1,j,k,3), fy(i,j+1,k,3), fz(i,j,k+1,3));

//     AMREX_DEVICE_PRINTF("dsdt = %.12f -- \n", dsdt_arr(i,j,k,3));
// #else
//     std::printf("flux = %.12f, %.12f, %.12f -- \n", fx(i,j,k,3), fy(i,j,k,3), fz(i,j,k,3));
//     std::printf("flux+1 = %.12f, %.12f, %.12f -- \n", fx(i+1,j,k,3), fy(i,j+1,k,3), fz(i,j+1,k,3));
//     std::printf("dsdt = %.12f -- \n", dsdt_arr(i,j,k,3));
// #endif
// }
  }

  // The operations following this assume we have returned the negative of the
  // divergence of fluxes.
  dsdt_arr(i, j, k, n) *= -1.0;
}

#endif