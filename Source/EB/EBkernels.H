#ifndef EBKERNELS_H
#define EBKERNELS_H

#include "IndexDefines.H"
#include "Reconstruction.H"
#include <Constants.H>
#include <EBR.H>

// TODO : use piecewise linear reconstruction with Green-Gauss approach
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void eb_recon_x(int i, int j, int k, int n,
                Array4<Real const> const &vfrc,
                amrex::Array4<amrex::Real> const &ql,
                amrex::Array4<amrex::Real> const &qr,
                amrex::Array4<amrex::Real const> const &q,
                amrex::Array4<amrex::EBCellFlag const> const &flag,
                Parm const &parm) noexcept {
  if (flag(i,j,k).isConnected(-1,0,0)) {
    ql(i, j, k, n) = q(i - 1, j, k, n);
    qr(i, j, k, n) = q(i, j, k, n);
    if (flag(i-1, j, k).isConnected(-1, 0, 0)) {
      ql(i,j,k,n) = q(i-1,j,k,n) + amrex::Real(0.5)*minmod(q(i,j,k,n)-q(i-1,j,k,n), q(i-1,j,k,n)-q(i-2,j,k,n));
    }
    if (flag(i,j,k).isConnected(1,0,0)) {
      qr(i,j,k,n) = q(i,j,k,n) - amrex::Real(0.5)*minmod(q(i+1,j,k,n)-q(i,j,k,n), q(i,j,k,n)-q(i-1,j,k,n));
    }
  }
} 
  
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void eb_recon_y(int i, int j, int k, int n,
                Array4<Real const> const &vfrc,
                amrex::Array4<amrex::Real> const &ql,
                amrex::Array4<amrex::Real> const &qr,
                amrex::Array4<amrex::Real const> const &q,
                amrex::Array4<amrex::EBCellFlag const> const &flag,
                Parm const &parm) noexcept {
  if (flag(i,j,k).isConnected(0,-1,0)) {
    ql(i, j, k, n) = q(i, j-1, k, n);
    qr(i, j, k, n) = q(i, j, k, n);
    if (flag(i, j-1, k).isConnected(0, -1, 0)) {
      ql(i,j,k,n) = q(i,j-1,k,n) + amrex::Real(0.5)*minmod(q(i,j,k,n)-q(i,j-1,k,n), q(i,j-1,k,n)-q(i,j-2,k,n));
    }
    if (flag(i,j,k).isConnected(0,1,0)) {
      qr(i,j,k,n) = q(i,j,k,n) - amrex::Real(0.5)*minmod(q(i,j+1,k,n)-q(i,j,k,n), q(i,j,k,n)-q(i,j-1,k,n));
    }
  }
} 
  
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void eb_recon_z(int i, int j, int k, int n,
                Array4<Real const> const &vfrc,
                amrex::Array4<amrex::Real> const &ql,
                amrex::Array4<amrex::Real> const &qr,
                amrex::Array4<amrex::Real const> const &q,
                amrex::Array4<amrex::EBCellFlag const> const &flag,
                Parm const &parm) noexcept {
  if (flag(i,j,k).isConnected(0,0,-1)) {
    ql(i, j, k, n) = q(i, j, k-1, n);
    qr(i, j, k, n) = q(i, j, k, n);
    if (flag(i, j, k-1).isConnected(0, 0, -1)) {
      ql(i,j,k,n) = q(i,j,k-1,n) + amrex::Real(0.5)*minmod(q(i,j,k,n)-q(i,j,k-1,n), q(i,j,k-1,n)-q(i,j,k-2,n));
    }
    if (flag(i,j,k).isConnected(0,0,1)) {
      qr(i,j,k,n) = q(i,j,k,n) - amrex::Real(0.5)*minmod(q(i,j,k+1,n)-q(i,j,k,n), q(i,j,k,n)-q(i,j,k-1,n));
    }
  }
}

// AUSM+
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
eb_flux (int i, int j, int k,
             amrex::Array4<amrex::Real const> const& ql,
             amrex::Array4<amrex::Real const> const& qr,
             amrex::Array4<amrex::Real> const& flxfab,
             int cdir, Parm const& parm) noexcept
{
  using amrex::Real;

  Real dl,ul,vl,wl,pl;
  Real dr,ur,vr,wr,pr;
  GpuArray<Real, NCONS> Fl, Fr;

  dl=ql(i,j,k,QRHO); 
  ul=ql(i,j,k,QU); 
  vl=ql(i,j,k,QV);
  wl=ql(i,j,k,QW); 
  pl=amrex::max(ql(i,j,k,QPRES), parm.smallp);
  dr=qr(i,j,k,QRHO); 
  ur=qr(i,j,k,QU); 
  vr=qr(i,j,k,QV);
  wr=qr(i,j,k,QW); 
  pr=amrex::max(qr(i,j,k,QPRES), parm.smallp);

  Real cL = ql(i,j,k,QC);
  Real cR = qr(i,j,k,QC);

  Real ML, MR;
  if (cdir == 0) {
    ML = ul/cL;
    MR = ur/cR;
  } else if (cdir == 1) {
    ML = vl/cL;
    MR = vr/cR;
  } else {
    ML = wl/cL;
    MR = wr/cR;
  }

  Real MLP, MRM, pLP, pRM;
  if (ML >= 1.0) {
    MLP = ML;
    pLP = pl;
  } else if (amrex::Math::abs(ML) < 1.0) {
    Real tmp1 = 0.25*(ML+1)*(ML+1);
    Real tmp2 = (ML*ML-1);
    MLP = tmp1+0.125*tmp2*tmp2;
    pLP = pl*(tmp1*(2-ML)+0.1875*ML*tmp2*tmp2);
  } else {
    MLP = 0.0;
    pLP = 0.0;
  }

   if (MR <= -1.0) {
    MRM = MR;
    pRM = pr;
  } else if (amrex::Math::abs(MR) < 1.0) {
    Real tmp1 = 0.25*(MR-1)*(MR-1);
    Real tmp2 = (MR*MR-1);
    MRM = -tmp1-0.125*tmp2*tmp2;
    pRM = pr*(tmp1*(2+MR)-0.1875*MR*tmp2*tmp2);
  } else {
    MRM = 0.0;
    pRM = 0.0;
  }
  Real Mn = MLP+MRM;
  Real pn = pLP+pRM;
  Real Mn_abs = amrex::Math::abs(Mn);

  Fl[URHO] = dl*cL;
  Fl[UMX] = dl*ul*cL;
  Fl[UMY] = dl*vl*cL;
  Fl[UMZ] = dl*wl*cL;
  Fl[UEDEN] = cL*(parm.eos_gamma*pl/(parm.eos_gamma-1.0) + 0.5*dl*(ul*ul+vl*vl+wl*wl));

  Fr[URHO] = dr*cR;
  Fr[UMX] = dr*ur*cR;
  Fr[UMY] = dr*vr*cR;
  Fr[UMZ] = dr*wr*cR;
  Fr[UEDEN] = cR*(parm.eos_gamma*pr/(parm.eos_gamma-1.0) + 0.5*dr*(ur*ur+vr*vr+wr*wr));

  for (int n=0; n<NCONS; ++n) {
    flxfab(i,j,k,n) = 0.5*Mn*(Fl[n]+Fr[n])-0.5*Mn_abs*(Fr[n]-Fl[n]);
  }
  flxfab(i,j,k,cdir+1) += pn;
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
  // drop too small cells
  if (flag(i, j, k).isCovered()) {
    dsdt_arr(i, j, k, n) = Real(0.0);
  } else if (flag(i,j,k).isRegular()) {
    dsdt_arr(i, j, k, n) = dxinv[0] * (fx(i + 1, j, k, n) - fx(i, j, k, n)) +
                       dxinv[1] * (fy(i, j + 1, k, n) - fy(i, j, k, n)) +
                       dxinv[2] * (fz(i, j, k + 1, n) - fz(i, j, k, n));
  } else {
    //TODO: 2nd-order correction for irregular flux or use Green-Gauss approach
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


// TODO: implement flux redistribution
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE 
void flux_redist(int i, int j, int k, int n, IntVect const &lo, IntVect const &hi,
                 Array4<Real> const& dsdt_arr, Array4<Real> const& divc_arr,
                 Array4<EBCellFlag const> const &flag, Array4<Real const> const &vfrac) {
  // try to compute non conservative flux
  Real totfrac = 0.0;
  Real totdivc = 0.0;
  Real divnc = 0.0;

  // amrex::Print() << "Location : " << i << " " << j << " " << k << " "<< n << ", vfrac: " << vfrac(i,j,k) << "\n";

  for (int ii=-1; ii<=1; ++ii) {
    for (int jj=-1; jj<=1; ++jj) {
      for (int kk=-1; kk<=1; ++kk) {
        if ( (lo[0] <= i+ii) && (i+ii <= hi[0]) && (lo[1] <= j+jj) && (j+jj <= hi[1]) && (lo[2] <= k+kk) && (k+kk <= hi[2])) {
          totfrac += vfrac(i+ii, j+jj, k+kk);
          totdivc += vfrac(i+ii, j+jj, k+kk) * divc_arr(i+ii, j+jj, k+kk,n);
          // amrex::Print() << i+l << " " << j+m << " " << k+n << ", " << vfrac(i+l,j+m,k+n) << " " << divc_arr(i+l,j+m,k+n) << "\n";
        }
      }
    }
  }
  divnc = totdivc/totfrac;
  dsdt_arr(i,j,k,n) = vfrac(i,j,k)*divc_arr(i,j,k,n); //+ (1.0-vfrac(i,j,k))*divnc;
  // amrex::Print() << "divc : " << divc_arr(i,j,k,n) << ", divnc : " << divnc << ", div : " << dsdt_arr(i,j,k,n) << "\n"; 

  // Real delta_M = vfrac(i,j,k)*(1.0-vfrac(i,j,k))*(divc_arr(i,j,k,n));

  // for (int ii=-1; ii<=1; ++ii) {
  //   for (int jj=-1; jj<=1; ++jj) {
  //     for (int kk=-1; kk<=1; ++kk) {
  //       if (ii!=0 || jj!= 0 || kk != 0) {
  //         if ( (lo[0] <= i+ii) && (i+ii <= hi[0]) && (lo[1] <= j+jj) && (j+jj <= hi[1]) && (lo[2] <= k+kk) && (k+kk <= hi[2]) && (vfrac(i+ii, j+jj, k+kk) != 0))
  //           dsdt_arr(i+ii,j+jj,k+kk,n) += vfrac(i+ii,j+jj,k+kk)*delta_M/(totfrac-vfrac(i,j,k));
  //       }
  //     }
  //   }
  // }
  // amrex::Print() << "After redist : " << dsdt_arr(i,j,k,n) << "\n"; 
}

#endif