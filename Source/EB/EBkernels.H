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
  ql(i, j, k, n) = q(i - 1, j, k, n);
  qr(i, j, k, n) = q(i, j, k, n);
  if (flag(i,j,k).isConnected(-1,0,0)) {
    if (flag(i, j, k).isConnected(-2, 0, 0)) {
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
  ql(i, j, k, n) = q(i, j-1, k, n);
  qr(i, j, k, n) = q(i, j, k, n);
  if (flag(i,j,k).isConnected(0,-1,0)) {
    if (flag(i, j, k).isConnected(0, -2, 0)) {
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
  ql(i, j, k, n) = q(i, j, k-1, n);
  qr(i, j, k, n) = q(i, j, k, n);
  if (flag(i,j,k).isConnected(0,0,-1)) {
    if (flag(i, j, k).isConnected(0, 0, -2)) {
      ql(i,j,k,n) = q(i,j,k-1,n) + amrex::Real(0.5)*minmod(q(i,j,k,n)-q(i,j,k-1,n), q(i,j,k-1,n)-q(i,j,k-2,n));
    }
    if (flag(i,j,k).isConnected(0,0,1)) {
      qr(i,j,k,n) = q(i,j,k,n) - amrex::Real(0.5)*minmod(q(i,j,k+1,n)-q(i,j,k,n), q(i,j,k,n)-q(i,j,k-1,n));
    }
  }
}

// Van Leer
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
  GpuArray<Real, NCONS> fp={0}, fm={0};
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

#ifndef CHEM
  Real gamma = parm.eos_gamma;
#else
  Real gamma = 0.5*(ql(i,j,k,QGAMA)+qr(i,j,k,QGAMA));
#endif

  Real cL = ql(i,j,k,QC);
  Real cR = qr(i,j,k,QC);

  Real ML, MR, UL, UR;
  int nx = 0, ny = 0, nz = 0;

  if (cdir == 0) {
    ML = ul/cL;
    MR = ur/cR;
    UL = ul;
    UR = ur;
    nx = 1;
  } else if (cdir == 1) {
    ML = vl/cL;
    MR = vr/cR;
    UL = vl;
    UR = vr;
    ny = 1;
  } else {
    ML = wl/cL;
    MR = wr/cR;
    UL = wl;
    UR = wr;
    nz = 1;
  }

	if (ML >= 1.0) {
		fp[0] = dl*UL;
		fp[1] = dl*UL*ul + nx * pl;
		fp[2] = dl*UL*vl + ny * pl;
		fp[3] = dl*UL*wl + nz * pl;
		fp[4] = UL*(gamma*pl / (gamma - 1.0) + 0.5*dl*(ul*ul + vl*vl + wl*wl));
	}
	else if (amrex::Math::abs(ML)<1.0) {
		Real Mp = 0.250*(1.0 + ML)*(1.0 + ML);
		Real tmp0 = dl*cL*Mp;
		fp[0] = tmp0;
		fp[1] = tmp0*(nx*(-UL+2.0*cL)/gamma + ul);
		fp[2] = tmp0*(ny*(-UL+2.0*cL)/gamma + vl);
		fp[3] = tmp0*(nz*(-UL+2.0*cL)/gamma + wl);
		fp[4] = tmp0*( ((gamma-1)*UL+2*cL) * ((gamma-1)*UL+2*cL)/(2*(gamma*gamma-1))
                  + (ul*ul + vl*vl + wl*wl - UL*UL)/2
    );
	}

	if (amrex::Math::abs(MR) < 1.0) {
		Real Mm = -0.250*(MR - 1.0) * (MR - 1.0);
		Real tmp0 = dr*cR*Mm;
		fm[0] = tmp0;
		fm[1] = tmp0*(nx*(-UR-2.0*cR)/gamma + ur);
		fm[2] = tmp0*(ny*(-UR-2.0*cR)/gamma + vr);
		fm[3] = tmp0*(nz*(-UR-2.0*cR)/gamma + wr);
		fm[4] = tmp0*( ((gamma-1)*UR-2*cR) * ((gamma-1)*UR-2*cR)/(2*(gamma*gamma-1))
                  + (ur*ur + vr*vr + wr*wr - UR*UR)/2
    );
	}
	else if (MR <= -1.0) {
		fm[0] = dr*UR;
		fm[1] = dr*UR*ur + nx * pr;
		fm[2] = dr*UR*vr + ny * pr;
		fm[3] = dr*UR*wr + nz * pr;
		fm[4] = UR*(gamma*pr / (gamma - 1.0) + 0.5*dr*(ur*ur + vr*vr + wr*wr));
	}
	for (int n = 0; n < NCONS; ++n) {
		flxfab(i,j,k,n) = fp[n] + fm[n];
	}
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE 
void eb_wallflux(GpuArray<Real, NPRIM> const &qw,
                 const amrex::Real axm, const amrex::Real axp,
                 const amrex::Real aym, const amrex::Real ayp,
                 const amrex::Real azm, const amrex::Real azp,
                 amrex::GpuArray<amrex::Real,NCONS>& fluxw, Parm const& parm) noexcept {
  GpuArray<Real, NCONS> fp, fm;
  Real d = qw[QRHO]; 
  Real u = qw[QU]; 
  Real v = qw[QV];
  Real w = qw[QW]; 
  Real p = qw[QPRES];
  Real c = qw[QC];

  Real apnorm = std::sqrt( (axm-axp)*(axm-axp) + (aym-ayp)*(aym-ayp) + (azm-azp)*(azm-azp) );
  Real un = u*(axm-axp) + v*(aym-ayp) + w*(azm-azp);
  Real lambda0 = amrex::Math::abs(un) + c*apnorm;

#ifndef CHEM
  Real gamma = parm.eos_gamma;
#else
  Real gamma = qw[QGAMA];
#endif
  Real E0=p/(gamma-1)+0.50*d*(u*u + v*v + w*w);
  
  fp[0]=0.5*(d*un+lambda0*d);
  fp[1]=0.5*(d*u*un+(axm-axp)*p + lambda0*d*u);
  fp[2]=0.5*(d*v*un+(aym-ayp)*p + lambda0*d*v);
  fp[3]=0.5*(d*w*un+(azm-azp)*p + lambda0*d*w);
  fp[4]=0.5*((E0+p)*un + lambda0*E0);

  fm[0]=0.5*( d*(-un)-lambda0*d);
  fm[1]=0.5*(d*u*un+(axm-axp)*p + lambda0*d*u);
  fm[2]=0.5*(d*v*un+(aym-ayp)*p + lambda0*d*v);
  fm[3]=0.5*(d*w*un+(azm-azp)*p + lambda0*d*w);
  fm[4]=0.5*(-(E0+p)*un - lambda0*E0);

  for (int n=0; n<NCONS; ++n) {
    fluxw[n] = fp[n] + fm[n];
  }
}

// only account for conservative update
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE 
void eb_compute_div_visc(
    int i, int j, int k,
    Array4<Real> const &q, Array4<Real> const& dsdt_arr,
    Array4<Real> const &fx, Array4<Real> const &fy, Array4<Real> const &fz,
    Array4<EBCellFlag const> const &flag, Array4<Real const> const &vfrc,
    Array4<Real const> const &bcent,
    Array4<Real const> const &apx, Array4<Real const> const &apy, Array4<Real const> const &apz,
    Array4<Real const> const &fcx, Array4<Real const> const &fcy, Array4<Real const> const &fcz,
    GpuArray<Real, AMREX_SPACEDIM> const &dxinv, Parm const &parm) noexcept {

  Real tmp = dxinv[0]/vfrc(i,j,k);
  // drop too small cells
  if (flag(i, j, k).isCovered()) {
    for (int n=0; n<NCONS; ++n) {
      dsdt_arr(i, j, k, n) = Real(0.0);
    }
  } else if (flag(i,j,k).isRegular()) {
    for (int n=0; n<NCONS; ++n) {
      dsdt_arr(i, j, k, n) = dxinv[0] * (fx(i + 1, j, k, n) - fx(i, j, k, n)) +
                        dxinv[1] * (fy(i, j + 1, k, n) - fy(i, j, k, n)) +
                        dxinv[2] * (fz(i, j, k + 1, n) - fz(i, j, k, n));
    }
  } else {
    //TODO: 2nd-order correction for irregular flux or use Green-Gauss approach
    for (int n=0; n<NCONS; ++n) {
      dsdt_arr(i, j, k, n) = tmp *
          (apx(i + 1, j, k) * fx(i+1,j,k,n) - apx(i, j, k) * fx(i,j,k,n) +
          apy(i, j + 1, k) * fy(i,j+1,k,n) - apy(i, j, k) * fy(i,j,k,n) +
          apz(i, j, k + 1) * fz(i,j,k+1,n) - apz(i, j, k) * fz(i,j,k,n));
    }

    GpuArray<Real, NCONS> flux_wall;
    GpuArray<Real, NPRIM> qw;

    for (int n=0; n<NPRIM; ++n) {
      qw[n] = q(i,j,k,n);
    }

    //TODO: Here is hyp wallflux, Still need diff wallflux
    eb_wallflux(qw, apx(i,j,k),apx(i+1,j,k),apy(i,j,k),apy(i,j+1,k),apz(i,j,k),apz(i,j,k+1),flux_wall,parm);
    // Here we assume dx == dy == dz
    for (int n=0; n<NCONS; ++n) {
      dsdt_arr(i, j, k, n) += flux_wall[n] * tmp;
    }
  }
  // The operations following this assume we have returned the negative of the
  // divergence of fluxes.
  for (int n=0; n<NCONS; ++n) {
    dsdt_arr(i, j, k, n) *= -1.0;
  }
}

// only account for conservative update
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE 
void eb_compute_div(
    int i, int j, int k,
    Array4<Real> const &q, Array4<Real> const& dsdt_arr,
    Array4<Real> const &fx, Array4<Real> const &fy, Array4<Real> const &fz,
    Array4<EBCellFlag const> const &flag, Array4<Real const> const &vfrc,
    Array4<Real const> const &bcent,
    Array4<Real const> const &apx, Array4<Real const> const &apy, Array4<Real const> const &apz,
    Array4<Real const> const &fcx, Array4<Real const> const &fcy, Array4<Real const> const &fcz,
    GpuArray<Real, AMREX_SPACEDIM> const &dxinv, Parm const &parm) noexcept {

  Real tmp = dxinv[0]/vfrc(i,j,k);
  // drop too small cells
  if (flag(i, j, k).isCovered()) {
    for (int n=0; n<NCONS; ++n) {
      dsdt_arr(i, j, k, n) = Real(0.0);
    }
  } else if (flag(i,j,k).isRegular()) {
    for (int n=0; n<NCONS; ++n) {
      dsdt_arr(i, j, k, n) = dxinv[0] * (fx(i + 1, j, k, n) - fx(i, j, k, n)) +
                        dxinv[1] * (fy(i, j + 1, k, n) - fy(i, j, k, n)) +
                        dxinv[2] * (fz(i, j, k + 1, n) - fz(i, j, k, n));
    }
  } else {
    //TODO: 2nd-order correction for irregular flux or use Green-Gauss approach
    for (int n=0; n<NCONS; ++n) {
      dsdt_arr(i, j, k, n) = tmp *
          (apx(i + 1, j, k) * fx(i+1,j,k,n) - apx(i, j, k) * fx(i,j,k,n) +
          apy(i, j + 1, k) * fy(i,j+1,k,n) - apy(i, j, k) * fy(i,j,k,n) +
          apz(i, j, k + 1) * fz(i,j,k+1,n) - apz(i, j, k) * fz(i,j,k,n));
    }

    GpuArray<Real, NCONS> flux_wall;
    // the slip wall flux
    flux_wall[URHO] = Real(0.0);
    flux_wall[UMX] = (apx(i,j,k)-apx(i+1,j,k))*q(i,j,k,QPRES);
    flux_wall[UMY] = (apy(i,j,k)-apy(i,j+1,k))*q(i,j,k,QPRES);
    flux_wall[UMZ] = (apz(i,j,k)-apz(i,j,k+1))*q(i,j,k,QPRES);
    flux_wall[UEDEN] = Real(0.0);
    
    // Here we assume dx == dy == dz
    for (int n=0; n<NCONS; ++n) {
      dsdt_arr(i, j, k, n) += flux_wall[n] * tmp;
    }

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
  for (int n=0; n<NCONS; ++n) {
    dsdt_arr(i, j, k, n) *= -1.0;
  }
}


// TODO: implement flux redistribution
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE 
void flux_redist(int i, int j, int k, IntVect const &lo, IntVect const &hi,
                 Array4<Real> const& dsdt_arr, Array4<Real> const& divc_arr,
                 Array4<EBCellFlag const> const &flag, Array4<Real const> const &vfrac) {
  // try to compute non conservative flux
  Real totfrac = vfrac(i,j,k);
  GpuArray<Real, NCONS> delta_M;

  for (int ii=-1; ii<=1; ++ii) {
    for (int jj=-1; jj<=1; ++jj) {
      if ( (lo[0] <= i+ii) && (i+ii <= hi[0]) && (lo[1] <= j+jj) && (j+jj <= hi[1]) ) {
        totfrac += vfrac(i+ii, j+jj, k);
      }
    }
  }

  // simply rescale
  for (int n=0; n<NCONS; ++n) {
    dsdt_arr(i,j,k,n) = vfrac(i,j,k)*divc_arr(i,j,k,n);
    delta_M[n] = (1.0-vfrac(i,j,k))*divc_arr(i,j,k,n);
  }

  // for (int ii=-1; ii<=1; ++ii) {
  //   for (int jj=-1; jj<=1; ++jj) {
  //     if (ii!=0 || jj!= 0) {
  //       if ( (lo[0] <= i+ii) && (i+ii <= hi[0]) && (lo[1] <= j+jj) && (j+jj <= hi[1]) ) {
  //         for (int n=0; n<NCONS; ++n) {
  //           AMREX_DEVICE_PRINTF("%d,%d,%d, divc=%f, dsdt=%f\n", ii,jj,k, divc_arr(i+ii,j+jj,k,n), dsdt_arr(i+ii,j+jj,k,n))
  //           // dsdt_arr(i+ii,j+jj,k,n) = divc_arr(i+ii,j+jj,k,n);// + vfrac(i+ii,j+jj,k)*delta_M[n]/(totfrac-vfrac(i,j,k));
  //         }
  //       }
  //     }
  //   }
  // }
}

#endif