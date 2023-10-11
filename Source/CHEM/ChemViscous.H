#ifndef VISCOUS_H
#define VISCOUS_H

#include "Constants.H"
#include "IndexDefines.H"
#include "LiDryer.H"

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
Real dot5(const Real (&x)[5], const Real *y) noexcept
{
    return x[0]*y[0] + x[1]*y[1] + x[2]*y[2] + x[3]*y[3] +
           x[4]*y[4];
}

/* viscosity: Pa-s
   conductivity: W/m/K */
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void mixtureProperties(Real T, const Real* X, Real &mu, Real &lambda, Parm const& parm) noexcept 
{
    Real mui[NSPECS], sqmui[NSPECS], lambdai[NSPECS];
    Real phi[NSPECS][NSPECS] = {{0.0}};
    Real sqT = sqrt(T), logT = log(T);
    Real sqsqT = sqrt(sqT);
    Real polytempvec[5] = {1.0, logT, logT*logT, logT*logT*logT, logT*logT*logT*logT};

    for (size_t k = 0; k < NSPECS; k++) {
        // the polynomial fit is done for sqrt(visc/sqrt(T))
        sqmui[k] = sqsqT * dot5(polytempvec, &parm.visccoeff[k*5]);
        lambdai[k] = sqT * dot5(polytempvec, &parm.conductcoeff[k*5]);
        mui[k] = (sqmui[k] * sqmui[k]);
    }

    Real sum1 = 0.0, sum2 = 0.0;
    for (size_t k = 0; k < NSPECS; k++) {
            sum1 += X[k] * lambdai[k];
            sum2 += X[k] / lambdai[k];
    }
    lambda = 0.5*(sum1 + 1.0/sum2);

    // Wilke fit, see Eq. (9-5.14) of Poling et al. (2001)
    for (size_t k = 0; k < NSPECS; k++) {
        for (size_t j = 0; j < k; j++) {
            Real wratiojk = parm.mw[j]/parm.mw[k];
            Real vratiokj = mui[k]/mui[j];

            Real factor1 = 1.0 + (sqmui[k]/sqmui[j]) * sqrt(sqrt(wratiojk));
            Real tmp = factor1*factor1 / sqrt(8.0+8.0*parm.mw[k]/parm.mw[j]);
            phi[k][j] = tmp;
            phi[j][k] = tmp / (vratiokj * wratiojk);
        }
        phi[k][k] = 1.0;
    }

    mu = 0;
    for (size_t k = 0; k < NSPECS; k++) {
        Real tmp = 0;
        for (size_t j = 0; j < NSPECS; j++) {
            tmp += X[j] * phi[k][j];
        }
        mu += X[k]*mui[k]/tmp;
    }
}

/* Returns the mixture-averaged diffusion coefficients [m^2/s].
   computed according to Eq. 12.178 in "Chemically Reacting Flow"*/
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void getMixDiffCoeffsMass(int i, int j, int k, Real T, Real p, const Real* X,
                          amrex::Array4<amrex::Real> const& Di, 
                          Parm const& parm) noexcept
{
    Real logT = log(T);
    Real polytempvec[5] = {1.0, logT, logT*logT, logT*logT*logT, logT*logT*logT*logT};
    Real bdiff[NSPECS][NSPECS];

    size_t ic = 0;
    for (size_t n = 0; n < NSPECS; n++) {
        for (size_t nn = n; nn < NSPECS; nn++) {
            bdiff[n][nn] = T * sqrt(T) *dot5(polytempvec,&parm.diffcoeff[ic*5]);
            bdiff[nn][n] = bdiff[n][nn];
            ic++;
        }
    }

    Real mmw = 0;
    for (int n=0;n<NSPECS; ++n) {
        mmw += X[n]*parm.mw[n];
    }
 
    for (size_t n=0; n<NSPECS; n++) {
        Real sum1 = 0.0;
        Real sum2 = 0.0;
        for (size_t nn=0; nn<NSPECS; nn++) {
            if (nn==n) {
                continue;
            }
            sum1 += X[nn] / bdiff[n][nn];
            sum2 += X[nn] * parm.mw[nn] / bdiff[n][nn];
        }
        sum1 *= p;
        sum2 *= p * X[n] / (mmw - parm.mw[n]*X[n]);
        Di(i,j,k,n) = 1.0 / (sum1 + sum2);
    }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
diffusion_x (int i, int j, int k,
             amrex::Array4<amrex::Real const> const& q,
             amrex::Array4<amrex::Real const> const& rhoi,
             amrex::Array4<amrex::Real const> const& D,
             amrex::Array4<amrex::Real> const& fx,
             amrex::Array4<amrex::Real> const& fx_diff,
             amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv) noexcept
{
    using amrex::Real;
    Real rhor = q(i,j,k,QRHO);
    Real rhol = q(i-1,j,k,QRHO);
    Real rho = 0.5*(rhol+rhor);
    Real di[NSPECS], Ytr[NSPECS], Ytl[NSPECS];

    for (int n=0; n<NSPECS; ++n) {
        Ytr[n] = rhoi(i,j,k,n) / rhor;
        Ytl[n] = rhoi(i-1,j,k,n) / rhol;
        di[n] = 0.5*(D(i,j,k,n)+D(i-1,j,k,n));
    }

    Real tmp;
    for (int n=0; n<NSPECS; ++n) {
        tmp = rho * di[n] * (Ytl[n] - Ytr[n]) * dxinv[0];
        fx_diff(i,j,k,n) = tmp;
        fx(i,j,k,n) += tmp;
    }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
diffusion_y (int i, int j, int k,
             amrex::Array4<amrex::Real const> const& q,
             amrex::Array4<amrex::Real const> const& rhoi,
             amrex::Array4<amrex::Real const> const& D,
             amrex::Array4<amrex::Real> const& fy,
             amrex::Array4<amrex::Real> const& fy_diff,
             amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv) noexcept
{
    using amrex::Real;
    Real rhor = q(i,j,k,QRHO);
    Real rhol = q(i,j-1,k,QRHO);
    Real rho = 0.5*(rhol+rhor);
    Real di[NSPECS], Ytr[NSPECS], Ytl[NSPECS];

    for (int n=0; n<NSPECS; ++n) {
        Ytr[n] = rhoi(i,j,k,n)/rhor;
        Ytl[n] = rhoi(i,j-1,k,n)/rhol;
        di[n] = 0.5*(D(i,j,k,n)+D(i,j-1,k,n));
    }

    Real tmp;
    for (int n=0; n<NSPECS; ++n) {
        tmp = rho * di[n] * (Ytl[n] - Ytr[n]) * dxinv[1];
        fy_diff(i,j,k,n) = tmp;
        fy(i,j,k,n) += tmp;
    }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
diffusion_z (int i, int j, int k,
             amrex::Array4<amrex::Real const> const& q,
             amrex::Array4<amrex::Real const> const& rhoi,
             amrex::Array4<amrex::Real const> const& D,
             amrex::Array4<amrex::Real> const& fz,
             amrex::Array4<amrex::Real> const& fz_diff,
             amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv) noexcept
{
    using amrex::Real;
    Real rhor = q(i,j,k,QRHO);
    Real rhol = q(i,j,k-1,QRHO);
    Real rho = 0.5*(rhol+rhor);
    Real di[NSPECS], Ytr[NSPECS], Ytl[NSPECS];

    for (int n=0; n<NSPECS; ++n) {
        Ytr[n] = rhoi(i,j,k,n)/rhor;
        Ytl[n] = rhoi(i,j,k-1,n)/rhol;
        di[n] = 0.5*(D(i,j,k,n)+D(i,j,k-1,n));
    }

    Real tmp;
    for (int n=0; n<NSPECS; ++n) {
        tmp = rho * di[n] * (Ytl[n] - Ytr[n]) * dxinv[2];
        fz_diff(i,j,k,n) = tmp;
        fz(i,j,k,n) += tmp;
    }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
compute_visc_x_multi(int i, int j, int k,
             amrex::Array4<amrex::Real const> const& q,
             amrex::Array4<amrex::Real const> const& rhoi,
             amrex::Array4<amrex::Real const> const& lambda,
             amrex::Array4<amrex::Real const> const& mu,
             amrex::Array4<amrex::Real> const& fx,
             amrex::Array4<amrex::Real const> const& fx_diff,
             amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv,
             Parm const& parm) noexcept
{
    using amrex::Real;
    // first compute temperature
    Real Tr = q(i,j,k,QT);
    Real Tl = q(i-1,j,k,QT);
    Real T = 0.5*(Tl + Tr);

    // lambda and mu
    Real mux = 0.5*(mu(i,j,k)+mu(i-1,j,k));
    Real lambdax = 0.5*(lambda(i,j,k)+lambda(i-1,j,k));

    // flux due to specie diffusion
    Real hi[NSPECS];
    speciesEnthalpy(T, hi, parm);
    Real tmp = 0;
    for (int n=0; n<NSPECS; ++n) {
        tmp += fx_diff(i,j,k,n) * hi[n];
    }

    Real dTdx = (Tr-Tl)*dxinv[0];
    Real dudx = (q(i,j,k,QU)-q(i-1,j,k,QU))*dxinv[0];
    Real dvdx = (q(i,j,k,QV)-q(i-1,j,k,QV))*dxinv[0];
    Real dwdx = (q(i,j,k,QW)-q(i-1,j,k,QW))*dxinv[0];
    Real dudy = (q(i,j+1,k,QU)+q(i-1,j+1,k,QU)-q(i,j-1,k,QU)-q(i-1,j-1,k,QU))*(Real(0.25)*dxinv[1]);
    Real dvdy = (q(i,j+1,k,QV)+q(i-1,j+1,k,QV)-q(i,j-1,k,QV)-q(i-1,j-1,k,QV))*(Real(0.25)*dxinv[1]);
    Real dudz = (q(i,j,k+1,QU)+q(i-1,j,k+1,QU)-q(i,j,k-1,QU)-q(i-1,j,k-1,QU))*(Real(0.25)*dxinv[2]);
    Real dwdz = (q(i,j,k+1,QW)+q(i-1,j,k+1,QW)-q(i,j,k-1,QW)-q(i-1,j,k-1,QW))*(Real(0.25)*dxinv[2]);
    Real divu = dudx + dvdy + dwdz;
    Real tauxx = mux*(Real(2.0)*dudx-parm.twoThirds*divu);
    Real tauxy = mux*(dudy+dvdx);
    Real tauxz = mux*(dudz+dwdx);
    fx(i,j,k,UMX)   += -tauxx;
    fx(i,j,k,UMY)   += -tauxy;
    fx(i,j,k,UMZ)   += -tauxz;
    fx(i,j,k,UEDEN) += -Real(0.5)*((q(i,j,k,QU)+q(i-1,j,k,QU))*tauxx \
                                  +(q(i,j,k,QV)+q(i-1,j,k,QV))*tauxy \
                                  +(q(i,j,k,QW)+q(i-1,j,k,QW))*tauxz)
                                  - lambdax*dTdx - tmp;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
compute_visc_y_multi(int i, int j, int k,
             amrex::Array4<amrex::Real const> const& q,
             amrex::Array4<amrex::Real const> const& rhoi,
             amrex::Array4<amrex::Real const> const& lambda,
             amrex::Array4<amrex::Real const> const& mu,
             amrex::Array4<amrex::Real> const& fy,
             amrex::Array4<amrex::Real const> const& fy_diff,
             amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv,
             Parm const& parm) noexcept
{
    using amrex::Real;
    // first compute temperature
    Real Tr = q(i,j,k,QT);
    Real Tl = q(i,j-1,k,QT);
    Real T = 0.5*(Tl + Tr);

    // lambda and mu
    Real muy = 0.5*(mu(i,j,k)+mu(i,j-1,k));
    Real lambday = 0.5*(lambda(i,j,k)+lambda(i,j-1,k));

    // flux due to specie diffusion
    Real hi[NSPECS];
    speciesEnthalpy(T, hi, parm);
    Real tmp = 0;
    for (int n=0; n<NSPECS; ++n) {
        tmp += fy_diff(i,j,k,n) * hi[n];
    }

    Real dTdy = (Tr-Tl)*dxinv[1];
    Real dudy = (q(i,j,k,QU)-q(i,j-1,k,QU))*dxinv[1];
    Real dvdy = (q(i,j,k,QV)-q(i,j-1,k,QV))*dxinv[1];
    Real dwdy = (q(i,j,k,QW)-q(i,j-1,k,QW))*dxinv[1];
    Real dudx = (q(i+1,j,k,QU)+q(i+1,j-1,k,QU)-q(i-1,j,k,QU)-q(i-1,j-1,k,QU))*(Real(0.25)*dxinv[0]);
    Real dvdx = (q(i+1,j,k,QV)+q(i+1,j-1,k,QV)-q(i-1,j,k,QV)-q(i-1,j-1,k,QV))*(Real(0.25)*dxinv[0]);
    Real dvdz = (q(i,j,k+1,QV)+q(i,j-1,k+1,QV)-q(i,j,k-1,QV)-q(i,j-1,k-1,QV))*(Real(0.25)*dxinv[2]);
    Real dwdz = (q(i,j,k+1,QW)+q(i,j-1,k+1,QW)-q(i,j,k-1,QW)-q(i,j-1,k-1,QW))*(Real(0.25)*dxinv[2]);
    Real divu = dudx + dvdy + dwdz;
    Real tauyy = muy*(Real(2.0)*dvdy-parm.twoThirds*divu);
    Real tauxy = muy*(dudy+dvdx);
    Real tauyz = muy*(dwdy+dvdz);
    fy(i,j,k,UMX)   += -tauxy;
    fy(i,j,k,UMY)   += -tauyy;
    fy(i,j,k,UMZ)   += -tauyz;
    fy(i,j,k,UEDEN) += -Real(0.5)*((q(i,j,k,QU)+q(i,j-1,k,QU))*tauxy \
                                  +(q(i,j,k,QV)+q(i,j-1,k,QV))*tauyy \
                                  +(q(i,j,k,QW)+q(i,j-1,k,QW))*tauyz)
                                  - lambday*dTdy - tmp;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
compute_visc_z_multi(int i, int j, int k,
             amrex::Array4<amrex::Real const> const& q,
             amrex::Array4<amrex::Real const> const& rhoi,
             amrex::Array4<amrex::Real const> const& lambda,
             amrex::Array4<amrex::Real const> const& mu,
             amrex::Array4<amrex::Real> const& fz,
             amrex::Array4<amrex::Real const> const& fz_diff,
             amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv,
             Parm const& parm) noexcept
{
    using amrex::Real;
    // first compute temperature
    Real Tr = q(i,j,k,QT);
    Real Tl = q(i,j,k-1,QT);
    Real T = 0.5*(Tl + Tr);

    // lambda and mu
    Real muz = 0.5*(mu(i,j,k)+mu(i,j,k-1));
    Real lambdaz = 0.5*(lambda(i,j,k)+lambda(i,j,k-1));

    // flux due to specie diffusion
    Real hi[NSPECS];
    speciesEnthalpy(T, hi, parm);
    Real tmp = 0;
    for (int n=0; n<NSPECS; ++n) {
        tmp += fz_diff(i,j,k,n) * hi[n];
    }

    Real dTdz = (Tr-Tl)*dxinv[2];
    Real dudz = (q(i,j,k,QU)-q(i,j,k-1,QU))*dxinv[2];
    Real dvdz = (q(i,j,k,QV)-q(i,j,k-1,QV))*dxinv[2];
    Real dwdz = (q(i,j,k,QW)-q(i,j,k-1,QW))*dxinv[2];
    Real dudx = (q(i+1,j,k,QU)+q(i+1,j,k-1,QU)-q(i-1,j,k,QU)-q(i-1,j,k-1,QU))*(Real(0.25)*dxinv[0]);
    Real dwdx = (q(i+1,j,k,QW)+q(i+1,j,k-1,QW)-q(i-1,j,k,QW)-q(i-1,j,k-1,QW))*(Real(0.25)*dxinv[0]);
    Real dvdy = (q(i,j+1,k,QV)+q(i,j+1,k-1,QV)-q(i,j-1,k,QV)-q(i,j-1,k-1,QV))*(Real(0.25)*dxinv[1]);
    Real dwdy = (q(i,j+1,k,QW)+q(i,j+1,k-1,QW)-q(i,j-1,k,QW)-q(i,j-1,k-1,QW))*(Real(0.25)*dxinv[1]);
    Real divu = dudx + dvdy + dwdz;
    Real tauxz = muz*(dudz+dwdx);
    Real tauyz = muz*(dvdz+dwdy);
    Real tauzz = muz*(Real(2.0)*dwdz-parm.twoThirds*divu);
    fz(i,j,k,UMX)   += -tauxz;
    fz(i,j,k,UMY)   += -tauyz;
    fz(i,j,k,UMZ)   += -tauzz;
    fz(i,j,k,UEDEN) += -Real(0.5)*((q(i,j,k,QU)+q(i,j,k-1,QU))*tauxz \
                                  +(q(i,j,k,QV)+q(i,j,k-1,QV))*tauyz \
                                  +(q(i,j,k,QW)+q(i,j,k-1,QW))*tauzz)
                                  - lambdaz*dTdz - tmp;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
compute_visc_x_multi_eb(int i, int j, int k,
             amrex::Array4<amrex::Real const> const& q,
             amrex::Array4<amrex::Real const> const& rhoi,
             amrex::Array4<amrex::Real const> const& lambda,
             amrex::Array4<amrex::Real const> const& mu,
             amrex::Array4<amrex::Real> const& fx,
             amrex::Array4<amrex::Real const> const& fx_diff,
             amrex::Array4<amrex::EBCellFlag const> const& flag,
             amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv,
             amrex::GpuArray<amrex::Real,3> const& weights,
             Parm const& parm) noexcept
{
    using amrex::Real;
    // first compute temperature
    Real Tr = q(i,j,k,QT);
    Real Tl = q(i-1,j,k,QT);
    Real T = 0.5*(Tl + Tr);

    // lambda and mu
    Real mux = 0.5*(mu(i,j,k)+mu(i-1,j,k));
    Real lambdax = 0.5*(lambda(i,j,k)+lambda(i-1,j,k));

    // flux due to specie diffusion
    Real hi[NSPECS];
    speciesEnthalpy(T, hi, parm);
    Real tmp = 0;
    for (int n=0; n<NSPECS; ++n) {
        tmp += fx_diff(i,j,k,n) * hi[n];
    }

    Real dTdx = (Tr-Tl)*dxinv[0];
    Real dudx = (q(i,j,k,QU)-q(i-1,j,k,QU))*dxinv[0];
    Real dvdx = (q(i,j,k,QV)-q(i-1,j,k,QV))*dxinv[0];
    Real dwdx = (q(i,j,k,QW)-q(i-1,j,k,QW))*dxinv[0];
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
    Real divu = dudx + dvdy + dwdz;
    Real tauxx = mux*(Real(2.0)*dudx-parm.twoThirds*divu);
    Real tauxy = mux*(dudy+dvdx);
    Real tauxz = mux*(dudz+dwdx);
    fx(i,j,k,UMX)   += -tauxx;
    fx(i,j,k,UMY)   += -tauxy;
    fx(i,j,k,UMZ)   += -tauxz;
    fx(i,j,k,UEDEN) += -Real(0.5)*((q(i,j,k,QU)+q(i-1,j,k,QU))*tauxx \
                                  +(q(i,j,k,QV)+q(i-1,j,k,QV))*tauxy \
                                  +(q(i,j,k,QW)+q(i-1,j,k,QW))*tauxz)
                                  - lambdax*dTdx - tmp;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
compute_visc_y_multi_eb(int i, int j, int k,
             amrex::Array4<amrex::Real const> const& q,
             amrex::Array4<amrex::Real const> const& rhoi,
             amrex::Array4<amrex::Real const> const& lambda,
             amrex::Array4<amrex::Real const> const& mu,
             amrex::Array4<amrex::Real> const& fy,
             amrex::Array4<amrex::Real const> const& fy_diff,
             amrex::Array4<amrex::EBCellFlag const> const& flag,
             amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv,
             amrex::GpuArray<amrex::Real,3> const& weights,
             Parm const& parm) noexcept
{
    using amrex::Real;
    // first compute temperature
    Real Tr = q(i,j,k,QT);
    Real Tl = q(i,j-1,k,QT);
    Real T = 0.5*(Tl + Tr);

    // lambda and mu
    Real muy = 0.5*(mu(i,j,k)+mu(i,j-1,k));
    Real lambday = 0.5*(lambda(i,j,k)+lambda(i,j-1,k));

    // flux due to specie diffusion
    Real hi[NSPECS];
    speciesEnthalpy(T, hi, parm);
    Real tmp = 0;
    for (int n=0; n<NSPECS; ++n) {
        tmp += fy_diff(i,j,k,n) * hi[n];
    }

    Real dTdy = (Tr-Tl)*dxinv[1];
    Real dudy = (q(i,j,k,QU)-q(i,j-1,k,QU))*dxinv[1];
    Real dvdy = (q(i,j,k,QV)-q(i,j-1,k,QV))*dxinv[1];
    Real dwdy = (q(i,j,k,QW)-q(i,j-1,k,QW))*dxinv[1];
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
    Real divu = dudx + dvdy + dwdz;
    Real tauyy = muy*(Real(2.0)*dvdy-parm.twoThirds*divu);
    Real tauxy = muy*(dudy+dvdx);
    Real tauyz = muy*(dwdy+dvdz);
    fy(i,j,k,UMX)   += -tauxy;
    fy(i,j,k,UMY)   += -tauyy;
    fy(i,j,k,UMZ)   += -tauyz;
    fy(i,j,k,UEDEN) += -Real(0.5)*((q(i,j,k,QU)+q(i,j-1,k,QU))*tauxy \
                                  +(q(i,j,k,QV)+q(i,j-1,k,QV))*tauyy \
                                  +(q(i,j,k,QW)+q(i,j-1,k,QW))*tauyz)
                                  - lambday*dTdy - tmp;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
compute_visc_z_multi_eb(int i, int j, int k,
             amrex::Array4<amrex::Real const> const& q,
             amrex::Array4<amrex::Real const> const& rhoi,
             amrex::Array4<amrex::Real const> const& lambda,
             amrex::Array4<amrex::Real const> const& mu,
             amrex::Array4<amrex::Real> const& fz,
             amrex::Array4<amrex::Real const> const& fz_diff,
             amrex::Array4<amrex::EBCellFlag const> const& flag,
             amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv,
             amrex::GpuArray<amrex::Real,3> const& weights,
             Parm const& parm) noexcept
{
    using amrex::Real;
    // first compute temperature
    Real Tr = q(i,j,k,QT);
    Real Tl = q(i,j,k-1,QT);
    Real T = 0.5*(Tl + Tr);

    // lambda and mu
    Real muz = 0.5*(mu(i,j,k)+mu(i,j,k-1));
    Real lambdaz = 0.5*(lambda(i,j,k)+lambda(i,j,k-1));

    // flux due to specie diffusion
    Real hi[NSPECS];
    speciesEnthalpy(T, hi, parm);
    Real tmp = 0;
    for (int n=0; n<NSPECS; ++n) {
        tmp += fz_diff(i,j,k,n) * hi[n];
    }

    Real dTdz = (Tr-Tl)*dxinv[2];
    Real dudz = (q(i,j,k,QU)-q(i,j,k-1,QU))*dxinv[2];
    Real dvdz = (q(i,j,k,QV)-q(i,j,k-1,QV))*dxinv[2];
    Real dwdz = (q(i,j,k,QW)-q(i,j,k-1,QW))*dxinv[2];
    int  ihip = i + (flag(i,j,k  ).isConnected( 1,0,0) ? 1 : 0 );
    int  ihim = i - (flag(i,j,k  ).isConnected(-1,0,0) ? 1 : 0 );
    int  ilop = i + (flag(i,j,k-1).isConnected( 1,0,0) ? 1 : 0 );
    int  ilom = i - (flag(i,j,k-1).isConnected(-1,0,0) ? 1 : 0 );

    Real   whi = weights[ihip-ihim];
    Real   wlo = weights[ilop-ilom];
    Real   dudx = (0.5*dxinv[0]) * ((q(ihip,j,k  ,QU)-q(ihim,j,k  ,QU))*whi
                                   +(q(ilop,j,k-1,QU)-q(ilom,j,k-1,QU))*wlo) ;
     Real  dwdx = (0.5*dxinv[0]) * ((q(ihip,j,k  ,QW)-q(ihim,j,k  ,QW))*whi
                                   +(q(ilop,j,k-1,QW)-q(ilom,j,k-1,QW))*wlo) ;


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
    Real divu = dudx + dvdy + dwdz;
    Real tauxz = muz*(dudz+dwdx);
    Real tauyz = muz*(dvdz+dwdy);
    Real tauzz = muz*(Real(2.0)*dwdz-parm.twoThirds*divu);
    fz(i,j,k,UMX)   += -tauxz;
    fz(i,j,k,UMY)   += -tauyz;
    fz(i,j,k,UMZ)   += -tauzz;
    fz(i,j,k,UEDEN) += -Real(0.5)*((q(i,j,k,QU)+q(i,j,k-1,QU))*tauxz \
                                  +(q(i,j,k,QV)+q(i,j,k-1,QV))*tauyz \
                                  +(q(i,j,k,QW)+q(i,j,k-1,QW))*tauzz)
                                  - lambdaz*dTdz - tmp;
}

#endif