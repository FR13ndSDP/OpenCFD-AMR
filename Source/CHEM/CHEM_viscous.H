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
void getMixDiffCoeffsMass(Real T, Real p, const Real* X, Real* d, Parm const& parm) noexcept
{
    Real logT = log(T);
    Real polytempvec[5] = {1.0, logT, logT*logT, logT*logT*logT, logT*logT*logT*logT};
    Real bdiff[NSPECS][NSPECS];

    size_t ic = 0;
    for (size_t i = 0; i < NSPECS; i++) {
        for (size_t j = i; j < NSPECS; j++) {
            bdiff[i][j] = T * sqrt(T) *dot5(polytempvec,&parm.diffcoeff[ic*5]);
            bdiff[j][i] = bdiff[i][j];
            ic++;
        }
    }

    Real mmw = 0;
    for (int n=0;n<NSPECS; ++n) {
        mmw += X[n]*parm.mw[n];
    }
 
    for (size_t k=0; k<NSPECS; k++) {
        Real sum1 = 0.0;
        Real sum2 = 0.0;
        for (size_t i=0; i<NSPECS; i++) {
            if (i==k) {
                continue;
            }
            sum1 += X[i] / bdiff[k][i];
            sum2 += X[i] * parm.mw[i] / bdiff[k][i];
        }
        sum1 *= p;
        sum2 *= p * X[k] / (mmw - parm.mw[k]*X[k]);
        d[k] = 1.0 / (sum1 + sum2);
    }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
diffusion_x (int i, int j, int k,
             amrex::Array4<amrex::Real const> const& q,
             amrex::Array4<amrex::Real const> const& rhoi,
             amrex::Array4<amrex::Real> const& fx,
             amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv,
             Parm const& parm) noexcept
{
    using amrex::Real;
    Real rhor = q(i,j,k,QRHO);
    Real rhol = q(i-1,j,k,QRHO);
    Real Tr = q(i,j,k,QT);
    Real Tl = q(i-1,j,k,QT);
    Real pr = q(i,j,k,QPRES);
    Real pl = q(i-1,j,k,QPRES);
    Real di[NSPECS], Ytr[NSPECS], Ytl[NSPECS], Yt[NSPECS], Xt[NSPECS];

    for (int n=0; n<NSPECS; ++n) {
        Ytr[n] = rhoi(i,j,k,n) / rhor;
        Ytl[n] = rhoi(i-1,j,k,n) / rhol;
        Yt[n] = 0.5 * (Ytr[n] + Ytl[n]);
    }
    CKYTX(Yt, Xt, parm);
    Real T = 0.5*(Tr + Tl);
    Real p = 0.5*(pr + pl);
    Real rho = 0.5*(rhor + rhol);
    getMixDiffCoeffsMass(T, p, Xt, di, parm);

    for (int n=0; n<NSPECS; ++n) {
        fx(i,j,k,n) += rho * di[n] * (Ytl[n] - Ytr[n]) * dxinv[0];
    }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
diffusion_y (int i, int j, int k,
             amrex::Array4<amrex::Real const> const& q,
             amrex::Array4<amrex::Real const> const& rhoi,
             amrex::Array4<amrex::Real> const& fy,
             amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv,
             Parm const& parm) noexcept
{
    using amrex::Real;
    Real rhor = q(i,j,k,QRHO);
    Real rhol = q(i,j-1,k,QRHO);
    Real Tr = q(i,j,k,QT);
    Real Tl = q(i,j-1,k,QT);
    Real pr = q(i,j,k,QPRES);
    Real pl = q(i,j-1,k,QPRES);
    Real di[NSPECS], Ytr[NSPECS], Ytl[NSPECS], Yt[NSPECS], Xt[NSPECS];

    for (int n=0; n<NSPECS; ++n) {
        Ytr[n] = rhoi(i,j,k,n) / rhor;
        Ytl[n] = rhoi(i,j-1,k,n) / rhol;
        Yt[n] = 0.5 * (Ytr[n] + Ytl[n]);
    }
    CKYTX(Yt, Xt, parm);
    Real T = 0.5*(Tr + Tl);
    Real p = 0.5*(pr + pl);
    Real rho = 0.5*(rhor + rhol);
    getMixDiffCoeffsMass(T, p, Xt, di, parm);

    for (int n=0; n<NSPECS; ++n) {
        fy(i,j,k,n) += rho * di[n] * (Ytl[n] - Ytr[n]) * dxinv[1];
    }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
diffusion_z (int i, int j, int k,
             amrex::Array4<amrex::Real const> const& q,
             amrex::Array4<amrex::Real const> const& rhoi,
             amrex::Array4<amrex::Real> const& fz,
             amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv,
             Parm const& parm) noexcept
{
    using amrex::Real;
    Real rhor = q(i,j,k,QRHO);
    Real rhol = q(i,j,k-1,QRHO);
    Real Tr = q(i,j,k,QT);
    Real Tl = q(i,j,k-1,QT);
    Real pr = q(i,j,k,QPRES);
    Real pl = q(i,j,k-1,QPRES);
    Real di[NSPECS], Ytr[NSPECS], Ytl[NSPECS], Yt[NSPECS], Xt[NSPECS];

    for (int n=0; n<NSPECS; ++n) {
        Ytr[n] = rhoi(i,j,k,n) / rhor;
        Ytl[n] = rhoi(i,j,k-1,n) / rhol;
        Yt[n] = 0.5 * (Ytr[n] + Ytl[n]);
    }
    CKYTX(Yt, Xt, parm);
    Real T = 0.5*(Tr + Tl);
    Real p = 0.5*(pr + pl);
    Real rho = 0.5*(rhor + rhol);
    getMixDiffCoeffsMass(T, p, Xt, di, parm);

    for (int n=0; n<NSPECS; ++n) {
        fz(i,j,k,n) += rho * di[n] * (Ytl[n] - Ytr[n]) * dxinv[2];
    }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
compute_visc_x_multi(int i, int j, int k,
             amrex::Array4<amrex::Real const> const& q,
             amrex::Array4<amrex::Real const> const& rhoi,
             amrex::Array4<amrex::Real> const& fx,
             amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv,
             Parm const& parm) noexcept
{
    using amrex::Real;
    // first compute temperature
    Real Tl = q(i,j,k,QT);
    Real Tr = q(i-1,j,k,QT);
    Real T = 0.5*(Tl + Tr);
    // derive mu and k
    Real mu, lambda;
    Real rhor = q(i,j,k,QRHO);
    Real rhol = q(i-1,j,k,QRHO);
    Real p = 0.5*(q(i,j,k,QPRES) + q(i-1,j,k,QPRES));
    Real di[NSPECS], Ytr[NSPECS], Ytl[NSPECS], Yt[NSPECS], Xt[NSPECS], hi[NSPECS];
    for (int n=0; n<NSPECS; ++n) {
        Ytr[n] = rhoi(i,j,k,n) / rhor;
        Ytl[n] = rhoi(i-1,j,k,n) / rhol;
        Yt[n] = 0.5 * (Ytr[n] + Ytl[n]);
    }
    CKYTX(Yt, Xt, parm);
    mixtureProperties(T, Xt, mu, lambda, parm);
    speciesEnthalpy(T, hi, parm);
    Real tmp = 0;
    getMixDiffCoeffsMass(T, p, Xt, di, parm);
    for (int n=0; n<NSPECS; ++n) {
        tmp += di[n] * (Ytl[n] - Ytr[n]) * dxinv[0] * hi[n];
    }
    tmp *= 0.5*(rhol + rhor);

    Real dTdx = (Tl-Tr)*dxinv[0];
    Real dudx = (q(i,j,k,QU)-q(i-1,j,k,QU))*dxinv[0];
    Real dvdx = (q(i,j,k,QV)-q(i-1,j,k,QV))*dxinv[0];
    Real dwdx = (q(i,j,k,QW)-q(i-1,j,k,QW))*dxinv[0];
    Real dudy = (q(i,j+1,k,QU)+q(i-1,j+1,k,QU)-q(i,j-1,k,QU)-q(i-1,j-1,k,QU))*(Real(0.25)*dxinv[1]);
    Real dvdy = (q(i,j+1,k,QV)+q(i-1,j+1,k,QV)-q(i,j-1,k,QV)-q(i-1,j-1,k,QV))*(Real(0.25)*dxinv[1]);
    Real dudz = (q(i,j,k+1,QU)+q(i-1,j,k+1,QU)-q(i,j,k-1,QU)-q(i-1,j,k-1,QU))*(Real(0.25)*dxinv[2]);
    Real dwdz = (q(i,j,k+1,QW)+q(i-1,j,k+1,QW)-q(i,j,k-1,QW)-q(i-1,j,k-1,QW))*(Real(0.25)*dxinv[2]);
    Real divu = dudx + dvdy + dwdz;
    Real tauxx = mu*(Real(2.0)*dudx-parm.twoThirds*divu);
    Real tauxy = mu*(dudy+dvdx);
    Real tauxz = mu*(dudz+dwdx);
    fx(i,j,k,UMX)   += -tauxx;
    fx(i,j,k,UMY)   += -tauxy;
    fx(i,j,k,UMZ)   += -tauxz;
    fx(i,j,k,UEDEN) += -Real(0.5)*((q(i,j,k,QU)+q(i-1,j,k,QU))*tauxx \
                                  +(q(i,j,k,QV)+q(i-1,j,k,QV))*tauxy \
                                  +(q(i,j,k,QW)+q(i-1,j,k,QW))*tauxz)
                                  - lambda*dTdx - tmp;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
compute_visc_y_multi(int i, int j, int k,
             amrex::Array4<amrex::Real const> const& q,
             amrex::Array4<amrex::Real const> const& rhoi,
             amrex::Array4<amrex::Real> const& fy,
             amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv,
             Parm const& parm) noexcept
{
    using amrex::Real;
    // first compute temperature
    Real Tl = q(i,j,k,QT);
    Real Tr = q(i,j-1,k,QT);
    Real T = 0.5*(Tl + Tr);
    // derive mu and k
    Real mu, lambda;
    Real rhor = q(i,j,k,QRHO);
    Real rhol = q(i,j-1,k,QRHO);
    Real p = 0.5*(q(i,j,k,QPRES) + q(i,j-1,k,QPRES));
    Real di[NSPECS], Ytr[NSPECS], Ytl[NSPECS], Yt[NSPECS], Xt[NSPECS], hi[NSPECS];
    for (int n=0; n<NSPECS; ++n) {
        Ytr[n] = rhoi(i,j,k,n) / rhor;
        Ytl[n] = rhoi(i,j-1,k,n) / rhol;
        Yt[n] = 0.5 * (Ytr[n] + Ytl[n]);
    }
    CKYTX(Yt, Xt, parm);
    mixtureProperties(T, Xt, mu, lambda, parm);
    speciesEnthalpy(T, hi, parm);
    Real tmp = 0;
    getMixDiffCoeffsMass(T, p, Xt, di, parm);
    for (int n=0; n<NSPECS; ++n) {
        tmp += di[n] * (Ytl[n] - Ytr[n]) * dxinv[1] * hi[n];
    }
    tmp *= 0.5*(rhol + rhor);

    Real dTdy = (Tl-Tr)*dxinv[1];
    Real dudy = (q(i,j,k,QU)-q(i,j-1,k,QU))*dxinv[1];
    Real dvdy = (q(i,j,k,QV)-q(i,j-1,k,QV))*dxinv[1];
    Real dwdy = (q(i,j,k,QW)-q(i,j-1,k,QW))*dxinv[1];
    Real dudx = (q(i+1,j,k,QU)+q(i+1,j-1,k,QU)-q(i-1,j,k,QU)-q(i-1,j-1,k,QU))*(Real(0.25)*dxinv[0]);
    Real dvdx = (q(i+1,j,k,QV)+q(i+1,j-1,k,QV)-q(i-1,j,k,QV)-q(i-1,j-1,k,QV))*(Real(0.25)*dxinv[0]);
    Real dvdz = (q(i,j,k+1,QV)+q(i,j-1,k+1,QV)-q(i,j,k-1,QV)-q(i,j-1,k-1,QV))*(Real(0.25)*dxinv[2]);
    Real dwdz = (q(i,j,k+1,QW)+q(i,j-1,k+1,QW)-q(i,j,k-1,QW)-q(i,j-1,k-1,QW))*(Real(0.25)*dxinv[2]);
    Real divu = dudx + dvdy + dwdz;
    Real tauyy = mu*(Real(2.0)*dvdy-parm.twoThirds*divu);
    Real tauxy = mu*(dudy+dvdx);
    Real tauyz = mu*(dwdy+dvdz);
    fy(i,j,k,UMX)   += -tauxy;
    fy(i,j,k,UMY)   += -tauyy;
    fy(i,j,k,UMZ)   += -tauyz;
    fy(i,j,k,UEDEN) += -Real(0.5)*((q(i,j,k,QU)+q(i,j-1,k,QU))*tauxy \
                                  +(q(i,j,k,QV)+q(i,j-1,k,QV))*tauyy \
                                  +(q(i,j,k,QW)+q(i,j-1,k,QW))*tauyz)
                                  - lambda*dTdy - tmp;
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
compute_visc_z_multi(int i, int j, int k,
             amrex::Array4<amrex::Real const> const& q,
             amrex::Array4<amrex::Real const> const& rhoi,
             amrex::Array4<amrex::Real> const& fz,
             amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dxinv,
             Parm const& parm) noexcept
{
    using amrex::Real;
    // first compute temperature
    Real Tl = q(i,j,k,QT);
    Real Tr = q(i,j,k-1,QT);
    Real T = 0.5*(Tl + Tr);
    // derive mu and k
    Real mu, lambda;
    Real rhor = q(i,j,k,QRHO);
    Real rhol = q(i,j,k-1,QRHO);
    Real p = 0.5*(q(i,j,k,QPRES) + q(i,j,k-1,QPRES));
    Real di[NSPECS], Ytr[NSPECS], Ytl[NSPECS], Yt[NSPECS], Xt[NSPECS], hi[NSPECS];
    for (int n=0; n<NSPECS; ++n) {
        Ytr[n] = rhoi(i,j,k,n) / rhor;
        Ytl[n] = rhoi(i,j,k-1,n) / rhol;
        Yt[n] = 0.5 * (Ytr[n] + Ytl[n]);
    }
    CKYTX(Yt, Xt, parm);
    mixtureProperties(T, Xt, mu, lambda, parm);
    speciesEnthalpy(T, hi, parm);
    Real tmp = 0;
    getMixDiffCoeffsMass(T, p, Xt, di, parm);
    for (int n=0; n<NSPECS; ++n) {
        tmp += di[n] * (Ytl[n] - Ytr[n]) * dxinv[2] * hi[n];
    }
    tmp *= 0.5*(rhol + rhor);

    Real dTdz = (Tl-Tr)*dxinv[2];
    Real dudz = (q(i,j,k,QU)-q(i,j,k-1,QU))*dxinv[2];
    Real dvdz = (q(i,j,k,QV)-q(i,j,k-1,QV))*dxinv[2];
    Real dwdz = (q(i,j,k,QW)-q(i,j,k-1,QW))*dxinv[2];
    Real dudx = (q(i+1,j,k,QU)+q(i+1,j,k-1,QU)-q(i-1,j,k,QU)-q(i-1,j,k-1,QU))*(Real(0.25)*dxinv[0]);
    Real dwdx = (q(i+1,j,k,QW)+q(i+1,j,k-1,QW)-q(i-1,j,k,QW)-q(i-1,j,k-1,QW))*(Real(0.25)*dxinv[0]);
    Real dvdy = (q(i,j+1,k,QV)+q(i,j+1,k-1,QV)-q(i,j-1,k,QV)-q(i,j-1,k-1,QV))*(Real(0.25)*dxinv[1]);
    Real dwdy = (q(i,j+1,k,QW)+q(i,j+1,k-1,QW)-q(i,j-1,k,QW)-q(i,j-1,k-1,QW))*(Real(0.25)*dxinv[1]);
    Real divu = dudx + dvdy + dwdz;
    Real tauxz = mu*(dudz+dwdx);
    Real tauyz = mu*(dvdz+dwdy);
    Real tauzz = mu*(Real(2.0)*dwdz-parm.twoThirds*divu);
    fz(i,j,k,UMX)   += -tauxz;
    fz(i,j,k,UMY)   += -tauyz;
    fz(i,j,k,UMZ)   += -tauzz;
    fz(i,j,k,UEDEN) += -Real(0.5)*((q(i,j,k,QU)+q(i,j,k-1,QU))*tauxz \
                                  +(q(i,j,k,QV)+q(i,j,k-1,QV))*tauyz \
                                  +(q(i,j,k,QW)+q(i,j,k-1,QW))*tauzz)
                                  - lambda*dTdz - tmp;
}
#endif