#ifndef EBKERNELS_H
#define EBKERNELS_H

#include "IndexDefines.H"
#include <Constants.H>
#include <EBR.H>

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void eb_recon(int i, int j, int k, amrex::Array4<amrex::Real> const &ql,
              amrex::Array4<amrex::Real> const &qr,
              amrex::Array4<amrex::Real const> const &q,
              amrex::Array4<amrex::EBCellFlag const> const &flag,
              Parm const &parm, int cdir) noexcept {
  if (cdir == 0) {
    if (flag(i, j, k).isConnected(-1, 0, 0)) {
      for (int n = 0; n < NPRIM; ++n) {
        ql(i, j, k, n) = q(i - 1, j, k, n);
        qr(i, j, k, n) = q(i, j, k, n);
      }
    } else {
      if (flag(i, j, k).isCovered()) {
        for (int n = 0; n < NPRIM; ++n) {
          ql(i, j, k, n) = q(i - 1, j, k, n);
        }
        qr(i, j, k, QRHO) = q(i - 1, j, k, QRHO);
        qr(i, j, k, QU) = -q(i - 1, j, k, QU);
        qr(i, j, k, QV) = q(i - 1, j, k, QV);
        qr(i, j, k, QW) = q(i - 1, j, k, QW);
        qr(i, j, k, QPRES) = q(i - 1, j, k, QPRES);
      } else {
        for (int n = 0; n < NPRIM; ++n) {
          qr(i, j, k, n) = q(i, j, k, n);
        }
        ql(i, j, k, QRHO) = q(i, j, k, QRHO);
        ql(i, j, k, QU) = -q(i, j, k, QU);
        ql(i, j, k, QV) = q(i, j, k, QV);
        ql(i, j, k, QW) = q(i, j, k, QW);
        ql(i, j, k, QPRES) = q(i, j, k, QPRES);
      }
    }
  } else if (cdir == 1) {
    if (flag(i, j, k).isConnected(0, -1, 0)) {
      for (int n = 0; n < NPRIM; ++n) {
        ql(i, j, k, n) = q(i, j - 1, k, n);
        qr(i, j, k, n) = q(i, j, k, n);
      }
    } else {
      if (flag(i, j, k).isCovered()) {
        for (int n = 0; n < NPRIM; ++n) {
          ql(i, j, k, n) = q(i, j - 1, k, n);
        }
        qr(i, j, k, QRHO) = q(i, j - 1, k, QRHO);
        qr(i, j, k, QU) = -q(i, j - 1, k, QU);
        qr(i, j, k, QV) = q(i, j - 1, k, QV);
        qr(i, j, k, QW) = q(i, j - 1, k, QW);
        qr(i, j, k, QPRES) = q(i, j - 1, k, QPRES);
      } else {
        for (int n = 0; n < NPRIM; ++n) {
          qr(i, j, k, n) = q(i, j, k, n);
        }
        ql(i, j, k, QRHO) = q(i, j, k, QRHO);
        ql(i, j, k, QU) = -q(i, j, k, QU);
        ql(i, j, k, QV) = q(i, j, k, QV);
        ql(i, j, k, QW) = q(i, j, k, QW);
        ql(i, j, k, QPRES) = q(i, j, k, QPRES);
      }
    }
  } else {
    if (flag(i, j, k).isConnected(0, 0, -1)) {
      for (int n = 0; n < NPRIM; ++n) {
        ql(i, j, k, n) = q(i, j, k - 1, n);
        qr(i, j, k, n) = q(i, j, k, n);
      }
    } else {
      if (flag(i, j, k).isCovered()) {
        for (int n = 0; n < NPRIM; ++n) {
          ql(i, j, k, n) = q(i, j, k - 1, n);
        }
        qr(i, j, k, QRHO) = q(i, j, k - 1, QRHO);
        qr(i, j, k, QU) = -q(i, j, k - 1, QU);
        qr(i, j, k, QV) = q(i, j, k - 1, QV);
        qr(i, j, k, QW) = q(i, j, k - 1, QW);
        qr(i, j, k, QPRES) = q(i, j, k - 1, QPRES);
      } else {
        for (int n = 0; n < NPRIM; ++n) {
          qr(i, j, k, n) = q(i, j, k, n);
        }
        ql(i, j, k, QRHO) = q(i, j, k, QRHO);
        ql(i, j, k, QU) = -q(i, j, k, QU);
        ql(i, j, k, QV) = q(i, j, k, QV);
        ql(i, j, k, QW) = q(i, j, k, QW);
        ql(i, j, k, QPRES) = q(i, j, k, QPRES);
      }
    }
  }
}

// Steger-Warming
AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void eb_flux(int i, int j, int k, amrex::Array4<amrex::Real const> const &ql,
             amrex::Array4<amrex::Real const> const &qr,
             amrex::Array4<amrex::Real> const &f, Parm const &parm,
             int cdir) noexcept {
  using amrex::Real;
  amrex::GpuArray<Real, NCONS> fm;
  amrex::GpuArray<Real, NCONS> fp;
  Real tmp = Real(1.0) / (Real(2.0) * parm.eos_gamma);

  Real uul;
  Real vvl;
  Real wwl;
  Real uur;
  Real vvr;
  Real wwr;

  Real dl = ql(i, j, k, QRHO);
  Real dr = qr(i, j, k, QRHO);
  Real pl = amrex::max(ql(i, j, k, QPRES), parm.smallp);
  Real pr = amrex::max(qr(i, j, k, QPRES), parm.smallp);
  if (cdir == 0) {
    uul = ql(i, j, k, QU);
    vvl = ql(i, j, k, QV);
    wwl = ql(i, j, k, QW);
    uur = qr(i, j, k, QU);
    vvr = qr(i, j, k, QV);
    wwr = qr(i, j, k, QW);
  } else if (cdir == 1) {
    uul = ql(i, j, k, QV);
    vvl = -ql(i, j, k, QU);
    wwl = ql(i, j, k, QW);
    uur = qr(i, j, k, QV);
    vvr = -qr(i, j, k, QU);
    wwr = qr(i, j, k, QW);
  } else {
    uul = ql(i, j, k, QW);
    vvl = ql(i, j, k, QV);
    wwl = -ql(i, j, k, QU);
    uur = qr(i, j, k, QW);
    vvr = qr(i, j, k, QV);
    wwr = -qr(i, j, k, QU);
  }

  Real al = std::sqrt(parm.eos_gamma * pl / dl);
  Real ar = std::sqrt(parm.eos_gamma * pr / dr);

  Real tmp1 = Real(2.0) * (parm.eos_gamma - Real(1.0));
  Real tmp3 =
      (Real(3.0) - parm.eos_gamma) / (Real(2.0) * (parm.eos_gamma - Real(1.0)));
  // eigenvalues---------
  Real E1P = (uul + amrex::Math::abs(uul)) * Real(0.5);
  Real E2P = (uul - al + amrex::Math::abs(uul - al)) * Real(0.5);
  Real E3P = (uul + al + amrex::Math::abs(uul + al)) * Real(0.5);
  Real tmp0 = dl * tmp;
  fp[0] = tmp0 * (tmp1 * E1P + E2P + E3P);
  fp[1] = tmp0 * (tmp1 * E1P * uul + E2P * (uul - al) + E3P * (uul + al));
  fp[2] = tmp0 * (tmp1 * E1P * vvl + E2P * vvl + E3P * vvl);
  fp[3] = tmp0 * (tmp1 * E1P * wwl + E2P * wwl + E3P * wwl);
  fp[4] = tmp0 *
          (E1P * (parm.eos_gamma - Real(1.0)) *
               (uul * uul + vvl * vvl + wwl * wwl) +
           E2P * ((uul - al) * (uul - al) + vvl * vvl + wwl * wwl) * Real(0.5) +
           E3P * ((uul + al) * (uul + al) + vvl * vvl + wwl * wwl) * Real(0.5) +
           tmp3 * al * al * (E2P + E3P));

  Real E1M = (uur - amrex::Math::abs(uur)) * Real(0.5);
  Real E2M = (uur - ar - amrex::Math::abs(uur - ar)) * Real(0.5);
  Real E3M = (uur + ar - amrex::Math::abs(uur + ar)) * Real(0.5);
  tmp0 = dr * tmp;
  fm[0] = tmp0 * (tmp1 * E1M + E2M + E3M);
  fm[1] = tmp0 * (tmp1 * E1M * uur + E2M * (uur - ar) + E3M * (uur + ar));
  fm[2] = tmp0 * (tmp1 * E1M * vvr + E2M * vvr + E3M * vvr);
  fm[3] = tmp0 * (tmp1 * E1M * wwr + E2M * wwr + E3M * wwr);
  fm[4] = tmp0 *
          (E1M * (parm.eos_gamma - Real(1.0)) *
               (uur * uur + vvr * vvr + wwr * wwr) +
           E2M * ((uur - ar) * (uur - ar) + vvr * vvr + wwr * wwr) * Real(0.5) +
           E3M * ((uur + ar) * (uur + ar) + vvr * vvr + wwr * wwr) * Real(0.5) +
           tmp3 * ar * ar * (E2M + E3M));

  if (cdir == 0) {
    for (int n = 0; n < NCONS; ++n) {
      f(i, j, k, n) = fp[n] + fm[n];
    }
  } else if (cdir == 1) {
    f(i, j, k, URHO) = fp[URHO] + fm[URHO];
    f(i, j, k, UMX) = -(fp[UMY] + fm[UMY]);
    f(i, j, k, UMY) = fp[UMX] + fm[UMX];
    f(i, j, k, UMZ) = fp[UMZ] + fm[UMZ];
    f(i, j, k, UEDEN) = fp[UEDEN] + fm[UEDEN];
  } else {
    f(i, j, k, URHO) = fp[URHO] + fm[URHO];
    f(i, j, k, UMX) = -(fp[UMZ] + fm[UMZ]);
    f(i, j, k, UMY) = fp[UMY] + fm[UMY];
    f(i, j, k, UMZ) = fp[UMX] + fm[UMX];
    f(i, j, k, UEDEN) = fp[UEDEN] + fm[UEDEN];
  }
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
riemann(const amrex::Real gamma, const amrex::Real smallp,
        const amrex::Real /*smallr*/, const amrex::Real rl,
        const amrex::Real ul, const amrex::Real pl, const amrex::Real ut1l,
        const amrex::Real ut2l, const amrex::Real rr, const amrex::Real ur,
        const amrex::Real pr, const amrex::Real ut1r, const amrex::Real ut2r,
        amrex::Real &flxrho, amrex::Real &flxu, amrex::Real &flxut,
        amrex::Real &flxutt, amrex::Real &flxe) noexcept {
  using amrex::Real;

  constexpr Real weakwv = Real(1.e-3);
  constexpr Real small = Real(1.e-6);

  Real clsql = gamma * pl * rl;
  Real clsqr = gamma * pr * rr;
  Real wl = std::sqrt(clsql);
  Real wr = std::sqrt(clsqr);
  Real cleft = wl / rl;
  Real cright = wr / rr;
  Real ccsmall = small * (cleft + cright);

  Real pstar = (wl * pr + wr * pl - wr * wl * (ur - ul)) / (wl + wr);
  pstar = amrex::max(pstar, smallp);
  Real pstnm1 = pstar;

  Real wlsq = (Real(0.5) * (gamma - Real(1.)) * (pstar + pl) + pstar) * rl;
  Real wrsq = (Real(0.5) * (gamma - Real(1.)) * (pstar + pr) + pstar) * rr;

  wl = std::sqrt(wlsq);
  wr = std::sqrt(wrsq);
  Real ustarp = ul - (pstar - pl) / wl;
  Real ustarm = ur + (pstar - pr) / wr;

  pstar = (wl * pr + wr * pl - wr * wl * (ur - ul)) / (wl + wr);
  pstar = amrex::max(pstar, smallp);

  Real ustar;
  for (int iter = 0; iter < 3; ++iter) {
    wlsq = (Real(0.5) * (gamma - Real(1.)) * (pstar + pl) + pstar) * rl;
    wrsq = (Real(0.5) * (gamma - Real(1.)) * (pstar + pr) + pstar) * rr;

    wl = Real(1.) / std::sqrt(wlsq);
    wr = Real(1.) / std::sqrt(wrsq);

    Real ustnm1 = ustarm;
    Real ustnp1 = ustarp;

    ustarm = ur - (pr - pstar) * wr;
    ustarp = ul + (pl - pstar) * wl;

    Real dpditer = std::abs(pstnm1 - pstar);
    Real zp = std::abs(ustarp - ustnp1);
    if (zp - weakwv * cleft < Real(0.0)) {
      zp = dpditer * wl;
    }
    Real zm = std::abs(ustarm - ustnm1);
    if (zm - weakwv * cright < Real(0.0)) {
      zm = dpditer * wr;
    }

    Real zz = zp + zm;
    Real denom = dpditer / amrex::max(zz, ccsmall);
    pstnm1 = pstar;
    pstar = pstar - denom * (ustarm - ustarp);
    pstar = amrex::max(pstar, smallp);
    ustar = Real(0.5) * (ustarm + ustarp);
  }

  Real ro, uo, po, sgnm, utrans1;
  Real utrans2;
  if (ustar > Real(0.)) {
    ro = rl;
    uo = ul;
    po = pl;
    sgnm = Real(1.);
    utrans1 = ut1l;
    utrans2 = ut2l;
  } else if (ustar < Real(0.)) {
    ro = rr;
    uo = ur;
    po = pr;
    sgnm = Real(-1.);
    utrans1 = ut1r;
    utrans2 = ut2r;
  } else {
    uo = Real(0.5) * (ur + ul);
    po = Real(0.5) * (pr + pl);
    ro = Real(2.) * (rl * rr) / (rl + rr);
    sgnm = Real(1.);
    utrans1 = Real(0.5) * (ut1l + ut1r);
    utrans2 = Real(0.5) * (ut2l + ut2r);
  }
  Real wosq = (Real(0.5) * (gamma - Real(1.)) * (pstar + po) + pstar) * ro;
  Real co = std::sqrt(gamma * po / ro);
  Real wo = std::sqrt(wosq);
  Real dpjmp = pstar - po;
  Real rstar = ro / (Real(1.) - ro * dpjmp / wosq);
  Real cstar = std::sqrt(gamma * pstar / rstar);
  Real spout = co - sgnm * uo;
  Real spin = cstar - sgnm * uo;
  if (pstar >= po) {
    spin = wo / ro - sgnm * uo;
    spout = spin;
  }
  Real ss = amrex::max(spout - spin, spout + spin);
  Real frac = Real(0.5) * (Real(1.) + (spin + spout) / amrex::max(ss, ccsmall));

  Real rgdnv, ugdnv, pgdnv;
  if (spout < Real(0.)) {
    rgdnv = ro;
    ugdnv = uo;
    pgdnv = po;
  } else if (spin >= Real(0.)) {
    rgdnv = rstar;
    ugdnv = ustar;
    pgdnv = pstar;
  } else {
    rgdnv = frac * rstar + (Real(1.) - frac) * ro;
    ugdnv = frac * ustar + (Real(1.) - frac) * uo;
    pgdnv = frac * pstar + (Real(1.) - frac) * po;
  }

  flxrho = rgdnv * ugdnv;
  flxu = rgdnv * ugdnv * ugdnv + pgdnv;
  flxut = rgdnv * ugdnv * utrans1;
  flxutt = rgdnv * ugdnv * utrans2;
  flxe = ugdnv * (Real(0.5) * rgdnv *
                      (ugdnv * ugdnv + utrans1 * utrans1 + utrans2 * utrans2) +
                  pgdnv / (gamma - Real(1.)) + pgdnv);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void compute_hyp_wallflux(
    const amrex::Real rho,
    AMREX_D_DECL(const amrex::Real u, const amrex::Real v, const amrex::Real w),
    const amrex::Real p, const amrex::Real axm, const amrex::Real axp,
    const amrex::Real aym, const amrex::Real ayp, const amrex::Real azm,
    const amrex::Real azp, amrex::GpuArray<amrex::Real, NCONS> &fluxw,
    Parm const &parm) noexcept {
  using amrex::Real;
  Real apnorm =
      std::sqrt((axm - axp) * (axm - axp) + (aym - ayp) * (aym - ayp) +
                (azm - azp) * (azm - azp));
  Real apnorminv = 1. / apnorm;
  Real un = u * (axm - axp) * apnorminv + v * (aym - ayp) * apnorminv +
            w * (azm - azp) * apnorminv;

  Real AMREX_D_DECL(momfluxn, momfluxt1, momfluxt2);

  riemann(parm.eos_gamma, parm.smallp, parm.smallr, rho, un, p, 0., 0., rho,
          -un, p, 0., 0., fluxw[URHO],
          AMREX_D_DECL(momfluxn, momfluxt1, momfluxt2), fluxw[UEDEN]);

  AMREX_D_TERM(fluxw[UMX] = (axm - axp) * momfluxn;
               , fluxw[UMY] = (aym - ayp) * momfluxn;
               , fluxw[UMZ] = (azm - azp) * momfluxn;);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE void eb_compute_div(
    int i, int j, int k, int n, IntVect const &blo, IntVect const &bhi,
    Array4<Real> const &q, Array4<Real> const &divu,
    AMREX_D_DECL(Array4<Real const> const &u, Array4<Real const> const &v,
                 Array4<Real const> const &w),
    AMREX_D_DECL(Array4<Real> const &fx, Array4<Real> const &fy,
                 Array4<Real> const &fz),
    Array4<EBCellFlag const> const &flag, Array4<Real const> const &vfrc,
    Array4<Real const> const &bcent, Array4<Real> const &redistwgt,
    AMREX_D_DECL(Array4<Real const> const &apx, Array4<Real const> const &apy,
                 Array4<Real const> const &apz),
    AMREX_D_DECL(Array4<Real const> const &fcx, Array4<Real const> const &fcy,
                 Array4<Real const> const &fcz),
    GpuArray<Real, AMREX_SPACEDIM> const &dxinv, Parm const &parm,
    int eb_weights_type) {
  AMREX_D_TERM(bool x_high = (i == bhi[0]);, bool y_high = (j == bhi[1]);
               , bool z_high = (k == bhi[2]));
  bool valid_cell = AMREX_D_TERM((blo[0] <= i) && (i <= bhi[0]),
                                 &&(blo[1] <= j) && (j <= bhi[1]),
                                 &&(blo[2] <= k) && (k <= bhi[2]));
  if (flag(i, j, k).isCovered()) {
    divu(i, j, k, n) = 0.0;
    if (valid_cell) {
      fx(i, j, k, n) = 0.;
      fy(i, j, k, n) = 0.;
      fz(i, j, k, n) = 0.;
      if (x_high) {
        fx(i + 1, j, k, n) = 0.;
      }
      if (y_high) {
        fy(i, j + 1, k, n) = 0.;
      }
      if (z_high) {
        fz(i, j, k + 1, n) = 0.;
      }
    }
  } else if (flag(i, j, k).isRegular()) {
    divu(i, j, k, n) = dxinv[0] * (u(i + 1, j, k, n) - u(i, j, k, n)) +
                       dxinv[1] * (v(i, j + 1, k, n) - v(i, j, k, n)) +
                       dxinv[2] * (w(i, j, k + 1, n) - w(i, j, k, n));
    if (valid_cell) {
      fx(i, j, k, n) = u(i, j, k, n);
      fy(i, j, k, n) = v(i, j, k, n);
      fz(i, j, k, n) = w(i, j, k, n);
      if (x_high) {
        fx(i + 1, j, k, n) = u(i + 1, j, k, n);
      }
      if (y_high) {
        fy(i, j + 1, k, n) = v(i, j + 1, k, n);
      }
      if (z_high) {
        fz(i, j, k + 1, n) = w(i, j, k + 1, n);
      }
    }
  } else {
    Real fxm = u(i, j, k, n);
    if (apx(i, j, k) != 0.0 && apx(i, j, k) != 1.0) {
      int jj = j + static_cast<int>(std::copysign(1.0_rt, fcx(i, j, k, 0)));
      int kk = k + static_cast<int>(std::copysign(1.0_rt, fcx(i, j, k, 1)));
      Real fracy = flag(i, j, k).isConnected(0, jj - j, 0)
                       ? std::abs(fcx(i, j, k, 0))
                       : 0.0_rt;
      Real fracz = flag(i, j, k).isConnected(0, 0, kk - k)
                       ? std::abs(fcx(i, j, k, 1))
                       : 0.0_rt;
      fxm = (1.0 - fracy) * (1.0 - fracz) * fxm +
            fracy * (1.0 - fracz) * u(i, jj, k, n) +
            fracz * (1.0 - fracy) * u(i, j, kk, n) +
            fracy * fracz * u(i, jj, kk, n);
    }
    if (valid_cell) {
      fx(i, j, k, n) = fxm;
    }

    Real fxp = u(i + 1, j, k, n);
    if (apx(i + 1, j, k) != 0.0 && apx(i + 1, j, k) != 1.0) {
      int jj = j + static_cast<int>(std::copysign(1.0_rt, fcx(i + 1, j, k, 0)));
      int kk = k + static_cast<int>(std::copysign(1.0_rt, fcx(i + 1, j, k, 1)));
      Real fracy = flag(i + 1, j, k).isConnected(0, jj - j, 0)
                       ? std::abs(fcx(i + 1, j, k, 0))
                       : 0.0_rt;
      Real fracz = flag(i + 1, j, k).isConnected(0, 0, kk - k)
                       ? std::abs(fcx(i + 1, j, k, 1))
                       : 0.0_rt;
      fxp = (1.0 - fracy) * (1.0 - fracz) * fxp +
            fracy * (1.0 - fracz) * u(i + 1, jj, k, n) +
            fracz * (1.0 - fracy) * u(i + 1, j, kk, n) +
            fracy * fracz * u(i + 1, jj, kk, n);
    }
    if (valid_cell && x_high) {
      fx(i + 1, j, k, n) = fxp;
    }

    Real fym = v(i, j, k, n);
    if (apy(i, j, k) != 0.0 && apy(i, j, k) != 1.0) {
      int ii = i + static_cast<int>(std::copysign(1.0_rt, fcy(i, j, k, 0)));
      int kk = k + static_cast<int>(std::copysign(1.0_rt, fcy(i, j, k, 1)));
      Real fracx = flag(i, j, k).isConnected(ii - i, 0, 0)
                       ? std::abs(fcy(i, j, k, 0))
                       : 0.0_rt;
      Real fracz = flag(i, j, k).isConnected(0, 0, kk - k)
                       ? std::abs(fcy(i, j, k, 1))
                       : 0.0_rt;
      fym = (1.0 - fracx) * (1.0 - fracz) * fym +
            fracx * (1.0 - fracz) * v(ii, j, k, n) +
            fracz * (1.0 - fracx) * v(i, j, kk, n) +
            fracx * fracz * v(ii, j, kk, n);
    }
    if (valid_cell) {
      fy(i, j, k, n) = fym;
    }

    Real fyp = v(i, j + 1, k, n);
    if (apy(i, j + 1, k) != 0.0 && apy(i, j + 1, k) != 1.0) {
      int ii = i + static_cast<int>(std::copysign(1.0_rt, fcy(i, j + 1, k, 0)));
      int kk = k + static_cast<int>(std::copysign(1.0_rt, fcy(i, j + 1, k, 1)));
      Real fracx = flag(i, j + 1, k).isConnected(ii - i, 0, 0)
                       ? std::abs(fcy(i, j + 1, k, 0))
                       : 0.0_rt;
      Real fracz = flag(i, j + 1, k).isConnected(0, 0, kk - k)
                       ? std::abs(fcy(i, j + 1, k, 1))
                       : 0.0_rt;
      fyp = (1.0 - fracx) * (1.0 - fracz) * fyp +
            fracx * (1.0 - fracz) * v(ii, j + 1, k, n) +
            fracz * (1.0 - fracx) * v(i, j + 1, kk, n) +
            fracx * fracz * v(ii, j + 1, kk, n);
    }
    if (valid_cell && y_high) {
      fy(i, j + 1, k, n) = fyp;
    }

    Real fzm = w(i, j, k, n);
    if (apz(i, j, k) != 0.0 && apz(i, j, k) != 1.0) {
      int ii = i + static_cast<int>(std::copysign(1.0_rt, fcz(i, j, k, 0)));
      int jj = j + static_cast<int>(std::copysign(1.0_rt, fcz(i, j, k, 1)));
      Real fracx = flag(i, j, k).isConnected(ii - i, 0, 0)
                       ? std::abs(fcz(i, j, k, 0))
                       : 0.0_rt;
      Real fracy = flag(i, j, k).isConnected(0, jj - j, 0)
                       ? std::abs(fcz(i, j, k, 1))
                       : 0.0_rt;

      fzm = (1.0 - fracx) * (1.0 - fracy) * fzm +
            fracx * (1.0 - fracy) * w(ii, j, k, n) +
            fracy * (1.0 - fracx) * w(i, jj, k, n) +
            fracx * fracy * w(ii, jj, k, n);
    }
    if (valid_cell) {
      fz(i, j, k, n) = fzm;
    }

    Real fzp = w(i, j, k + 1, n);
    if (apz(i, j, k + 1) != 0.0 && apz(i, j, k + 1) != 1.0) {
      int ii = i + static_cast<int>(std::copysign(1.0_rt, fcz(i, j, k + 1, 0)));
      int jj = j + static_cast<int>(std::copysign(1.0_rt, fcz(i, j, k + 1, 1)));
      Real fracx = flag(i, j, k + 1).isConnected(ii - i, 0, 0)
                       ? std::abs(fcz(i, j, k + 1, 0))
                       : 0.0_rt;
      Real fracy = flag(i, j, k + 1).isConnected(0, jj - j, 0)
                       ? std::abs(fcz(i, j, k + 1, 1))
                       : 0.0_rt;
      fzp = (1.0 - fracx) * (1.0 - fracy) * fzp +
            fracx * (1.0 - fracy) * w(ii, j, k + 1, n) +
            fracy * (1.0 - fracx) * w(i, jj, k + 1, n) +
            fracx * fracy * w(ii, jj, k + 1, n);
    }
    if (valid_cell && z_high) {
      fz(i, j, k + 1, n) = fzp;
    }

    divu(i, j, k, n) =
        (1.0 / vfrc(i, j, k)) *
        (dxinv[0] * (apx(i + 1, j, k) * fxp - apx(i, j, k) * fxm) +
         dxinv[1] * (apy(i, j + 1, k) * fyp - apy(i, j, k) * fym) +
         dxinv[2] * (apz(i, j, k + 1) * fzp - apz(i, j, k) * fzm));

    GpuArray<Real, NCONS> flux_hyp_wall;
    compute_hyp_wallflux(
        q(i, j, k, QRHO),
        AMREX_D_DECL(q(i, j, k, QU), q(i, j, k, QV), q(i, j, k, QW)),
        q(i, j, k, QPRES), apx(i, j, k), apx(i + 1, j, k), apy(i, j, k),
        apy(i, j + 1, k), apz(i, j, k), apz(i, j, k + 1), flux_hyp_wall, parm);

    // Here we assume dx == dy == dz
    divu(i, j, k, n) += flux_hyp_wall[n] * dxinv[0] / vfrc(i, j, k);
  }

  // The operations following this assume we have returned the negative of the
  // divergence of fluxes.
  divu(i, j, k, n) *= -1.0;

  // Go ahead and make the redistwgt array here since we'll need it in
  // flux_redistribute
  if (eb_weights_type == 0) {
    redistwgt(i, j, k) = 1.0;
  } else if (eb_weights_type == 1) {
    redistwgt(i, j, k) =
        q(i, j, k, QPRES) / (parm.eos_gamma - 1) +
        0.5 * q(i, j, k, QRHO) *
            (q(i, j, k, QU) * q(i, j, k, QU) + q(i, j, k, QV) * q(i, j, k, QV) +
             q(i, j, k, QW) * q(i, j, k, QW));
  } else if (eb_weights_type == 2) {
    redistwgt(i, j, k) = q(i, j, k, QRHO);
  } else if (eb_weights_type == 3) {
    redistwgt(i, j, k) = vfrc(i, j, k);
  }
}

#endif