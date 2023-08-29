#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>

using namespace std;

const int NSPECS = 9, NREACS = 19;

// kg/mol
static const double mw[NSPECS] = {2.016 * 1e-3,   /*H2 */
                                  31.998 * 1e-3,   /*O2 */
                                  18.015 * 1e-3,   /*H2O */
                                  1.008 * 1e-3,   /*H */
                                  15.999 * 1e-3,   /*O */
                                  17.007 * 1e-3,   /*OH */
                                  33.006 * 1e-3,   /*HO2 */
                                  34.014 * 1e-3,   /*H2O2 */
                                  28.014 * 1e-3}; /*N2 */

static const double H2_lo[7] = {3.29812,      8.24944e-4,    -81.43015e-8,
                                -9.47543e-11, 413.48718e-15, -0.01013e+5,
                                -3.29409};
static const double H2_hi[7] = {2.99142,      7.00064e-4,  -5.63383e-8,
                                -0.92316e-11, 1.58275e-15, -0.00835e+5,
                                -1.35511};
static const double O2_lo[7] = {3.21294,       11.27486e-4,   -57.56150e-8,
                                131.38770e-11, -876.8554e-15, -0.01005e+5,
                                6.03474};
static const double O2_hi[7] = {3.69758,     6.13520e-4,   -12.58842e-8,
                                1.77528e-11, -1.13644e-15, -0.01234e+5,
                                3.18917};
static const double H2O_lo[7] = {3.38684,      34.74982e-4,   -635.4696e-8,
                                 696.8581e-11, -2506.588e-15, -0.30208e+5,
                                 2.59023};
static const double H2O_hi[7] = {2.67215,      30.56293e-4,  -87.30260e-8,
                                 12.00996e-11, -6.39162e-15, -0.29899e+5,
                                 6.86282};
static const double H_lo[7] = {2.5, 0.0, 0.0, 0.0, 0.0, 0.25472e+5, -0.46012};
static const double H_hi[7] = {2.5, 0.0, 0.0, 0.0, 0.0, 0.25472e+5, -0.46012};
static const double O_lo[7] = {2.94643,      -16.3817e-4,   242.10320e-8,
                               -160.284e-11, 389.06961e-15, 0.29148e+5,
                               2.96399};
static const double O_hi[7] = {2.54206,     -0.27551e-4,  -0.31028e-8,
                               0.45511e-11, -0.43681e-15, 0.29231e+5,
                               4.92031};
static const double OH_lo[7] = {3.63727,       1.85091e-4,    -167.6165e-8,
                                238.72031e-11, -843.1442e-15, 0.03607e+5,
                                1.35886};
static const double OH_hi[7] = {2.88273,     10.13974e-4,  -22.76877e-8,
                                2.17468e-11, -0.51263e-15, 0.03887e+5,
                                5.59571};
static const double HO2_lo[7] = {2.97996,       49.96697e-4,   -379.0997e-8,
                                 235.41919e-11, -808.9024e-15, 0.00176e+5,
                                 9.22272};
static const double HO2_hi[7] = {4.07219,     21.31296e-4,  -53.08145e-8,
                                 6.11227e-11, -2.84116e-15, -0.00158e+5,
                                 3.47603};
static const double H2O2_lo[7] = {3.38875,        65.69226e-4,  -14.85013e-8,
                                  -462.58060e-11, 2471.515e-15, -0.17663e+5,
                                  6.78536};
static const double H2O2_hi[7] = {4.57317,      43.36136e-4,   -147.4689e-8,
                                  23.48904e-11, -14.31654e-15, -0.18007e+5,
                                  0.50114};
static const double N2_lo[7] = {3.29868,       14.08240e-4,   -396.3222e-8,
                                564.15149e-11, -2444.855e-15, -0.01021e+5,
                                3.95037};
static const double N2_hi[7] = {2.92664,      14.87977e-4,  -56.84761e-8,
                                10.09704e-11, -6.75335e-15, -0.00923e+5,
                                5.98053};

// enthalpy of production at lo
// K
static const double HP[NSPECS] = {H2_lo[5], O2_lo[5], H2O_lo[5], H_lo[5], O_lo[5], OH_lo[5], HO2_lo[5], H2O2_lo[5], N2_lo[5]};

// used to get initial T
static const double A0[NSPECS] = {H2_lo[0]-1.0, O2_lo[0]-1.0, H2O_lo[0]-1.0, H_lo[0]-1.0, O_lo[0]-1.0, OH_lo[0]-1.0, HO2_lo[0]-1.0, H2O2_lo[0]-1.0, N2_lo[0]-1.0};

// Pa
static const double Patm = 101325.0, Ru = 8.31446261815324;

void CKPY(double *rhoi, double T, double &P) {
  double YOW = 0; /* for computing mean MW */
  for (int i = 0; i < NSPECS; ++i) {
    YOW += rhoi[i] / mw[i];
  }
  P = Ru * T * YOW; /*P = rho*R*T/W */

  return;
}

void CKRHOY(double P, double T, double *y, double &rho) {
  double YOW = 0;
  double tmp[NSPECS];

  for (int i = 0; i < NSPECS; i++) {
    tmp[i] = y[i] / mw[i];
  }
  for (int i = 0; i < NSPECS; i++) {
    YOW += tmp[i];
  }

  rho = P / (Ru * T * YOW); /*rho = P*W/(R*T) */
  return;
}

void CKYTX(double *y, double *x) {
  double YOW = 0;
  double tmp[NSPECS];

  for (int i = 0; i < NSPECS; i++) {
    tmp[i] = y[i] / mw[i];
  }
  for (int i = 0; i < NSPECS; i++) {
    YOW += tmp[i];
  }

  double YOWINV = 1.0 / YOW;

  for (int i = 0; i < NSPECS; i++) {
    x[i] = y[i] / mw[i] * YOWINV;
  }
  return;
}

void CKXTY(double *x, double *y) {
  double XW = 0.0; /*See Eq 4, 9 in CK Manual */
  /*Compute mean molecular wt first */
  for (int i = 0; i < NSPECS; ++i) {
    XW += x[i] * mw[i];
  }
  /*Now compute conversion */
  for (int i = 0; i < NSPECS; ++i) {
    y[i] = x[i] * mw[i] / XW;
  }

  return;
}

// gi/T, gi = g/Ri
void gibbs(double *species, double *tc) {
  /*temperature */
  double T = tc[1];
  double invT = 1 / T;

  /*species with midpoint at T=1000 kelvin */
  if (T < 1000) {
    /*species 0: H2 */
    species[0] = H2_lo[0] * (1.0 - tc[0]) - H2_lo[1] / 2.0 * tc[1] -
                 H2_lo[2] / 6.0 * tc[2] - H2_lo[3] / 12.0 * tc[3] -
                 H2_lo[4] / 20.0 * tc[4] + H2_lo[5] * invT - H2_lo[6];
    /*species 1: O2 */
    species[1] = O2_lo[0] * (1.0 - tc[0]) - O2_lo[1] / 2.0 * tc[1] -
                 O2_lo[2] / 6.0 * tc[2] - O2_lo[3] / 12.0 * tc[3] -
                 O2_lo[4] / 20.0 * tc[4] + O2_lo[5] * invT - O2_lo[6];
    /*species 2: H2O */
    species[2] = H2O_lo[0] * (1.0 - tc[0]) - H2O_lo[1] / 2.0 * tc[1] -
                 H2O_lo[2] / 6.0 * tc[2] - H2O_lo[3] / 12.0 * tc[3] -
                 H2O_lo[4] / 20.0 * tc[4] + H2O_lo[5] * invT - H2O_lo[6];
    /*species 3: H */
    species[3] = H_lo[0] * (1.0 - tc[0]) - H_lo[1] / 2.0 * tc[1] -
                 H_lo[2] / 6.0 * tc[2] - H_lo[3] / 12.0 * tc[3] -
                 H_lo[4] / 20.0 * tc[4] + H_lo[5] * invT - H_lo[6];
    /*species 4: O */
    species[4] = O_lo[0] * (1.0 - tc[0]) - O_lo[1] / 2.0 * tc[1] -
                 O_lo[2] / 6.0 * tc[2] - O_lo[3] / 12.0 * tc[3] -
                 O_lo[4] / 20.0 * tc[4] + O_lo[5] * invT - O_lo[6];
    /*species 5: OH */
    species[5] = OH_lo[0] * (1.0 - tc[0]) - OH_lo[1] / 2.0 * tc[1] -
                 OH_lo[2] / 6.0 * tc[2] - OH_lo[3] / 12.0 * tc[3] -
                 OH_lo[4] / 20.0 * tc[4] + OH_lo[5] * invT - OH_lo[6];
    /*species 6: HO2 */
    species[6] = HO2_lo[0] * (1.0 - tc[0]) - HO2_lo[1] / 2.0 * tc[1] -
                 HO2_lo[2] / 6.0 * tc[2] - HO2_lo[3] / 12.0 * tc[3] -
                 HO2_lo[4] / 20.0 * tc[4] + HO2_lo[5] * invT - HO2_lo[6];
    /*species 7: H2O2 */
    species[7] = H2O2_lo[0] * (1.0 - tc[0]) - H2O2_lo[1] / 2.0 * tc[1] -
                 H2O2_lo[2] / 6.0 * tc[2] - H2O2_lo[3] / 12.0 * tc[3] -
                 H2O2_lo[4] / 20.0 * tc[4] + H2O2_lo[5] * invT - H2O2_lo[6];
    /*species 8: N2 */
    species[8] = N2_lo[0] * (1.0 - tc[0]) - N2_lo[1] / 2.0 * tc[1] -
                 N2_lo[2] / 6.0 * tc[2] - N2_lo[3] / 12.0 * tc[3] -
                 N2_lo[4] / 20.0 * tc[4] + N2_lo[5] * invT - N2_lo[6];
  } else {
    /*species 0: H2 */
    species[0] = H2_hi[0] * (1.0 - tc[0]) - H2_hi[1] / 2.0 * tc[1] -
                 H2_hi[2] / 6.0 * tc[2] - H2_hi[3] / 12.0 * tc[3] -
                 H2_hi[4] / 20.0 * tc[4] + H2_hi[5] * invT - H2_hi[6];
    /*species 1: O2 */
    species[1] = O2_hi[0] * (1.0 - tc[0]) - O2_hi[1] / 2.0 * tc[1] -
                 O2_hi[2] / 6.0 * tc[2] - O2_hi[3] / 12.0 * tc[3] -
                 O2_hi[4] / 20.0 * tc[4] + O2_hi[5] * invT - O2_hi[6];
    /*species 2: H2O */
    species[2] = H2O_hi[0] * (1.0 - tc[0]) - H2O_hi[1] / 2.0 * tc[1] -
                 H2O_hi[2] / 6.0 * tc[2] - H2O_hi[3] / 12.0 * tc[3] -
                 H2O_hi[4] / 20.0 * tc[4] + H2O_hi[5] * invT - H2O_hi[6];
    /*species 3: H */
    species[3] = H_hi[0] * (1.0 - tc[0]) - H_hi[1] / 2.0 * tc[1] -
                 H_hi[2] / 6.0 * tc[2] - H_hi[3] / 12.0 * tc[3] -
                 H_hi[4] / 20.0 * tc[4] + H_hi[5] * invT - H_hi[6];
    /*species 4: O */
    species[4] = O_hi[0] * (1.0 - tc[0]) - O_hi[1] / 2.0 * tc[1] -
                 O_hi[2] / 6.0 * tc[2] - O_hi[3] / 12.0 * tc[3] -
                 O_hi[4] / 20.0 * tc[4] + O_hi[5] * invT - O_hi[6];
    /*species 5: OH */
    species[5] = OH_hi[0] * (1.0 - tc[0]) - OH_hi[1] / 2.0 * tc[1] -
                 OH_hi[2] / 6.0 * tc[2] - OH_hi[3] / 12.0 * tc[3] -
                 OH_hi[4] / 20.0 * tc[4] + OH_hi[5] * invT - OH_hi[6];
    /*species 6: HO2 */
    species[6] = HO2_hi[0] * (1.0 - tc[0]) - HO2_hi[1] / 2.0 * tc[1] -
                 HO2_hi[2] / 6.0 * tc[2] - HO2_hi[3] / 12.0 * tc[3] -
                 HO2_hi[4] / 20.0 * tc[4] + HO2_hi[5] * invT - HO2_hi[6];
    /*species 7: H2O2 */
    species[7] = H2O2_hi[0] * (1.0 - tc[0]) - H2O2_hi[1] / 2.0 * tc[1] -
                 H2O2_hi[2] / 6.0 * tc[2] - H2O2_hi[3] / 12.0 * tc[3] -
                 H2O2_hi[4] / 20.0 * tc[4] + H2O2_hi[5] * invT - H2O2_hi[6];
    /*species 8: N2 */
    species[8] = N2_hi[0] * (1.0 - tc[0]) - N2_hi[1] / 2.0 * tc[1] -
                 N2_hi[2] / 6.0 * tc[2] - N2_hi[3] / 12.0 * tc[3] -
                 N2_hi[4] / 20.0 * tc[4] + N2_hi[5] * invT - N2_hi[6];
  }

  return;
}

void vproductionRate(double *wdot, double (&Arate)[NSPECS][NSPECS], double *sc,
                     double T) {
  double k_f_s[21], Kc_s[21], mixture, gi_T[NSPECS];
  double tc[5], invT;
  const double eps = 1e-20;

  tc[0] = log(T);
  tc[1] = T;
  tc[2] = T * (T);
  tc[3] = tc[2] * tc[1];
  tc[4] = tc[2] * tc[2];
  invT = 1.0 / T;

  k_f_s[0] = 3.55e+15 * exp(-0.41 * tc[0] - 8353.444256079512 * invT);
  k_f_s[1] = 50800.0 * exp(2.67 * tc[0] - 3165.2508657072367 * invT);
  k_f_s[2] = 2.16e+08 * exp(1.51 * tc[0] - 1726.0429999007667 * invT);
  k_f_s[3] = 2.97e+06 * exp(2.02 * tc[0] - 6743.141748883461 * invT);
  k_f_s[4] = 4.58e+19 * exp(-1.4 * tc[0] - 52526.05490660117 * invT);
  k_f_s[5] = 6.16e+15 * exp(-0.5 * tc[0]);
  k_f_s[6] = 4.71e+18 * exp(-1 * tc[0]);
  k_f_s[7] = 3.8e+22 * exp(-2 * tc[0]);
  k_f_s[8] = 1.48e+12 * exp(0.6 * tc[0]);
  k_f_s[9] = 1.66e+13 * exp(-412.64001746898793 * invT);
  k_f_s[10] = 7.08e+13 * exp(-150.96586004962973 * invT);
  k_f_s[11] = 3.25e+13;
  k_f_s[12] = 2.89e+13 * exp(+251.60976674938289 * invT);
  k_f_s[13] = 4.2e+14 * exp(-6028.570011315214 * invT);
  k_f_s[14] = 1.3e+11 * exp(+820.2478396029883 * invT);
  k_f_s[15] = 2.95e+14 * exp(-24355.825421340265 * invT);
  k_f_s[16] = 2.41e+13 * exp(-1997.7815479901 * invT);
  k_f_s[17] = 4.82e+13 * exp(-4000.595291315188 * invT);
  k_f_s[18] = 9.55e+06 * exp(2 * tc[0] - 1997.7815479901* invT);
  k_f_s[19] = 1.0e+12;
  k_f_s[20] = 5.8e+14 * exp(- 4810.778740248201 * invT);

  /*compute the Gibbs free energy */

  gibbs(gi_T, tc);

  double RsT = Ru / Patm * 1e6 * T;

  Kc_s[0] = exp((gi_T[3] + gi_T[1]) - (gi_T[4] + gi_T[5]));
  Kc_s[1] = exp((gi_T[4] + gi_T[0]) - (gi_T[3] + gi_T[5]));
  Kc_s[2] = exp((gi_T[0] + gi_T[5]) - (gi_T[2] + gi_T[3]));
  Kc_s[3] = exp((gi_T[4] + gi_T[2]) - (gi_T[5] + gi_T[5]));
  Kc_s[4] = 1.0 / RsT * exp((gi_T[0]) - (gi_T[3] + gi_T[3]));
  Kc_s[5] = RsT * exp((gi_T[4] + gi_T[4]) - (gi_T[1]));
  Kc_s[6] = RsT * exp((gi_T[4] + gi_T[3]) - (gi_T[5]));
  Kc_s[7] = RsT * exp((gi_T[3] + gi_T[5]) - (gi_T[2]));
  Kc_s[8] = RsT * exp((gi_T[3] + gi_T[1]) - (gi_T[6]));
  Kc_s[9] = exp((gi_T[6] + gi_T[3]) - (gi_T[0] + gi_T[1]));
  Kc_s[10] = exp((gi_T[6] + gi_T[3]) - (gi_T[5] + gi_T[5]));
  Kc_s[11] = exp((gi_T[6] + gi_T[4]) - (gi_T[1] + gi_T[5]));
  Kc_s[12] = exp((gi_T[6] + gi_T[5]) - (gi_T[2] + gi_T[1]));
  Kc_s[13] = exp((gi_T[6] + gi_T[6]) - (gi_T[7] + gi_T[1]));
  Kc_s[14] = exp((gi_T[6] + gi_T[6]) - (gi_T[7] + gi_T[1]));
  Kc_s[15] = 1.0 / RsT * exp((gi_T[7]) - (gi_T[5] + gi_T[5]));
  Kc_s[16] = exp((gi_T[7] + gi_T[3]) - (gi_T[2] + gi_T[5]));
  Kc_s[17] = exp((gi_T[7] + gi_T[3]) - (gi_T[6] + gi_T[0]));
  Kc_s[18] = exp((gi_T[7] + gi_T[4]) - (gi_T[5] + gi_T[6]));
  Kc_s[19] = exp((gi_T[7] + gi_T[5]) - (gi_T[6] + gi_T[2]));
  Kc_s[20] = exp((gi_T[7] + gi_T[5]) - (gi_T[6] + gi_T[2]));

  mixture = 0.0;

  for (int n = 0; n < NSPECS; n++) {
    mixture += sc[n];
    wdot[n] = 0.0;
    for (int l = 0; l < NSPECS; l++) {
      Arate[n][l] = 0.0;
    }
  }

  double q_f[NREACS] = {0}, q_r[NREACS] = {0}, phi_f, phi_r, k_f, k_r, Kc;
  double alpha, redP, logPred, logFcent;
  double troe_c, troe_n, troe, F_troe;
  int vf[NREACS][NSPECS]={{0}}, vr[NREACS][NSPECS] = {{0}};

  /*reaction 0: H + O2 <=> O + OH */
  phi_f = sc[3] * sc[1];
  k_f = k_f_s[0];
  q_f[0] = phi_f * k_f;
  phi_r = sc[4] * sc[5];
  Kc = Kc_s[0];
  k_r = k_f / Kc;
  q_r[0] = phi_r * k_r;
  vf[0][3] = 1;
  vf[0][1] = 1;
  vr[0][4] = 1;
  vr[0][5] = 1;

  /*reaction 1: O + H2 <=> H + OH */
  phi_f = sc[4] * sc[0];
  k_f = k_f_s[1];
  q_f[1] = phi_f * k_f;
  phi_r = sc[3] * sc[5];
  Kc = Kc_s[1];
  k_r = k_f / Kc;
  q_r[1] = phi_r * k_r;
  vf[1][4] = 1;
  vf[1][0] = 1;
  vr[1][3] = 1;
  vr[1][5] = 1;

  /*reaction 2: H2 + OH <=> H2O + H */
  phi_f = sc[0] * sc[5];
  k_f = k_f_s[2];
  q_f[2] = phi_f * k_f;
  phi_r = sc[2] * sc[3];
  Kc = Kc_s[2];
  k_r = k_f / Kc;
  q_r[2] = phi_r * k_r;
  vf[2][0] = 1;
  vf[2][5] = 1;
  vr[2][2] = 1;
  vr[2][3] = 1;

  /*reaction 3: O + H2O <=> OH + OH */
  phi_f = sc[4] * sc[2];
  k_f = k_f_s[3];
  q_f[3] = phi_f * k_f;
  phi_r = sc[5] * sc[5];
  Kc = Kc_s[3];
  k_r = k_f / Kc;
  q_r[3] = phi_r * k_r;
  vf[3][4] = 1;
  vf[3][2] = 1;
  vr[3][5] = 2;

  /*reaction 4: H2 + M <=> H + H + M */
  phi_f = sc[0];
  alpha = mixture + 1.5 * sc[0] + 11 * sc[2];
  k_f = alpha * k_f_s[4];
  q_f[4] = phi_f * k_f;
  phi_r = sc[3] * sc[3];
  Kc = Kc_s[4];
  k_r = k_f / Kc;
  q_r[4] = phi_r * k_r;
  vf[4][0] = 1;
  vr[4][3] = 2;

  /*reaction 5: O + O + M <=> O2 + M */
  phi_f = sc[4] * sc[4];
  alpha = mixture + 1.5 * sc[0] + 11 * sc[2];
  k_f = alpha * k_f_s[5];
  q_f[5] = phi_f * k_f;
  phi_r = sc[1];
  Kc = Kc_s[5];
  k_r = k_f / Kc;
  q_r[5] = phi_r * k_r;
  vf[5][4] = 2;
  vr[5][1] = 1;

  /*reaction 6: O + H + M <=> OH + M */
  phi_f = sc[4] * sc[3];
  alpha = mixture + 1.5 * sc[0] + 11 * sc[2];
  k_f = alpha * k_f_s[6];
  q_f[6] = phi_f * k_f;
  phi_r = sc[5];
  Kc = Kc_s[6];
  k_r = k_f / Kc;
  q_r[6] = phi_r * k_r;
  vf[6][4] = 1;
  vf[6][3] = 1;
  vr[6][5] = 1;

  /*reaction 7: H + OH + M <=> H2O + M */
  phi_f = sc[3] * sc[5];
  alpha = mixture + 1.5 * sc[0] + 11 * sc[2];
  k_f = alpha * k_f_s[7];
  q_f[7] = phi_f * k_f;
  phi_r = sc[2];
  Kc = Kc_s[7];
  k_r = k_f / Kc;
  q_r[7] = phi_r * k_r;
  vf[7][3] = 1;
  vf[7][5] = 1;
  vr[7][2] = 1;

  /*reaction 8: H + O2 (+M) <=> HO2 (+M) */
  phi_f = sc[3] * sc[1];
  alpha = mixture + sc[0] + 10 * sc[2] + -0.22 * sc[1];
  k_f = k_f_s[8];
  redP =
      alpha / k_f * 6.37e+20 * exp(-1.72 * tc[0] - 261.6741574193582 * invT);
  logPred = log10(redP);
  logFcent = log10(0.8);
  troe_c = -0.4 - 0.67 * logFcent;
  troe_n = 0.75 - 1.27 * logFcent;
  troe = (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
  F_troe = pow(10.0, logFcent / (1.0 + troe * troe));
  k_f *= (redP / (1.0 + redP)) * F_troe;
  q_f[8] = phi_f * k_f;
  phi_r = sc[6];
  Kc = Kc_s[8];
  k_r = k_f / Kc;
  q_r[8] = phi_r * k_r;
  vf[8][3] = 1;
  vf[8][1] = 1;
  vr[8][6] = 1;

  /*reaction 9: HO2 + H <=> H2 + O2 */
  phi_f = sc[6] * sc[3];
  k_f = k_f_s[9];
  q_f[9] = phi_f * k_f;
  phi_r = sc[0] * sc[1];
  Kc = Kc_s[9];
  k_r = k_f / Kc;
  q_r[9] = phi_r * k_r;
  vf[9][6] = 1;
  vf[9][3] = 1;
  vr[9][0] = 1;
  vr[9][1] = 1;

  /*reaction 10: HO2 + H <=> OH + OH */
  phi_f = sc[6] * sc[3];
  k_f = k_f_s[10];
  q_f[10] = phi_f * k_f;
  phi_r = sc[5] * sc[5];
  Kc = Kc_s[10];
  k_r = k_f / Kc;
  q_r[10] = phi_r * k_r;
  vf[10][6] = 1;
  vf[10][3] = 1;
  vr[10][5] = 2;

  /*reaction 11: HO2 + O <=> O2 + OH */
  phi_f = sc[6] * sc[4];
  k_f = k_f_s[11];
  q_f[11] = phi_f * k_f;
  phi_r = sc[1] * sc[5];
  Kc = Kc_s[11];
  k_r = k_f / Kc;
  q_r[11] = phi_r * k_r;
  vf[11][6] = 1;
  vf[11][4] = 1;
  vr[11][1] = 1;
  vr[11][5] = 1;

  /*reaction 12: HO2 + OH <=> H2O + O2 */
  phi_f = sc[6] * sc[5];
  k_f = k_f_s[12];
  q_f[12] = phi_f * k_f;
  phi_r = sc[2] * sc[1];
  Kc = Kc_s[12];
  k_r = k_f / Kc;
  q_r[12] = phi_r * k_r;
  vf[12][6] = 1;
  vf[12][5] = 1;
  vr[12][2] = 1;
  vr[12][1] = 1;

  /*reaction 13: HO2 + HO2 <=> H2O2 + O2 */ /*DUP*/
  phi_f = sc[6] * sc[6];
  k_f = k_f_s[13];
  q_f[13] = phi_f * k_f;
  phi_r = sc[7] * sc[1];
  Kc = Kc_s[13];
  k_r = k_f / Kc;
  q_r[13] = phi_r * k_r;
  vf[13][6] = 2;
  vr[13][7] = 1;
  vr[13][1] = 1;

  /*reaction 13: HO2 + HO2 <=> H2O2 + O2 */ /*DUP*/
  phi_f = sc[6] * sc[6];
  k_f = k_f_s[14];
  q_f[13] += phi_f * k_f;
  phi_r = sc[7] * sc[1];
  Kc = Kc_s[14];
  k_r = k_f / Kc;
  q_r[13] += phi_r * k_r;

  /*reaction 14: H2O2 (+M) <=> OH + OH (+M) */
  phi_f = sc[7];
  alpha = mixture + 1.5 * sc[0] + 11 * sc[2];
  k_f = k_f_s[15];
  redP = alpha / k_f * 1.20e+17 * exp(-22846.166820843966 * invT);
  logPred = log10(redP);
  logFcent = log10(0.5);
  troe_c = -0.4 - 0.67 * logFcent;
  troe_n = 0.75 - 1.27 * logFcent;
  troe = (troe_c + logPred) / (troe_n - 0.14 * (troe_c + logPred));
  F_troe = pow(10.0, logFcent / (1.0 + troe * troe));
  k_f *= (redP / (1.0 + redP)) * F_troe;
  q_f[14] = phi_f * k_f;
  phi_r = sc[5] * sc[5];
  Kc = Kc_s[15];
  k_r = k_f / Kc;
  q_r[14] = phi_r * k_r;
  vf[14][7] = 1;
  vr[14][5] = 2;

  /*reaction 15: H2O2 + H <=> H2O + OH */
  phi_f = sc[7] * sc[3];
  k_f = k_f_s[16];
  q_f[15] = phi_f * k_f;
  phi_r = sc[2] * sc[5];
  Kc = Kc_s[16];
  k_r = k_f / Kc;
  q_r[15] = phi_r * k_r;
  vf[15][7] = 1;
  vf[15][3] = 1;
  vr[15][2] = 1;
  vr[15][5] = 1;

  /*reaction 16: H2O2 + H <=> HO2 + H2 */
  phi_f = sc[7] * sc[3];
  k_f = k_f_s[17];
  q_f[16] = phi_f * k_f;
  phi_r = sc[6] * sc[0];
  Kc = Kc_s[17];
  k_r = k_f / Kc;
  q_r[16] = phi_r * k_r;
  vf[16][7] = 1;
  vf[16][3] = 1;
  vr[16][6] = 1;
  vr[16][0] = 1;

  /*reaction 17: H2O2 + O <=> OH + HO2 */
  phi_f = sc[7] * sc[4];
  k_f = k_f_s[18];
  q_f[17] = phi_f * k_f;
  phi_r = sc[5] * sc[6];
  Kc = Kc_s[18];
  k_r = k_f / Kc;
  q_r[17] = phi_r * k_r;
  vf[17][7] = 1;
  vf[17][4] = 1;
  vr[17][5] = 1;
  vr[17][6] = 1;

  /*reaction 18: H2O2 + OH <=> HO2 + H2O */ /*DUP*/
  phi_f = sc[7] * sc[5];
  k_f = k_f_s[19];
  q_f[18] = phi_f * k_f;
  phi_r = sc[6] * sc[2];
  Kc = Kc_s[19];
  k_r = k_f / Kc;
  q_r[18] = phi_r * k_r;
  vf[18][7] = 1;
  vf[18][5] = 1;
  vr[18][6] = 1;
  vr[18][2] = 1;

  /*reaction 18: H2O2 + OH <=> HO2 + H2O */ /*DUP*/
  phi_f = sc[7] * sc[5];
  k_f = k_f_s[20];
  q_f[18] += phi_f * k_f;
  phi_r = sc[6] * sc[2];
  Kc = Kc_s[20];
  k_r = k_f / Kc;
  q_r[18] += phi_r * k_r;

  for (int m = 0; m < NREACS; ++m) {

    double wf1 = q_f[m];
    double wr1 = q_r[m];

    for (int n = 0; n < NSPECS; ++n) {
      wdot[n] += (wf1 - wr1) * (vr[m][n] - vf[m][n]);
    }

    for (int n = 0; n < NSPECS; ++n) {
      double Awf = vf[m][n] * wf1 / (sc[n] + eps);
      double Awr = vr[m][n] * wr1 / (sc[n] + eps);
      for (int l = 0; l < NSPECS; ++l) {
        Arate[l][n] += (Awf - Awr) * (vr[m][l] - vf[m][l]);
      }
    }
  }
}

/*compute the e/(RT) at the given temperature */
/*do not include enthalpy of production*/
/*tc contains precomputed powers of T, tc[0] = log(T) */
void speciesInternalEnergy(double *species, double *tc) {

  /*temperature */
  double T = tc[1];
  double invT = 1.0 / T;

  /*species with midpoint at T=1000 kelvin */
  if (T < 1000) {
    /*species 0: H2 */
    species[0] = H2_lo[0] - 1.0 + H2_lo[1] / 2.0 * tc[1] +
                 H2_lo[2] / 3.0 * tc[2] + H2_lo[3] / 4.0 * tc[3] +
                 H2_lo[4] / 5.0 * tc[4];
    /*species 1: O2 */
    species[1] = O2_lo[0] - 1.0 + O2_lo[1] / 2.0 * tc[1] +
                 O2_lo[2] / 3.0 * tc[2] + O2_lo[3] / 4.0 * tc[3] +
                 O2_lo[4] / 5.0 * tc[4];
    /*species 2: H2O */
    species[2] = H2O_lo[0] - 1.0 + H2O_lo[1] / 2.0 * tc[1] +
                 H2O_lo[2] / 3.0 * tc[2] + H2O_lo[3] / 4.0 * tc[3] +
                 H2O_lo[4] / 5.0 * tc[4];
    /*species 3: H */
    species[3] = H_lo[0] - 1.0 + H_lo[1] / 2.0 * tc[1] + H_lo[2] / 3.0 * tc[2] +
                 H_lo[3] / 4.0 * tc[3] + H_lo[4] / 5.0 * tc[4];
    /*species 4: O */
    species[4] = O_lo[0] - 1.0 + O_lo[1] / 2.0 * tc[1] + O_lo[2] / 3.0 * tc[2] +
                 O_lo[3] / 4.0 * tc[3] + O_lo[4] / 5.0 * tc[4];
    /*species 5: OH */
    species[5] = OH_lo[0] - 1.0 + OH_lo[1] / 2.0 * tc[1] +
                 OH_lo[2] / 3.0 * tc[2] + OH_lo[3] / 4.0 * tc[3] +
                 OH_lo[4] / 5.0 * tc[4];
    /*species 6: HO2 */
    species[6] = HO2_lo[0] - 1.0 + HO2_lo[1] / 2.0 * tc[1] +
                 HO2_lo[2] / 3.0 * tc[2] + HO2_lo[3] / 4.0 * tc[3] +
                 HO2_lo[4] / 5.0 * tc[4];
    /*species 7: H2O2 */
    species[7] = H2O2_lo[0] - 1.0 + H2O2_lo[1] / 2.0 * tc[1] +
                 H2O2_lo[2] / 3.0 * tc[2] + H2O2_lo[3] / 4.0 * tc[3] +
                 H2O2_lo[4] / 5.0 * tc[4];
    /*species 8: N2 */
    species[8] = N2_lo[0] - 1.0 + N2_lo[1] / 2.0 * tc[1] +
                 N2_lo[2] / 3.0 * tc[2] + N2_lo[3] / 4.0 * tc[3] +
                 N2_lo[4] / 5.0 * tc[4];
  } else {
    species[0] = H2_hi[0] - 1.0 + H2_hi[1] / 2.0 * tc[1] +
                 H2_hi[2] / 3.0 * tc[2] + H2_hi[3] / 4.0 * tc[3] +
                 H2_hi[4] / 5.0 * tc[4] + (H2_hi[5] - H2_lo[5]) * invT;
    /*species 1: O2 */
    species[1] = O2_hi[0] - 1.0 + O2_hi[1] / 2.0 * tc[1] +
                 O2_hi[2] / 3.0 * tc[2] + O2_hi[3] / 4.0 * tc[3] +
                 O2_hi[4] / 5.0 * tc[4] + (O2_hi[5] - O2_lo[5]) * invT;
    /*species 2: H2O */
    species[2] = H2O_hi[0] - 1.0 + H2O_hi[1] / 2.0 * tc[1] +
                 H2O_hi[2] / 3.0 * tc[2] + H2O_hi[3] / 4.0 * tc[3] +
                 H2O_hi[4] / 5.0 * tc[4] + (H2O_hi[5] - H2O_lo[5])* invT;
    /*species 3: H */
    species[3] = H_hi[0] - 1.0 + H_hi[1] / 2.0 * tc[1] + H_hi[2] / 3.0 * tc[2] +
                 H_hi[3] / 4.0 * tc[3] + H_hi[4] / 5.0 * tc[4] +
                 (H_hi[5] - H_lo[5])* invT;
    /*species 4: O */
    species[4] = O_hi[0] - 1.0 + O_hi[1] / 2.0 * tc[1] + O_hi[2] / 3.0 * tc[2] +
                 O_hi[3] / 4.0 * tc[3] + O_hi[4] / 5.0 * tc[4] +
                 (O_hi[5] - O_lo[5])* invT;
    /*species 5: OH */
    species[5] = OH_hi[0] - 1.0 + OH_hi[1] / 2.0 * tc[1] +
                 OH_hi[2] / 3.0 * tc[2] + OH_hi[3] / 4.0 * tc[3] +
                 OH_hi[4] / 5.0 * tc[4] + (OH_hi[5] - OH_lo[5])* invT;
    /*species 6: HO2 */
    species[6] = HO2_hi[0] - 1.0 + HO2_hi[1] / 2.0 * tc[1] +
                 HO2_hi[2] / 3.0 * tc[2] + HO2_hi[3] / 4.0 * tc[3] +
                 HO2_hi[4] / 5.0 * tc[4] + (HO2_hi[5] - HO2_lo[5])* invT;
    /*species 7: H2O2 */
    species[7] = H2O2_hi[0] - 1.0 + H2O2_hi[1] / 2.0 * tc[1] +
                 H2O2_hi[2] / 3.0 * tc[2] + H2O2_hi[3] / 4.0 * tc[3] +
                 H2O2_hi[4] / 5.0 * tc[4] + (H2O2_hi[5] - H2O2_lo[5])* invT;
    /*species 8: N2 */
    species[8] = N2_hi[0] - 1.0 + N2_hi[1] / 2.0 * tc[1] +
                 N2_hi[2] / 3.0 * tc[2] + N2_hi[3] / 4.0 * tc[3] +
                 N2_hi[4] / 5.0 * tc[4] + (N2_hi[5] - N2_lo[5])* invT;
  }
  return;
}

/*get mean internal energy in volume unit */
/* SI unit*/
/* J/m^3*/
void CKUBMS(double T, double *rhoi, double &ubms) {
  double result = 0;
  double tc[] = {0, T, T * T, T * T * T, T * T * T * T}; /*temperature cache */
  double ums[NSPECS]; /* temporary energy array */
  double RT = Ru * T; /*R*T */
  speciesInternalEnergy(ums, tc);
  /*perform dot product + scaling by wt */
  for (int n = 0; n < NSPECS; ++n) {
    result += rhoi[n] * ums[n] / mw[n];
  }

  ubms = result * RT;
}

/*compute Cp/R at the given temperature */
/*tc contains precomputed powers of T, tc[0] = log(T) */
void cp_R(double *species, double *tc) {

  /*temperature */
  double T = tc[1];

  /*species with midpoint at T=1000 kelvin */
  if (T < 1000) {
    /*species 0: H2 */
    species[0] = H2_lo[0] + H2_lo[1] * tc[1] + H2_lo[2] * tc[2] +
                 H2_lo[3] * tc[3] + H2_lo[4] * tc[4];
    /*species 1: O2 */
    species[1] = O2_lo[0] + O2_lo[1] * tc[1] + O2_lo[2] * tc[2] +
                 O2_lo[3] * tc[3] + O2_lo[4] * tc[4];
    /*species 2: H2O */
    species[2] = H2O_lo[0] + H2O_lo[1] * tc[1] + H2O_lo[2] * tc[2] +
                 H2O_lo[3] * tc[3] + H2O_lo[4] * tc[4];
    /*species 3: H */
    species[3] = H_lo[0] + H_lo[1] * tc[1] + H_lo[2] * tc[2] +
                 H_lo[3] * tc[3] + H_lo[4] * tc[4];
    /*species 4: O */
    species[4] = O_lo[0] + O_lo[1] * tc[1] + O_lo[2] * tc[2] +
                 O_lo[3] * tc[3] + O_lo[4] * tc[4];
    /*species 5: OH */
    species[5] = OH_lo[0] + OH_lo[1] * tc[1] + OH_lo[2] * tc[2] +
                 OH_lo[3] * tc[3] + OH_lo[4] * tc[4];
    /*species 6: HO2 */
    species[6] = HO2_lo[0] + HO2_lo[1] * tc[1] + HO2_lo[2] * tc[2] +
                 HO2_lo[3] * tc[3] + HO2_lo[4] * tc[4];
    /*species 7: H2O2 */
    species[7] = H2O2_lo[0] + H2O2_lo[1] * tc[1] + H2O2_lo[2] * tc[2] +
                 H2O2_lo[3] * tc[3] + H2O2_lo[4] * tc[4];
    /*species 8: N2 */
    species[8] = N2_lo[0] + N2_lo[1] * tc[1] + N2_lo[2] * tc[2] +
                 N2_lo[3] * tc[3] + N2_lo[4] * tc[4];
  } else {
    /*species 0: H2 */
    species[0] = H2_hi[0] + H2_hi[1] * tc[1] + H2_hi[2] * tc[2] +
                 H2_hi[3] * tc[3] + H2_hi[4] * tc[4];
    /*species 1: O2 */
    species[1] = O2_hi[0] + O2_hi[1] * tc[1] + O2_hi[2] * tc[2] +
                 O2_hi[3] * tc[3] + O2_hi[4] * tc[4];
    /*species 2: H2O */
    species[2] = H2O_hi[0] + H2O_hi[1] * tc[1] + H2O_hi[2] * tc[2] +
                 H2O_hi[3] * tc[3] + H2O_hi[4] * tc[4];
    /*species 3: H */
    species[3] = H_hi[0] + H_hi[1] * tc[1] + H_hi[2] * tc[2] +
                 H_hi[3] * tc[3] + H_hi[4] * tc[4];
    /*species 4: O */
    species[4] = O_hi[0] + O_hi[1] * tc[1] + O_hi[2] * tc[2] +
                 O_hi[3] * tc[3] + O_hi[4] * tc[4];
    /*species 5: OH */
    species[5] = OH_hi[0] + OH_hi[1] * tc[1] + OH_hi[2] * tc[2] +
                 OH_hi[3] * tc[3] + OH_hi[4] * tc[4];
    /*species 6: HO2 */
    species[6] = HO2_hi[0] + HO2_hi[1] * tc[1] + HO2_hi[2] * tc[2] +
                 HO2_hi[3] * tc[3] + HO2_hi[4] * tc[4];
    /*species 7: H2O2 */
    species[7] = H2O2_hi[0] + H2O2_hi[1] * tc[1] + H2O2_hi[2] * tc[2] +
                 H2O2_hi[3] * tc[3] + H2O2_hi[4] * tc[4];
    /*species 8: N2 */
    species[8] = N2_hi[0] + N2_hi[1] * tc[1] + N2_hi[2] * tc[2] +
                 N2_hi[3] * tc[3] + N2_hi[4] * tc[4];
  }
  return;
}

/*J/(m^3 K)*/
void CKCPBS(double T, double *rhoi, double &cvbs) {
  double result = 0;
  double tc[] = {0, T, T * T, T * T * T, T * T * T * T}; /*temperature cache */
  double cpor[NSPECS];                                   /* temporary storage */
  cp_R(cpor, tc);
  /*multiply by y/molecularweight */
  for (int n = 0; n < NSPECS; ++n) {
    result += cpor[n] * rhoi[n] / mw[n];
  }

  cvbs =  result * Ru;
}

/*J/(m^3 K)*/
void CKCVBS(double T, double *rhoi, double &cvbs) {
  double result = 0;
  double tc[] = {0, T, T * T, T * T * T, T * T * T * T}; /*temperature cache */
  double cpor[NSPECS];                                   /* temporary storage */
  cp_R(cpor, tc);
  /*multiply by y/molecularweight */
  for (int n = 0; n < NSPECS; ++n) {
    result += (cpor[n] - 1.0) * rhoi[n] / mw[n];
  }

  cvbs =  result * Ru;
}

/*J/(m^3 K)*/
void CKGAMMA(double T, double *rhoi, double &gamma) {
  double cp = 0, cv = 0;
  double tc[] = {0, T, T * T, T * T * T, T * T * T * T}; /*temperature cache */
  double cpor[NSPECS];                                   /* temporary storage */
  cp_R(cpor, tc);
  /*multiply by y/molecularweight */
  for (int n = 0; n < NSPECS; ++n) {
    cp += cpor[n] * rhoi[n] / mw[n];
    cv += (cpor[n]-1.0) * rhoi[n] / mw[n];
  }

  gamma = cp/cv;
}

/* get temperature given internal energy in mass units and mass fracs */
void GET_T_GIVEN_EY(double e, double *rhoi, double &T) {
  const int maxiter = 200;
  const double tol = 1e-7;
  double ein = e;
  double tmin = 250;  /*max lower bound for thermo def */
  double tmax = 3500; /*min upper bound for thermo def */
  double e1, emin, emax, cv, t1, dt;
  int i; /* loop counter */

  CKUBMS(tmin, rhoi, emin);
  CKUBMS(tmax, rhoi, emax);
  if (ein < emin) {
    /*Linear Extrapolation below tmin */
    CKCVBS(tmin, rhoi, cv);
    T = tmin - (emin - ein) / cv;
    return;
  }
  if (ein > emax) {
    /*Linear Extrapolation above tmax */
    CKCVBS(tmax, rhoi, cv);
    T = tmax - (emax - ein) / cv;
    return;
  }

  double As=0;
  for (int n=0; n<NSPECS; ++n) {
    As += A0[n]*Ru/mw[n]*rhoi[n];
  }

  // initial value
  t1 = e/As;

  if (t1 < tmin || t1 > tmax) {
    t1 = tmin + (tmax - tmin) / (emax - emin) * (ein - emin);
  }
  for (i = 0; i < maxiter; ++i) {
    CKUBMS(t1, rhoi, e1);
    CKCVBS(t1, rhoi, cv);
    dt = (ein - e1) / cv;
    if (dt > 100.0) {
      dt = 100.0;
    } else if (dt < -100.0) {
      dt = -100.0;
    } else if (fabs(dt) < tol) {
      break;
    } else if (t1+dt == t1) {
      break;
    }
    t1 += dt;
  }
  T = t1;
  return;
}

void gauss(double *x, double (&A)[NSPECS][NSPECS], double *b) {
  int i, j, k;
  double U[NSPECS][NSPECS + 1];

  // Copy A to U and augment with vector b.
  for (i = 0; i < NSPECS; i++) {
    U[i][NSPECS] = b[i];
    for (j = 0; j < NSPECS; j++)
      U[i][j] = A[i][j];
  }

  // Factorisation stage
  for (k = 0; k < NSPECS; k++) {
    // Find the best pivot
    int p = k;
    double maxPivot = 0.0;
    for (i = k; i < NSPECS; i++) {
      if (fabs(U[i][k]) > maxPivot) {
        maxPivot = fabs(U[i][k]);
        p = i;
      }
    }

    // Swap rows k and p
    if (p != k) {
      for (i = k; i < NSPECS + 1; i++)
        std::swap(U[p][i], U[k][i]);
    }

    // Elimination of variables
    for (i = k + 1; i < NSPECS; i++) {
      double m = U[i][k] / U[k][k];
      for (j = k; j < NSPECS + 1; j++)
        U[i][j] -= m * U[k][j];
    }
  }

  // Back substitution
  for (int k = NSPECS - 1; k >= 0; k--) {
    double sum = U[k][NSPECS];
    for (int j = k + 1; j < NSPECS; j++)
      sum -= U[k][j] * x[j];
    x[k] = sum / U[k][k];
  }
}

int main() {
  double dt = 1e-5;
  double T = 305;
  double p = Patm;
  double rho;
  double Et = 0;
  double t_end = 1e-3;
  double time, gamma;

  double rhoi[NSPECS] = {0.0}, Xt[NSPECS] = {0.0}, Yt[NSPECS] = {0.0},
         rhoi_1[NSPECS] = {0.0};
  double wdot[NSPECS], Arate[NSPECS][NSPECS], A1[NSPECS][NSPECS],
      drho[NSPECS];

  ofstream ofile;
  ofile.open("molfrac.csv", ios::out);

  Xt[0] = 0.15;
  Xt[8] = 0.85;

  CKXTY(Xt, Yt);

  CKRHOY(p, T, Yt, rho);
  for (int n = 0; n < NSPECS; ++n) {
    rhoi[n] = Yt[n] * rho;
  }

  CKPY(rhoi, T, p);

  CKUBMS(T, rhoi, Et);
  cout.precision(18);
  cout << "rho = " << rho << endl;
  cout << "E = " << Et << endl;

  // int niter = ceil(t_end / dt);

  // for (int iter = 0; iter < niter; ++iter) {

  //   GET_T_GIVEN_EY(Et, rhoi, T);

  //   double c[9];
  //   for (int n = 0; n < NSPECS; n++) {
  //     c[n] = rhoi[n] / mw[n] * 1e-6;
  //   }

  //   /*call productionRate */
  //   vproductionRate(wdot, Arate, c, T);

  //   for (int i = 0; i < NSPECS; ++i) {
  //     for (int j = 0; j < NSPECS; ++j) {
  //       A1[i][j] = (i == j ? 1.0 : 0.0);
  //       A1[i][j] -= Arate[i][j] * mw[i] / mw[j] * dt;
  //     }
  //   }

  //   for (int n = 0; n < NSPECS; ++n) {
  //     rhoi_1[n] = wdot[n] * mw[n] * 1e6 * dt;
  //   }

  //   gauss(drho, A1, rhoi_1);

  //   for (int n = 0; n < NSPECS; ++n) {
  //     rhoi[n] += drho[n];
  //     if (rhoi[n] < 0) rhoi[n] =0;
  //   }

  //   // update Yt
  //   for (int n = 0; n < NSPECS; ++n) {
  //     Yt[n] = rhoi[n] / rho;
  //   }

  //   for (int n=0; n<NSPECS; ++n) {
  //     Et -= HP[n] * Ru / mw[n] * drho[n];
  //   }

  //   time = (iter + 1) * dt;

  //   // output 
  //   CKYTX(Yt, Xt);
  //   CKPY(rhoi, T, p);
  //   CKGAMMA(T, rhoi, gamma);
  //   // CKUBMS(T, rhoi, Et);
  //   cout.precision(12);
  //   cout << "---------------\nTime = " << time << " s" << endl;
  //   cout << "T = " << T << " K" << endl;
  //   cout << "P = " << p << " Pa" << endl;
  //   cout << "E = " << Et << endl;
  //   cout << "gamma = " << gamma << " " << 1.0 + p/Et << endl;
  //   cout << "C = " << sqrt(gamma*p/rho) << " " << sqrt((1.0+p/Et)*p/rho) << endl;
  //   cout << "Mol frac of H2:  " << Xt[0] << endl;
  //   cout << "Mol frac of O2:  " << Xt[1] << endl;
  //   cout << "Mol frac of H2O: " << Xt[2] << endl;
  //   ofile << time << " " << T << " " << Yt[0] << " " << Yt[1] << " " << Yt[2] << endl;
  // }
  ofile.close();
  return 0;
}