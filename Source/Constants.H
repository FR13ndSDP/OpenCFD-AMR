#ifndef EBR_PARM_H_
#define EBR_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

struct Parm
{
    amrex::Real eos_gamma = 1.4;
    amrex::Real eos_m = 28.97e-3;  // mean molecular weight
    amrex::Real Patm = 101325.0;

    amrex::Real cv;
    amrex::Real cp;
    amrex::Real Ru = 8.31446261815324;
    amrex::Real Rg;
    amrex::Real kOverMu;

    amrex::Real Pr  = 0.72;     // Prandtl number
    amrex::Real C_s = 1.458e-6; // constant in Sutherland's law
    amrex::Real T_s = 110.4;    // Sutherland temperature

    amrex::Real smallr = 1.e-19;
    amrex::Real smallp = 1.e-10;
    amrex::Real eps = 1.e-40;
    
    amrex::Real twoThirds = 2.0/3.0;
    amrex::Real oneSix = 1.0/6.0;

#ifdef CHEM
    amrex::Real* mw = nullptr;

    amrex::Real* H2_lo = nullptr;
    amrex::Real* H2_hi = nullptr;
    amrex::Real* O2_lo = nullptr;
    amrex::Real* O2_hi = nullptr;
    amrex::Real* H2O_lo = nullptr;
    amrex::Real* H2O_hi = nullptr;
    amrex::Real* H_lo = nullptr;
    amrex::Real* H_hi = nullptr;
    amrex::Real* O_lo = nullptr;
    amrex::Real* O_hi = nullptr;
    amrex::Real* OH_lo = nullptr;
    amrex::Real* OH_hi = nullptr;
    amrex::Real* HO2_lo = nullptr;
    amrex::Real* HO2_hi = nullptr;
    amrex::Real* H2O2_lo = nullptr;
    amrex::Real* H2O2_hi = nullptr;
    amrex::Real* N2_lo = nullptr;
    amrex::Real* N2_hi = nullptr;

    amrex::Real* HP = nullptr;
    amrex::Real* A0 = nullptr;

    amrex::Real* visccoeff = nullptr;
    amrex::Real* conductcoeff = nullptr;
    amrex::Real* diffcoeff = nullptr;

    Parm();
    ~Parm();
#endif

    void Initialize ();
};

#endif
