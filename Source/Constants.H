#ifndef EBR_PARM_H_
#define EBR_PARM_H_

#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

struct Parm
{
    amrex::Real eos_gamma = 1.4;
    amrex::Real eos_m = 28.97e-3;  // mean molecular weight

    amrex::Real cv;
    amrex::Real cp;
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

    amrex::Real frac_t = 0.1;   // eb small cell volume fraction threshold

    void Initialize ();
};

#endif
