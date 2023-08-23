#ifndef EBR_PROB_PARM_H_
#define EBR_PROB_PARM_H_

#include <AMReX_REAL.H>

struct ProbParm
{
    amrex::Real p = 0.2*101325.0;
    amrex::Real T = 295;
    amrex::Real T0 = 1500;
    amrex::Real p0 = 20*101325.0;
};

#endif
