#ifndef EBR_PROB_PARM_H_
#define EBR_PROB_PARM_H_

#include <AMReX_REAL.H>

struct ProbParm
{
    amrex::Real p = 0.3*101325.0;
    amrex::Real T = 880;
};

#endif
