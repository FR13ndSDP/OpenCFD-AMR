#ifndef EBR_PROB_PARM_H_
#define EBR_PROB_PARM_H_

#include <AMReX_REAL.H>
using namespace amrex::literals;

struct ProbParm
{
    amrex::Real p = 2.5;
    amrex::Real rho_in = 2.0;
    amrex::Real rho_out = 1.0;
};

#endif
