#ifndef EBR_PROB_PARM_H_
#define EBR_PROB_PARM_H_

#include <AMReX_REAL.H>
using namespace amrex::literals;

struct ProbParm
{
    amrex::Real p_l = 1.0;
    amrex::Real p_r = 0.1;
    amrex::Real rho_l = 1.0;
    amrex::Real rho_r = 0.125;
};

#endif
