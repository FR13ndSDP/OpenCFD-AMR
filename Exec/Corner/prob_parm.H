#ifndef EBR_PROB_PARM_H_
#define EBR_PROB_PARM_H_

#include <AMReX_REAL.H>
using namespace amrex::literals;

struct ProbParm
{
    amrex::Real p_l = 230.6;
    amrex::Real p_r = 230.6;
    amrex::Real rho_l = 0.004;
    amrex::Real rho_r = 0.004;
    amrex::Real u_l = 1988.665381606468;
    amrex::Real u_r = 1988.665381606468;

    amrex::Real* inflow_state = nullptr;

    ProbParm ();
    ~ProbParm ();
};

#endif
