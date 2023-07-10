#ifndef EBR_PROB_PARM_H_
#define EBR_PROB_PARM_H_

#include <AMReX_REAL.H>
using namespace amrex::literals;

struct ProbParm
{
    amrex::Real p_l = 2026.5;
    amrex::Real p_r = 2026.5;
    amrex::Real rho_l = 0.08938;
    amrex::Real rho_r = 0.08938;
    amrex::Real u_l = 534.49;
    amrex::Real u_r = 534.49;

    amrex::Real* inflow_state = nullptr;

    ProbParm ();
    ~ProbParm ();
};

#endif
