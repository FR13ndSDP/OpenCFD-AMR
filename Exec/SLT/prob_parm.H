#ifndef EBR_PROB_PARM_H_
#define EBR_PROB_PARM_H_

#include <AMReX_REAL.H>

using namespace amrex::literals;

struct ProbParm
{
    amrex::Real p_static = 402.948;
    amrex::Real p_out = 96321.53451;
    amrex::Real u = 853.8283218615844;
    amrex::Real v = 150.5529700372286;
    amrex::Real rho = 0.027;

    amrex::Real* inflow_state = nullptr;

    ProbParm ();
    ~ProbParm ();
};

#endif
