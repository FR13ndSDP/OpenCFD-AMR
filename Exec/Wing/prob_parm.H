#ifndef EBR_PROB_PARM_H_
#define EBR_PROB_PARM_H_

#include <AMReX_REAL.H>

using namespace amrex::literals;

struct ProbParm
{
    amrex::Real p_static = 96321.53451;
    amrex::Real p_out = 96321.53451;
    amrex::Real u = 57.73297950529637;
    amrex::Real v = 5.559053646172991;
    amrex::Real rho = 1.158493;

    amrex::Real* inflow_state = nullptr;

    ProbParm ();
    ~ProbParm ();
};

#endif
