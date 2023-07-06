#ifndef EBR_PROB_PARM_H_
#define EBR_PROB_PARM_H_

#include <AMReX_REAL.H>
using namespace amrex::literals;

struct ProbParm
{
    amrex::Real p_l = 116.5;
    amrex::Real p_r = 1.0;
    amrex::Real rho_l = 8.0;
    amrex::Real rho_r = 1.4;
    amrex::Real u_l = 8.25*amrex::Math::cospi(1.0/6.0);
    amrex::Real u_r = 0.0;
    amrex::Real v_l = -8.25*amrex::Math::sinpi(1.0/6.0);
    amrex::Real v_r = 0.0;
};

#endif
