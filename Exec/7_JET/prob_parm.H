#ifndef EBR_PROB_PARM_H_
#define EBR_PROB_PARM_H_

#include <AMReX_REAL.H>

struct ProbParm
{
    // pure H2 jet
    
    amrex::Real p_ambient = 101325.0;
    amrex::Real T_ambient = 1150;
    amrex::Real u_ambient = 20;
    amrex::Real T_jet = 305;
    amrex::Real p_jet = 101325.0;
    amrex::Real u_jet = 900;

    amrex::Real* inner_state = nullptr;
    amrex::Real* outer_state = nullptr;
    amrex::Real* inner_spec = nullptr;
    amrex::Real* outer_spec = nullptr;

    ProbParm ();
    ~ProbParm ();
};

#endif
