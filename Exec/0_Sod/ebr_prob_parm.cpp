#include "ebr_prob_parm.H"
#include "EBR.H"
#include "EBR_index_macros.H"

#include <AMReX_Arena.H>

ProbParm::ProbParm ()
{
    inflow_state = (amrex::Real*)The_Arena()->alloc(sizeof(Real)*NCONS);
}

ProbParm::~ProbParm ()
{
    The_Arena()->free(inflow_state);
}
