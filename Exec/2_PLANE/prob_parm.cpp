#include "prob_parm.H"
#include "IndexDefines.H"

#include <AMReX_Arena.H>

ProbParm::ProbParm ()
{
    inflow_state = (amrex::Real*)amrex::The_Arena()->alloc(sizeof(amrex::Real)*NCONS);
}

ProbParm::~ProbParm ()
{
    amrex::The_Arena()->free(inflow_state);
}
