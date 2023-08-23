#include "prob_parm.H"
#include "IndexDefines.H"

#include <AMReX_Arena.H>

ProbParm::ProbParm ()
{
    left_state = (amrex::Real*)amrex::The_Arena()->alloc(sizeof(amrex::Real)*NCONS);
    right_state = (amrex::Real*)amrex::The_Arena()->alloc(sizeof(amrex::Real)*NCONS);
}

ProbParm::~ProbParm ()
{
    amrex::The_Arena()->free(left_state);
    amrex::The_Arena()->free(right_state);
}