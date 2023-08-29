#include "prob_parm.H"
#include "IndexDefines.H"

#include <AMReX_Arena.H>

ProbParm::ProbParm ()
{
    inner_state = (amrex::Real*)amrex::The_Arena()->alloc(sizeof(amrex::Real)*NCONS);
    outer_state = (amrex::Real*)amrex::The_Arena()->alloc(sizeof(amrex::Real)*NCONS);

    inner_spec = (amrex::Real*)amrex::The_Arena()->alloc(sizeof(amrex::Real)*NSPECS);
    outer_spec = (amrex::Real*)amrex::The_Arena()->alloc(sizeof(amrex::Real)*NSPECS);
}

ProbParm::~ProbParm ()
{
    amrex::The_Arena()->free(inner_state);
    amrex::The_Arena()->free(outer_state);
    amrex::The_Arena()->free(inner_spec);
    amrex::The_Arena()->free(outer_spec);
}