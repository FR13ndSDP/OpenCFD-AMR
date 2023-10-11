#include <AMReX_PROB_AMR_F.H>
#include <AMReX_ParmParse.H>
#include "prob_parm.H"
#include "EBR.H"
#include "IndexDefines.H"

extern "C" {
    void amrex_probinit (const int* /*init*/,
                         const int* /*name*/,
                         const int* /*namelen*/,
                         const amrex_real* /*problo*/,
                         const amrex_real* /*probhi*/)
    {
        amrex::ParmParse pp("prob");

        pp.query("p_l", EBR::h_prob_parm->p_l);
        pp.query("p_r", EBR::h_prob_parm->p_r);
        pp.query("rho_l", EBR::h_prob_parm->rho_l);
        pp.query("rho_r", EBR::h_prob_parm->rho_r);

        Gpu::copy(amrex::Gpu::hostToDevice, EBR::h_prob_parm, EBR::h_prob_parm+1, EBR::d_prob_parm);
    }
}
