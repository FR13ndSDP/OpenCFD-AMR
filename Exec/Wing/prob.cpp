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

        Gpu::HostVector<Real> inflow_state(6);
        inflow_state[0] = EBR::h_prob_parm->rho;
        inflow_state[1] = EBR::h_prob_parm->u;
        inflow_state[2] = EBR::h_prob_parm->v;
        inflow_state[3] = 0;
        inflow_state[4] = EBR::h_prob_parm->p_static;
        inflow_state[5] = EBR::h_prob_parm->p_out;

#ifdef AMREX_USE_GPU
        // Cannot use Gpu::copy because ProbParm is not trivailly copyable.
        Gpu::htod_memcpy_async(EBR::d_prob_parm, EBR::h_prob_parm, sizeof(ProbParm));
#else
        std::memcpy(EBR::d_prob_parm, EBR::h_prob_parm, sizeof(ProbParm));
#endif

        Gpu::copyAsync(Gpu::hostToDevice, inflow_state.data(),
                       inflow_state.data() + 6,
                       EBR::h_prob_parm->inflow_state);
        Gpu::streamSynchronize();
    }
}
