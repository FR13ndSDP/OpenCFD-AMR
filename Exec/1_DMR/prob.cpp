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
        pp.query("u_l", EBR::h_prob_parm->u_l);
        pp.query("u_r", EBR::h_prob_parm->u_r);

        Gpu::HostVector<Real> left_state(NCONS);
        Gpu::HostVector<Real> right_state(NCONS);
        left_state[0] = EBR::h_prob_parm->rho_l;
        left_state[1] = EBR::h_prob_parm->u_l*EBR::h_prob_parm->rho_l;
        left_state[2] = EBR::h_prob_parm->v_l*EBR::h_prob_parm->rho_l;
        left_state[3] = 0;
        left_state[4] = EBR::h_prob_parm->p_l/0.4 + 0.5*EBR::h_prob_parm->rho_l* \
                          (EBR::h_prob_parm->u_l*EBR::h_prob_parm->u_l + EBR::h_prob_parm->v_l*EBR::h_prob_parm->v_l);

        right_state[0] = EBR::h_prob_parm->rho_r;
        right_state[1] = EBR::h_prob_parm->u_r*EBR::h_prob_parm->rho_r;
        right_state[2] = EBR::h_prob_parm->v_r*EBR::h_prob_parm->rho_r;
        right_state[3] = 0;
        right_state[4] = EBR::h_prob_parm->p_r/0.4 + 0.5*EBR::h_prob_parm->rho_r* \
                          (EBR::h_prob_parm->u_r*EBR::h_prob_parm->u_r + EBR::h_prob_parm->v_r*EBR::h_prob_parm->v_r);

#ifdef AMREX_USE_GPU
        // Cannot use Gpu::copy because ProbParm is not trivailly copyable.
        Gpu::htod_memcpy_async(EBR::d_prob_parm, EBR::h_prob_parm, sizeof(ProbParm));
#else
        std::memcpy(EBR::d_prob_parm, EBR::h_prob_parm, sizeof(ProbParm));
#endif

        Gpu::copyAsync(Gpu::hostToDevice, left_state.data(),
                       left_state.data() + NCONS,
                       EBR::h_prob_parm->left_state);

        Gpu::copyAsync(Gpu::hostToDevice, right_state.data(),
                       right_state.data() + NCONS,
                       EBR::h_prob_parm->right_state);

        Gpu::streamSynchronize();
    }
}
