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
        Gpu::HostVector<Real> inner_state(NCONS);
        Gpu::HostVector<Real> outer_state(NCONS);
        Gpu::HostVector<Real> inner_spec(NSPECS);
        Gpu::HostVector<Real> outer_spec(NSPECS);
        inner_state[0] = 0.963512265437802995;
        inner_state[1] = 0.963512265437802995 * 900;
        inner_state[2] = 0;
        inner_state[3] = 0;
        inner_state[4] = 245428.076382951345 + 0.5*0.963512265437802995*900*900;

        outer_state[0] = 0.305731395505996983;
        outer_state[1] = 0.305731395505996983 * 20;
        outer_state[2] = 0;
        outer_state[3] = 0;
        outer_state[4] = 271976.980564150028 + 0.5*0.305731395505996983*20*20;

        for (int n=0; n<NSPECS; ++n) {
            inner_spec[n] = 0;
            outer_spec[n] = 0;
        }
        inner_spec[0] = 0.963512265437802995 * 0.012540;
        inner_spec[8] = 0.963512265437802995 * 0.987460;
        outer_spec[1] = 0.305731395505996983 * 0.232909;
        outer_spec[8] = 0.305731395505996983 * 0.767091;
#ifdef AMREX_USE_GPU
        // Cannot use Gpu::copy because ProbParm is not trivailly copyable.
        Gpu::htod_memcpy_async(EBR::d_prob_parm, EBR::h_prob_parm, sizeof(ProbParm));
#else
        std::memcpy(EBR::d_prob_parm, EBR::h_prob_parm, sizeof(ProbParm));
#endif

        Gpu::copyAsync(Gpu::hostToDevice, inner_state.data(),
                       inner_state.data() + NCONS,
                       EBR::h_prob_parm->inner_state);

        Gpu::copyAsync(Gpu::hostToDevice, outer_state.data(),
                       outer_state.data() + NCONS,
                       EBR::h_prob_parm->outer_state);

        Gpu::copyAsync(Gpu::hostToDevice, inner_spec.data(),
                       inner_spec.data() + NSPECS,
                       EBR::h_prob_parm->inner_spec);

        Gpu::copyAsync(Gpu::hostToDevice, outer_spec.data(),
                       outer_spec.data() + NSPECS,
                       EBR::h_prob_parm->outer_spec);
        Gpu::streamSynchronize();
    }
}
