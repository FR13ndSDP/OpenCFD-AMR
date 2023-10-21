#ifndef EBR_PROB_PARM_H_
#define EBR_PROB_PARM_H_

#include <AMReX_REAL.H>
#include <Eigen/Core>
#include <Eigen/Dense>
#include "AMReX_GpuQualifiers.H"

using namespace amrex::literals;

struct ProbParm
{
    amrex::Real p_static = 96321.53451;
    amrex::Real p_out = 96321.53451;
    amrex::Real u = 57.73297950529637;
    amrex::Real v = 5.559053646172991;
    amrex::Real rho = 1.158493;

    amrex::Real* inflow_state = nullptr;

    ProbParm ();
    ~ProbParm ();
};

// AMREX_GPU_DEVICE
// AMREX_FORCE_INLINE
// double SoC(const Eigen::MatrixXd& D, int r, int c, double dx) noexcept {
//     double result = (D(r,c) - D(r,c+1))/(2*dx); //二阶中心有限差分格式
//     return result;
// }

// AMREX_GPU_DEVICE
// AMREX_FORCE_INLINE
// double SoOsB(const Eigen::MatrixXd& D,  int r, double dx) noexcept {
//     double result = (D(r,2) - 4*D(r,1) + 3*D(r,0))/(2*dx); //二阶单边后向有限差分格式
//     return result;
// }

#endif
