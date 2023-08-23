#include <EBR.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_PhysBCFunct.H>
#include <AMReX_Math.H>

using namespace amrex;

struct FillExtDir
{
    Real* left_state = nullptr;
    Real* right_state = nullptr;

    AMREX_GPU_DEVICE
    void operator() (const IntVect& iv, Array4<Real> const& dest,
                     const int dcomp, const int numcomp,
                     GeometryData const& geom, const Real time,
                     const BCRec* bcr, const int bcomp,
                     const int /*orig_comp*/) const
    {
        const Box& domain_box = geom.Domain();
        const BCRec& bc = bcr[bcomp+0];

        const Real* prob_lo = geom.ProbLo();
        const Real* dx      = geom.CellSize();

        int i = iv[0];
        int j = iv[1];
        int k = iv[2];

        Real x = prob_lo[0] + (i+Real(0.5))*dx[0];

        Real x0 = Real(1.0)/Real(6.0) + std::sqrt(1/3.0) + time*10.0/amrex::Math::sinpi(1.0/3.0);
        if (bc.hi(1) == BCType::ext_dir && j > domain_box.bigEnd(1))
        {
            if (x < x0) {
                for (int nc = 0; nc < numcomp; ++nc)
                    dest(i,j,k,dcomp+nc) = left_state[dcomp+nc];
            } else {
                for (int nc = 0; nc < numcomp; ++nc)
                    dest(i,j,k,dcomp+nc) = right_state[dcomp+nc];
            }
        }

        if (bc.lo(1) == BCType::reflect_even && j < domain_box.smallEnd(1))
        {
            if (x < Real(1.0/6.0)){
                for (int nc = 0; nc < numcomp; ++nc)
                    dest(i,j,k,dcomp+nc) = left_state[dcomp+nc]; 
            }
        }

        if (bc.lo(0) == BCType::ext_dir && i < domain_box.smallEnd(0))
        {
            for (int nc = 0; nc < numcomp; ++nc)
                dest(i,j,k,dcomp+nc) = left_state[dcomp+nc]; 
        }
    }
};

// bx                  : Cells outside physical domain and inside bx are filled.
// data, dcomp, numcomp: Fill numcomp components of data starting from dcomp.
// bcr, bcomp          : bcr[bcomp] specifies BC for component dcomp and so on.
// scomp               : component index for dcomp as in the descriptor set up in EBR::variableSetUp.

void ebr_bcfill (Box const& bx, FArrayBox& data,
                 const int dcomp, const int numcomp,
                 Geometry const& geom, const Real time,
                 const Vector<BCRec>& bcr, const int bcomp,
                 const int scomp)
{
    GpuBndryFuncFab<FillExtDir> gpu_bndry_func(FillExtDir{EBR::h_prob_parm->left_state, EBR::h_prob_parm->right_state});
    gpu_bndry_func(bx,data,dcomp,numcomp,geom,time,bcr,bcomp,scomp);
}
