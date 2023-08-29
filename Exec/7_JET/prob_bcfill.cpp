#include <EBR.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_PhysBCFunct.H>

using namespace amrex;

struct FillExtDir_state
{
    Real* inner_state = nullptr;
    Real* outer_state = nullptr;

    AMREX_GPU_DEVICE
    void operator() (const IntVect& iv, Array4<Real> const& dest,
                     const int dcomp, const int numcomp,
                     GeometryData const& geom, const Real /*time*/,
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

        Real y = prob_lo[1] + (j+Real(0.5))*dx[1];
        Real z = prob_lo[2] + (k+Real(0.5))*dx[2];
        Real r = std::sqrt(y*y + z*z);

        if (bc.lo(0) == BCType::ext_dir && i < domain_box.smallEnd(0) && r <= 0.0003)
        {
            for (int nc = 0; nc < numcomp; ++nc)
                dest(i,j,k,dcomp+nc) = inner_state[dcomp+nc]; 
        }

        if (bc.lo(0) == BCType::ext_dir && i < domain_box.smallEnd(0) && r > 0.0003)
        {
            for (int nc = 0; nc < numcomp; ++nc)
                dest(i,j,k,dcomp+nc) = outer_state[dcomp+nc]; 
        }
    }
};

struct FillExtDir_spec
{
    Real* inner_spec = nullptr;
    Real* outer_spec = nullptr;
    AMREX_GPU_DEVICE
    void operator() (const IntVect& iv, Array4<Real> const& dest,
                     const int dcomp, const int numcomp,
                     GeometryData const& geom, const Real /*time*/,
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

        Real y = prob_lo[1] + (j+Real(0.5))*dx[1];
        Real z = prob_lo[2] + (k+Real(0.5))*dx[2];
        Real r = std::sqrt(y*y + z*z);

        if (bc.lo(0) == BCType::ext_dir && i < domain_box.smallEnd(0) && r <= 0.0003)
        {
            for (int nc = 0; nc < numcomp; ++nc)
                dest(i,j,k,dcomp+nc) = inner_spec[dcomp+nc]; 
        }

        if (bc.lo(0) == BCType::ext_dir && i < domain_box.smallEnd(0) && r > 0.0003)
        {
            for (int nc = 0; nc < numcomp; ++nc)
                dest(i,j,k,dcomp+nc) = outer_spec[dcomp+nc]; 
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
    GpuBndryFuncFab<FillExtDir_state> gpu_bndry_func(FillExtDir_state{EBR::h_prob_parm->inner_state, EBR::h_prob_parm->outer_state});
    gpu_bndry_func(bx,data,dcomp,numcomp,geom,time,bcr,bcomp,scomp);
}


void spec_bcfill (Box const& bx, FArrayBox& data,
                 const int dcomp, const int numcomp,
                 Geometry const& geom, const Real time,
                 const Vector<BCRec>& bcr, const int bcomp,
                 const int scomp)
{
    GpuBndryFuncFab<FillExtDir_spec> gpu_bndry_func(FillExtDir_spec{EBR::h_prob_parm->inner_spec, EBR::h_prob_parm->outer_spec});
    gpu_bndry_func(bx,data,dcomp,numcomp,geom,time,bcr,bcomp,scomp);
}