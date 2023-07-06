#include <EBR.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_PhysBCFunct.H>

using namespace amrex;

struct FillExtDir
{
    
    AMREX_GPU_DEVICE
    void operator() (const IntVect& iv, Array4<Real> const& dest,
                     const int dcomp, const int numcomp,
                     GeometryData const& geom, const Real /*time*/,
                     const BCRec* bcr, const int bcomp,
                     const int /*orig_comp*/) const
    { 
        
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
    GpuBndryFuncFab<FillExtDir> gpu_bndry_func(FillExtDir{});
    gpu_bndry_func(bx,data,dcomp,numcomp,geom,time,bcr,bcomp,scomp);
}
