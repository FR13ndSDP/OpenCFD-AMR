#include <IndexDefines.H>
#include <EBR.H>
#include <EBkernels.H>
#include <Kernels.H>
#include <FluxSplit.H>
#include <AMReX_EBFArrayBox.H>
#include <AMReX_EB_Redistribution.H>
#include <AMReX_MultiCutFab.H>

#include <AMReX_EBMultiFabUtil_3D_C.H>

using namespace amrex;

void
EBR::eb_compute_dSdt_box (const Box& bx,
                          Array4<Real const> const& s_arr,
                          Array4<Real      > const& dsdt_arr,
                          std::array<FArrayBox*, AMREX_SPACEDIM> const& flux,
                          Array4<EBCellFlag const> const& flag,
                          Array4<Real       const> const& vfrac,
                          Array4<Real       const> const& apx,
                          Array4<Real       const> const& apy,
                          Array4<Real       const> const& apz,
                          Array4<Real       const> const& fcx,
                          Array4<Real       const> const& fcy,
                          Array4<Real       const> const& fcz,
                          Array4<Real       const> const& bcent,
                          int as_crse,
                          Array4<Real            > const& drho_as_crse,
                          Array4<int        const> const& rrflag_as_crse,
                          int as_fine,
                          Array4<Real            > const& dm_as_fine,
                          Array4<int        const> const& lev_mask,
                          Real dt)
{
    BL_PROFILE("EBR::eb_compute_dSdt_box()");

    const Box& bxg1 = amrex::grow(bx,1);
    const Box& bxg = amrex::grow(bx,NUM_GROW);

    const auto dxinv = geom.InvCellSizeArray();

    // Primitive variables
    FArrayBox qtmp(bxg, NPRIM, The_Async_Arena());
    auto const& q = qtmp.array();

    // left and right state, async arena
    const Box& nodebox = amrex::surroundingNodes(bx);
    FArrayBox qltmp(nodebox, NPRIM, The_Async_Arena());
    FArrayBox qrtmp(nodebox, NPRIM, The_Async_Arena());
    qltmp.setVal<RunOn::Device>(Real(1.0));
    qrtmp.setVal<RunOn::Device>(Real(1.0));
    auto const& ql = qltmp.array();
    auto const& qr = qrtmp.array();

    AMREX_D_TERM(auto const& fxfab = flux[0]->array();,
                 auto const& fyfab = flux[1]->array();,
                 auto const& fzfab = flux[2]->array(););

    Parm const* lparm = d_parm;

    // Initialize dsdt
    ParallelFor(bx, NCONS,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        dsdt_arr(i,j,k,n) = Real(0.0);
    });

    // Initialize dm_as_fine to 0
    if (as_fine)
    {
        ParallelFor(bxg1, NCONS,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
           dm_as_fine(i,j,k,n) = 0.;
        });
    }

    ParallelFor(bxg,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        c2prim(i, j, k, s_arr, q, *lparm);
    });

    // x-direction
    int cdir = 0;
    const Box& xflxbx = amrex::surroundingNodes(bx,cdir);
    ParallelFor(xflxbx, NPRIM,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        eb_recon_x(i, j, k, n, ql, qr, q, flag, *lparm);
    });

    ParallelFor(xflxbx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        compute_flux_x(i, j, k, ql, qr, fxfab, *lparm);
    });

    // y-direction
    cdir = 1;
    const Box& yflxbx = amrex::surroundingNodes(bx,cdir);
    ParallelFor(yflxbx, NPRIM,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        eb_recon_y(i, j, k, n, ql, qr, q, flag, *lparm);
    });

    ParallelFor(yflxbx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        compute_flux_y(i, j, k, ql, qr, fyfab, *lparm);
    });

    // z-direction
    cdir = 2;
    const Box& zflxbx = amrex::surroundingNodes(bx,cdir);
    ParallelFor(zflxbx, NPRIM,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        eb_recon_z(i, j, k, n, ql, qr, q, flag, *lparm);
    });

    ParallelFor(zflxbx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        compute_flux_z(i, j, k, ql, qr, fzfab, *lparm);
    });

    ParallelFor(bx, NCONS,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
       eb_compute_div(i,j,k,n,q,dsdt_arr,
                      fxfab, fyfab, fzfab,
                      flag, vfrac, bcent,
                      apx, apy, apz,
                      fcx, fcy, fcz, dxinv, *lparm);
    });

    if (do_gravity) {
        const Real g = -9.8;
        const int irho = Density;
        const int imz = Zmom;
        const int irhoE = Eden;
        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            dsdt_arr(i,j,k,imz) += g * s_arr(i,j,k,irho);
            dsdt_arr(i,j,k,irhoE) += g * s_arr(i,j,k,imz);
        });
    }

#ifdef AMREX_USE_GPU
    Gpu::streamSynchronize();
#endif
}
