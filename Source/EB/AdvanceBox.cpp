#include "AMReX_Print.H"
#include <IndexDefines.H>
#include <EBR.H>
#include <EBkernels.H>
#include <Kernels.H>
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
                          AMREX_D_DECL(
                          Array4<Real       const> const& apx,
                          Array4<Real       const> const& apy,
                          Array4<Real       const> const& apz),
                          AMREX_D_DECL(
                          Array4<Real       const> const& fcx,
                          Array4<Real       const> const& fcy,
                          Array4<Real       const> const& fcz),
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
    const Box& bxg2 = amrex::grow(bx,2);
    const Box& bxg3 = amrex::grow(bx,3);
    const Box& bxg4 = amrex::grow(bx,4);

    const auto dxinv = geom.InvCellSizeArray();

    // Quantities for redistribution
    FArrayBox divc,redistwgt;
    divc.resize(bxg2,NCONS);
    redistwgt.resize(bxg2,1);

    // Set to zero just in case
    divc.setVal<RunOn::Device>(0.0);
    redistwgt.setVal<RunOn::Device>(0.0);

    // Primitive variables
    FArrayBox qtmp(bxg4, NPRIM, The_Async_Arena());
    auto const& q = qtmp.array();

    // left and right state, async arena
    const Box& nodebox = amrex::surroundingNodes(bx);
    FArrayBox qltmp(nodebox, NPRIM, The_Async_Arena());
    FArrayBox qrtmp(nodebox, NPRIM, The_Async_Arena());
    auto const& ql = qltmp.array();
    auto const& qr = qrtmp.array();

    FArrayBox flux_tmp[AMREX_SPACEDIM];
    for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
        flux_tmp[idim].resize(amrex::surroundingNodes(bxg3,idim),NCONS);
        flux_tmp[idim].setVal<RunOn::Device>(0.);
    }

    Parm const* lparm = d_parm;

    // Initialize dm_as_fine to 0
    if (as_fine)
    {
        ParallelFor(bxg1, NCONS,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
           dm_as_fine(i,j,k,n) = 0.;
        });
    }

    amrex::Print() << "here 1\n";

    ParallelFor(bxg4,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        c2prim(i, j, k, s_arr, q, *lparm);
    });

    for (int cdir=0; cdir<3; ++cdir) {
        const Box& flxbx = amrex::surroundingNodes(bx,cdir);
        ParallelFor(flxbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            eb_recon(i, j, k, ql, qr, q, flag, *lparm, cdir);
        });

        auto const& f_arr = flux_tmp[cdir].array();
        ParallelFor(flxbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            eb_flux(i, j, k, ql, qr, f_arr, *lparm, cdir);
        });
    }

    amrex::Print() << "here 2\n";
    

    // These are the fluxes we computed above -- they live at face centers
    AMREX_D_TERM(auto const& fx_in_arr = flux_tmp[0].array();,
                 auto const& fy_in_arr = flux_tmp[1].array();,
                 auto const& fz_in_arr = flux_tmp[2].array(););

    // These are the fluxes on face centroids -- they are defined in eb_compute_div
    //    and are the fluxes that go into the flux registers
    AMREX_D_TERM(auto const& fx_out_arr = flux[0]->array();,
                 auto const& fy_out_arr = flux[1]->array();,
                 auto const& fz_out_arr = flux[2]->array(););

    auto const& blo = bx.smallEnd();
    auto const& bhi = bx.bigEnd();

    // Because we are going to redistribute, we put the divergence into divc
    //    rather than directly into dsdt_arr
    auto const& divc_arr = divc.array();

    auto l_eb_weights_type = eb_weights_type;

    auto const& redistwgt_arr = redistwgt.array();

    amrex::Print() << "here 3\n";
    
    ParallelFor(bxg2, NCONS,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
       // This does the divergence but not the redistribution -- we will do that later
       // We do compute the weights here though
       eb_compute_div(i,j,k,n,blo,bhi,q,divc_arr,
                      AMREX_D_DECL(fx_in_arr ,fy_in_arr ,fz_in_arr),
                      AMREX_D_DECL(fx_out_arr,fy_out_arr,fz_out_arr),
                      flag, vfrac, bcent, redistwgt_arr,
                      AMREX_D_DECL(apx, apy, apz),
                      AMREX_D_DECL(fcx, fcy, fcz), dxinv, *lparm, l_eb_weights_type);
    });

    amrex::Print() << "here 4\n";

    // Now do redistribution
    int icomp = 0;
    int ncomp = NCONS;
    bool use_wts_in_divnc = false;
    amrex_flux_redistribute(bx, dsdt_arr, divc_arr, redistwgt_arr, vfrac, flag,
                            as_crse, drho_as_crse, rrflag_as_crse,
                            as_fine, dm_as_fine, lev_mask, geom, use_wts_in_divnc,
                            level_mask_notcovered, icomp, ncomp, dt);
    // apply_flux_redistribution(bx, dsdt_arr, divc_arr, redistwgt_arr, icomp, ncomp, flag, vfrac, geom);

    Gpu::streamSynchronize();
}
