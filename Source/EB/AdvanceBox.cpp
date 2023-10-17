#include <IndexDefines.H>
#include <EBR.H>
#include <EBkernels.H>
#include <EBdiffusion.H>
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
                          int as_crse, int as_fine,
                          Array4<Real            > const& dm_as_fine,
                          Real dt)
{
    BL_PROFILE("EBR::eb_compute_dSdt_box()");

    const Box& bxg1 = amrex::grow(bx,1);
    const Box& bxg = amrex::grow(bx,NUM_GROW);

    const auto dxinv = geom.InvCellSizeArray();

    GpuArray<Real,3> weights;
    weights[0] = 0.;
    weights[1] = 1.;
    weights[2] = 0.5;

    // Quantities for redistribution
    FArrayBox divc,redistwgt;
    divc.resize(bx,NCONS);
    redistwgt.resize(bx,1);

    // Set to zero just in case
    divc.setVal<RunOn::Device>(0.0);
    redistwgt.setVal<RunOn::Device>(0.0);

    // Because we are going to redistribute, we put the divergence into divc
    //    rather than directly into dsdt_arr
    auto const& divc_arr = divc.array();

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

    auto const& fxfab = flux[0]->array();
    auto const& fyfab = flux[1]->array();
    auto const& fzfab = flux[2]->array();

    Parm const* lparm = d_parm;

    // Initialize
    ParallelFor(bx, NCONS,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        dsdt_arr(i,j,k,n) = Real(0.0);
        divc_arr(i,j,k,n) = Real(0.0);
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
        eb_recon_x(i, j, k, n, vfrac,ql, qr, q, flag, *lparm);
    });

    ParallelFor(xflxbx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        eb_flux(i, j, k, ql, qr, fxfab, cdir, *lparm);
    });

    if (do_visc) {
        ParallelFor(xflxbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            eb_compute_visc_x(i,j,k,q,fxfab,flag,dxinv,weights,*lparm);
        });
    }

    // y-direction
    cdir = 1;
    const Box& yflxbx = amrex::surroundingNodes(bx,cdir);
    ParallelFor(yflxbx, NPRIM,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        eb_recon_y(i, j, k, n, vfrac, ql, qr, q, flag, *lparm);
    });

    ParallelFor(yflxbx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        eb_flux(i, j, k, ql, qr, fyfab, cdir, *lparm);
    });

    if (do_visc) {
        ParallelFor(yflxbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            eb_compute_visc_y(i,j,k,q,fyfab,flag,dxinv,weights,*lparm);
        });
    }

    // z-direction
    cdir = 2;
    const Box& zflxbx = amrex::surroundingNodes(bx,cdir);
    ParallelFor(zflxbx, NPRIM,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        eb_recon_z(i, j, k, n, vfrac, ql, qr, q, flag, *lparm);
    });

    ParallelFor(zflxbx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        eb_flux(i, j, k, ql, qr, fzfab, cdir, *lparm);
    });

    if (do_visc) {
        ParallelFor(zflxbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            eb_compute_visc_z(i,j,k,q,fzfab,flag,dxinv,weights,*lparm);
        });
    }

    if (do_visc) {
        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            eb_compute_div_visc(i,j,k,q, divc_arr,
                            fxfab, fyfab, fzfab,
                            flag, vfrac, bcent,
                            apx, apy, apz,
                            fcx, fcy, fcz, dxinv, *lparm);
        });
    } else {
        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            eb_compute_div(i,j,k,q, divc_arr,
                            fxfab, fyfab, fzfab,
                            flag, vfrac, bcent,
                            apx, apy, apz,
                            fcx, fcy, fcz, dxinv, *lparm);
        });  
    }

    if (do_redistribute) {
        auto const &lo = bx.smallEnd();
        auto const &hi = bx.bigEnd();
        // Now do redistribution
        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // make sure the cell is small
            if (vfrac(i,j,k) < 0.5 && vfrac(i,j,k) > 0.0) {
                flux_redist(i,j,k,lo,hi,dsdt_arr,divc_arr,flag, vfrac);
            } else {
                for (int n=0; n<NCONS; ++n) {
                    dsdt_arr(i,j,k,n) = divc_arr(i,j,k,n);
                }
            }
        });
    } else {
        ParallelFor(bx, NCONS,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
            dsdt_arr(i,j,k,n) = divc_arr(i,j,k,n);
        });
    }

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
}

// TODO: implement state redistribution
void EBR::state_redist(MultiFab& State, int ng)
{

}