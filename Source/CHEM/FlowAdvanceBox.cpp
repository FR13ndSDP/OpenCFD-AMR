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
#include "ChemViscous.H"

using namespace amrex;

void
EBR::ebchem_compute_dSdt_box (const Box& bx,
                          Array4<Real const> const& s_arr,
                          Array4<Real const> const& spec_arr,
                          Array4<Real      > const& dsdt_arr,
                          Array4<Real      > const& dsdt_spec_arr,
                          std::array<FArrayBox*, AMREX_SPACEDIM> const& flux,
                          std::array<FArrayBox*, AMREX_SPACEDIM> const& flux_spec,
                          std::array<FArrayBox*, AMREX_SPACEDIM> const& flux_diffuse,
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
                          Array4<Real            > const& dm_as_fine_spec,
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
    FArrayBox rholtmp(nodebox, NSPECS, The_Async_Arena());
    FArrayBox rhortmp(nodebox, NSPECS, The_Async_Arena());
    qltmp.setVal<RunOn::Device>(Real(1.0));
    qrtmp.setVal<RunOn::Device>(Real(1.0));
    rholtmp.setVal<RunOn::Device>(Real(1.0));
    rhortmp.setVal<RunOn::Device>(Real(1.0));
    auto const& ql = qltmp.array();
    auto const& qr = qrtmp.array();
    auto const& rhol = rholtmp.array();
    auto const& rhor = rhortmp.array();

    // lambda mu and D
    FArrayBox lambda_tmp(bxg, 1, The_Async_Arena());
    FArrayBox mu_tmp(bxg, 1, The_Async_Arena());
    FArrayBox D_tmp(bxg, NSPECS, The_Async_Arena());
    auto const& lambda = lambda_tmp.array();
    auto const& mu = mu_tmp.array();
    auto const& D = D_tmp.array();

    auto const& fxfab = flux[0]->array();
    auto const& fyfab = flux[1]->array();
    auto const& fzfab = flux[2]->array();

    auto const& fxfab_spec = flux_spec[0]->array();
    auto const& fyfab_spec = flux_spec[1]->array();
    auto const& fzfab_spec = flux_spec[2]->array();

    auto const& fxfab_diff = flux_diffuse[0]->array();
    auto const& fyfab_diff = flux_diffuse[1]->array();
    auto const& fzfab_diff = flux_diffuse[2]->array();

    Parm const* lparm = d_parm;

    // Initialize
    ParallelFor(bx, NCONS,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        dsdt_arr(i,j,k,n) = Real(0.0);
        divc_arr(i,j,k,n) = Real(0.0);
    });
    ParallelFor(bx, NSPECS,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        dsdt_spec_arr(i,j,k,n) = Real(0.0);
    });

    // Initialize dm_as_fine to 0
    if (as_fine)
    {
        ParallelFor(bxg1, NCONS,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
           dm_as_fine(i,j,k,n) = 0.;
        });
        ParallelFor(bxg1, NSPECS,
        [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
        {
           dm_as_fine_spec(i,j,k,n) = 0.;
        });
    }

    ParallelFor(bxg,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        c2prim_rgas(i, j, k, s_arr, spec_arr, q, *lparm);
    });

    if (do_visc) {
        ParallelFor(bxg, 
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real Yt[NSPECS], Xt[NSPECS];
            Real T = q(i,j,k,QT);
            Real p = q(i,j,k,QPRES);
            for (int n=0; n<NSPECS; ++n) {
                Yt[n] = spec_arr(i,j,k,n) / q(i,j,k,QRHO);
            }
            CKYTX(Yt, Xt, *lparm);
            mixtureProperties(T, Xt, mu(i,j,k), lambda(i,j,k), *lparm);
            getMixDiffCoeffsMass(i,j,k,T, p, Xt, D, *lparm);
        });
    }

    // x-direction
    int cdir = 0;
    const Box& xflxbx = amrex::surroundingNodes(bx,cdir);
    ParallelFor(xflxbx, NPRIM,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        eb_recon_x(i, j, k, n, vfrac,ql, qr, q, flag, *lparm);
    });
    ParallelFor(xflxbx, NSPECS,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        eb_recon_x(i,j,k,n,vfrac, rhol, rhor, spec_arr, flag, *lparm);
    });

    ParallelFor(xflxbx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        eb_flux(i, j, k, ql, qr, fxfab, cdir, *lparm);
    });
    ParallelFor(xflxbx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        using amrex::Real;

        GpuArray<Real, NSPECS> fp={0}, fm={0};
        Real ul, ur;
        
        ul=ql(i,j,k,QU); 
        ur=qr(i,j,k,QU); 

        Real cL = ql(i,j,k,QC);
        Real cR = qr(i,j,k,QC);

        Real ML, MR, UL, UR;

        ML = ul/cL;
        MR = ur/cR;
        UL = ul;
        UR = ur;

        if (ML >= 1.0) {
            for (int n=0; n<NSPECS; ++n) {
                fp[n] = rhol(i,j,k,n)*UL;
            }
        } else if (amrex::Math::abs(ML)<1.0) {
            Real Mp = 0.250*(1.0 + ML)*(1.0 + ML);
            for (int n=0; n<NSPECS; ++n) {
                Real tmp0 = rhol(i,j,k,n)*cL*Mp;
                fp[n] = tmp0;
            }
        }

        if (amrex::Math::abs(MR) < 1.0) {
            Real Mm = -0.250*(MR - 1.0) * (MR - 1.0);
            for (int n=0; n<NSPECS; ++n) {
                Real tmp0 = rhor(i,j,k,n)*cR*Mm;
                fm[n] = tmp0;
            }
        }
        else if (MR <= -1.0) {
            for (int n=0; n<NSPECS; ++n) {
                fm[n] = rhor(i,j,k,n)*UR;
            }
        }

        for (int n = 0; n < NSPECS; ++n) {
            fxfab_spec(i,j,k,n) = fp[n] + fm[n];
        }  
    });

    if (do_visc) {
        ParallelFor(xflxbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            diffusion_x(i,j,k,q,spec_arr,D,fxfab_spec,fxfab_diff, dxinv);
        });
        ParallelFor(xflxbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            compute_visc_x_multi_eb(i,j,k,q,spec_arr,lambda,mu,fxfab,fxfab_diff,flag,dxinv,weights,*lparm);
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
    ParallelFor(yflxbx, NSPECS,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        eb_recon_y(i, j, k, n, vfrac, rhol, rhor, spec_arr, flag, *lparm);
    });

    ParallelFor(yflxbx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        eb_flux(i, j, k, ql, qr, fyfab, cdir, *lparm);
    });
    ParallelFor(yflxbx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        using amrex::Real;

        GpuArray<Real, NSPECS> fp={0}, fm={0};
        Real ul, ur;
        
        ul=ql(i,j,k,QV); 
        ur=qr(i,j,k,QV); 

        Real cL = ql(i,j,k,QC);
        Real cR = qr(i,j,k,QC);

        Real ML, MR, UL, UR;

        ML = ul/cL;
        MR = ur/cR;
        UL = ul;
        UR = ur;

        if (ML >= 1.0) {
            for (int n=0; n<NSPECS; ++n) {
                fp[n] = rhol(i,j,k,n)*UL;
            }
        } else if (amrex::Math::abs(ML)<1.0) {
            Real Mp = 0.250*(1.0 + ML)*(1.0 + ML);
            for (int n=0; n<NSPECS; ++n) {
                Real tmp0 = rhol(i,j,k,n)*cL*Mp;
                fp[n] = tmp0;
            }
        }

        if (amrex::Math::abs(MR) < 1.0) {
            Real Mm = -0.250*(MR - 1.0) * (MR - 1.0);
            for (int n=0; n<NSPECS; ++n) {
                Real tmp0 = rhor(i,j,k,n)*cR*Mm;
                fm[n] = tmp0;
            }
        }
        else if (MR <= -1.0) {
            for (int n=0; n<NSPECS; ++n) {
                fm[n] = rhor(i,j,k,n)*UR;
            }
        }

        for (int n = 0; n < NSPECS; ++n) {
            fyfab_spec(i,j,k,n) = fp[n] + fm[n];
        }    
    });

    if (do_visc) {
        ParallelFor(yflxbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            diffusion_y(i,j,k,q,spec_arr,D,fyfab_spec,fyfab_diff, dxinv);
        });
        ParallelFor(yflxbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            compute_visc_y_multi_eb(i,j,k,q,spec_arr,lambda,mu,fyfab,fyfab_diff,flag,dxinv,weights,*lparm);
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
    ParallelFor(zflxbx, NSPECS,
    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    {
        eb_recon_z(i, j, k, n, vfrac, rhol, rhor, spec_arr, flag, *lparm);
    });

    ParallelFor(zflxbx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        eb_flux(i, j, k, ql, qr, fzfab, cdir, *lparm);
    });
    ParallelFor(zflxbx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        using amrex::Real;

        GpuArray<Real, NSPECS> fp={0}, fm={0};
        Real ul, ur;
        
        ul=ql(i,j,k,QW); 
        ur=qr(i,j,k,QW); 

        Real cL = ql(i,j,k,QC);
        Real cR = qr(i,j,k,QC);

        Real ML, MR, UL, UR;

        ML = ul/cL;
        MR = ur/cR;
        UL = ul;
        UR = ur;

        if (ML >= 1.0) {
            for (int n=0; n<NSPECS; ++n) {
                fp[n] = rhol(i,j,k,n)*UL;
            }
        } else if (amrex::Math::abs(ML)<1.0) {
            Real Mp = 0.250*(1.0 + ML)*(1.0 + ML);
            for (int n=0; n<NSPECS; ++n) {
                Real tmp0 = rhol(i,j,k,n)*cL*Mp;
                fp[n] = tmp0;
            }
        }

        if (amrex::Math::abs(MR) < 1.0) {
            Real Mm = -0.250*(MR - 1.0) * (MR - 1.0);
            for (int n=0; n<NSPECS; ++n) {
                Real tmp0 = rhor(i,j,k,n)*cR*Mm;
                fm[n] = tmp0;
            }
        }
        else if (MR <= -1.0) {
            for (int n=0; n<NSPECS; ++n) {
                fm[n] = rhor(i,j,k,n)*UR;
            }
        }

        for (int n = 0; n < NSPECS; ++n) {
            fzfab_spec(i,j,k,n) = fp[n] + fm[n];
        }    
    });

    if (do_visc) {
        ParallelFor(zflxbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            diffusion_z(i,j,k,q,spec_arr,D,fzfab_spec,fzfab_diff,dxinv);
        });
        ParallelFor(zflxbx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            compute_visc_z_multi_eb(i,j,k,q,spec_arr,lambda,mu,fzfab,fzfab_diff,flag,dxinv,weights,*lparm);
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
        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real tmp = dxinv[0]/vfrac(i,j,k);
            // drop too small cells
            if (flag(i, j, k).isCovered()) {
                for (int n=0; n<NSPECS; ++n) {
                dsdt_spec_arr(i, j, k, n) = Real(0.0);
                }
            } else if (flag(i,j,k).isRegular()) {
                for (int n=0; n<NSPECS; ++n) {
                dsdt_spec_arr(i, j, k, n) = dxinv[0] * (fxfab_spec(i + 1, j, k, n) - fxfab_spec(i, j, k, n)) +
                                    dxinv[1] * (fyfab_spec(i, j + 1, k, n) - fyfab_spec(i, j, k, n)) +
                                    dxinv[2] * (fzfab_spec(i, j, k + 1, n) - fzfab_spec(i, j, k, n));
                }
            } else {
                //TODO: 2nd-order correction for irregular flux or use Green-Gauss approach
                for (int n=0; n<NSPECS; ++n) {
                dsdt_spec_arr(i, j, k, n) = tmp *
                    (apx(i + 1, j, k)* fxfab_spec(i+1,j,k,n) - apx(i, j, k) * fxfab_spec(i,j,k,n) +
                    apy(i, j + 1, k) * fyfab_spec(i,j+1,k,n) - apy(i, j, k) * fyfab_spec(i,j,k,n) +
                    apz(i, j, k + 1) * fzfab_spec(i,j,k+1,n) - apz(i, j, k) * fzfab_spec(i,j,k,n));
                }

                GpuArray<Real, NSPECS> flux_wall;
                Real apnorm = std::sqrt(
                    (apx(i,j,k)-apx(i+1,j,k))*(apx(i,j,k)-apx(i+1,j,k)) + 
                    (apy(i,j,k)-apy(i,j+1,k))*(apy(i,j,k)-apy(i,j+1,k)) + 
                    (apz(i,j,k)-apz(i,j,k+1))*(apz(i,j,k)-apz(i,j,k+1)) );
                Real un = q(i,j,k,QU)*(apx(i,j,k)-apx(i+1,j,k)) + q(i,j,k,QV)*(apy(i,j,k)-apy(i,j+1,k)) + q(i,j,k,QW)*(apz(i,j,k)-apz(i,j,k+1));
                Real lambda0 = amrex::Math::abs(un) + q(i,j,k,QC)*apnorm;

                for (int n=0; n<NSPECS; ++n) {
                    Real fp = 0.5*(spec_arr(i,j,k,n)*un+lambda0*spec_arr(i,j,k,n));
                    Real fm = -0.5*(spec_arr(i,j,k,n)*un+lambda0*spec_arr(i,j,k,n));
                    flux_wall[n] = fp + fm;
                }
                // Here we assume dx == dy == dz
                for (int n=0; n<NSPECS; ++n) {
                    dsdt_spec_arr(i, j, k, n) += flux_wall[n] * tmp;
                }
            }

            for (int n=0; n<NSPECS; ++n) {
                dsdt_spec_arr(i, j, k, n) *= -1.0;
            }
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
        ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real tmp = dxinv[0]/vfrac(i,j,k);
            // drop too small cells
            if (flag(i, j, k).isCovered()) {
                for (int n=0; n<NSPECS; ++n) {
                dsdt_spec_arr(i, j, k, n) = Real(0.0);
                }
            } else if (flag(i,j,k).isRegular()) {
                for (int n=0; n<NSPECS; ++n) {
                dsdt_spec_arr(i, j, k, n) = dxinv[0] * (fxfab_spec(i + 1, j, k, n) - fxfab_spec(i, j, k, n)) +
                                    dxinv[1] * (fyfab_spec(i, j + 1, k, n) - fyfab_spec(i, j, k, n)) +
                                    dxinv[2] * (fzfab_spec(i, j, k + 1, n) - fzfab_spec(i, j, k, n));
                }
            } else {
                //TODO: 2nd-order correction for irregular flux or use Green-Gauss approach
                for (int n=0; n<NSPECS; ++n) {
                dsdt_spec_arr(i, j, k, n) = tmp *
                    (apx(i + 1, j, k)* fxfab_spec(i+1,j,k,n) - apx(i, j, k) * fxfab_spec(i,j,k,n) +
                    apy(i, j + 1, k) * fyfab_spec(i,j+1,k,n) - apy(i, j, k) * fyfab_spec(i,j,k,n) +
                    apz(i, j, k + 1) * fzfab_spec(i,j,k+1,n) - apz(i, j, k) * fzfab_spec(i,j,k,n));
                }
            }
            for (int n=0; n<NSPECS; ++n) {
                dsdt_spec_arr(i, j, k, n) *= -1.0;
            }
        });  
    }

    // TODO: redist for EB-CHEM
    // if (do_redistribute) {
    //     auto const &lo = bx.smallEnd();
    //     auto const &hi = bx.bigEnd();
    //     // Now do redistribution
    //     ParallelFor(bx,
    //     [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    //     {
    //         // make sure the cell is small
    //         if (vfrac(i,j,k) < 0.5 && vfrac(i,j,k) > 0.0) {
    //             flux_redist(i,j,k,lo,hi,dsdt_arr,divc_arr,flag, vfrac);
    //         } else {
    //             for (int n=0; n<NCONS; ++n) {
    //                 dsdt_arr(i,j,k,n) = divc_arr(i,j,k,n);
    //             }
    //         }
    //     });
    // } else {
    //     ParallelFor(bx, NCONS,
    //     [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
    //     {
    //         dsdt_arr(i,j,k,n) = divc_arr(i,j,k,n);
    //     });
    // }
}
