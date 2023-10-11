#include "EBR.H"
#include "Reconstruction.H"
#include "Kernels.H"
#include "FluxSplit.H"
#include "ChemViscous.H"

using namespace amrex;

void EBR::flow_advance_multi(Real time, Real dt, int iteration, int ncycle)
{
    BL_PROFILE("EBR::flow_advance");

    for (int i=0; i<NUM_STATE_TYPE; i++)
    {
        state[i].allocOldData();
        state[i].swapTimeLevels(dt);
    }

    MultiFab& S_new = get_new_data(State_Type);
    MultiFab& S_old = get_old_data(State_Type);

    MultiFab& Spec_new = get_new_data(Spec_Type);
    MultiFab& Spec_old = get_old_data(Spec_Type);

    MultiFab& C_new = get_new_data(Cost_Type);
    C_new.setVal(0.0);

    // rhs
    MultiFab dSdt(grids, dmap, NUM_STATE, 0, MFInfo(), Factory());
    MultiFab dSdt_spec(grids, dmap, NSPECS, 0, MFInfo(), Factory());
    // state with ghost cell
    MultiFab Sborder(grids, dmap, NUM_STATE, NUM_GROW, MFInfo(), Factory());
    MultiFab Spec_border(grids, dmap, NSPECS, NUM_GROW, MFInfo(), Factory());

    EBFluxRegister* fine = nullptr; 
    EBFluxRegister* current = nullptr; 
    EBFluxRegister* fine_spec = nullptr; 
    EBFluxRegister* current_spec = nullptr; 

    // reflux for flow
    if (do_reflux && level < parent->finestLevel())
    {
        EBR& fine_level = getLevel(level+1);
        fine = &fine_level.flux_reg;
        fine_spec = &fine_level.flux_reg_spec;
        fine->reset();
        fine_spec->reset();
    }

    if (do_reflux && level > 0)
    {
        current = &flux_reg;
        current_spec = &flux_reg_spec;
    }

    // add Euler here for debug
    if (time_integration == "Euler")
    {
        FillPatch(*this, Sborder, NUM_GROW, time, State_Type, 0, NUM_STATE);
        FillPatch(*this, Spec_border, NUM_GROW, time, Spec_Type, 0, NSPECS);

        compute_dSdt_multi(Sborder, Spec_border, dSdt, dSdt_spec, dt, fine, current, fine_spec, current_spec);
        MultiFab::LinComb(S_new, 1.0, Sborder, 0, dt, dSdt, 0, 0, NUM_STATE, 0);
        MultiFab::LinComb(Spec_new, 1.0, Spec_border, 0, dt, dSdt_spec, 0, 0, NSPECS, 0);
    } else if (time_integration == "RK2")
    {
        // RK2 stage 1
        FillPatch(*this, Sborder, NUM_GROW, time, State_Type, 0, NUM_STATE);
        FillPatch(*this, Spec_border, NUM_GROW, time, Spec_Type, 0, NSPECS);
        compute_dSdt_multi(Sborder, Spec_border, dSdt, dSdt_spec, 0.5*dt, fine, current, fine_spec, current_spec);
        // U^* = U^n + dt * dUdt^n
        // S_new = 1 * Sborder + dt * dSdt
        MultiFab::LinComb(S_new, 1.0, Sborder, 0, dt, dSdt, 0, 0, NUM_STATE, 0);
        MultiFab::LinComb(Spec_new, 1.0, Spec_border, 0, dt, dSdt_spec, 0, 0, NSPECS, 0);

        // Rk2 stage 2
        // after fillpatch Sborder is U^*
        FillPatch(*this, Sborder, NUM_GROW, time+dt, State_Type, 0, NUM_STATE);
        FillPatch(*this, Spec_border, NUM_GROW, time+dt, Spec_Type, 0, NSPECS);
        compute_dSdt_multi(Sborder, Spec_border, dSdt, dSdt_spec, 0.5*dt, fine, current, fine_spec, current_spec);
        // S_new = 0.5 * U^* + 0.5 * U^n + 0.5*dt*dUdt^*
        MultiFab::LinComb(S_new, 0.5, Sborder, 0, 0.5, S_old, 0, 0, NUM_STATE, 0);
        MultiFab::LinComb(Spec_new, 0.5, Spec_border, 0, 0.5, Spec_old, 0, 0, NSPECS, 0);
        MultiFab::Saxpy(S_new, 0.5*dt, dSdt, 0, 0, NUM_STATE, 0);
        MultiFab::Saxpy(Spec_new, 0.5*dt, dSdt_spec, 0, 0, NSPECS, 0);
    } else {
        amrex::Error("No such time scheme !!!");
    }

}

void EBR::compute_dSdt_multi(const amrex::MultiFab &S, amrex::MultiFab &Spec, amrex::MultiFab &dSdt, amrex::MultiFab &dSdt_spec, amrex::Real dt, amrex::EBFluxRegister *fine, amrex::EBFluxRegister *current, amrex::EBFluxRegister *fine_spec, amrex::EBFluxRegister *current_spec)
{
    BL_PROFILE("EBR::compute_dSdt");

    const auto dx = geom.CellSize();
    const auto dxinv = geom.InvCellSizeArray();
    const int ncomp = dSdt.nComp();
    const int nspec = dSdt_spec.nComp();

    auto const& fact = dynamic_cast<EBFArrayBoxFactory const&>(S.Factory());
    auto const& flags = fact.getMultiEBCellFlagFab();

    Parm const* lparm = d_parm;

    MultiFab& cost = get_new_data(Cost_Type);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        FArrayBox dm_as_fine(Box::TheUnitBox(), ncomp);
        FArrayBox dm_as_fine_spec(Box::TheUnitBox(), nspec);

        GpuArray<FArrayBox,AMREX_SPACEDIM> flux;
        GpuArray<FArrayBox,AMREX_SPACEDIM> flux_spec;
        // energy flux due to species diffusion
        GpuArray<FArrayBox,AMREX_SPACEDIM> flux_diffuse;

        for (MFIter mfi(S, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            auto wt = amrex::second();

            const Box& bx = mfi.tilebox();

            const auto& flag = flags[mfi];

            if (flag.getType(bx) == FabType::covered) {
                dSdt[mfi].setVal<RunOn::Device>(0.0, bx , 0, ncomp);
                dSdt_spec[mfi].setVal<RunOn::Device>(0.0, bx , 0, nspec);
            } else {
                for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
                    flux[idim].resize(amrex::surroundingNodes(bx,idim),ncomp);
                    flux_spec[idim].resize(amrex::surroundingNodes(bx,idim),nspec);
                    flux[idim].setVal<RunOn::Device>(0.0);
                    flux_spec[idim].setVal<RunOn::Device>(0.0);
                    if (do_visc) {
                        flux_diffuse[idim].resize(amrex::surroundingNodes(bx,idim),nspec);
                        flux_diffuse[idim].setVal<RunOn::Device>(0.0);
                    }
                }

                auto const& sfab = S.array(mfi);
                auto const& rhoi = Spec.array(mfi);
                auto const& dsdtfab = dSdt.array(mfi);
                auto const& dsdtfab_spec = dSdt_spec.array(mfi);
                auto const& fxfab = flux[0].array();
                auto const& fyfab = flux[1].array();
                auto const& fzfab = flux[2].array();
                auto const& fxfab_spec = flux_spec[0].array();
                auto const& fyfab_spec = flux_spec[1].array();
                auto const& fzfab_spec = flux_spec[2].array();
                auto const& fxfab_diff = flux_diffuse[0].array();
                auto const& fyfab_diff = flux_diffuse[1].array();
                auto const& fzfab_diff = flux_diffuse[2].array();

                if (flag.getType(amrex::grow(bx,NUM_GROW)) == FabType::regular) {
                    // primitives, async arena
                    const Box& bxg = amrex::grow(bx,NUM_GROW);
                    FArrayBox qtmp(bxg, NPRIM, The_Async_Arena());
                    auto const& q = qtmp.array();

                    // lambda mu and D
                    FArrayBox lambda_tmp(bxg, 1, The_Async_Arena());
                    FArrayBox mu_tmp(bxg, 1, The_Async_Arena());
                    FArrayBox D_tmp(bxg, NSPECS, The_Async_Arena());
                    auto const& lambda = lambda_tmp.array();
                    auto const& mu = mu_tmp.array();
                    auto const& D = D_tmp.array();

                    // positive and negative fluxes
                    FArrayBox fptmp(bxg, ncomp, The_Async_Arena());
                    FArrayBox fmtmp(bxg, ncomp, The_Async_Arena());
                    FArrayBox fptmp_spec(bxg, nspec, The_Async_Arena());
                    FArrayBox fmtmp_spec(bxg, nspec, The_Async_Arena());
                    auto const& fp = fptmp.array();
                    auto const& fm = fmtmp.array();
                    auto const& fp_spec = fptmp_spec.array();
                    auto const& fm_spec = fmtmp_spec.array();

                    // For real gas
                    ParallelFor(bxg, 
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        c2prim_rgas(i,j,k,sfab,rhoi,q,*lparm);
                    });

                    if (do_visc) {
                        ParallelFor(bxg, 
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            Real Yt[NSPECS], Xt[NSPECS];
                            Real T = q(i,j,k,QT);
                            Real p = q(i,j,k,QPRES);
                            for (int n=0; n<NSPECS; ++n) {
                                Yt[n] = rhoi(i,j,k,n) / q(i,j,k,QRHO);
                            }
                            CKYTX(Yt, Xt, *lparm);
                            mixtureProperties(T, Xt, mu(i,j,k), lambda(i,j,k), *lparm);
                            getMixDiffCoeffsMass(i,j,k,T, p, Xt, D, *lparm);
                        });
                    }

                    // X-direction
                    int cdir = 0;
                    const Box& xflxbx = amrex::surroundingNodes(bx, cdir);

                    // flux split
                    ParallelFor(bxg,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        flux_split_x(i,j,k,fp,fm,q,sfab,*lparm);
                    });
                    ParallelFor(bxg,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        Real un = q(i,j,k,QU);
                        Real c = q(i,j,k,QC);
                        Real gamma = q(i,j,k,QGAMA);

                        Real E1 = un;
                        Real E2 = un - c;
                        Real E3 = un + c;
                        Real E1P = (E1 + amrex::Math::abs(E1)) * Real(0.5);
                        Real E2P = (E2 + amrex::Math::abs(E2)) * Real(0.5);
                        Real E3P = (E3 + amrex::Math::abs(E3)) * Real(0.5);

                        Real E1M = E1 - E1P;
                        Real E2M = E2 - E2P;
                        Real E3M = E3 - E3P;

                        Real tmp1 = Real(1.0)/(Real(2.0) * gamma);
                        Real tmp2 = Real(2.0) * (gamma - Real(1.0));
                        Real tmp2_p = tmp2* E1P + E2P + E3P;
                        Real tmp2_m = tmp2* E1M + E2M + E3M;
                        
                        for (int n=0; n<nspec; ++n) {
                            Real tmp0 = rhoi(i,j,k,n)*tmp1;
                            fp_spec(i,j,k,n) = tmp0 * tmp2_p;
                            fm_spec(i,j,k,n) = tmp0 * tmp2_m;
                        }
                    });

                    ParallelFor(xflxbx, ncomp,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                    {
                        reconstruction_x(i,j,k,n,fp,fm,fxfab,*lparm);
                    });
                    ParallelFor(xflxbx, nspec,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                    {
                        reconstruction_x(i,j,k,n,fp_spec,fm_spec,fxfab_spec,*lparm);
                    });

                    if (do_visc) {
                        ParallelFor(xflxbx,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            diffusion_x(i,j,k,q,rhoi,D,fxfab_spec,fxfab_diff, dxinv);
                        });
                        ParallelFor(xflxbx,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            compute_visc_x_multi(i,j,k,q,rhoi,lambda,mu,fxfab,fxfab_diff,dxinv,*lparm);
                        });
                    }

                    // Y-direction
                    cdir = 1;
                    const Box& yflxbx = amrex::surroundingNodes(bx, cdir);

                    ParallelFor(bxg,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        flux_split_y(i,j,k,fp,fm,q,sfab,*lparm);
                    });
                    ParallelFor(bxg,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        Real un = q(i,j,k,QV);
                        Real c = q(i,j,k,QC);
                        Real gamma = q(i,j,k,QGAMA);

                        Real E1 = un;
                        Real E2 = un - c;
                        Real E3 = un + c;
                        Real E1P = (E1 + amrex::Math::abs(E1)) * Real(0.5);
                        Real E2P = (E2 + amrex::Math::abs(E2)) * Real(0.5);
                        Real E3P = (E3 + amrex::Math::abs(E3)) * Real(0.5);

                        Real E1M = E1 - E1P;
                        Real E2M = E2 - E2P;
                        Real E3M = E3 - E3P;

                        Real tmp1 = Real(1.0)/(Real(2.0) * gamma);
                        Real tmp2 = Real(2.0) * (gamma - Real(1.0));
                        Real tmp2_p = tmp2* E1P + E2P + E3P;
                        Real tmp2_m = tmp2* E1M + E2M + E3M;
                        
                        for (int n=0; n<nspec; ++n) {
                            Real tmp0 = rhoi(i,j,k,n)*tmp1;
                            fp_spec(i,j,k,n) = tmp0 * tmp2_p;
                            fm_spec(i,j,k,n) = tmp0 * tmp2_m;
                        }
                    });

                    ParallelFor(yflxbx, ncomp,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                    {
                        reconstruction_y(i,j,k,n,fp,fm,fyfab,*lparm);
                    });
                    ParallelFor(yflxbx, nspec,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                    {
                        reconstruction_y(i,j,k,n,fp_spec,fm_spec,fyfab_spec,*lparm);
                    });

                    if (do_visc) {
                        ParallelFor(yflxbx,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            diffusion_y(i,j,k,q,rhoi,D,fyfab_spec,fyfab_diff, dxinv);
                        });
                        ParallelFor(yflxbx,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            compute_visc_y_multi(i,j,k,q,rhoi,lambda,mu,fyfab,fyfab_diff,dxinv,*lparm);
                        });
                    }

                    // Z-direction
                    cdir = 2;
                    const Box& zflxbx = amrex::surroundingNodes(bx, cdir);

                    ParallelFor(bxg,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        flux_split_z(i,j,k,fp,fm,q,sfab,*lparm);
                    });
                    ParallelFor(bxg,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        Real un = q(i,j,k,QW);
                        Real c = q(i,j,k,QC);
                        Real gamma = q(i,j,k,QGAMA);

                        Real E1 = un;
                        Real E2 = un - c;
                        Real E3 = un + c;
                        Real E1P = (E1 + amrex::Math::abs(E1)) * Real(0.5);
                        Real E2P = (E2 + amrex::Math::abs(E2)) * Real(0.5);
                        Real E3P = (E3 + amrex::Math::abs(E3)) * Real(0.5);

                        Real E1M = E1 - E1P;
                        Real E2M = E2 - E2P;
                        Real E3M = E3 - E3P;

                        Real tmp1 = Real(1.0)/(Real(2.0) * gamma);
                        Real tmp2 = Real(2.0) * (gamma - Real(1.0));
                        Real tmp2_p = tmp2* E1P + E2P + E3P;
                        Real tmp2_m = tmp2* E1M + E2M + E3M;
                        
                        for (int n=0; n<nspec; ++n) {
                            Real tmp0 = rhoi(i,j,k,n)*tmp1;
                            fp_spec(i,j,k,n) = tmp0 * tmp2_p;
                            fm_spec(i,j,k,n) = tmp0 * tmp2_m;
                        }
                    });

                    ParallelFor(zflxbx, ncomp,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                    {
                        reconstruction_z(i,j,k,n,fp,fm,fzfab,*lparm);
                    });
                    ParallelFor(zflxbx, nspec,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                    {
                        reconstruction_z(i,j,k,n,fp_spec,fm_spec,fzfab_spec,*lparm);
                    });

                    if (do_visc) {
                        ParallelFor(zflxbx,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            diffusion_z(i,j,k,q,rhoi,D,fzfab_spec,fzfab_diff,dxinv);
                        });
                        ParallelFor(zflxbx,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            compute_visc_z_multi(i,j,k,q,rhoi,lambda,mu,fzfab,fzfab_diff,dxinv,*lparm);
                        });
                    }

                    ParallelFor(bx, ncomp,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                    {
                        divop(i,j,k,n,dsdtfab,fxfab, fyfab, fzfab, dxinv);
                    });
                    ParallelFor(bx, nspec,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                    {
                        divop(i,j,k,n,dsdtfab_spec,fxfab_spec, fyfab_spec, fzfab_spec, dxinv);
                    });
                    
                    // TODO: reflux for EB is too complicated!
                    if (do_reflux)
                    {
                        // the flux registers from the coarse or fine grid perspective
                        // NOTE: the flux register associated with flux_reg[lev] is associated
                        // with the lev/lev-1 interface (and has grid spacing associated with lev-1)
                        if (current) {
                        // update the lev/lev-1 flux register (index lev)
                            // for (int i=0; i<AMREX_SPACEDIM; i++)
                                current->FineAdd(mfi, {&flux[0], &flux[1], &flux[2]}, dx, dt, RunOn::Device);
                                current_spec->FineAdd(mfi, {&flux_spec[0], &flux_spec[1], &flux_spec[2]}, dx, dt, RunOn::Device);
                        }

                        if (fine) {
                        // update the lev+1/lev flux register (index lev+1)
                            // for (int i=0; i<AMREX_SPACEDIM; i++)
                                fine->CrseAdd(mfi, {&flux[0], &flux[1], &flux[2]}, dx, dt, RunOn::Device);
                                fine_spec->CrseAdd(mfi, {&flux_spec[0], &flux_spec[1], &flux_spec[2]}, dx, dt, RunOn::Device);
                        }
                    }
                } else {

                    if (current) {
                        dm_as_fine.resize(amrex::grow(bx,1),ncomp);
                        dm_as_fine_spec.resize(amrex::grow(bx,1),nspec);
                    }

                    int as_fine = (current != nullptr);
                    int as_crse = (fine != nullptr);

                    Array4<Real const> vf_arr = (*volfrac).array(mfi);
                    Array4<Real const> bcent_arr = (*bndrycent).array(mfi);

                    Array4<Real const> const& apx = areafrac[0]->const_array(mfi);
                    Array4<Real const> const& apy = areafrac[1]->const_array(mfi);
                    Array4<Real const> const& apz = areafrac[2]->const_array(mfi);
                    Array4<Real const> const& fcx = facecent[0]->const_array(mfi);
                    Array4<Real const> const& fcy = facecent[1]->const_array(mfi);
                    Array4<Real const> const& fcz = facecent[2]->const_array(mfi);

                    ebchem_compute_dSdt_box(bx, sfab, rhoi, dsdtfab, dsdtfab_spec,
                                       {&flux[0],&flux[1],&flux[2]}, 
                                       {&flux_spec[0],&flux_spec[1],&flux_spec[2]}, 
                                       {&flux_diffuse[0],&flux_diffuse[1],&flux_diffuse[2]}, 
                                        flags.const_array(mfi), vf_arr,
                                        apx, apy, apz, fcx, fcy, fcz, bcent_arr,
                                        as_crse, as_fine, dm_as_fine.array(), dm_as_fine_spec.array(), dt);

                    if (do_reflux) {
                        if (fine) {
                            fine->CrseAdd(mfi, {&flux[0],&flux[1],&flux[2]}, dx,dt,
                                                (*volfrac)[mfi],
                                                {&((*areafrac[0])[mfi]),
                                                &((*areafrac[1])[mfi]),
                                                &((*areafrac[2])[mfi])},
                                                RunOn::Device);
                            fine_spec->CrseAdd(mfi, {&flux_spec[0], &flux_spec[1], &flux_spec[2]}, dx, dt,
                                                    (*volfrac)[mfi],
                                                    {&((*areafrac[0])[mfi]),
                                                    &((*areafrac[1])[mfi]),
                                                    &((*areafrac[2])[mfi])},
                                                    RunOn::Device);
                        }

                        if (current) {
                            current->FineAdd(mfi, {&flux[0],&flux[1],&flux[2]}, dx,dt,
                                                (*volfrac)[mfi],
                                                {&((*areafrac[0])[mfi]),
                                                &((*areafrac[1])[mfi]),
                                                &((*areafrac[2])[mfi])},
                                                dm_as_fine,
                                                RunOn::Device);
                            current_spec->FineAdd(mfi, {&flux_spec[0],&flux_spec[1],&flux_spec[2]}, dx,dt,
                                                (*volfrac)[mfi],
                                                {&((*areafrac[0])[mfi]),
                                                &((*areafrac[1])[mfi]),
                                                &((*areafrac[2])[mfi])},
                                                dm_as_fine_spec,
                                                RunOn::Device);
                        }
                    }
                }
            }
#ifdef AMREX_USE_GPU
            // sync here to avoid out of if loop synchronize
            Gpu::streamSynchronize();
#endif
            wt = (amrex::second() - wt) / bx.d_numPts();
            cost[mfi].plus<RunOn::Device>(wt, bx);
        }
    }
}
