#include "EBR.H"
#include "Reconstruction.H"
#include "Kernels.H"
#include "FluxSplit.H"
#include "CHEM_viscous.H"

using namespace amrex;

void EBR::chemical_advance(Real dt)
{
    BL_PROFILE("EBR::chemical_advance");
    MultiFab& S_new = get_new_data(State_Type);
    MultiFab& Spec_new = get_new_data(Spec_Type);

    for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();

        auto const& sfab = S_new.array(mfi);
        auto const& rhoi = Spec_new.array(mfi);

        Parm const* lparm = d_parm;

        ParallelFor<NTHREADS>(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real T, ei;
            Real dei = 0;
            Real rhoi_0[NSPECS], wdot[NSPECS];

            Real c[NSPECS], Arate[NSPECS][NSPECS], A1[NSPECS][NSPECS], rhoi_1[NSPECS], drho[NSPECS];

            Real rho = amrex::max(sfab(i,j,k,URHO), lparm->smallr);
            Real rhoinv = Real(1.0)/rho;
            Real ux = sfab(i,j,k,UMX)*rhoinv;
            Real uy = sfab(i,j,k,UMY)*rhoinv;
            Real uz = sfab(i,j,k,UMZ)*rhoinv;
            Real kineng = Real(0.5)*(ux*ux+uy*uy+uz*uz);
            ei = sfab(i,j,k,UEDEN) - rho * kineng;

            for (int n=0; n<NSPECS; ++n) {
                rhoi_0[n] = rhoi(i,j,k,n);
            }

            GET_T_GIVEN_EY(ei, rhoi_0, T);

            for (int n = 0; n < NSPECS; n++) {
                c[n] = rhoi_0[n] / mw[n] * 1e-6;
            }

            /*call productionRate */
            vproductionRate(wdot, Arate, c, T);

            for (int i = 0; i < NSPECS; ++i) {
            for (int j = 0; j < NSPECS; ++j) {
                A1[i][j] = (i == j ? 1.0 : 0.0);
                A1[i][j] -= Arate[i][j] * mw[i] / mw[j] * dt;
            }
            }

            for (int n = 0; n < NSPECS; ++n) {
                rhoi_1[n] = wdot[n] * mw[n] * 1e6 * dt;
            }

            gauss(drho, A1, rhoi_1);

            for (int n = 0; n < NSPECS; ++n) {
                rhoi(i,j,k,n) += drho[n];
                if (rhoi(i,j,k,n) < 0) rhoi(i,j,k,n) = Real(0.0);
            }

            for (int n=0; n<NSPECS; ++n) {
                dei += -HP[n] * Ru / mw[n] * drho[n];
            }

            sfab(i,j,k,UEDEN) += dei;
        });
    }
}

void EBR::specie_advance_multi(Real time, Real dt, int iteration, int ncycle)
{
    BL_PROFILE("EBR::specie_advance");

    state[Spec_Type].allocOldData();
    state[Spec_Type].swapTimeLevels(dt);
    
    // reflux for specs
    EBFluxRegister* fine_spec = nullptr; 
    EBFluxRegister* current_spec = nullptr; 

    if (do_reflux && level < parent->finestLevel())
    {
        EBR& fine_level = getLevel(level+1);
        fine_spec = &fine_level.flux_reg_spec;
        fine_spec->reset();
    }

    if (do_reflux && level > 0)
    {
        current_spec = &flux_reg_spec;
    }

    // do scalar transport
    MultiFab& Spec_new = get_new_data(Spec_Type);
    MultiFab& Spec_old = get_old_data(Spec_Type);
    MultiFab Spec_border(grids, dmap, NSPECS, NUM_GROW, MFInfo(), Factory());
    MultiFab dSdt_spec(grids, dmap, NSPECS, 0, MFInfo(), Factory());

    // state with ghost cell
    MultiFab Sborder(grids, dmap, NUM_STATE, NUM_GROW, MFInfo(), Factory());

    // add Euler here for debug
    if (time_integration == "Euler")
    {
        FillPatch(*this, Sborder, NUM_GROW, time, State_Type, 0, NUM_STATE);
        FillPatch(*this, Spec_border, NUM_GROW, time, Spec_Type, 0, NSPECS);

        scalar_dSdt(Spec_border, Sborder, dSdt_spec, dt, fine_spec, current_spec);
        MultiFab::LinComb(Spec_new, 1.0, Spec_border, 0, dt, dSdt_spec, 0, 0, NSPECS, 0);
    }
    else if (time_integration == "RK2")
    {
        // RK2 stage 1
        FillPatch(*this, Sborder, NUM_GROW, time, State_Type, 0, NUM_STATE);
        FillPatch(*this, Spec_border, NUM_GROW, time, Spec_Type, 0, NSPECS);
        scalar_dSdt(Spec_border, Sborder, dSdt_spec, dt, fine_spec, current_spec);
        MultiFab::LinComb(Spec_new, 1.0, Spec_border, 0, dt, dSdt_spec, 0, 0, NSPECS, 0);

        // Rk2 stage 2
        // after fillpatch Sborder is U^*
        FillPatch(*this, Sborder, NUM_GROW, time+dt, State_Type, 0, NUM_STATE);
        FillPatch(*this, Spec_border, NUM_GROW, time+dt, Spec_Type, 0, NSPECS);
        scalar_dSdt(Spec_border, Sborder, dSdt_spec, 0.5*dt, fine_spec, current_spec);
        MultiFab::LinComb(Spec_new, 0.5, Spec_border, 0, 0.5, Spec_old, 0, 0, NSPECS, 0);
        MultiFab::Saxpy(Spec_new, 0.5*dt, dSdt_spec, 0, 0, NSPECS, 0);
    }
}

void EBR::flow_advance_multi(Real time, Real dt, int iteration, int ncycle)
{
    BL_PROFILE("EBR::flow_advance");

    state[State_Type].allocOldData();
    state[State_Type].swapTimeLevels(dt);

    MultiFab& S_new = get_new_data(State_Type);
    MultiFab& S_old = get_old_data(State_Type);

    // rhs
    MultiFab dSdt(grids, dmap, NUM_STATE, 0, MFInfo(), Factory());
    // state with ghost cell
    MultiFab Sborder(grids, dmap, NUM_STATE, NUM_GROW, MFInfo(), Factory());

    EBFluxRegister* fine = nullptr; 
    EBFluxRegister* current = nullptr; 

    // reflux for flow
    if (do_reflux && level < parent->finestLevel())
    {
        EBR& fine_level = getLevel(level+1);
        fine = &fine_level.flux_reg;
        fine->reset();
    }

    if (do_reflux && level > 0)
    {
        current = &flux_reg;
    }

    // species field
    MultiFab Spec_border(grids, dmap, NSPECS, NUM_GROW, MFInfo(), Factory());

    // add Euler here for debug
    if (time_integration == "Euler")
    {
        FillPatch(*this, Sborder, NUM_GROW, time, State_Type, 0, NUM_STATE);
        FillPatch(*this, Spec_border, NUM_GROW, time, Spec_Type, 0, NSPECS);

        compute_dSdt_multi(Sborder, Spec_border, dSdt, dt, fine, current);
        MultiFab::LinComb(S_new, 1.0, Sborder, 0, dt, dSdt, 0, 0, NUM_STATE, 0);
    }
    else if (time_integration == "RK2")
    {
        // RK2 stage 1
        FillPatch(*this, Sborder, NUM_GROW, time, State_Type, 0, NUM_STATE);
        FillPatch(*this, Spec_border, NUM_GROW, time, Spec_Type, 0, NSPECS);
        compute_dSdt_multi(Sborder, Spec_border, dSdt, 0.5*dt, fine, current);
        // U^* = U^n + dt * dUdt^n
        // S_new = 1 * Sborder + dt * dSdt
        MultiFab::LinComb(S_new, 1.0, Sborder, 0, dt, dSdt, 0, 0, NUM_STATE, 0);

        // Rk2 stage 2
        // after fillpatch Sborder is U^*
        FillPatch(*this, Sborder, NUM_GROW, time+dt, State_Type, 0, NUM_STATE);
        FillPatch(*this, Spec_border, NUM_GROW, time+dt, Spec_Type, 0, NSPECS);
        compute_dSdt_multi(Sborder, Spec_border, dSdt, 0.5*dt, fine, current);
        // S_new = 0.5 * U^* + 0.5 * U^n + 0.5*dt*dUdt^*
        MultiFab::LinComb(S_new, 0.5, Sborder, 0, 0.5, S_old, 0, 0, NUM_STATE, 0);
        MultiFab::Saxpy(S_new, 0.5*dt, dSdt, 0, 0, NUM_STATE, 0);
    }

    // rescaling di
    MultiFab& Spec_new = get_new_data(Spec_Type);
    for (MFIter mfi(Spec_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        auto const& rhoi = Spec_new.array(mfi);
        auto const& sfab = S_new.array(mfi);

        ParallelFor<NTHREADS>(bx, 
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real rho0 = 0;
            for (int n=0; n<NSPECS; ++n) {
                if (rhoi(i,j,k,n) < 0)
                    rhoi(i,j,k,n) = Real(0.0);
                rho0 += rhoi(i,j,k,n);
            }
            Real tmp = sfab(i,j,k,URHO)/rho0;
            for (int n=0; n<NSPECS; ++n) {
                rhoi(i,j,k,n) *= tmp;
            }
        }); 
    }
}

void EBR::compute_dSdt_multi(const amrex::MultiFab &S, amrex::MultiFab &Spec, amrex::MultiFab &dSdt, amrex::Real dt, amrex::EBFluxRegister *fine, amrex::EBFluxRegister *current)
{
    BL_PROFILE("EBR::compute_dSdt");

    const auto dx = geom.CellSize();
    const auto dxinv = geom.InvCellSizeArray();
    const int ncomp = dSdt.nComp();

    Parm const* lparm = d_parm;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        FArrayBox dm_as_fine(Box::TheUnitBox(), ncomp);

        GpuArray<FArrayBox,AMREX_SPACEDIM> flux;

        for (MFIter mfi(S, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();

            for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
                flux[idim].resize(amrex::surroundingNodes(bx,idim),ncomp);
                flux[idim].setVal<RunOn::Device>(0.0);
            }

            auto const& sfab = S.array(mfi);
            auto const& rhoi = Spec.array(mfi);
            auto const& dsdtfab = dSdt.array(mfi);
            auto const& fxfab = flux[0].array();
            auto const& fyfab = flux[1].array();
            auto const& fzfab = flux[2].array();

            // primitives, async arena
            const Box& bxg = amrex::grow(bx,NUM_GROW);
            FArrayBox qtmp(bxg, NPRIM, The_Async_Arena());
            auto const& q = qtmp.array();

            // positive and negative fluxes
            FArrayBox fptmp(bxg, ncomp, The_Async_Arena());
            FArrayBox fmtmp(bxg, ncomp, The_Async_Arena());
            auto const& fp = fptmp.array();
            auto const& fm = fmtmp.array();

            // For real gas
            ParallelFor<NTHREADS>(bxg, 
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                c2prim_rgas(i,j,k,sfab,rhoi,q,*lparm);
            });

            // X-direction
            int cdir = 0;
            const Box& xflxbx = amrex::surroundingNodes(bx, cdir);

            // flux split
            ParallelFor<NTHREADS>(bxg,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                flux_split_x(i,j,k,fp,fm,q,*lparm);
            });

            ParallelFor<NTHREADS>(xflxbx, ncomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {
                reconstruction_x(i,j,k,n,fp,fm,fxfab,*lparm);
            });

            if (do_visc) {
                ParallelFor<NTHREADS>(xflxbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    compute_visc_x_multi(i,j,k,q,rhoi,fxfab,dxinv,*lparm);
                });
            }

            // Y-direction
            cdir = 1;
            const Box& yflxbx = amrex::surroundingNodes(bx, cdir);

            ParallelFor<NTHREADS>(bxg,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                flux_split_y(i,j,k,fp,fm,q,*lparm);
            });

            ParallelFor<NTHREADS>(yflxbx, ncomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {
                reconstruction_y(i,j,k,n,fp,fm,fyfab,*lparm);
            });

            if (do_visc) {
                ParallelFor<NTHREADS>(yflxbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    compute_visc_y_multi(i,j,k,q,rhoi,fyfab,dxinv,*lparm);
                });
            }

            // Z-direction
            cdir = 2;
            const Box& zflxbx = amrex::surroundingNodes(bx, cdir);

            ParallelFor<NTHREADS>(bxg,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                flux_split_z(i,j,k,fp,fm,q,*lparm);
            });

            ParallelFor<NTHREADS>(zflxbx, ncomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {
                reconstruction_z(i,j,k,n,fp,fm,fzfab,*lparm);
            });

            if (do_visc) {
                ParallelFor<NTHREADS>(zflxbx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    compute_visc_z_multi(i,j,k,q,rhoi,fzfab,dxinv,*lparm);
                });
            }

            ParallelFor<NTHREADS>(bx, ncomp,
            [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
            {
                divop(i,j,k,n,dsdtfab,AMREX_D_DECL(fxfab, fyfab, fzfab), dxinv);
            });

            if (do_gravity) {
                const Real g = -9.8;
                const int irho = Density;
                const int imz = Zmom;
                const int irhoE = Eden;
                ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    dsdtfab(i,j,k,imz) += g * sfab(i,j,k,irho);
                    dsdtfab(i,j,k,irhoE) += g * sfab(i,j,k,imz);
                });
            }
#ifdef AMREX_USE_GPU
            // sync here to avoid out of if loop synchronize
            Gpu::streamSynchronize();
#endif
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
                }

                if (fine) {
                // update the lev+1/lev flux register (index lev+1)
                    // for (int i=0; i<AMREX_SPACEDIM; i++)
                        fine->CrseAdd(mfi, {&flux[0], &flux[1], &flux[2]}, dx, dt, RunOn::Device);
                }
            }
        }
    }
}
