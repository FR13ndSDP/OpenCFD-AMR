#include <EBR.H>
#include <EBR_index_macros.H>
#include <kernels.H>

using namespace amrex;

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
amrex::Real
ebr_estdt (amrex::Box const& bx, amrex::Array4<Real const> const& state,
           amrex::GpuArray<amrex::Real,AMREX_SPACEDIM> const& dx,
           Parm const& parm) noexcept
{
    using amrex::Real;

    Real dt = std::numeric_limits<Real>::max();
    const auto lo = amrex::lbound(bx);
    const auto hi = amrex::ubound(bx);

    for         (int k = lo.z; k <= hi.z; ++k) {
        for     (int j = lo.y; j <= hi.y; ++j) {
            for (int i = lo.x; i <= hi.x; ++i) {
                Real rho = state(i,j,k,URHO);
                Real mx  = state(i,j,k,UMX);
                Real my  = state(i,j,k,UMY);
                Real mz  = state(i,j,k,UMY);
                Real E  = state(i,j,k,UEDEN);
                Real rhoinv = Real(1.0)/amrex::max(rho,parm.smallr);
                Real vx = mx*rhoinv;
                Real vy = my*rhoinv;
                Real vz = mz*rhoinv;
                Real p = amrex::max((parm.eos_gamma-Real(1.0))*(E-Real(0.5)*rho*(vx*vx+vy*vy+vz*vz)), parm.smallp);
                Real cs = std::sqrt(parm.eos_gamma*p*rhoinv);
                Real dtx = dx[0]/(std::abs(vx)+cs);
                Real dty = dx[1]/(std::abs(vy)+cs);
                Real dtz = dx[2]/(std::abs(vz)+cs);
                dt = amrex::min(dt,amrex::min(dtx,amrex::min(dty,dtz)));
            }
        }
    }

    return dt;
}

Real EBR::estTimeStep()
{
    BL_PROFILE("EBR::estTimeStep()");

    const auto dx = geom.CellSizeArray();
    const MultiFab& S = get_new_data(State_Type);
    Parm const* lparm = d_parm;

    // TODO: consider EB
    // auto const& fact = dynamic_cast<EBFArrayBoxFactory const&>(S.Factory());
    // auto const& flags = fact.getMultiEBCellFlagFab();

    Real estdt = amrex::ReduceMin(S, 0,
    [=] AMREX_GPU_HOST_DEVICE (Box const& bx, Array4<Real const> const& fab) -> Real
    {
        return ebr_estdt(bx, fab, dx, *lparm);
    });

    estdt *= cfl;
    ParallelDescriptor::ReduceRealMin(estdt);

    return estdt;
}

Real EBR::advance(Real time, Real dt, int iteration, int ncycle)
{
    BL_PROFILE("EBR::advance");

    for (int i=0; i<NUM_STATE_TYPE; i++)
    {
        state[i].allocOldData();
        state[i].swapTimeLevels(dt);
    }

    MultiFab& S_new = get_new_data(State_Type);
    MultiFab& S_old = get_old_data(State_Type);
    // rhs
    MultiFab dSdt(grids, dmap, NUM_STATE, 0, MFInfo(), Factory());
    // state with ghost cell
    MultiFab Sborder(grids, dmap, NUM_STATE, NUM_GROW, MFInfo(), Factory());

    EBFluxRegister* fine = nullptr; 
    EBFluxRegister* current = nullptr; 

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

    // TODO: Use implicit time integration
    // TODO: use spectral deferred correction method
    // TODO: or consider using builtin RK3/4 interface to maintain accuracy

    // add Euler here for debug
    if (time_integration == "Euler")
    {
        FillPatch(*this, Sborder, NUM_GROW, time, State_Type, 0, NUM_STATE);
        compute_dSdt(Sborder, dSdt, dt, fine, current);
        MultiFab::LinComb(S_new, 1.0, Sborder, 0, dt, dSdt, 0, 0, NUM_STATE, 0);
    }
    else if (time_integration == "RK2")
    {
        // RK2 stage 1
        FillPatch(*this, Sborder, NUM_GROW, time, State_Type, 0, NUM_STATE);
        compute_dSdt(Sborder, dSdt, 0.5*dt, fine, current);
        // U^* = U^n + dt * dUdt^n
        // S_new = 1 * Sborder + dt * dSdt
        MultiFab::LinComb(S_new, 1.0, Sborder, 0, dt, dSdt, 0, 0, NUM_STATE, 0);

        // Rk2 stage 2
        // after fillpatch Sborder is U^*
        FillPatch(*this, Sborder, NUM_GROW, time+dt, State_Type, 0, NUM_STATE);
        compute_dSdt(Sborder, dSdt, 0.5*dt, fine, current);
        // S_new = 0.5 * U^* + 0.5 * U^n + 0.5*dt*dUdt^*
        MultiFab::LinComb(S_new, 0.5, Sborder, 0, 0.5, S_old, 0, 0, NUM_STATE, 0);
        MultiFab::Saxpy(S_new, 0.5*dt, dSdt, 0, 0, NUM_STATE, 0);
    } else
    {
        // // RK3 stage 1
        // FillPatch(*this, Sborder, NUM_GROW, time, State_Type, 0, NUM_STATE);
        // compute_dSdt(Sborder, dSdt, Real(dt/6.0), fine, current);
        // // U^* = U^n + dt * dUdt^n
        // // S_new = 1 * Sborder + dt * dSdt
        // MultiFab::LinComb(S_new, 1.0, Sborder, 0, dt, dSdt, 0, 0, NUM_STATE, 0);

        // // Rk3 stage 2
        // // after fillpatch Sborder is U^*
        // FillPatch(*this, Sborder, NUM_GROW, time+dt, State_Type, 0, NUM_STATE);
        // compute_dSdt(Sborder, dSdt, Real(dt/6.0), fine, current);
        // // S_new = 0.25 * U^* + 0.75 * U^n + 0.25*dt*dUdt^*
        // MultiFab::LinComb(S_new, 0.25, Sborder, 0, 0.75, S_old, 0, 0, NUM_STATE, 0);
        // MultiFab::Saxpy(S_new, 0.25*dt, dSdt, 0, 0, NUM_STATE, 0);

        // // Rk3 stage 3
        // // after fillpatch Sborder is U^*
        // FillPatch(*this, Sborder, NUM_GROW, time+dt, State_Type, 0, NUM_STATE);
        // compute_dSdt(Sborder, dSdt, Real(2.0*dt/3.0), fine, current);
        // MultiFab::LinComb(S_new, 2.0/3.0, Sborder, 0, 1.0/3.0, S_old, 0, 0, NUM_STATE, 0);
        // MultiFab::Saxpy(S_new, 2.0/3.0*dt, dSdt, 0, 0, NUM_STATE, 0);
        RK(3, State_Type, time, dt, iteration, ncycle,
        // Given state S, compute dSdt. dtsub is needed for flux register operations
        [&] (int /*stage*/, MultiFab& dSdt, MultiFab const& S,
                Real /*t*/, Real dtsub) {
            compute_dSdt(S, dSdt, dtsub, fine, current);
        });
    }
    return dt;
}

// compute rhs flux
// bx with no ghost cell
// flux is defined on box face
void EBR::compute_dSdt(const amrex::MultiFab &S, amrex::MultiFab &dSdt, amrex::Real dt, amrex::EBFluxRegister *fine, amrex::EBFluxRegister *current)
{
    BL_PROFILE("EBR::compute_dSdt");

    const auto dx = geom.CellSize();
    const auto dxinv = geom.InvCellSizeArray();
    const int ncomp = dSdt.nComp();

    auto const& fact = dynamic_cast<EBFArrayBoxFactory const&>(S.Factory());
    auto const& flags = fact.getMultiEBCellFlagFab();

    Parm const* lparm = d_parm;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        FArrayBox dm_as_fine(Box::TheUnitBox(), ncomp);
        FArrayBox fab_drho_as_crse(Box::TheUnitBox(), ncomp);
        IArrayBox fab_rrflag_as_crse(Box::TheUnitBox());

        GpuArray<FArrayBox,AMREX_SPACEDIM> flux;

        for (MFIter mfi(S, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();

            const auto& flag = flags[mfi];

            if (flag.getType(bx) == FabType::covered) {
                dSdt[mfi].setVal<RunOn::Gpu>(0.0, bx , 0, ncomp);
            } else {
                // flux is used to store centroid flux needed for reflux
                // for flux_x in x direction is nodal, in other directions centroid
                // for flux_y in y ...
                for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
                    flux[idim].resize(amrex::surroundingNodes(bx,idim),ncomp);
                }

                auto const& sfab = S.array(mfi);
                auto const& dsdtfab = dSdt.array(mfi);
                AMREX_D_TERM(auto const& fxfab = flux[0].array();,
                             auto const& fyfab = flux[1].array();,
                             auto const& fzfab = flux[2].array(););

                // no cut cell around
                if (flag.getType(amrex::grow(bx,1)) == FabType::regular)
                {
                    // primitives, async arena
                    const Box& bxg = amrex::grow(bx,NUM_GROW);
                    FArrayBox qtmp(bxg, NPRIM, The_Async_Arena());
                    auto const& q = qtmp.array();

                    // left and right state, async arena
                    const Box& nodebox = amrex::surroundingNodes(bx);
                    FArrayBox qltmp(nodebox, NPRIM, The_Async_Arena());
                    FArrayBox qrtmp(nodebox, NPRIM, The_Async_Arena());
                    auto const& ql = qltmp.array();
                    auto const& qr = qrtmp.array();

                    ParallelFor<NUM_THREADS>(bxg, 
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        c2prim(i,j,k,sfab,q,*lparm);
                    });

                    // X-direction
                    int cdir = 0;
                    const Box& xflxbx = amrex::surroundingNodes(bx, cdir);

                    ParallelFor<NUM_THREADS>(xflxbx, NPRIM,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                    {
                        reconstruction_x(i,j,k,n,ql,qr,q,*lparm);
                    });

                    ParallelFor<NUM_THREADS>(xflxbx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        compute_flux_x(i,j,k,ql,qr,fxfab,*lparm);
                    });

                    if (do_visc) {
                        ParallelFor<NUM_THREADS>(xflxbx,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            compute_visc_x(i,j,k,q,fxfab,dxinv,*lparm);
                        });
                    }

                    // Y-direction
                    cdir = 1;
                    const Box& yflxbx = amrex::surroundingNodes(bx, cdir);

                    ParallelFor<NUM_THREADS>(yflxbx, NPRIM,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                    {
                        reconstruction_y(i,j,k,n,ql,qr,q,*lparm);
                    });

                    ParallelFor<NUM_THREADS>(yflxbx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        compute_flux_y(i,j,k,ql,qr,fyfab,*lparm);
                    });

                    if (do_visc) {
                        ParallelFor<NUM_THREADS>(yflxbx,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            compute_visc_y(i,j,k,q,fyfab,dxinv,*lparm);
                        });
                    }

                    // Z-direction
                    cdir = 2;
                    const Box& zflxbx = amrex::surroundingNodes(bx, cdir);

                    ParallelFor<NUM_THREADS>(zflxbx, NPRIM,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                    {
                        reconstruction_z(i,j,k,n,ql,qr,q,*lparm);
                    });

                    ParallelFor<NUM_THREADS>(zflxbx,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        compute_flux_z(i,j,k,ql,qr,fzfab,*lparm);
                    });

                    if (do_visc) {
                        ParallelFor<NUM_THREADS>(zflxbx,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            compute_visc_z(i,j,k,q,fzfab,dxinv,*lparm);
                        });
                    }

                    ParallelFor<NUM_THREADS>(bx, NCONS,
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

                    // TODO: reflux for EB is too complicated!
                    if (do_reflux)
                    {
                        // the flux registers from the coarse or fine grid perspective
                        // NOTE: the flux register associated with flux_reg[lev] is associated
                        // with the lev/lev-1 interface (and has grid spacing associated with lev-1)
                        if (current) {
                        // update the lev/lev-1 flux register (index lev)
                            // for (int i=0; i<AMREX_SPACEDIM; i++)
                                current->FineAdd(mfi, {&flux[0], &flux[1], &flux[2]}, dx, dt, RunOn::Gpu);
                        }

                        if (fine) {
                        // update the lev+1/lev flux register (index lev+1)
                            // for (int i=0; i<AMREX_SPACEDIM; i++)
                                fine->CrseAdd(mfi, {&flux[0], &flux[1], &flux[2]}, dx, dt, RunOn::Gpu);
                        }
                    }
#ifdef AMREX_USE_GPU
                    // sync here to avoid out of if loop synchronize
                    Gpu::streamSynchronize();
#endif
                }
                else
                {
                    // cut cells and its neighbor

                    // FArrayBox* p_drho_as_crse = (fine) ?
                    //         fine->getCrseData(mfi) : &fab_drho_as_crse;
                    // const IArrayBox* p_rrflag_as_crse = (fine) ?
                    //     fine->getCrseFlag(mfi) : &fab_rrflag_as_crse;

                    // if (current) {
                    //     dm_as_fine.resize(amrex::grow(bx,1),ncomp);
                    // }

                    // int as_fine = (fine != nullptr);
                    // int as_crse = (current != nullptr);

                    // eb_compute_dudt(BL_TO_FORTRAN_BOX(bx),
                    //                 BL_TO_FORTRAN_ANYD(dSdt[mfi]),
                    //                 BL_TO_FORTRAN_ANYD(S[mfi]),
                    //                 BL_TO_FORTRAN_ANYD(flux[0]),
                    //                 BL_TO_FORTRAN_ANYD(flux[1]),
                    //                 BL_TO_FORTRAN_ANYD(flux[2]),
                    //                 BL_TO_FORTRAN_ANYD(flag),
                    //                 BL_TO_FORTRAN_ANYD((*volfrac)[mfi]),
                    //                 BL_TO_FORTRAN_ANYD((*bndrycent)[mfi]),
                    //                 BL_TO_FORTRAN_ANYD((*areafrac[0])[mfi]),
                    //                 BL_TO_FORTRAN_ANYD((*areafrac[1])[mfi]),
                    //                 BL_TO_FORTRAN_ANYD((*areafrac[2])[mfi]),
                    //                 BL_TO_FORTRAN_ANYD((*facecent[0])[mfi]),
                    //                 BL_TO_FORTRAN_ANYD((*facecent[1])[mfi]),
                    //                 BL_TO_FORTRAN_ANYD((*facecent[2])[mfi]),
                    //                 &as_fine,
                    //                 BL_TO_FORTRAN_ANYD(*p_drho_as_crse),
                    //                 BL_TO_FORTRAN_ANYD(*p_rrflag_as_crse),
                    //                 &as_crse,
                    //                 BL_TO_FORTRAN_ANYD(dm_as_fine),
                    //                 BL_TO_FORTRAN_ANYD(level_mask[mfi]),
                    //                 dx, &dt,&level);

                    // if (fine) {
                    //     fine->CrseAdd(mfi, {&flux[0],&flux[1],&flux[2]}, dx,dt,
                    //                         (*volfrac)[mfi],
                    //                         {&((*areafrac[0])[mfi]),
                    //                          &((*areafrac[1])[mfi]),
                    //                          &((*areafrac[2])[mfi])},
                    //                         RunOn::Cpu);
                    // }

                    // if (current) {
                    //     current->FineAdd(mfi, {&flux[0],&flux[1],&flux[2]}, dx,dt,
                    //                         (*volfrac)[mfi],
                    //                         {&((*areafrac[0])[mfi]),
                    //                          &((*areafrac[1])[mfi]),
                    //                          &((*areafrac[2])[mfi])},
                    //                         dm_as_fine,
                    //                         RunOn::Cpu);
                    // }
                }
            }
        }
    }
}
