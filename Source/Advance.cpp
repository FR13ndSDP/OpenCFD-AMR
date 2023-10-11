#include <EBR.H>
#include <IndexDefines.H>
#include <Kernels.H>
#include <Reconstruction.H>
#include <FluxSplit.H>
#include <Diffusion.H>

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
                Real mz  = state(i,j,k,UMZ);
                Real E  = state(i,j,k,UEDEN);
                Real rhoinv = Real(1.0)/amrex::max(rho,parm.smallr);
                Real vx = mx*rhoinv;
                Real vy = my*rhoinv;
                Real vz = mz*rhoinv;
// #ifdef CHEM
//                 Real p;
//                 Real T = state(i,j,k,UTemp);
//                 CKPY(rhoi, T, p);
//                 Real e = E-0.5_rt*rho*(vx*vx+vy*vy+vz*vz);
//                 Real gamma = 1.0_rt + p/e;
//                 Real cs = std::sqrt(gamma*p/rho);
// #else
                Real p = amrex::max((parm.eos_gamma-Real(1.0))*(E-Real(0.5)*rho*(vx*vx+vy*vy+vz*vz)), parm.smallp);
                Real cs = std::sqrt(parm.eos_gamma*p*rhoinv);
// #endif
                Real dtx = dx[0]/(std::abs(vx)+cs);
                Real dty = dx[1]/(std::abs(vy)+cs);
                Real dtz = dx[2]/(std::abs(vz)+cs);
                dt = amrex::min(dt,amrex::min(dtx,amrex::min(dty,dtz)));
            }
        }
    }

    //TODO: fix dt for chem now
#ifdef CHEM
    dt = 1e-8;
#endif
    return dt;
}

Real EBR::estTimeStep()
{
    BL_PROFILE("EBR::estTimeStep()");

    const auto dx = geom.CellSizeArray();
    const MultiFab& S = get_new_data(State_Type);
    Parm const* lparm = d_parm;

    Real estdt = amrex::ReduceMin(S, 0,
    [=] AMREX_GPU_HOST_DEVICE (Box const& bx, Array4<Real const> const& fab) -> Real
    {
        return ebr_estdt(bx, fab, dx, *lparm);
    });

    estdt *= cfl;
    ParallelDescriptor::ReduceRealMin(estdt);

    return estdt;
}

void EBR::flow_advance(Real time, Real dt, int iteration, int ncycle)
{
    BL_PROFILE("EBR::flow_advance");

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

    MultiFab& C_new = get_new_data(Cost_Type);
    C_new.setVal(0.0);

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
        RK(3, State_Type, time, dt, iteration, ncycle,
        // Given state S, compute dSdt. dtsub is needed for flux register operations
        [&] (int /*stage*/, MultiFab& dSdt, MultiFab const& S,
                Real /*t*/, Real dtsub) {
            compute_dSdt(S, dSdt, dtsub, fine, current);
        },
        // TODO: implement state redistribution
       [&] (int /*stage*/, MultiFab& S) { state_redist(S,0); });
    }
}

Real EBR::advance(Real time, Real dt, int iteration, int ncycle)
{
    BL_PROFILE("EBR::advance");

#ifndef CHEM
    flow_advance(time, dt, iteration, ncycle);
#else
    int iter = level<=2? 4/int(pow(2,level)):1;
    Real dt1 = 0.5*dt;
    Real dt2 = dt1/iter;

    /*
        Operator splitting: Introduction to Computational Astrophisical Hydrodynamics, Michael Zingale, p213.
    */

    for (int n=0; n<iter; ++n) {
        chemical_advance(dt2);
    }
    flow_advance_multi(time, dt, iteration, ncycle);
    for (int n=0; n<iter; ++n) {
        chemical_advance(dt2);
    }
#endif

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

    MultiFab& cost = get_new_data(Cost_Type);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    {
        FArrayBox dm_as_fine(Box::TheUnitBox(), ncomp);

        GpuArray<FArrayBox,AMREX_SPACEDIM> flux;

        for (MFIter mfi(S, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            auto wt = amrex::second();

            const Box& bx = mfi.tilebox();

            const auto& flag = flags[mfi];

            if (flag.getType(bx) == FabType::covered) {
                dSdt[mfi].setVal<RunOn::Device>(0.0, bx , 0, ncomp);
            } else {
                // flux is used to store centroid flux needed for reflux
                // for flux_x in x direction is nodal, in other directions centroid
                // for flux_y in y ...
                for (int idim=0; idim < AMREX_SPACEDIM; ++idim) {
                    flux[idim].resize(amrex::surroundingNodes(bx,idim),ncomp);
                    flux[idim].setVal<RunOn::Device>(0.0);
                }

                auto const& sfab = S.array(mfi);
                auto const& dsdtfab = dSdt.array(mfi);
                auto const& fxfab = flux[0].array();
                auto const& fyfab = flux[1].array();
                auto const& fzfab = flux[2].array();

                // no cut cell around
                if (flag.getType(amrex::grow(bx,NUM_GROW)) == FabType::regular)
                {
                    // primitives, async arena
                    const Box& bxg = amrex::grow(bx,NUM_GROW);
                    FArrayBox qtmp(bxg, NPRIM, The_Async_Arena());
                    auto const& q = qtmp.array();

                    // positive and negative fluxes
                    FArrayBox fptmp(bxg, ncomp, The_Async_Arena());
                    FArrayBox fmtmp(bxg, ncomp, The_Async_Arena());
                    auto const& fp = fptmp.array();
                    auto const& fm = fmtmp.array();

                    // For perfect gas
                    ParallelFor(bxg, 
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        c2prim(i,j,k,sfab,q,*lparm);
                    });

                    // X-direction
                    int cdir = 0;
                    const Box& xflxbx = amrex::surroundingNodes(bx, cdir);

                    // flux split
                    ParallelFor(bxg,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                    {
                        flux_split_x(i,j,k,fp,fm,q,sfab,*lparm);
                    });

                    ParallelFor(xflxbx, ncomp,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                    {
                        reconstruction_x(i,j,k,n,fp,fm,fxfab,*lparm);
                    });

                    if (do_visc) {
                        ParallelFor(xflxbx,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            compute_visc_x(i,j,k,q,fxfab,dxinv,*lparm);
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

                    ParallelFor(yflxbx, ncomp,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                    {
                        reconstruction_y(i,j,k,n,fp,fm,fyfab,*lparm);
                    });

                    if (do_visc) {
                        ParallelFor(yflxbx,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            compute_visc_y(i,j,k,q,fyfab,dxinv,*lparm);
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

                    ParallelFor(zflxbx, ncomp,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                    {
                        reconstruction_z(i,j,k,n,fp,fm,fzfab,*lparm);
                    });

                    if (do_visc) {
                        ParallelFor(zflxbx,
                        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                        {
                            compute_visc_z(i,j,k,q,fzfab,dxinv,*lparm);
                        });
                    }

                    ParallelFor(bx, ncomp,
                    [=] AMREX_GPU_DEVICE (int i, int j, int k, int n) noexcept
                    {
                        divop(i,j,k,n,dsdtfab,fxfab, fyfab, fzfab, dxinv);
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
                                current->FineAdd(mfi, {&flux[0], &flux[1], &flux[2]}, dx, dt, RunOn::Device);
                        }

                        if (fine) {
                        // update the lev+1/lev flux register (index lev+1)
                            // for (int i=0; i<AMREX_SPACEDIM; i++)
                                fine->CrseAdd(mfi, {&flux[0], &flux[1], &flux[2]}, dx, dt, RunOn::Device);
                        }
                    }
                }
                else
                {
                    // cut cells and its neighbor

                    if (current) {
                        dm_as_fine.resize(amrex::grow(bx,1),ncomp);
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

                    eb_compute_dSdt_box(bx, sfab, dsdtfab, 
                                       {&flux[0],&flux[1],&flux[2]}, 
                                        flags.const_array(mfi), vf_arr,
                                        apx, apy, apz, fcx, fcy, fcz, bcent_arr,
                                        as_crse, as_fine, dm_as_fine.array(), dt);

                    if (do_reflux) {
                        if (fine) {
                            fine->CrseAdd(mfi, {&flux[0],&flux[1],&flux[2]}, dx,dt,
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
                        }
                    }
                }
            }
#ifdef AMREX_USE_GPU
            Gpu::streamSynchronize();
#endif 
            wt = (amrex::second() - wt) / bx.d_numPts();
            cost[mfi].plus<RunOn::Device>(wt, bx);
        }
    }
}
