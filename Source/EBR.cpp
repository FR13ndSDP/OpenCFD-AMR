#include <EBR.H>
#include <Constants.H>

#include <AMReX_EBMultiFabUtil.H>
#include <AMReX_ParmParse.H>
#include <AMReX_EBFArrayBox.H>
#include <AMReX_MultiCutFab.H>

using namespace amrex;

#if __cplusplus < 201703L
constexpr int EBR::level_mask_interior;
constexpr int EBR::level_mask_covered;
constexpr int EBR::level_mask_notcovered;
constexpr int EBR::level_mask_physbnd;
constexpr int EBR::NUM_GROW;
#endif

BCRec     EBR::phys_bc;

int       EBR::verbose = 0;
IntVect   EBR::hydro_tile_size {1024,16,16};
Real      EBR::cfl       = 0.3;
bool      EBR::do_reflux = true;
bool      EBR::do_visc = true;
bool      EBR::do_gravity = false;
bool      EBR::do_redistribute = false;
bool      EBR::IO_HDF5 = false;
int       EBR::plot_int = -1;
std::string EBR::plot_file = "plt";
Real      EBR::stop_time = 0.0;
int       EBR::max_step = -1;
int       EBR::refine_max_dengrad_lev   = -1;
Real      EBR::refine_dengrad           = 1.0e10;
RealBox*  EBR::dp_refine_boxes;
std::string EBR::time_integration       = "RK2";
Vector<RealBox> EBR::refine_boxes;
int       EBR::refine_cutcells = 1;

EBR::EBR ()
= default;

EBR::EBR (Amr&            papa,
          int             lev,
          const Geometry& level_geom,
          const BoxArray& bl,
          const DistributionMapping& dm,
          Real            time)
    : AmrLevel(papa,lev,level_geom,bl,dm,time)
{
    if (do_reflux && level > 0) {
        flux_reg.define(bl, papa.boxArray(level-1),
                        dm, papa.DistributionMap(level-1),
                        level_geom, papa.Geom(level-1),
                        papa.refRatio(level-1), level, NUM_STATE);
#ifdef CHEM
        flux_reg_spec.define(bl, papa.boxArray(level-1),
                        dm, papa.DistributionMap(level-1),
                        level_geom, papa.Geom(level-1),
                        papa.refRatio(level-1), level, NSPECS);
#endif
    }

    buildMetrics();
}

EBR::~EBR ()
= default;

void
EBR::init (AmrLevel& old)
{
    auto& oldlev = dynamic_cast<EBR&>(old);

    Real dt_new    = parent->dtLevel(level);
    Real cur_time  = oldlev.state[State_Type].curTime();
    Real prev_time = oldlev.state[State_Type].prevTime();
    Real dt_old    = cur_time - prev_time;
    setTimeLevel(cur_time,dt_old,dt_new);

    MultiFab& S_new = get_new_data(State_Type);
    FillPatch(old,S_new,0,cur_time,State_Type,0,NUM_STATE);

    MultiFab& C_new = get_new_data(Cost_Type);
    FillPatch(old,C_new,0,cur_time,Cost_Type,0,1);

#ifdef CHEM
    MultiFab& Spec_new = get_new_data(Spec_Type);
    FillPatch(old,Spec_new,0,cur_time,Spec_Type,0,NSPECS);
#endif
}

void
EBR::init ()
{
    Real dt        = parent->dtLevel(level);
    Real cur_time  = getLevel(level-1).state[State_Type].curTime();
    Real prev_time = getLevel(level-1).state[State_Type].prevTime();
    Real dt_old = (cur_time - prev_time)/static_cast<Real>(parent->MaxRefRatio(level-1));
    setTimeLevel(cur_time,dt_old,dt);

    MultiFab& S_new = get_new_data(State_Type);
    FillCoarsePatch(S_new, 0, cur_time, State_Type, 0, NUM_STATE);

#ifdef CHEM
    MultiFab& Spec_new = get_new_data(Spec_Type);
    FillCoarsePatch(Spec_new,0,cur_time,Spec_Type,0,NSPECS);
#endif
}

void
EBR::initData ()
{
    BL_PROFILE("EBR::initData()");

    const auto geomdata = geom.data();
    MultiFab& S_new = get_new_data(State_Type);

    auto const& fact = dynamic_cast<EBFArrayBoxFactory const&>(S_new.Factory());
    auto const& flags = fact.getMultiEBCellFlagFab();

    Parm const* lparm = d_parm;
    ProbParm const* lprobparm = d_prob_parm;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
    for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& box = mfi.validbox();
        const Box& bxg = amrex::grow(box,NUM_GROW);

        auto sfab = S_new.array(mfi);

        const auto& flag_array = flags.const_array(mfi);
        Array4<Real const> vf_arr = (*volfrac).array(mfi);

        amrex::ParallelFor(bxg,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            // This is in EXE files
            ebr_initdata(i, j, k, sfab, geomdata, vf_arr, flag_array, *lparm, *lprobparm);
        });
    }

    MultiFab& C_new = get_new_data(Cost_Type);
    C_new.setVal(1.0);

#ifdef CHEM
    MultiFab& Spec_new = get_new_data(Spec_Type);

    for (MFIter mfi(Spec_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& box = mfi.validbox();
        auto sfab = Spec_new.array(mfi);
        auto sfab_state = S_new.array(mfi);

        amrex::ParallelFor(box,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            ebr_initspec(i, j, k, sfab_state, sfab, geomdata, *lparm);
        });
    }
#endif
#ifdef AMREX_USE_HDF5
    if (IO_HDF5) {
        writeHDF5PlotFile(0, 0.0);
    }
#endif
}

void EBR::buildMetrics()
{
    // TODO: this is critical
    // make sure dx == dy == dz
    const Real* dx = geom.CellSize();
    if (std::abs(dx[0]-dx[1]) > 1.e-12*dx[0] || std::abs(dx[0]-dx[2]) > 1.e-12*dx[0]) {
        amrex::Abort("EBR: must have dx == dy == dz\n");
    }

    const auto& ebfactory = dynamic_cast<EBFArrayBoxFactory const&>(Factory());

    volfrac = &(ebfactory.getVolFrac());
    bndrycent = &(ebfactory.getBndryCent());
    areafrac = ebfactory.getAreaFrac();
    facecent = ebfactory.getFaceCent();
}

void
EBR::computeInitialDt (int                   finest_level,
                       int                   sub_cycle,
                       Vector<int>&           n_cycle,
                       const Vector<IntVect>& ref_ratio,
                       Vector<Real>&          dt_level,
                       Real                  stop_time)
{
  //
  // Grids have been constructed, compute dt for all levels.
  //
  if (level > 0) {
    return;
  }

  Real dt_0 = std::numeric_limits<Real>::max();
  int n_factor = 1;
  for (int i = 0; i <= finest_level; i++)
  {
    dt_level[i] = getLevel(i).initialTimeStep();
    n_factor   *= n_cycle[i];
    dt_0 = std::min(dt_0,n_factor*dt_level[i]);
  }

  //
  // Limit dt's by the value of stop_time.
  //
  const Real eps = 0.001*dt_0;
  Real cur_time  = state[State_Type].curTime();
  if (stop_time >= 0.0) {
    if ((cur_time + dt_0) > (stop_time - eps))
      dt_0 = stop_time - cur_time;
  }

  n_factor = 1;
  for (int i = 0; i <= finest_level; i++)
  {
    n_factor *= n_cycle[i];
    dt_level[i] = dt_0/n_factor;
  }
}

void
EBR::computeNewDt (int                    finest_level,
                   int                    sub_cycle,
                   Vector<int>&           n_cycle,
                   const Vector<IntVect>& ref_ratio,
                   Vector<Real>&          dt_min,
                   Vector<Real>&          dt_level,
                   Real                   stop_time,
                   int                    post_regrid_flag)
{
    BL_PROFILE("EBR::computeNewDt()");

    //
    // We are at the end of a coarse grid timecycle.
    // Compute the timesteps for the next iteration.
    //
    if (level > 0) {
        return;
    }

    for (int i = 0; i <= finest_level; i++)
    {
        dt_min[i] = getLevel(i).estTimeStep();
    }

    if (post_regrid_flag == 1)
    {
        //
        // Limit dt's by pre-regrid dt
        //
        for (int i = 0; i <= finest_level; i++)
        {
            dt_min[i] = std::min(dt_min[i],dt_level[i]);
        }
    }
    else
    {
        //
        // Limit dt's by change_max * old dt
        //
        static Real change_max = 1.1;
        for (int i = 0; i <= finest_level; i++)
        {
            dt_min[i] = std::min(dt_min[i],change_max*dt_level[i]);
        }
    }

    //
    // Find the minimum over all levels
    //
    Real dt_0 = std::numeric_limits<Real>::max();
    int n_factor = 1;
    for (int i = 0; i <= finest_level; i++)
    {
        n_factor *= n_cycle[i];
        dt_0 = std::min(dt_0,n_factor*dt_min[i]);
    }

    //
    // Limit dt's by the value of stop_time.
    //
    const Real eps = 0.001*dt_0;
    Real cur_time  = state[State_Type].curTime();
    if (stop_time >= 0.0) {
        if ((cur_time + dt_0) > (stop_time - eps)) {
            dt_0 = stop_time - cur_time;
        }
    }

    n_factor = 1;
    for (int i = 0; i <= finest_level; i++)
    {
        n_factor *= n_cycle[i];
        dt_level[i] = dt_0/n_factor;
    }
}

void
EBR::post_regrid (int lbase, int new_finest)
{
}

void
EBR::post_timestep (int iteration)
{
    BL_PROFILE("EBR::post_timestep");

    MultiFab& S_crse = get_new_data(State_Type);
#ifdef CHEM
    MultiFab& Spec_crse = get_new_data(Spec_Type);
#endif
    if (do_reflux && level < parent->finestLevel()) {
        EBR& fine_level = getLevel(level+1);
        MultiFab& S_fine = fine_level.get_new_data(State_Type);
        fine_level.flux_reg.Reflux(S_crse, *volfrac, S_fine, *fine_level.volfrac);
#ifdef CHEM
        MultiFab& Spec_fine = fine_level.get_new_data(Spec_Type);
        fine_level.flux_reg_spec.Reflux(Spec_crse, *volfrac, Spec_fine, *fine_level.volfrac);
#endif
    }

// TODO: mass conservation need varification
#ifdef CHEM
    // rescaling di for all levels
    for (MFIter mfi(Spec_crse, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        auto const& rhoi = Spec_crse.array(mfi);
        auto const& sfab = S_crse.array(mfi);

        ParallelFor(bx, 
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            Real rho0 = 0;
            for (int n=0; n<NSPECS; ++n) {
                if (rhoi(i,j,k,n) < 0) {
                    rhoi(i,j,k,n) = Real(0.0);
                }
                rho0 += rhoi(i,j,k,n);
            }
            Real tmp = sfab(i,j,k,URHO)/rho0;
            for (int n=0; n<NSPECS; ++n) {
                rhoi(i,j,k,n) *= tmp;
            }
        }); 
    }
#endif

    if (level < parent->finestLevel()) {
        avgDown();
    }
}

void
EBR::postCoarseTimeStep (Real time)
{
    BL_PROFILE("EBR::postCoarseTimeStep()");

    // post coarse timestep synctime not needed
    // This only computes sum on level 0
    if (verbose >= 2) {
        printTotal();
    }

#ifdef AMREX_USE_HDF5
    if (IO_HDF5) {
        // the step after one step
        int step = parent->levelSteps(0);

        if (step%plot_int == 0 || time == stop_time || step == max_step) {
            writeHDF5PlotFile(step, time);
        }
    }
#endif
}

void
EBR::printTotal () const
{
    const MultiFab& S_new = get_new_data(State_Type);
    MultiFab mf(grids, dmap, 1, 0);
    Array<Real,5> tot;
    for (int comp = 0; comp < 5; ++comp) {
        MultiFab::Copy(mf, S_new, comp, 0, 1, 0);
        tot[comp] = mf.sum(0,true) * geom.ProbSize();
    }
#ifdef BL_LAZY
    Lazy::QueueReduction( [=] () mutable {
#endif
            ParallelDescriptor::ReduceRealSum(tot.data(), 5, ParallelDescriptor::IOProcessorNumber());
            amrex::Print().SetPrecision(17) << "\n[EBR] Total mass       is " << tot[0] << "\n"
                                            <<   "      Total x-momentum is " << tot[1] << "\n"
                                            <<   "      Total y-momentum is " << tot[2] << "\n"
                                            <<   "      Total z-momentum is " << tot[3] << "\n"
                                            <<   "      Total energy     is " << tot[4] << "\n";
#ifdef BL_LAZY
        });
#endif

#ifdef CHEM
    const MultiFab& Spec_new = get_new_data(Spec_Type);
    Array<Real,NSPECS> tot_spec;
    for (int comp = 0; comp < NSPECS; ++comp) {
        MultiFab::Copy(mf, Spec_new, comp, 0, 1, 0);
        tot_spec[comp] = mf.sum(0,true) * geom.ProbSize();
    }
#ifdef BL_LAZY
    Lazy::QueueReduction( [=] () mutable {
#endif
            ParallelDescriptor::ReduceRealSum(tot_spec.data(), NSPECS, ParallelDescriptor::IOProcessorNumber());
            amrex::Print().SetPrecision(17) << "\n[CHEM] Total spec0 is " << tot_spec[0] << "\n"
                                            <<   "       Total spec1 is " << tot_spec[1] << "\n"
                                            <<   "       Total spec2 is " << tot_spec[2] << "\n"
                                            <<   "       Total spec3 is " << tot_spec[3] << "\n"
                                            <<   "       Total spec4 is " << tot_spec[4] << "\n"
                                            <<   "       Total spec5 is " << tot_spec[5] << "\n"
                                            <<   "       Total spec6 is " << tot_spec[6] << "\n"
                                            <<   "       Total spec7 is " << tot_spec[7] << "\n"
                                            <<   "       Total spec8 is " << tot_spec[8] << "\n";
#ifdef BL_LAZY
        });
#endif
#endif
}

void
EBR::post_init (Real)
{
    BL_PROFILE("EBR::post_init()");

    if (level > 0) return;
    for (int k = parent->finestLevel()-1; k >= 0; --k) {
        getLevel(k).avgDown();
    }

    if (verbose >= 2) {
        printTotal();
    }
}

void
EBR::post_restart ()
{
}

void
EBR::read_params ()
{
    ParmParse pp("ebr");

    pp.query("v", verbose);

    Vector<int> tilesize(AMREX_SPACEDIM);
    if (pp.queryarr("hydro_tile_size", tilesize, 0, AMREX_SPACEDIM))
    {
        for (int i=0; i<AMREX_SPACEDIM; i++) hydro_tile_size[i] = tilesize[i];
    }

    pp.query("cfl", cfl);

    Vector<int> lo_bc(AMREX_SPACEDIM), hi_bc(AMREX_SPACEDIM);
    pp.getarr("lo_bc", lo_bc, 0, AMREX_SPACEDIM);
    pp.getarr("hi_bc", hi_bc, 0, AMREX_SPACEDIM);
    for (int i = 0; i < AMREX_SPACEDIM; ++i) {
        phys_bc.setLo(i,lo_bc[i]);
        phys_bc.setHi(i,hi_bc[i]);
    }

    pp.query("do_reflux", do_reflux);

    pp.query("do_gravity", do_gravity);

    pp.query("do_visc", do_visc);

    pp.query("IO_HDF5", IO_HDF5);
    pp.query("plot_int", plot_int);
    pp.query("plot_file", plot_file);

    pp.query("refine_max_dengrad_lev", refine_max_dengrad_lev);
    pp.query("refine_dengrad", refine_dengrad);
    pp.query("refine_cutcells", refine_cutcells);

    int irefbox = 0;
    Vector<Real> refboxlo, refboxhi;
    while (pp.queryarr(("refine_box_lo_"+std::to_string(irefbox)).c_str(), refboxlo))
    {
        pp.getarr(("refine_box_hi_"+std::to_string(irefbox)).c_str(), refboxhi);
        refine_boxes.emplace_back(refboxlo.data(), refboxhi.data());
        ++irefbox;
    }

    if (!refine_boxes.empty()) {
#ifdef AMREX_USE_GPU
        dp_refine_boxes = (RealBox*)The_Arena()->alloc(sizeof(RealBox)*refine_boxes.size());
        Gpu::htod_memcpy_async(dp_refine_boxes, refine_boxes.data(), sizeof(RealBox)*refine_boxes.size());
#else
        dp_refine_boxes = refine_boxes.data();
#endif
    }

    pp.query("time_integration", time_integration);
    pp.query("eos_gamma", h_parm->eos_gamma);
    pp.query("eos_m"   , h_parm->eos_m);
    pp.query("Pr"       , h_parm->Pr);
    pp.query("C_s"      , h_parm->C_s);
    pp.query("T_s"      , h_parm->T_s);

    h_parm->Initialize();
#ifdef AMREX_USE_GPU
        // Cannot use Gpu::copy because ProbParm is not trivailly copyable.
        Gpu::htod_memcpy_async(EBR::d_parm, EBR::h_parm, sizeof(Parm));
#else
        std::memcpy(EBR::d_parm, EBR::h_parm, sizeof(Parm));
#endif

    pp.query("do_redistribute", do_redistribute);

    ParmParse pg;
    pg.query("stop_time",stop_time);
    pg.query("max_step",max_step);

    amrex::Gpu::streamSynchronize();
}

void
EBR::avgDown ()
{
    BL_PROFILE("EBR::avgDown()");

    if (level == parent->finestLevel()) return;

    auto& fine_lev = getLevel(level+1);

    MultiFab& S_crse =          get_new_data(State_Type);
    MultiFab& S_fine = fine_lev.get_new_data(State_Type);

    MultiFab volume(S_fine.boxArray(), S_fine.DistributionMap(), 1, 0);
    volume.setVal(1.0);
    amrex::EB_average_down(S_fine, S_crse, volume, fine_lev.volFrac(),
                           0, S_fine.nComp(), fine_ratio);

#ifdef CHEM
    MultiFab& Spec_crse =          get_new_data(Spec_Type);
    MultiFab& Spec_fine = fine_lev.get_new_data(Spec_Type);

    amrex::EB_average_down(Spec_fine, Spec_crse, volume, fine_lev.volFrac(),
                           0, Spec_fine.nComp(), fine_ratio);
#endif
}

Real EBR::initialTimeStep()
{
    return estTimeStep();
}
