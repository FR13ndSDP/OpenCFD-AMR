#ifndef EBR_H_
#define EBR_H_

#include <prob.H>

#include <AMReX_AmrLevel.H>
#include <AMReX_EBFluxRegister.H>
#include <AMReX_EBCellFlag.H>
#include <AMReX_LayoutData.H>

using namespace amrex;
class EBR : public amrex::AmrLevel {
public:
  EBR();
  EBR(amrex::Amr &papa, int lev, const amrex::Geometry &level_geom,
      const amrex::BoxArray &bl, const amrex::DistributionMapping &dm,
      amrex::Real time);
  ~EBR() override;

  // disable copy
  EBR(const EBR &rhs) = delete;
  EBR &operator=(const EBR &rhs) = delete;

  // Restart from a checkpoint file.
  virtual void restart(amrex::Amr &papa, std::istream &is,
                       bool bReadSpecial = false) override;

  // Write checkpoint
  virtual void checkPoint(const std::string &dir, std::ostream &os,
                          amrex::VisMF::How how = amrex::VisMF::NFiles,
                          bool dump_old = true) override;

  virtual std::string thePlotFileType() const override {
    return {"HyperCLaw-V1.1"};
  }

  // Write state for debugging
  void printState(const MultiFab &mf);

  // Write a plotfile to specified directory.
  virtual void writePlotFile(const std::string &dir, std::ostream &os,
                             amrex::VisMF::How how) override;

  void writeHDF5PlotFile(int step, Real time);

  // Initialize data on this level from another EBR (during regrid).
  virtual void init(amrex::AmrLevel &old) override;

  // Initialize data on this level after regridding if old level did not
  // previously exist
  virtual void init() override;

  // Initialize grid data at problem start-up.
  virtual void initData() override;

  // Advance grids at this level in time.
  virtual amrex::Real advance(amrex::Real time, amrex::Real dt, int iteration,
                              int ncycle) override;

  virtual void computeInitialDt(int finest_level, int sub_cycle,
                                amrex::Vector<int> &n_cycle,
                                const amrex::Vector<amrex::IntVect> &ref_ratio,
                                amrex::Vector<amrex::Real> &dt_level,
                                amrex::Real stop_time) override;

  virtual void computeNewDt(int finest_level, int sub_cycle,
                            amrex::Vector<int> &n_cycle,
                            const amrex::Vector<amrex::IntVect> &ref_ratio,
                            amrex::Vector<amrex::Real> &dt_min,
                            amrex::Vector<amrex::Real> &dt_level,
                            amrex::Real stop_time,
                            int post_regrid_flag) override;

  virtual void post_regrid(int lbase, int new_finest) override;

  // Do work after timestep().
  virtual void post_timestep(int iteration) override;

  // After a full time step
  virtual void postCoarseTimeStep(amrex::Real time) override;

  // Do work after init().
  virtual void post_init(amrex::Real stop_time) override;

  virtual void post_restart() override;

  // Error estimation for regridding.
  virtual void errorEst(amrex::TagBoxArray &tb, int clearval, int tagval,
                        amrex::Real time, int n_error_buf = 0,
                        int ngrow = 0) override;

  // For load balance
  int WorkEstType() override { return Cost_Type; }

  static int numGrow() { return NUM_GROW; }

  // Get volume fraction of cut cell
  const amrex::MultiFab &volFrac() const { return *volfrac; }

  // Define data descriptions
  static void variableSetUp();

  // Cleanup data descriptions at end of run
  static void variableCleanUp();

# if !defined(AMREX_USE_CUDA)
protected:
#endif

  static void read_params();

  // call class Amr function to get AmrLevel at lev
  EBR &getLevel(int lev) { return dynamic_cast<EBR &>(parent->getLevel(lev)); }

  // interpolate from fine to coarse
  // and update temprature
  void avgDown();

  // build geometry information
  void buildMetrics();

  amrex::Real estTimeStep();

  // Compute initial time step.
  amrex::Real initialTimeStep();

  // Advance flow field at this level in time.
  void flow_advance(amrex::Real time, amrex::Real dt, int iteration, int ncycle);

#ifdef CHEM

  // Advance chemical reaction at this level in time.
  void chemical_advance(amrex::Real dt);

  // Advance multispecies flow field at this level in time.
  void flow_advance_multi(amrex::Real time, amrex::Real dt, int iteration, int ncycle);

  // Compute RHS with Multispecies
  void compute_dSdt_multi(const amrex::MultiFab &S, amrex::MultiFab &Spec, amrex::MultiFab &dSdt, amrex::MultiFab &dSdt_spec,
                          amrex::Real dt, amrex::EBFluxRegister *fine, amrex::EBFluxRegister *current, 
                          amrex::EBFluxRegister *fine_spec, amrex::EBFluxRegister *current_spec);

  void ebchem_compute_dSdt_box (const amrex::Box& bx,
                            Array4<Real const> const& s_arr,
                            Array4<Real const> const& spec_arr,
                            Array4<Real      > const& dsdt_arr,
                            Array4<Real      > const& dsdt_spec_arr,
                            std::array<FArrayBox*, AMREX_SPACEDIM> const& flux,
                            std::array<FArrayBox*, AMREX_SPACEDIM> const& flux_spec,
                            std::array<FArrayBox*, AMREX_SPACEDIM> const& flux_diffuse,
                            Array4<EBCellFlag const> const& flag,
                            Array4<Real       const> const& vfrac,
                            Array4<Real const> const& apx,
                            Array4<Real const> const& apy,
                            Array4<Real const> const& apz,
                            Array4<Real const> const& fcx,
                            Array4<Real const> const& fcy,
                            Array4<Real const> const& fcz,
                            Array4<Real       const> const& bcent,
                            int as_crse, int as_fine,
                            Array4<Real     > const& dm_as_fine,
                            Array4<Real     > const& dm_as_fine_spec,
                            Real dt);
#endif

  // core function
  void compute_dSdt(const amrex::MultiFab &S, amrex::MultiFab &dSdt,
                    amrex::Real dt, amrex::EBFluxRegister *fine,
                    amrex::EBFluxRegister *current);

  void eb_compute_dSdt_box (const amrex::Box& bx,
                            Array4<Real const> const& s_arr,
                            Array4<Real      > const& dsdt_arr,
                            std::array<FArrayBox*, AMREX_SPACEDIM> const& flux,
                            Array4<EBCellFlag const> const& flag,
                            Array4<Real       const> const& vfrac,
                            Array4<Real const> const& apx,
                            Array4<Real const> const& apy,
                            Array4<Real const> const& apz,
                            Array4<Real const> const& fcx,
                            Array4<Real const> const& fcy,
                            Array4<Real const> const& fcz,
                            Array4<Real       const> const& bcent,
                            int as_crse, int as_fine,
                            Array4<Real     > const& dm_as_fine,
                            Real dt);
  
  void state_redist(amrex::MultiFab& state, int ng);

  // print total momentum and energy
  // only used when verbose>=2
  void printTotal() const;

  // geometric properties
  const amrex::MultiFab *volfrac{};
  const amrex::MultiCutFab *bndrycent{};
  std::array<const amrex::MultiCutFab *, AMREX_SPACEDIM> areafrac{};
  std::array<const amrex::MultiCutFab *, AMREX_SPACEDIM> facecent{};

  amrex::EBFluxRegister flux_reg;

  #ifdef CHEM
  amrex::EBFluxRegister flux_reg_spec;
  #endif

  // Max 7-th order stencil based scheme
  static constexpr int NUM_GROW = 4;

  // conservative states
  enum StateVariable { Density = 0, Xmom, Ymom, Zmom, Eden, NUM_STATE };

#ifdef CHEM
  enum StateType {State_Type = 0, Spec_Type, Cost_Type, NUM_STATE_TYPE};
#else
  enum StateType { State_Type = 0, Cost_Type, NUM_STATE_TYPE };
#endif

  static amrex::BCRec phys_bc;

  // Parameters
  static int verbose;
  static amrex::IntVect hydro_tile_size;
  static amrex::Real cfl;
  static bool do_reflux;
  static bool do_visc;
  static bool do_gravity;
  static bool do_redistribute;
  static bool IO_HDF5;
  static int plot_int;
  static std::string plot_file; 
  static amrex::Real stop_time;
  static int max_step;
  static int refine_max_dengrad_lev;
  static amrex::Real refine_dengrad;
  static amrex::Vector<amrex::RealBox> refine_boxes;
  static amrex::RealBox* dp_refine_boxes;
  static std::string time_integration;
  static int refine_cutcells;

public:

    // Host/Device problem parameters
    static Parm* h_parm;
    static Parm* d_parm;
    static ProbParm* h_prob_parm;
    static ProbParm* d_prob_parm;
};

void ebr_bcfill (amrex::Box const& bx, amrex::FArrayBox& data,
                 int dcomp, int numcomp,
                 amrex::Geometry const& geom, amrex::Real time,
                 const amrex::Vector<amrex::BCRec>& bcr, int bcomp,
                 int scomp);

#ifdef CHEM
void spec_bcfill (amrex::Box const& bx, amrex::FArrayBox& data,
                 int dcomp, int numcomp,
                 amrex::Geometry const& geom, amrex::Real time,
                 const amrex::Vector<amrex::BCRec>& bcr, int bcomp,
                 int scomp);
#endif
#endif
