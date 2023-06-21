#include <AMReX_LevelBld.H>
#include <EBR.H>

using namespace amrex;

class EBRBld : public LevelBld {
  virtual void variableSetUp() override;
  virtual void variableCleanUp() override;
  virtual AmrLevel *operator()() override;
  virtual AmrLevel *operator()(Amr &papa, int lev, const Geometry &level_geom,
                               const BoxArray &ba,
                               const DistributionMapping &dm,
                               Real time) override;
};

EBRBld ebr_bld;

LevelBld *getLevelBld() { return &ebr_bld; }

void EBRBld::variableSetUp() { EBR::variableSetUp(); }

void EBRBld::variableCleanUp() { EBR::variableCleanUp(); }

AmrLevel *EBRBld::operator()() { return new EBR; }

AmrLevel *EBRBld::operator()(Amr &papa, int lev, const Geometry &level_geom,
                             const BoxArray &ba, const DistributionMapping &dm,
                             Real time) {
  return new EBR(papa, lev, level_geom, ba, dm, time);
}
