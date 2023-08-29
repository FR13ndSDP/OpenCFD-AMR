#include <EBR.H>

#ifdef AMREX_USE_HDF5
#include <AMReX_PlotFileUtilHDF5.H>
#endif

using namespace amrex;

void EBR::restart(Amr &papa, std::istream &is, bool bReadSpecial)
{
    AmrLevel::restart(papa, is, bReadSpecial);

    if (do_reflux && level > 0)
    {
        flux_reg.define(grids, papa.boxArray(level - 1),
                        dmap, papa.DistributionMap(level - 1), geom, papa.Geom(level - 1), papa.refRatio(level - 1), level, NUM_STATE);

#ifdef CHEM
        flux_reg_spec.define(grids, papa.boxArray(level - 1),
                        dmap, papa.DistributionMap(level - 1), geom, papa.Geom(level - 1), papa.refRatio(level - 1), level, NSPECS);
#endif
    }

    buildMetrics();
}

void EBR::checkPoint(const std::string &dir, std::ostream &os, VisMF::How how, bool dump_old)
{
    AmrLevel::checkPoint(dir, os, how, dump_old);
}

void EBR::writePlotFile(const std::string &dir, std::ostream &os, VisMF::How how)
{
    BL_PROFILE("EBR::writePlotFile()");
    AmrLevel::writePlotFile(dir, os, how);
}

#ifdef AMREX_USE_HDF5
void EBR::writeHDF5PlotFile(int step, Real time)
{
    int nlevs = parent->finestLevel() + 1;
    Vector<int> level_steps;
    Vector<IntVect> ratio_list;
    Vector<std::string> name_list;
    Vector<MultiFab> mf_list(nlevs);
    Vector<Geometry> geom_list;

    for (int lev=0; lev<nlevs; ++lev)
    {
        EBR& this_level = getLevel(lev);
        level_steps.push_back(parent->levelSteps(lev));
        ratio_list.push_back({2,2,2});
        geom_list.push_back(this_level.geom);
        mf_list[lev] = MultiFab(this_level.boxArray(), this_level.dmap, NCONS, 0);
        MultiFab::Copy(mf_list[lev], this_level.state[State_Type].newData(), 0, 0, NCONS, 0);
    }

    name_list = {"rho", "rhoU", "rhoV", "rhoW", "E"};

    const std::string& pltfile = amrex::Concatenate(plot_file, step);
    WriteMultiLevelPlotfileHDF5(pltfile, nlevs,
                                GetVecOfConstPtrs(mf_list), name_list,
                                geom_list, time,
                                level_steps, ratio_list);
}
#endif

void EBR::printState(const MultiFab &mf)
{
    const IntVect ng(2, 2, 2);
    const IntVect cell(64, 64, 0);

    print_state(mf, cell, -1, ng);
}
