#include <EBR.H>

using namespace amrex;

void EBR::restart(Amr &papa, std::istream &is, bool bReadSpecial)
{
    AmrLevel::restart(papa, is, bReadSpecial);

    if (do_reflux && level > 0)
    {
        flux_reg.define(grids, papa.boxArray(level - 1),
                        dmap, papa.DistributionMap(level - 1), geom, papa.Geom(level - 1), papa.refRatio(level - 1), level, NUM_STATE);
    }

    buildMetrics();
}

void EBR::checkPoint(const std::string &dir, std::ostream &os, VisMF::How how, bool dump_old)
{
    AmrLevel::checkPoint(dir, os, how, dump_old);
}

// TODO: Add proper HDF5 output
void EBR::writePlotFile(const std::string &dir, std::ostream &os, VisMF::How how)
{
    BL_PROFILE("EBR::writePlotFile()");
    AmrLevel::writePlotFile(dir, os, how);
}

// void EBR::writeHDF5PlotFile (const std::string &plotfilename,
//                                   int nlevels,
//                                   const Vector<const MultiFab*> &mf,
//                                   const Vector<std::string> &varnames,
//                                   const Vector<Geometry> &geom,
//                                   Real time,
//                                   const Vector<int> &level_steps,
//                                   const Vector<IntVect> &ref_ratio,
//                                   const std::string &compression);
// {
//     WriteMultiLevelPlotfileHDF5();
// }

void EBR::printState(const MultiFab &mf)
{
    const IntVect ng(2, 2, 2);
    const IntVect cell(64, 64, 0);

    print_state(mf, cell, -1, ng);
}
