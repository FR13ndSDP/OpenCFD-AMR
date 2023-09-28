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
    // int i, n;
    // //
    // // The list of indices of State to write to plotfile.
    // // first component of pair is state_type,
    // // second component of pair is component # within the state_type
    // //
//     std::vector<std::pair<int,int> > plot_var_map;
//     // only plot state_type
//     for (int comp = 0; comp < desc_lst[0].nComp();comp++)
//     {
//         if (amrex::Amr::isStatePlotVar(desc_lst[0].name(comp)) &&
//             desc_lst[0].getType() == IndexType::TheCellType())
//         {
//             plot_var_map.emplace_back(0,comp);
//         }
//     }

//     int num_derive = 0;
//     std::vector<std::string> derive_names;
//     const std::list<DeriveRec>& dlist = derive_lst.dlist();
//     for (auto const& d : dlist)
//     {
//         if (amrex::Amr::isDerivePlotVar(d.name()))
//         {
//             derive_names.push_back(d.name());
//             num_derive += d.numDerive();
//         }
//     }

//     int n_data_items = static_cast<int>(plot_var_map.size()) + num_derive;

// #ifdef AMREX_USE_EB
//     if (EB2::TopIndexSpaceIfPresent()) {
//         n_data_items += 1;
//     }
// #endif

//     // get the time from the first State_Type
//     // if the State_Type is ::Interval, this will get t^{n+1/2} instead of t^n
//     Real cur_time = state[0].curTime();

//     if (level == 0 && ParallelDescriptor::IOProcessor())
//     {
//         //
//         // The first thing we write out is the plotfile type.
//         //
//         os << thePlotFileType() << '\n';

//         if (n_data_items == 0) {
//             amrex::Error("Must specify at least one valid data item to plot");
//         }

//         os << n_data_items << '\n';

//         //
//         // Names of variables
//         //
//         for (i =0; i < static_cast<int>(plot_var_map.size()); i++)
//         {
//             int typ = plot_var_map[i].first;
//             int comp = plot_var_map[i].second;
//             os << desc_lst[typ].name(comp) << '\n';
//         }

//         // derived
//         for (auto const& dname : derive_names) {
//             const DeriveRec* rec = derive_lst.get(dname);
//             for (i = 0; i < rec->numDerive(); ++i) {
//                 os << rec->variableName(i) << '\n';
//             }
//         }

// #ifdef AMREX_USE_EB
//         if (EB2::TopIndexSpaceIfPresent()) {
//             os << "vfrac\n";
//         }
// #endif

//         os << AMREX_SPACEDIM << '\n';
//         os << parent->cumTime() << '\n';
//         int f_lev = parent->finestLevel();
//         os << f_lev << '\n';
//         for (i = 0; i < AMREX_SPACEDIM; i++) {
//             os << Geom().ProbLo(i) << ' ';
//         }
//         os << '\n';
//         for (i = 0; i < AMREX_SPACEDIM; i++) {
//             os << Geom().ProbHi(i) << ' ';
//         }
//         os << '\n';
//         for (i = 0; i < f_lev; i++) {
//             os << parent->refRatio(i)[0] << ' ';
//         }
//         os << '\n';
//         for (i = 0; i <= f_lev; i++) {
//             os << parent->Geom(i).Domain() << ' ';
//         }
//         os << '\n';
//         for (i = 0; i <= f_lev; i++) {
//             os << parent->levelSteps(i) << ' ';
//         }
//         os << '\n';
//         for (i = 0; i <= f_lev; i++)
//         {
//             for (int k = 0; k < AMREX_SPACEDIM; k++) {
//                 os << parent->Geom(i).CellSize()[k] << ' ';
//             }
//             os << '\n';
//         }
//         os << (int) Geom().Coord() << '\n';
//         os << "0\n"; // Write bndry data.

//     }
//     // Build the directory to hold the MultiFab at this level.
//     // The name is relative to the directory containing the Header file.
//     //
//     static const std::string BaseName = "/Cell";
//     char buf[64];
//     snprintf(buf, sizeof buf, "Level_%d", level);
//     std::string sLevel = buf;
//     //
//     // Now for the full pathname of that directory.
//     //
//     std::string FullPath = dir;
//     if ( ! FullPath.empty() && FullPath[FullPath.size()-1] != '/')
//     {
//         FullPath += '/';
//     }
//     FullPath += sLevel;
//     //
//     // Only the I/O processor makes the directory if it doesn't already exist.
//     //
//     if ( ! levelDirectoryCreated) {
//         if (ParallelDescriptor::IOProcessor()) {
//             if ( ! amrex::UtilCreateDirectory(FullPath, 0755)) {
//                 amrex::CreateDirectoryFailed(FullPath);
//             }
//         }
//         // Force other processors to wait until directory is built.
//         ParallelDescriptor::Barrier();
//     }

//     if (ParallelDescriptor::IOProcessor())
//     {
//         os << level << ' ' << grids.size() << ' ' << cur_time << '\n';
//         os << parent->levelSteps(level) << '\n';

//         for (i = 0; i < grids.size(); ++i)
//         {
//             RealBox gridloc = RealBox(grids[i],geom.CellSize(),geom.ProbLo());
//             for (n = 0; n < AMREX_SPACEDIM; n++) {
//                 os << gridloc.lo(n) << ' ' << gridloc.hi(n) << '\n';
// }
//         }
//         //
//         // The full relative pathname of the MultiFabs at this level.
//         // The name is relative to the Header file containing this name.
//         // It's the name that gets written into the Header.
//         //
//         if (n_data_items > 0)
//         {
//             std::string PathNameInHeader = sLevel;
//             PathNameInHeader += BaseName;
//             os << PathNameInHeader << '\n';
//         }

// #ifdef AMREX_USE_EB
//         if (EB2::TopIndexSpaceIfPresent()) {
//             // volfrac threshold for amrvis
//             if (level == parent->finestLevel()) {
//                 for (int lev = 0; lev <= parent->finestLevel(); ++lev) {
//                     os << "1.0e-6\n";
//                 }
//             }
//         }
// #endif
//     }
//     //
//     // We combine all of the multifabs -- state, derived, etc -- into one
//     // multifab -- plotMF.
//     int       cnt   = 0;
//     const int nGrow = 0;
//     MultiFab  plotMF(grids,dmap,n_data_items,nGrow,MFInfo(),Factory());
//     MultiFab* this_dat = nullptr;
//     //
//     // Cull data from state variables -- use no ghost cells.
//     //
//     for (i = 0; i < static_cast<int>(plot_var_map.size()); i++)
//     {
//         int typ  = plot_var_map[i].first;
//         int comp = plot_var_map[i].second;
//         this_dat = &state[typ].newData();
//         MultiFab::Copy(plotMF,*this_dat,comp,cnt,1,nGrow);
//         cnt++;
//     }

//     // derived
//     if (!derive_names.empty())
//     {
//         for (auto const& dname : derive_names)
//         {
//             derive(dname, cur_time, plotMF, cnt);
//             cnt += derive_lst.get(dname)->numDerive();
//         }
//     }

// #ifdef AMREX_USE_EB
//     if (EB2::TopIndexSpaceIfPresent()) {
//         plotMF.setVal(0.0, cnt, 1, nGrow);
//         auto *factory = static_cast<EBFArrayBoxFactory*>(m_factory.get());
//         MultiFab::Copy(plotMF,factory->getVolFrac(),0,cnt,1,nGrow);
//     }
// #endif

//     //
//     // Use the Full pathname when naming the MultiFab.
//     //
//     std::string TheFullPath = FullPath;
//     TheFullPath += BaseName;
//     if (AsyncOut::UseAsyncOut()) {
//         VisMF::AsyncWrite(plotMF,TheFullPath);
//     } else {
//         VisMF::Write(plotMF,TheFullPath,how,true);
//     }

//     levelDirectoryCreated = false;  // ---- now that the plotfile is finished
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
