#include <EBR.H>
#include <Tagging.H>
#include <AMReX_EBMultiFabUtil.H>
#include <AMReX_EBFArrayBox.H>
#include <AMReX_EBAmrUtil.H>

using namespace amrex;

void
EBR::errorEst (TagBoxArray& tags, int, int, Real time, int, int)
{
    BL_PROFILE("EBR::errorEst()");

    if (refine_cutcells) {
        const MultiFab& S_new = get_new_data(State_Type);
        amrex::TagCutCells(tags, S_new);
    }

    if (!refine_boxes.empty())
    {
        const Real* problo = geom.ProbLo();
        const Real* dx = geom.CellSize();

#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(tags, TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            auto& fab = tags[mfi];
            const Box& bx = fab.box();
            for (BoxIterator bi(bx); bi.ok(); ++bi)
            {
                const IntVect& cell = bi();
                RealVect pos {AMREX_D_DECL((cell[0]+Real(0.5))*dx[0]+problo[0],
                                           (cell[1]+Real(0.5))*dx[1]+problo[1],
                                           (cell[2]+Real(0.5))*dx[2]+problo[2])};
                for (const auto& rbx : refine_boxes) {
                    if (rbx.contains(pos)) {
                        fab(cell) = TagBox::SET;
                    }
                }
            }
        }
    }

    if (level < refine_max_dengrad_lev)
    {
        int ng = 1;
        const auto& rho = derive("rho", time, ng);
        const MultiFab& S_new = get_new_data(State_Type);

        const char   tagval = TagBox::SET;
        const char clearval = TagBox::CLEAR;
        // make a local copy to ensure captured by lambda expression
        const Real dengrad_threshold = refine_dengrad;
        auto const& fact = dynamic_cast<EBFArrayBoxFactory const&>(S_new.Factory());
        auto const& flags = fact.getMultiEBCellFlagFab();

        const auto geomdata = geom.data();
        
#ifdef AMREX_USE_OMP
#pragma omp parallel if (Gpu::notInLaunchRegion())
#endif
        for (MFIter mfi(*rho,TilingIfNotGPU()); mfi.isValid(); ++mfi)
        {
            const Box& bx = mfi.tilebox();
            const auto rhofab = (*rho)[mfi].array();
            auto tag = tags.array(mfi);
            const auto& flag = flags[mfi];

            if (flag.getType(bx) != FabType::covered)
            {
                amrex::ParallelFor(bx,
                [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
                {
                    tag_dengrad(i,j,k,tag,rhofab,geomdata,dengrad_threshold,tagval,clearval);
                });
            }
        }
    }
}