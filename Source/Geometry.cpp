#include <AMReX_EB2.H>
#include <AMReX_EB2_IF.H>

#include <AMReX_ParmParse.H>

using namespace amrex;

void
initialize_Geom (const Geometry& geom, const int required_coarsening_level,
                const int max_coarsening_level)
{
    BL_PROFILE("initializeGeom");

    ParmParse ppeb2("eb2");
    std::string geom_type;
    ppeb2.get("geom_type", geom_type);

    if (geom_type == "none")
    {
        EB2::AllRegularIF allreg;
        auto gshop = EB2::makeShop(allreg);
        EB2::Build(gshop, geom, max_coarsening_level, max_coarsening_level, 4);
    }
    else
    {
        // Disable build coarse level by coarsening, which cause problem for complex geometry AMR
        EB2::Build(geom, max_coarsening_level, max_coarsening_level, 4, false);
    }
}
