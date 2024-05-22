#ifndef TAGGING_H
#define TAGGING_H

#include <AMReX_FArrayBox.H>
#include <AMReX_TagBox.H>
#include <cmath>

AMREX_GPU_HOST_DEVICE
AMREX_FORCE_INLINE
void
tag_dengrad (int i, int j, int k,
                  amrex::Array4<char> const& tag,
                  amrex::Array4<amrex::Real const> const& rho,
                  amrex::GeometryData const& /*geomdata*/,
                  amrex::Real dengrad_threshold, char tagval, char clearval) noexcept
{
    using amrex::Real;

    // const Real* prob_lo = geomdata.ProbLo();
    // const Real* dx      = geomdata.CellSize();

    // Real x = prob_lo[0] + (i+Real(0.5))*dx[0];
    // Real y = prob_lo[1] + (j+Real(0.5))*dx[1];

    Real ax = amrex::Math::abs(rho(i+1,j,k) - rho(i,j,k));
    Real ay = amrex::Math::abs(rho(i,j+1,k) - rho(i,j,k));
    Real az = amrex::Math::abs(rho(i,j,k+1) - rho(i,j,k));
    ax = amrex::max(ax,amrex::Math::abs(rho(i,j,k) - rho(i-1,j,k)));
    ay = amrex::max(ay,amrex::Math::abs(rho(i,j,k) - rho(i,j-1,k)));
    az = amrex::max(az,amrex::Math::abs(rho(i,j,k) - rho(i,j,k-1)));
    if ((amrex::max(ax,ay,az) >= dengrad_threshold)) {
        tag(i,j,k) = tagval;
    } else {
        tag(i,j,k) = clearval;
    }
}

#endif
