#include <Constants.H>
#include <IndexDefines.H>
#include <AMReX_GpuContainers.H>

using amrex::Real;

void Parm::Initialize ()
{
    Rg = Ru/eos_m;
    cv = Rg / (eos_gamma-Real(1.0));
    cp = eos_gamma * cv;
    kOverMu = cp/Pr;
}