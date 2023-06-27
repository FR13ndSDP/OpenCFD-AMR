#include <EBR_parm.H>

void Parm::Initialize ()
{
    constexpr amrex::Real Ru = amrex::Real(8.31451);
    Rg = Ru/eos_m;
    cv = Rg / (eos_gamma-amrex::Real(1.0));
    cp = eos_gamma * cv;
    kOverMu = cp/Pr;
}
