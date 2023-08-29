#include <Constants.H>

void Parm::Initialize ()
{
    Rg = Ru/eos_m;
    cv = Rg / (eos_gamma-amrex::Real(1.0));
    cp = eos_gamma * cv;
    kOverMu = cp/Pr;
}
