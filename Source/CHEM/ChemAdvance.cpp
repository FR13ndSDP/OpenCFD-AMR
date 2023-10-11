#include "EBR.H"
#include "LiDryer.H"

using namespace amrex;

// semi-implicit
void EBR::chemical_advance(Real dt)
{
    BL_PROFILE("EBR::chemical_advance");
    MultiFab& S_new = get_new_data(State_Type);
    MultiFab& Spec_new = get_new_data(Spec_Type);

    auto const& fact = dynamic_cast<EBFArrayBoxFactory const&>(S_new.Factory());
    auto const& flags = fact.getMultiEBCellFlagFab();

    for (MFIter mfi(S_new, TilingIfNotGPU()); mfi.isValid(); ++mfi)
    {
        const Box& bx = mfi.tilebox();
        const auto& flag = flags[mfi];

        if (flag.getType(bx) != FabType::covered) {
            auto const& sfab = S_new.array(mfi);
            auto const& rhoi = Spec_new.array(mfi);

            Parm const* lparm = d_parm;

            ParallelFor(bx,
            [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
            {
                Real T, ei;
                Real dei = 0;
                Real rhoi_0[NSPECS], wdot[NSPECS];

                Real c[NSPECS], Arate[NSPECS][NSPECS], A1[NSPECS][NSPECS], rhoi_1[NSPECS], drho[NSPECS];

                Real rho = amrex::max(sfab(i,j,k,URHO), lparm->smallr);
                Real rhoinv = Real(1.0)/rho;
                Real ux = sfab(i,j,k,UMX)*rhoinv;
                Real uy = sfab(i,j,k,UMY)*rhoinv;
                Real uz = sfab(i,j,k,UMZ)*rhoinv;
                Real kineng = Real(0.5)*(ux*ux+uy*uy+uz*uz);
                ei = sfab(i,j,k,UEDEN) - rho * kineng;

                for (int n=0; n<NSPECS; ++n) {
                    rhoi_0[n] = rhoi(i,j,k,n);
                }

                GET_T_GIVEN_EY(ei, rhoi_0, T, *lparm);

                for (int n = 0; n < NSPECS; n++) {
                    c[n] = rhoi_0[n] / lparm->mw[n] * 1e-6;
                }

                /*call productionRate */
                vproductionRate(wdot, Arate, c, T, *lparm);

                for (int i = 0; i < NSPECS; ++i) {
                for (int j = 0; j < NSPECS; ++j) {
                    A1[i][j] = (i == j ? 1.0 : 0.0);
                    A1[i][j] -= Arate[i][j] * lparm->mw[i] / lparm->mw[j] * dt;
                }
                }

                for (int n = 0; n < NSPECS; ++n) {
                    rhoi_1[n] = wdot[n] * lparm->mw[n] * 1e6 * dt;
                }

                gauss(drho, A1, rhoi_1);

                for (int n = 0; n < NSPECS; ++n) {
                    rhoi(i,j,k,n) += drho[n];
                    if (rhoi(i,j,k,n) < 0) {
                        rhoi(i,j,k,n) = Real(0.0);
                    }
                }

                for (int n=0; n<NSPECS; ++n) {
                    dei += -lparm->HP[n] * lparm->Ru / lparm->mw[n] * drho[n];
                }

                sfab(i,j,k,UEDEN) += dei;
            });
        }
    }
}

