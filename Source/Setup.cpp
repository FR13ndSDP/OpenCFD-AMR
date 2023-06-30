#include <EBR.H>

#include <IndexDefines.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>

using namespace amrex;

Parm* EBR::h_parm = nullptr;
Parm* EBR::d_parm = nullptr;
ProbParm* EBR::h_prob_parm = nullptr;
ProbParm* EBR::d_prob_parm = nullptr;

static Box the_same_box (const Box& b) { return b; }
//static Box grow_box_by_one (const Box& b) { return amrex::grow(b,1); }

// TODO: Add fix temperature wall boudary
//
// Components are:
//  Interior, Inflow, Outflow,  Symmetry,     SlipWall,     NoSlipWall(adiabatic)
//
static int scalar_bc[] =
{
    BCType::int_dir, BCType::ext_dir, BCType::foextrap, BCType::reflect_even, BCType::reflect_even, BCType::reflect_even
};

static int norm_vel_bc[] =
{
    BCType::int_dir, BCType::ext_dir, BCType::foextrap, BCType::reflect_odd,  BCType::reflect_odd,  BCType::reflect_odd
};

static int tang_vel_bc[] =
{
    BCType::int_dir, BCType::ext_dir, BCType::foextrap, BCType::reflect_even, BCType::reflect_even, BCType::reflect_odd
};

static void set_scalar_bc (BCRec& bc, const BCRec& phys_bc)
{
    const int* lo_bc = phys_bc.lo();
    const int* hi_bc = phys_bc.hi();
    for (int i = 0; i < AMREX_SPACEDIM; i++)
    {
        bc.setLo(i,scalar_bc[lo_bc[i]]);
        bc.setHi(i,scalar_bc[hi_bc[i]]);
    }
}

static void set_x_vel_bc(BCRec& bc, const BCRec& phys_bc)
{
    const int* lo_bc = phys_bc.lo();
    const int* hi_bc = phys_bc.hi();
    bc.setLo(0,norm_vel_bc[lo_bc[0]]);
    bc.setHi(0,norm_vel_bc[hi_bc[0]]);

    bc.setLo(1,tang_vel_bc[lo_bc[1]]);
    bc.setHi(1,tang_vel_bc[hi_bc[1]]);

    bc.setLo(2,tang_vel_bc[lo_bc[2]]);
    bc.setHi(2,tang_vel_bc[hi_bc[2]]);
}

static
void
set_y_vel_bc(BCRec& bc, const BCRec& phys_bc)
{
    const int* lo_bc = phys_bc.lo();
    const int* hi_bc = phys_bc.hi();
    bc.setLo(0,tang_vel_bc[lo_bc[0]]);
    bc.setHi(0,tang_vel_bc[hi_bc[0]]);

    bc.setLo(1,norm_vel_bc[lo_bc[1]]);
    bc.setHi(1,norm_vel_bc[hi_bc[1]]);

    bc.setLo(2,tang_vel_bc[lo_bc[2]]);
    bc.setHi(2,tang_vel_bc[hi_bc[2]]);
}

static
void
set_z_vel_bc(BCRec& bc, const BCRec& phys_bc)
{
    const int* lo_bc = phys_bc.lo();
    const int* hi_bc = phys_bc.hi();
    bc.setLo(0,tang_vel_bc[lo_bc[0]]);
    bc.setHi(0,tang_vel_bc[hi_bc[0]]);

    bc.setLo(1,tang_vel_bc[lo_bc[1]]);
    bc.setHi(1,tang_vel_bc[hi_bc[1]]);

    bc.setLo(2,norm_vel_bc[lo_bc[2]]);
    bc.setHi(2,norm_vel_bc[hi_bc[2]]);
}

void ebr_derpres (const Box& bx, FArrayBox& pfab, int dcomp, int ncomp,
                  const FArrayBox& datfab, const Geometry& /*geomdata*/,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{
    amrex::ignore_unused(ncomp);
    auto const dat = datfab.array();
    auto       p    = pfab.array();
    Parm const* parm = EBR::d_parm;
    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        p(i,j,k,dcomp) = (parm->eos_gamma-1.)*(dat(i,j,k,UEDEN)-0.5/dat(i,j,k,URHO)* \
                         (dat(i,j,k,UMX)*dat(i,j,k,UMX) + \
                          dat(i,j,k,UMY)*dat(i,j,k,UMY) + \
                          dat(i,j,k,UMZ)*dat(i,j,k,UMZ)));
    });
}

void ebr_dervel (const Box& bx, FArrayBox& velfab, int dcomp, int ncomp,
                  const FArrayBox& datfab, const Geometry& /*geomdata*/,
                  Real /*time*/, const int* /*bcrec*/, int /*level*/)
{
    amrex::ignore_unused(ncomp);
    AMREX_ASSERT(ncomp == AMREX_SPACEDIM);
    auto const dat = datfab.array();
    auto       vel = velfab.array();
    amrex::ParallelFor(bx,
    [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        AMREX_D_TERM(vel(i,j,k,dcomp  ) = dat(i,j,k,UMX)/dat(i,j,k,URHO);,
                     vel(i,j,k,dcomp+1) = dat(i,j,k,UMY)/dat(i,j,k,URHO);,
                     vel(i,j,k,dcomp+2) = dat(i,j,k,UMZ)/dat(i,j,k,URHO);)
    });
}

void
EBR::variableSetUp ()
{
    h_parm = new Parm{};
    h_prob_parm = new ProbParm{};
    d_parm = (Parm*)The_Arena()->alloc(sizeof(Parm));
    d_prob_parm = (ProbParm*)The_Arena()->alloc(sizeof(ProbParm));

    read_params();

    bool state_data_extrap = false;
    bool store_in_checkpoint = true;
    // TODO: use cell_cons_interp here, what's the effect of others
    desc_lst.addDescriptor(State_Type,IndexType::TheCellType(),
                           StateDescriptor::Point,NUM_GROW,NUM_STATE,
                           &cell_cons_interp,state_data_extrap,store_in_checkpoint);

    Vector<BCRec>       bcs(NUM_STATE);
    Vector<std::string> name(NUM_STATE);
    BCRec bc;
    int cnt = 0;
    set_scalar_bc(bc,phys_bc); bcs[cnt] = bc; name[cnt] = "rho";
    cnt++; set_x_vel_bc(bc,phys_bc);  bcs[cnt] = bc; name[cnt] = "xmom";
    cnt++; set_y_vel_bc(bc,phys_bc);  bcs[cnt] = bc; name[cnt] = "ymom";
    cnt++; set_z_vel_bc(bc,phys_bc);  bcs[cnt] = bc; name[cnt] = "zmom";
    cnt++; set_scalar_bc(bc,phys_bc); bcs[cnt] = bc; name[cnt] = "E";

    StateDescriptor::BndryFunc bndryfunc(ebr_bcfill);
    bndryfunc.setRunOnGPU(true);

    desc_lst.setComponent(State_Type,
                          Density,
                          name,
                          bcs,
                          bndryfunc);

    // DEFINE DERIVED QUANTITIES

    // derive_lst.add("T",IndexType::TheCellType(),1,
    //                nc_dertemp,the_same_box);
    // derive_lst.addComponent("T",desc_lst,State_Type,Density,1);
    // derive_lst.addComponent("T",desc_lst,State_Type,Xmom,1);
    // derive_lst.addComponent("T",desc_lst,State_Type,Ymom,1);
    // derive_lst.addComponent("T",desc_lst,State_Type,Zmom,1);
    // derive_lst.addComponent("T",desc_lst,State_Type,Eden,1);

    derive_lst.add("pressure",IndexType::TheCellType(),1,
                   ebr_derpres,the_same_box);
    derive_lst.addComponent("pressure",desc_lst,State_Type,Density,1);
    derive_lst.addComponent("pressure",desc_lst,State_Type,Xmom,1);
    derive_lst.addComponent("pressure",desc_lst,State_Type,Ymom,1);
    derive_lst.addComponent("pressure",desc_lst,State_Type,Zmom,1);
    derive_lst.addComponent("pressure",desc_lst,State_Type,Eden,1);

    // Velocities
    // get velocity by momentum/density
    derive_lst.add("velocity",IndexType::TheCellType(),AMREX_SPACEDIM,
                   {AMREX_D_DECL("ux", "uy", "uz")}, ebr_dervel,the_same_box);
    derive_lst.addComponent("velocity",desc_lst,State_Type,Density,1+AMREX_SPACEDIM);
}

void
EBR::variableCleanUp ()
{
    delete h_parm;
    delete h_prob_parm;
    The_Arena()->free(d_parm);
    The_Arena()->free(d_prob_parm);
    desc_lst.clear();
    derive_lst.clear();
}
