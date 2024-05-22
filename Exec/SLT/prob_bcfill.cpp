#include <EBR.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Geometry.H>
#include <AMReX_PhysBCFunct.H>

using namespace amrex;

struct FillExtDir
{
    Real* inflow_state = nullptr;
    
    AMREX_GPU_DEVICE
    void operator() (const IntVect& iv, Array4<Real> const& dest,
                     const int dcomp, const int numcomp,
                     GeometryData const& geom, const Real /*time*/,
                     const BCRec* bcr, const int bcomp,
                     const int /*orig_comp*/) const
    {
        const Box& domain_box = geom.Domain();

        const BCRec& bc = bcr[bcomp+0];

        int i = iv[0];
        int j = iv[1];
        int k = iv[2];

        // farfield
        // x+
        if (bc.hi(0) == BCType::ext_dir and i > domain_box.bigEnd(0))
        {
            int nx = domain_box.bigEnd(0);
            Real rho_d = dest(2*nx-i, j, k, dcomp+0);
            Real mx_d = dest(2*nx-i, j, k, dcomp+1);
            Real my_d = dest(2*nx-i, j, k, dcomp+2);
            Real mz_d = dest(2*nx-i, j, k, dcomp+3);
            Real Eden_d = dest(2*nx-i, j, k, dcomp+4);
            Real u_d = mx_d/rho_d;
            Real v_d = my_d/rho_d;
            Real w_d = mz_d/rho_d;
            Real p_d = 0.4 * (Eden_d - 0.5*rho_d*(u_d*u_d+v_d*v_d+w_d*w_d));
            Real c_d = sqrt(1.4 * p_d/rho_d);
        
            Real p_b = inflow_state[5];
            Real rho_b = rho_d + (p_b - p_d)/(c_d * c_d);
            Real u_b = u_d + (p_d-p_b)/(rho_d * c_d);
            Real v_b = v_d;
            Real w_b = w_d;

            rho_d = 2 * rho_b - rho_d;
            u_d = 2 * u_b - u_d;
            v_d = 2 * v_b - v_d;
            w_d = 2 * w_b - w_d;
            p_d = 2 * p_b - p_d;
            
            dest(i,j,k,dcomp + 0) = rho_d;
            dest(i,j,k,dcomp + 1) = rho_d * u_d;
            dest(i,j,k,dcomp + 2) = rho_d * v_d;
            dest(i,j,k,dcomp + 3) = rho_d * w_d;
            dest(i,j,k,dcomp + 4) = p_d/0.4 + 0.5*rho_d*(u_d*u_d+v_d*v_d+w_d*w_d);
        }

        // y+
        if (bc.hi(1) == BCType::ext_dir and j > domain_box.bigEnd(1))
        {
            int ny = domain_box.bigEnd(1);
            Real rho_d = dest(i, 2*ny-j, k, dcomp+0);
            Real mx_d = dest(i, 2*ny-j, k, dcomp+1);
            Real my_d = dest(i, 2*ny-j, k, dcomp+2);
            Real mz_d = dest(i, 2*ny-j, k, dcomp+3);
            Real Eden_d = dest(i, 2*ny-j, k, dcomp+4);
            Real u_d = mx_d/rho_d;
            Real v_d = my_d/rho_d;
            Real w_d = mz_d/rho_d;
            Real p_d = 0.4 * (Eden_d - 0.5*rho_d*(u_d*u_d+v_d*v_d+w_d*w_d));
            Real c_d = std::sqrt(1.4 * p_d/rho_d);

            Real p_b = 0.5*(inflow_state[4] + p_d - rho_d*c_d*(inflow_state[2]-v_d));
            Real rho_b = inflow_state[0] + (p_b - inflow_state[4])/(c_d*c_d);
            Real u_b = inflow_state[1];
            Real v_b = inflow_state[2] - (inflow_state[4]-p_b)/(rho_d * c_d);
            Real w_b = inflow_state[3];

            rho_d = 2 * rho_b - rho_d;
            u_d = 2 * u_b - u_d;
            v_d = 2 * v_b - v_d;
            w_d = 2 * w_b - w_d;
            p_d = 2 * p_b - p_d;
            
            dest(i,j,k,dcomp + 0) = rho_d;
            dest(i,j,k,dcomp + 1) = rho_d * u_d;
            dest(i,j,k,dcomp + 2) = rho_d * v_d;
            dest(i,j,k,dcomp + 3) = rho_d * w_d;
            dest(i,j,k,dcomp + 4) = p_d/0.4 + 0.5*rho_d*(u_d*u_d+v_d*v_d+w_d*w_d);
        }

        // x-
        if (bc.lo(0) == BCType::ext_dir and i < domain_box.smallEnd(0))
        {
            Real rho_d = dest(-i, j, k, dcomp+0);
            Real mx_d = dest(-i, j, k, dcomp+1);
            Real my_d = dest(-i, j, k, dcomp+2);
            Real mz_d = dest(-i, j, k, dcomp+3);
            Real Eden_d = dest(-i, j, k, dcomp+4);
            Real u_d = mx_d/rho_d;
            Real v_d = my_d/rho_d;
            Real w_d = mz_d/rho_d;
            Real p_d = 0.4 * (Eden_d - 0.5*rho_d*(u_d*u_d+v_d*v_d+w_d*w_d));
            Real c_d = std::sqrt(1.4 * p_d/rho_d);

            Real p_b = 0.5*(inflow_state[4] + p_d - rho_d*c_d*(u_d - inflow_state[1]));
            Real rho_b = inflow_state[0] + (p_b - inflow_state[4])/(c_d*c_d);
            Real u_b = inflow_state[1]+ (inflow_state[4]-p_b)/(rho_d * c_d);
            Real v_b = inflow_state[2];
            Real w_b = inflow_state[3];

            rho_d = 2 * rho_b - rho_d;
            u_d = 2 * u_b - u_d;
            v_d = 2 * v_b - v_d;
            w_d = 2 * w_b - w_d;
            p_d = 2 * p_b - p_d;
            
            dest(i,j,k,dcomp + 0) = rho_d;
            dest(i,j,k,dcomp + 1) = rho_d * u_d;
            dest(i,j,k,dcomp + 2) = rho_d * v_d;
            dest(i,j,k,dcomp + 3) = rho_d * w_d;
            dest(i,j,k,dcomp + 4) = p_d/0.4 + 0.5*rho_d*(u_d*u_d+v_d*v_d+w_d*w_d);
        }

        // y-
        if (bc.lo(1) == BCType::ext_dir and j < domain_box.smallEnd(1))
        {
            Real rho_d = dest(i, -j, k, dcomp+0);
            Real mx_d = dest(i, -j, k, dcomp+1);
            Real my_d = dest(i, -j, k, dcomp+2);
            Real mz_d = dest(i, -j, k, dcomp+3);
            Real Eden_d = dest(i, -j, k, dcomp+4);
            Real u_d = mx_d/rho_d;
            Real v_d = my_d/rho_d;
            Real w_d = mz_d/rho_d;
            Real p_d = 0.4 * (Eden_d - 0.5*rho_d*(u_d*u_d+v_d*v_d+w_d*w_d));
            Real c_d = std::sqrt(1.4 * p_d/rho_d);

            Real p_b = 0.5*(inflow_state[4] + p_d - rho_d*c_d*(v_d - inflow_state[2]));
            Real rho_b = inflow_state[0] + (p_b - inflow_state[4])/(c_d*c_d);
            Real u_b = inflow_state[1];
            Real v_b = inflow_state[2] + (inflow_state[4]-p_b)/(rho_d * c_d);
            Real w_b = inflow_state[3];

            rho_d = 2 * rho_b - rho_d;
            u_d = 2 * u_b - u_d;
            v_d = 2 * v_b - v_d;
            w_d = 2 * w_b - w_d;
            p_d = 2 * p_b - p_d;
            
            dest(i,j,k,dcomp + 0) = rho_d;
            dest(i,j,k,dcomp + 1) = rho_d * u_d;
            dest(i,j,k,dcomp + 2) = rho_d * v_d;
            dest(i,j,k,dcomp + 3) = rho_d * w_d;
            dest(i,j,k,dcomp + 4) = p_d/0.4 + 0.5*rho_d*(u_d*u_d+v_d*v_d+w_d*w_d);
        }
    }
};

// bx                  : Cells outside physical domain and inside bx are filled.
// data, dcomp, numcomp: Fill numcomp components of data starting from dcomp.
// bcr, bcomp          : bcr[bcomp] specifies BC for component dcomp and so on.
// scomp               : component index for dcomp as in the descriptor set up in EBR::variableSetUp.

void ebr_bcfill (Box const& bx, FArrayBox& data,
                 const int dcomp, const int numcomp,
                 Geometry const& geom, const Real time,
                 const Vector<BCRec>& bcr, const int bcomp,
                 const int scomp)
{
    GpuBndryFuncFab<FillExtDir> gpu_bndry_func(FillExtDir{EBR::h_prob_parm->inflow_state});
    gpu_bndry_func(bx,data,dcomp,numcomp,geom,time,bcr,bcomp,scomp);
}
