module probdata_module
  use amrex_fort_module, only : rt => amrex_real
  implicit none
  real(rt), save :: p_l   = 101325.d0
  real(rt), save :: rho_l = 0.3069989396d0
  real(rt), save :: u_l   = 20.d0
  real(rt), save :: v_l   = 0.d0
  real(rt), save :: u_driven = 904.d0
  real(rt), save :: p_driven = 101325.d0
  real(rt), save :: rho_driven = 1.1575369852d0
end module probdata_module


subroutine amrex_probinit (init,name,namlen,problo,probhi) bind(c)
  use amrex_fort_module, only : rt => amrex_real
  use amrex_parmparse_module
  use probdata_module
  implicit none
  integer, intent(in) :: init, namlen
  integer, intent(in) :: name(namlen)
  real(rt), intent(in) :: problo(*), probhi(*)
  type(amrex_parmparse) :: pp
  call amrex_parmparse_build(pp,"prob")
  call pp%query("p_l",p_l)
  call pp%query("rho_l",rho_l)
  call pp%query("u_l",u_l)
  call amrex_parmparse_destroy(pp)
end subroutine amrex_probinit


subroutine initdata_f(level, time, lo, hi, u, ulo, uhi, dx, prob_lo, prob_hi) bind(C, name="initdata_f")
  use amrex_fort_module, only : rt => amrex_real
  use nc_module, only : nvar, urho, umx, umy, umz, ueden, gamma, cv
  use probdata_module
  implicit none
  integer, intent(in) :: level, lo(3), hi(3), ulo(3), uhi(3)
  real(rt), intent(in) :: time
  real(rt), intent(inout) :: u(ulo(1):uhi(1), ulo(2):uhi(2), ulo(3):uhi(3),nvar)
  real(rt), intent(in) :: dx(3), prob_lo(3), prob_hi(3)

  integer :: i,j,k

  do k = lo(3), hi(3)
    do j = lo(2), hi(2)
      do i = lo(1), hi(1)

        u(i,j,k,urho) = rho_l
        u(i,j,k,umx) = rho_l * u_l
        u(i,j,k,umy:umz) = 0.d0
        u(i,j,k,ueden) = P_l/(gamma - 1.d0) + 0.5d0*rho_l*(u_l*u_l)
      end do
    end do
  end do

end subroutine initdata_f
