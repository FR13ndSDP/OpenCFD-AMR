module bc_fill_module

  implicit none

  public

contains

  ! All subroutines in this file must be threadsafe because they are called
  ! inside OpenMP parallel regions.

  ! fill boundary for all components
  ! xlo is the location of corner node in ghost grid (local grid)
  subroutine nc_hypfill(adv,adv_lo,adv_hi,domlo,domhi,delta,xlo,time,bc) &
    bind(C, name="nc_hypfill")

    use nc_module, only: NVAR, gamma, cv
    use amrex_fort_module, only: dim=>amrex_spacedim, rt=>amrex_real
    use amrex_filcc_module
    use amrex_bc_types_module, only : amrex_bc_reflect_even, amrex_bc_ext_dir
    use probdata_module

    implicit none

    integer          :: adv_lo(3),adv_hi(3)
    integer          :: bc(dim,2,*)
    integer          :: domlo(3), domhi(3)
    double precision :: delta(3), xlo(3), time
    double precision :: adv(adv_lo(1):adv_hi(1),adv_lo(2):adv_hi(2),adv_lo(3):adv_hi(3),NVAR)

    integer          :: i,j,k,n
    real(rt)         :: x, y, loc, x_start, x0, pi = 3.1415926

    loc = 1.d0/6.d0+1.d0/sqrt(3.0)+10.d0/sin(pi/3.d0)*time
    x_start = -5.d0*delta(1)
    x0 = 1.d0/6.d0

    do n = 1,NVAR
      call amrex_filcc(adv(:,:,:,n),adv_lo(1),adv_lo(2),adv_lo(3),adv_hi(1),adv_hi(2),adv_hi(3),domlo,domhi,delta,xlo,bc(:,:,n))
    enddo

    ! for DMR
    !bc (direction, lo_or_hi, var) index starts from 1
    if ( bc(2,1,1).eq.amrex_bc_reflect_even) then
      do       k = adv_lo(3), adv_hi(3)
        do    j = adv_lo(2), domlo(2)-1
          do i = adv_lo(1), adv_hi(1)
            x = x_start + delta(1) * (i+5) + 0.5*delta(1)

            if (x<x0) then
              adv(i,j,k,1) = rho_l
              adv(i,j,k,2) = rho_l * u_l
              adv(i,j,k,3) = rho_l * v_l
              adv(i,j,k,4) = 0.d0
              adv(i,j,k,5) = p_l/(gamma-1.d0) + 0.5d0*(u_l*u_l + v_l*v_l)*rho_l
            end if
          end do
        end do
      end do
    end if

    if (bc(2,2,1).eq.amrex_bc_ext_dir) then
      do     k = adv_lo(3), adv_hi(3)
        do   j = domhi(2)+1, adv_hi(2)
          do i = adv_lo(1), adv_hi(1)
            x = x_start + delta(1) * (i+5) + 0.5*delta(1)
            if (x < loc) then
              adv(i,j,k,1) = rho_l
              adv(i,j,k,2) = rho_l * u_l
              adv(i,j,k,3) = rho_l * v_l
              adv(i,j,k,4) = 0.d0
              adv(i,j,k,5) = p_l/(gamma-1.d0) + 0.5d0*(u_l*u_l + v_l*v_l)*rho_l
            else
              adv(i,j,k,1) = rho_r
              adv(i,j,k,2) = rho_r * u_r
              adv(i,j,k,3) = rho_r * v_r
              adv(i,j,k,4) = 0.d0
              adv(i,j,k,5) = p_r/(gamma-1.d0) + 0.5d0*(u_r*u_r + v_r*v_r)*rho_r
            end if
          end do
        end do
      end do
    end if

  end subroutine nc_hypfill

  ! fill boundary for density, the tag variable
  ! called for filling boundary of nc_tagging
  subroutine nc_denfill(adv,adv_lo,adv_hi,domlo,domhi,delta,xlo,time,bc) &
    bind(C, name="nc_denfill")

    use amrex_fort_module, only: dim=>amrex_spacedim
    use amrex_filcc_module
    use amrex_bc_types_module, only : amrex_bc_ext_dir, amrex_bc_reflect_even
    use probdata_module
    implicit none

    include 'AMReX_bc_types.fi'

    integer          :: adv_lo(3),adv_hi(3)
    integer          :: bc(dim,2)
    integer          :: domlo(3), domhi(3)
    double precision :: delta(3), xlo(3), time
    double precision :: adv(adv_lo(1):adv_hi(1),adv_lo(2):adv_hi(2),adv_lo(3):adv_hi(3))

    integer :: i,j,k
    real(rt):: x, y, loc, x_start, x0, pi=3.1415926

    loc = 1.d0/6.d0+1.d0/sqrt(3.0)+10.d0/sin(pi/3.d0)*time
    x_start = -1.d0*delta(1)
    x0 = 1.d0/6.d0

    call amrex_filcc(adv,adv_lo(1),adv_lo(2),adv_lo(3),adv_hi(1),adv_hi(2),adv_hi(3),domlo,domhi,delta,xlo,bc)

    ! for DMR
    if ( bc(2,1).eq.amrex_bc_reflect_even) then
      do       k = adv_lo(3), adv_hi(3)
        do    j = adv_lo(2), domlo(2)-1
          do i = adv_lo(1), adv_hi(1)
            x = x_start + delta(1) * (i+1) + 0.5*delta(1)

            if (x<x0) then
              adv(i,j,k) = rho_l
            end if
          end do
        end do
      end do
    end if

    if (bc(2,2).eq.amrex_bc_ext_dir) then
      do     k = adv_lo(3), adv_hi(3)
        do   j = domhi(2)+1, adv_hi(2)
          do i = adv_lo(1), adv_hi(1)
            x = x_start + delta(1) * (i+1) + 0.5*delta(1)

            if (x < loc) then
              adv(i,j,k) = rho_l
            else 
              adv(i,j,k) = rho_r
            end if
          end do
        end do
      end do
    end if

  end subroutine nc_denfill

end module bc_fill_module
