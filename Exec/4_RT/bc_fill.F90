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

    do n = 1,NVAR
      call amrex_filcc(adv(:,:,:,n),adv_lo(1),adv_lo(2),adv_lo(3),adv_hi(1),adv_hi(2),adv_hi(3),domlo,domhi,delta,xlo,bc(:,:,n))
    enddo

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

    call amrex_filcc(adv,adv_lo(1),adv_lo(2),adv_lo(3),adv_hi(1),adv_hi(2),adv_hi(3),domlo,domhi,delta,xlo,bc)

  end subroutine nc_denfill

end module bc_fill_module
