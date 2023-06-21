module nc_ibm_module
  use amrex_fort_module, only : rt=>amrex_real
  use nc_module, only : gamma, cv, smallp, smallr, nvar, qvar, qrho, qu, &
    qv, qw, qp, urho, ueden, umx, umy, umz, nghost_plm, nextra_eb
  use amrex_mempool_module, only : amrex_allocate, amrex_deallocate
  use amrex_error_module, only : amrex_abort
  use amrex_ebcellflag_module, only : is_regular_cell, is_covered_cell, is_single_valued_cell, &
  get_neighbor_cells

  implicit none
  private

  public :: eb_compute_dudt

contains

  subroutine eb_compute_dudt (lo,hi, dudt, utlo, uthi, &
    u,ulo,uhi,fx,fxlo,fxhi,fy,fylo,fyhi,fz,fzlo,fzhi,flag,fglo,fghi, &
    volfrac,vlo,vhi, bcent,blo,bhi, &
    apx,axlo,axhi,apy,aylo,ayhi,apz,azlo,azhi, &
    centx,cxlo,cxhi,centy,cylo,cyhi,centz,czlo,czhi, &
    as_crse_in, rr_drho_crse, rdclo, rdchi, rr_flag_crse, rfclo, rfchi, &
    as_fine_in, dm_as_fine, dflo, dfhi, &
    levmsk, lmlo, lmhi, &
    dx,dt,level) &
    bind(c,name='eb_compute_dudt')
    use nc_dudt_module, only : c2prim
    integer, dimension(3), intent(in) :: lo,hi,utlo,uthi,ulo,uhi, &
      vlo,vhi,axlo,axhi,aylo,ayhi,azlo,azhi, &
      cxlo,cxhi,cylo,cyhi,czlo,czhi, &
      fglo,fghi, blo, bhi, fxlo,fxhi, fylo,fyhi, fzlo,fzhi, &
      rdclo, rdchi, rfclo, rfchi, dflo, dfhi, lmlo, lmhi
    integer, intent(in) :: as_crse_in, as_fine_in
    real(rt), intent(inout) :: dudt(utlo(1):uthi(1),utlo(2):uthi(2),utlo(3):uthi(3),nvar)
    real(rt), intent(in   ) :: u ( ulo(1): uhi(1), ulo(2): uhi(2), ulo(3): uhi(3),nvar)
    real(rt), intent(inout) :: fx(fxlo(1):fxhi(1),fxlo(2):fxhi(2),fxlo(3):fxhi(3),nvar)
    real(rt), intent(inout) :: fy(fylo(1):fyhi(1),fylo(2):fyhi(2),fylo(3):fyhi(3),nvar)
    real(rt), intent(inout) :: fz(fzlo(1):fzhi(1),fzlo(2):fzhi(2),fzlo(3):fzhi(3),nvar)
    integer , intent(in) ::  flag(fglo(1):fghi(1),fglo(2):fghi(2),fglo(3):fghi(3))
    real(rt), intent(in) :: volfrac(vlo(1):vhi(1),vlo(2):vhi(2),vlo(3):vhi(3))
    real(rt), intent(in) :: bcent  (blo(1):bhi(1),blo(2):bhi(2),blo(3):bhi(3),3)
    real(rt), intent(in) :: apx(axlo(1):axhi(1),axlo(2):axhi(2),axlo(3):axhi(3))
    real(rt), intent(in) :: apy(aylo(1):ayhi(1),aylo(2):ayhi(2),aylo(3):ayhi(3))
    real(rt), intent(in) :: apz(azlo(1):azhi(1),azlo(2):azhi(2),azlo(3):azhi(3))
    real(rt), intent(in) :: centx(cxlo(1):cxhi(1),cxlo(2):cxhi(2),cxlo(3):cxhi(3),2)
    real(rt), intent(in) :: centy(cylo(1):cyhi(1),cylo(2):cyhi(2),cylo(3):cyhi(3),2)
    real(rt), intent(in) :: centz(czlo(1):czhi(1),czlo(2):czhi(2),czlo(3):czhi(3),2)
    real(rt), intent(inout) :: rr_drho_crse(rdclo(1):rdchi(1),rdclo(2):rdchi(2),rdclo(3):rdchi(3),nvar)
    integer,  intent(in) ::  rr_flag_crse(rfclo(1):rfchi(1),rfclo(2):rfchi(2),rfclo(3):rfchi(3))
    real(rt), intent(out) :: dm_as_fine(dflo(1):dfhi(1),dflo(2):dfhi(2),dflo(3):dfhi(3),nvar)
    integer,  intent(in) ::  levmsk (lmlo(1):lmhi(1),lmlo(2):lmhi(2),lmlo(3):lmhi(3))
    real(rt), intent(in) :: dx(3), dt
    integer, intent(in) :: level

    integer :: qlo(3), qhi(3), dvlo(3), dvhi(3), dmlo(3), dmhi(3)
    integer :: lfxlo(3), lfylo(3), lfzlo(3), lfxhi(3), lfyhi(3), lfzhi(3)
    integer :: clo(3), chi(3)
    real(rt), pointer, contiguous :: q(:,:,:,:)
    real(rt), dimension(:,:,:), pointer, contiguous :: divc, dm, optmp, rediswgt
    real(rt), dimension(:,:,:,:), pointer, contiguous :: fhx,fhy,fhz
    real(rt), dimension(:,:,:), pointer, contiguous :: lambda, mu, xi
    integer, parameter :: nghost = nextra_eb + 3 ! 3 because of wall flux

    integer :: k,n, i, j
    logical :: as_crse, as_fine

    as_crse = as_crse_in .ne. 0
    as_fine = as_fine_in .ne. 0

    qlo = lo - nghost
    qhi = hi + nghost
    call amrex_allocate(q, qlo(1),qhi(1), qlo(2),qhi(2), qlo(3),qhi(3), 1,qvar)

    dvlo = lo-2
    dvhi = hi+2
    call amrex_allocate(divc, dvlo, dvhi)
    call amrex_allocate(optmp, dvlo, dvhi)
    call amrex_allocate(rediswgt, dvlo, dvhi)

    dmlo(1:3) = lo - 1
    dmhi(1:3) = hi + 1
    call amrex_allocate(dm, dmlo, dmhi)

    lfxlo = lo - nextra_eb - 1;  lfxlo(1) = lo(1)-nextra_eb
    lfxhi = hi + nextra_eb + 1
    call amrex_allocate(fhx, lfxlo(1),lfxhi(1),lfxlo(2),lfxhi(2),lfxlo(3),lfxhi(3),1,5)

    lfylo = lo - nextra_eb - 1;  lfylo(2) = lo(2)-nextra_eb
    lfyhi = hi + nextra_eb + 1
    call amrex_allocate(fhy, lfylo(1),lfyhi(1),lfylo(2),lfyhi(2),lfylo(3),lfyhi(3),1,5)

    lfzlo = lo - nextra_eb - 1;  lfzlo(3) = lo(3)-nextra_eb
    lfzhi = hi + nextra_eb + 1
    call amrex_allocate(fhz, lfzlo(1),lfzhi(1),lfzlo(2),lfzhi(2),lfzlo(3),lfzhi(3),1,5)

    ! NUM_GROW is 5
    call c2prim(qlo, qhi, u, ulo, uhi, q, qlo, qhi)

    ! for the eb cell , should be verified
    ! TODO
    ! q: -5 20
    ! lo: 0
    ! hi: 15
    ! lfxlo: -2 -3 -3
    ! lfxhi: 18
    ! fg: -5 20
    ! dudt: with no ghost
    ! u : with 5 ghost cells
    ! fxlo: 0
    ! fxhi: 16 15 15
    call strange_flux(q, qlo, qhi, lo, hi, dx, &
      fhx, lfxlo, lfxhi, fhy, lfylo, lfyhi, fhz, lfzlo, lfzhi,&
      flag, fglo, fghi)

    ! no viscous for now

    fx(:,:,:,:) = &
      fhx(fxlo(1):fxhi(1),fxlo(2):fxhi(2),fxlo(3):fxhi(3),:)
    fy(      fylo(1):fyhi(1),fylo(2):fyhi(2),fylo(3):fyhi(3),:) = &
      fhy(fylo(1):fyhi(1),fylo(2):fyhi(2),fylo(3):fyhi(3),:)
    fz(      fzlo(1):fzhi(1),fzlo(2):fzhi(2),fzlo(3):fzhi(3),:) = &
      fhz(fzlo(1):fzhi(1),fzlo(2):fzhi(2),fzlo(3):fzhi(3),:)

    dm_as_fine = 0.d0

    ! fx fy fz updated here to correctly reflux
    ! only for cut cells
    call compute_eb_divop(lo,hi,5,dx,dt,fhx,lfxlo,lfxhi,fhy,lfylo,lfyhi,fhz,lfzlo,lfzhi,&
      fx, fxlo, fxhi, fy, fylo, fyhi, fz, fzlo, fzhi, &
      dudt,utlo,uthi, q,qlo,qhi, &
      divc, optmp, rediswgt, dvlo,dvhi, &
      dm,dmlo,dmhi, &
      volfrac,vlo,vhi, &
      apx,axlo,axhi,apy,aylo,ayhi,apz,azlo,azhi, &
      centx(:,:,:,1),cxlo,cxhi, centx(:,:,:,2),cxlo,cxhi, &
      centy(:,:,:,1),cylo,cyhi, centy(:,:,:,2),cylo,cyhi, &
      centz(:,:,:,1),czlo,czhi, centz(:,:,:,2),czlo,czhi, &
      flag,fglo,fghi, &
      as_crse, rr_drho_crse, rdclo, rdchi, rr_flag_crse, rfclo, rfchi, &
      as_fine, dm_as_fine, dflo, dfhi, &
      levmsk, lmlo, lmhi)

    
    ! call write_slice('fluxx', fxlo, fxhi, fx(:,:,1,1))

    call amrex_deallocate(fhy)
    call amrex_deallocate(fhx)
    call amrex_deallocate(fhz)
    call amrex_deallocate(dm)
    call amrex_deallocate(rediswgt)
    call amrex_deallocate(optmp)
    call amrex_deallocate(divc)
    call amrex_deallocate(q)
  end subroutine eb_compute_dudt


  !TODO: use information of flag to calculate normal direction flux
  ! flag and q: with 5 ghost cells
  ! lo and hi : 0 15
  ! flux: with 3 ghost but in main direction with 2, 3
  subroutine strange_flux(q, qd_lo, qd_hi, &
    lo, hi, dx, &
    flux1, fd1_lo, fd1_hi, &
    flux2, fd2_lo, fd2_hi, &
    flux3, fd3_lo, fd3_hi, &
    flag, fg_lo, fg_hi)
    use fluxsplit_module, only : flux_split

    integer, intent(in) :: qd_lo(3), qd_hi(3)
    integer, intent(in) :: lo(3), hi(3)
    integer, intent(in) :: fd1_lo(3), fd1_hi(3)
    integer, intent(in) :: fd2_lo(3), fd2_hi(3)
    integer, intent(in) :: fd3_lo(3), fd3_hi(3)
    integer, intent(in) :: fg_lo(3), fg_hi(3)
    real(rt), intent(in) :: dx(3)
    real(rt), intent(in   ) ::     q( qd_lo(1): qd_hi(1), qd_lo(2): qd_hi(2), qd_lo(3): qd_hi(3),QVAR)
    real(rt), intent(inout) :: flux1(fd1_lo(1):fd1_hi(1),fd1_lo(2):fd1_hi(2),fd1_lo(3):fd1_hi(3),5)
    real(rt), intent(inout) :: flux2(fd2_lo(1):fd2_hi(1),fd2_lo(2):fd2_hi(2),fd2_lo(3):fd2_hi(3),5)
    real(rt), intent(inout) :: flux3(fd3_lo(1):fd3_hi(1),fd3_lo(2):fd3_hi(2),fd3_lo(3):fd3_hi(3),5)
    integer,  intent(in   ) ::  flag( fg_lo(1): fg_hi(1), fg_lo(2): fg_hi(2), fg_lo(3): fg_hi(3))

    real(rt), dimension(5) :: qL, qR, flux_loc
    real(rt) :: tmp
    integer :: i,j,k,n
    integer :: nbr(-1:1,-1:1,-1:1)

    ! X-direction
    do k = fd1_lo(3) ,fd1_hi(3)
      do j = fd1_lo(2), fd1_hi(2)
        do i = fd1_lo(1), fd1_hi(1)
          do n = 1,QVAR
            qL(n) = q(i-1,j,k, n)
            qR(n) = q(i,j,k, n)
          end do
          call flux_split(qL, qR, flux_loc)

          flux1(i,j,k,:) = flux_loc

        end do
      end do
    end do

    ! Y-direction
    do k = fd2_lo(3), fd2_hi(3)
      do j = fd2_lo(2), fd2_hi(2)
        do i = fd2_lo(1), fd2_hi(1)
          do n = 1,QVAR
            qL(n) = q(i,j-1,k, n)
            qR(n) = q(i,j,k, n)
          end do
          ! rotate here
          tmp = qL(qu)
          qL(qu) = qL(qv)
          qL(qv) = -tmp

          tmp = qR(qu)
          qR(qu) = qR(qv)
          qR(qv) = -tmp
          call flux_split(qL, qR, flux_loc)

          ! revert
          tmp = flux_loc(qu)
          flux_loc(qu) = -flux_loc(qv)
          flux_loc(qv) = tmp

          flux2(i,j,k,:) = flux_loc

        end do
      end do
    end do

    ! Z-direction
    do k = fd3_lo(3), fd3_hi(3)
      do j = fd3_lo(2), fd3_hi(2)
        do i = fd3_lo(1), fd3_hi(1)
          do n = 1,QVAR
            qL(n) = q(i,j,k-1, n)
            qR(n) = q(i,j,k, n)
          end do
          ! exchange u and w
          tmp = qL(qu)
          qL(qu) = qL(qw)
          qL(qw) = -tmp

          tmp = qR(qu)
          qR(qu) = qR(qw)
          qR(qw) = -tmp

          call flux_split(qL, qR, flux_loc)

          ! revert
          tmp = flux_loc(qu)
          flux_loc(qu) = -flux_loc(qw)
          flux_loc(qw) = tmp

          flux3(i,j,k,:) = flux_loc

        end do
      end do
    end do

  end subroutine strange_flux

  pure logical function is_inside (i,j,k,lo,hi)
    integer, intent(in) :: i,j,k,lo(3),hi(3)
    is_inside = i.ge.lo(1) .and. i.le.hi(1) &
      .and.  j.ge.lo(2) .and. j.le.hi(2) &
      .and.  k.ge.lo(3) .and. k.le.hi(3)
  end function is_inside


  subroutine compute_eb_divop (lo,hi,ncomp, dx, dt, &
    fluxx,fxlo,fxhi, &       ! flux at face center
    fluxy,fylo,fyhi, &
    fluxz,fzlo,fzhi, &
    fctrdx, fcxlo, fcxhi, &     ! flux for reflux, defined on valid cells
    fctrdy, fcylo, fcyhi, &
    fctrdz, fczlo, fczhi, &
    ebdivop, oplo, ophi, &
    q, qlo, qhi, &
    divc, optmp, rediswgt, dvlo, dvhi, &
    delm, dmlo, dmhi, &
    vfrac, vlo, vhi, &
    apx, axlo, axhi, &
    apy, aylo, ayhi, &
    apz, azlo, azhi, &
    centx_y, cxylo, cxyhi, &
    centx_z, cxzlo, cxzhi, &
    centy_x, cyxlo, cyxhi, &
    centy_z, cyzlo, cyzhi, &
    centz_x, czxlo, czxhi, &
    centz_y, czylo, czyhi, &
    cellflag, cflo, cfhi,  &
    as_crse, rr_drho_crse, rdclo, rdchi, rr_flag_crse, rfclo, rfchi, &
    as_fine, dm_as_fine, dflo, dfhi, &
    levmsk, lmlo, lmhi)


    use amrex_eb_flux_reg_nd_module, only : crse_cell, crse_fine_boundary_cell, &
      covered_by_fine=>fine_cell, reredistribution_threshold

    integer, intent(in), dimension(3) :: lo, hi, fxlo,fxhi,fylo,fyhi,fzlo,fzhi,oplo,ophi,&
      dvlo,dvhi,dmlo,dmhi,axlo,axhi,aylo,ayhi,azlo,azhi,cxylo,cxyhi,cxzlo,cxzhi,&
      cyxlo,cyxhi,cyzlo,cyzhi,czxlo,czxhi,czylo,czyhi,vlo,vhi,cflo,cfhi, qlo,qhi, &
      fcxlo, fcxhi, fcylo, fcyhi, fczlo, fczhi, &
      rdclo, rdchi, rfclo, rfchi, dflo, dfhi, lmlo, lmhi
    logical, intent(in) :: as_crse, as_fine
    integer, intent(in) :: ncomp
    real(rt), intent(in) :: dx(3), dt
    real(rt), intent(in   ) :: fluxx ( fxlo(1): fxhi(1), fxlo(2): fxhi(2), fxlo(3): fxhi(3),ncomp)
    real(rt), intent(in   ) :: fluxy ( fylo(1): fyhi(1), fylo(2): fyhi(2), fylo(3): fyhi(3),ncomp)
    real(rt), intent(in   ) :: fluxz ( fzlo(1): fzhi(1), fzlo(2): fzhi(2), fzlo(3): fzhi(3),ncomp)
    real(rt), intent(inout) :: fctrdx(fcxlo(1):fcxhi(1),fcxlo(2):fcxhi(2),fcxlo(3):fcxhi(3),ncomp)
    real(rt), intent(inout) :: fctrdy(fcylo(1):fcyhi(1),fcylo(2):fcyhi(2),fcylo(3):fcyhi(3),ncomp)
    real(rt), intent(inout) :: fctrdz(fczlo(1):fczhi(1),fczlo(2):fczhi(2),fczlo(3):fczhi(3),ncomp)
    real(rt), intent(inout) :: ebdivop(oplo(1):ophi(1),oplo(2):ophi(2),oplo(3):ophi(3),ncomp)
    real(rt), intent(in) ::   q(qlo(1):qhi(1),qlo(2):qhi(2),qlo(3):qhi(3),qvar)
    real(rt) :: divc    (dvlo(1):dvhi(1),dvlo(2):dvhi(2),dvlo(3):dvhi(3))
    real(rt) :: optmp   (dvlo(1):dvhi(1),dvlo(2):dvhi(2),dvlo(3):dvhi(3))
    real(rt) :: rediswgt(dvlo(1):dvhi(1),dvlo(2):dvhi(2),dvlo(3):dvhi(3))
    real(rt) :: delm    (dmlo(1):dmhi(1),dmlo(2):dmhi(2),dmlo(3):dmhi(3))
    real(rt), intent(in) :: vfrac(vlo(1):vhi(1),vlo(2):vhi(2),vlo(3):vhi(3))
    real(rt), intent(in) :: apx(axlo(1):axhi(1),axlo(2):axhi(2),axlo(3):axhi(3))
    real(rt), intent(in) :: apy(aylo(1):ayhi(1),aylo(2):ayhi(2),aylo(3):ayhi(3))
    real(rt), intent(in) :: apz(azlo(1):azhi(1),azlo(2):azhi(2),azlo(3):azhi(3))
    real(rt), intent(in) :: centx_y(cxylo(1):cxyhi(1),cxylo(2):cxyhi(2),cxylo(3):cxyhi(3))
    real(rt), intent(in) :: centx_z(cxzlo(1):cxzhi(1),cxzlo(2):cxzhi(2),cxzlo(3):cxzhi(3))
    real(rt), intent(in) :: centy_x(cyxlo(1):cyxhi(1),cyxlo(2):cyxhi(2),cyxlo(3):cyxhi(3))
    real(rt), intent(in) :: centy_z(cyzlo(1):cyzhi(1),cyzlo(2):cyzhi(2),cyzlo(3):cyzhi(3))
    real(rt), intent(in) :: centz_x(czxlo(1):czxhi(1),czxlo(2):czxhi(2),czxlo(3):czxhi(3))
    real(rt), intent(in) :: centz_y(czylo(1):czyhi(1),czylo(2):czyhi(2),czylo(3):czyhi(3))
    integer, intent(in) :: cellflag(cflo(1):cfhi(1),cflo(2):cfhi(2),cflo(3):cfhi(3))
    real(rt), intent(inout) :: rr_drho_crse(rdclo(1):rdchi(1),rdclo(2):rdchi(2),rdclo(3):rdchi(3),ncomp)
    integer,  intent(in) ::  rr_flag_crse(rfclo(1):rfchi(1),rfclo(2):rfchi(2),rfclo(3):rfchi(3))
    real(rt), intent(out) :: dm_as_fine(dflo(1):dfhi(1),dflo(2):dfhi(2),dflo(3):dfhi(3),ncomp)
    integer,  intent(in) ::  levmsk (lmlo(1):lmhi(1),lmlo(2):lmhi(2),lmlo(3):lmhi(3))

    logical :: valid_cell, valid_dst_cell
    logical :: as_crse_crse_cell, as_crse_covered_cell, as_fine_valid_cell, as_fine_ghost_cell
    integer :: i,j,k,n,ii,jj,kk, nbr(-1:1,-1:1,-1:1), iii,jjj,kkk
    integer :: nwalls, iwall
    real(rt) :: fxp,fxm,fyp,fym,fzp,fzm,divnc, vtot,wtot, fracx,fracy,fracz,dxinv(3)
    real(rt) :: divwn, drho
    real(rt), pointer, contiguous :: divhyp(:,:)

    dxinv = 1.d0/dx

    nwalls = 0
    do       k = lo(3)-2, hi(3)+2
      do    j = lo(2)-2, hi(2)+2
        do i = lo(1)-2, hi(1)+2
          if (is_single_valued_cell(cellflag(i,j,k))) then
            nwalls = nwalls+1
          end if
        end do
      end do
    end do

    call amrex_allocate(divhyp, 1,5, 1,nwalls)

    do n = 1, ncomp

      !
      ! First, we compute conservative divergence on (lo-2,hi+2)
      !
      iwall = 0
      do       k = lo(3)-2, hi(3)+2
        do    j = lo(2)-2, hi(2)+2
          do i = lo(1)-2, hi(1)+2
            if (is_regular_cell(cellflag(i,j,k))) then
              divc(i,j,k) = (fluxx(i,j,k,n)-fluxx(i+1,j,k,n))*dxinv(1) &
                +        (fluxy(i,j,k,n)-fluxy(i,j+1,k,n))*dxinv(2) &
                +        (fluxz(i,j,k,n)-fluxz(i,j,k+1,n))*dxinv(3)
            else if (is_covered_cell(cellflag(i,j,k))) then
              divc(i,j,k) = 0.d0
              if (is_inside(i,j,k,lo,hi)) then
                fctrdx(i,j,k,n) = 0.d0
                fctrdx(i+1,j,k,n) = 0.d0
                fctrdy(i,j,k,n) = 0.d0
                fctrdy(i,j+1,k,n) = 0.d0
                fctrdz(i,j,k,n) = 0.d0
                fctrdz(i,j,k+1,n) = 0.d0
              end if
            else if (is_single_valued_cell(cellflag(i,j,k))) then

              valid_cell = is_inside(i,j,k,lo,hi)

              call get_neighbor_cells(cellflag(i,j,k),nbr)
              
              ! linear interpolation of face flux
              ! x-direction lo face
              if (apx(i,j,k).lt.1.d0) then
                if (centx_y(i,j,k).le.0.d0) then
                  fracy = -centx_y(i,j,k)*nbr(0,-1,0)
                  if(centx_z(i,j,k).le. 0.0d0)then
                    fracz = - centx_z(i,j,k)*nbr(0,0,-1)
                    fxm = (1.d0-fracz)*(     fracy *fluxx(i,j-1,k  ,n)  + &
                    &             (1.d0-fracy)*fluxx(i,j  ,k  ,n)) + &
                    &      fracz *(     fracy *fluxx(i,j-1,k-1,n)  + &
                    &             (1.d0-fracy)*fluxx(i,j  ,k-1,n))
                  else
                    fracz =  centx_z(i,j,k)*nbr(0,0,1)
                    fxm = (1.d0-fracz)*(     fracy *fluxx(i,j-1,k  ,n)  + &
                    &             (1.d0-fracy)*fluxx(i,j  ,k  ,n)) + &
                    &      fracz *(     fracy *fluxx(i,j-1,k+1,n)  + &
                    &             (1.d0-fracy)*fluxx(i,j  ,k+1,n))
                  endif
                else
                  fracy = centx_y(i,j,k)*nbr(0,1,0)
                  if(centx_z(i,j,k).le. 0.0d0)then
                    fracz = -centx_z(i,j,k)*nbr(0,0,-1)
                    fxm = (1.d0-fracz)*(     fracy *fluxx(i,j+1,k  ,n)  + &
                    &             (1.d0-fracy)*fluxx(i,j  ,k  ,n)) + &
                    &      fracz *(     fracy *fluxx(i,j+1,k-1,n)  + &
                    &             (1.d0-fracy)*fluxx(i,j  ,k-1,n))
                  else
                    fracz = centx_z(i,j,k)*nbr(0,0,1)
                    fxm = (1.d0-fracz)*(     fracy *fluxx(i,j+1,k  ,n)  + &
                    &             (1.d0-fracy)*fluxx(i,j  ,k  ,n)) + &
                    &      fracz *(     fracy *fluxx(i,j+1,k+1,n)  + &
                    &             (1.d0-fracy)*fluxx(i,j  ,k+1,n))
                  endif
                end if
              else
                fxm = fluxx(i,j,k,n)
              end if

              if (valid_cell) fctrdx(i,j,k,n) = fxm

              ! x-direction hi face
              if (apx(i+1,j,k).lt.1.d0) then
                if (centx_y(i+1,j,k).le.0.d0) then
                  fracy = -centx_y(i+1,j,k)*nbr(0,-1,0)
                  if(centx_z(i+1,j,k).le. 0.0d0)then
                    fracz = - centx_z(i+1,j,k)*nbr(0,0,-1)
                    fxp = (1.d0-fracz)*(     fracy *fluxx(i+1,j-1,k  ,n)  + &
                    &             (1.d0-fracy)*fluxx(i+1,j  ,k  ,n)) + &
                    &      fracz *(     fracy *fluxx(i+1,j-1,k-1,n)  + &
                    &             (1.d0-fracy)*fluxx(i+1,j  ,k-1,n))
                  else
                    fracz =  centx_z(i+1,j,k)*nbr(0,0,1)
                    fxp = (1.d0-fracz)*(     fracy *fluxx(i+1,j-1,k  ,n)  + &
                    &             (1.d0-fracy)*fluxx(i+1,j  ,k  ,n)) + &
                    &      fracz *(     fracy *fluxx(i+1,j-1,k+1,n)  + &
                    &             (1.d0-fracy)*fluxx(i+1,j  ,k+1,n))
                  endif
                else
                  fracy = centx_y(i+1,j,k)*nbr(0,1,0)
                  if(centx_z(i+1,j,k).le. 0.0d0)then
                    fracz = -centx_z(i+1,j,k)*nbr(0,0,-1)
                    fxp = (1.d0-fracz)*(     fracy *fluxx(i+1,j+1,k  ,n)  + &
                    &             (1.d0-fracy)*fluxx(i+1,j  ,k  ,n)) + &
                    &      fracz *(     fracy *fluxx(i+1,j+1,k-1,n)  + &
                    &             (1.d0-fracy)*fluxx(i+1,j  ,k-1,n))
                  else
                    fracz = centx_z(i+1,j,k)*nbr(0,0,1)
                    fxp = (1.d0-fracz)*(     fracy *fluxx(i+1,j+1,k  ,n)  + &
                    &             (1.d0-fracy)*fluxx(i+1,j  ,k  ,n)) + &
                    &      fracz *(     fracy *fluxx(i+1,j+1,k+1,n)  + &
                    &             (1.d0-fracy)*fluxx(i+1,j  ,k+1,n))
                  endif
                end if
              else
                fxp = fluxx(i+1,j,k,n)
              end if

              if (valid_cell) fctrdx(i+1,j,k,n) = fxp

              ! y-direction lo face
              if (apy(i,j,k).lt.1.d0) then
                if (centy_x(i,j,k).le.0.d0) then
                  fracx = - centy_x(i,j,k)*nbr(-1,0,0)
                  if(centy_z(i,j,k).le. 0.0d0)then
                    fracz = - centy_z(i,j,k)*nbr(0,0,-1)
                    fym = (1.d0-fracz)*(     fracx *fluxy(i-1,j,k  ,n)  + &
                    &             (1.d0-fracx)*fluxy(i  ,j,k  ,n)) + &
                    &      fracz *(     fracx *fluxy(i-1,j,k-1,n)  + &
                    &             (1.d0-fracx)*fluxy(i  ,j,k-1,n))
                  else
                    fracz =  centy_z(i,j,k)*nbr(0,0,1)
                    fym = (1.d0-fracz)*(     fracx *fluxy(i-1,j,k  ,n)  + &
                    &             (1.d0-fracx)*fluxy(i  ,j,k  ,n)) + &
                    &      fracz *(     fracx *fluxy(i-1,j,k+1,n)  + &
                    &             (1.d0-fracx)*fluxy(i  ,j,k+1,n))
                  endif
                else
                  fracx =  centy_x(i,j,k)*nbr(1,0,0)
                  if(centy_z(i,j,k).le. 0.0d0)then
                    fracz = -centy_z(i,j,k)*nbr(0,0,-1)
                    fym = (1.d0-fracz)*(     fracx *fluxy(i+1,j,k  ,n)  + &
                    &             (1.d0-fracx)*fluxy(i  ,j,k  ,n)) + &
                    &      fracz *(     fracx *fluxy(i+1,j,k-1,n)  + &
                    &             (1.d0-fracx)*fluxy(i  ,j,k-1,n))
                  else
                    fracz = centy_z(i,j,k)*nbr(0,0,1)
                    fym = (1.d0-fracz)*(     fracx *fluxy(i+1,j,k  ,n)  + &
                    &             (1.d0-fracx)*fluxy(i  ,j,k  ,n)) + &
                    &      fracz *(     fracx *fluxy(i+1,j,k+1,n)  + &
                    &             (1.d0-fracx)*fluxy(i  ,j,k+1,n))
                  endif
                endif
              else
                fym = fluxy(i,j,k,n)
              end if

              if (valid_cell) fctrdy(i,j,k,n) = fym

              ! y-direction hi face
              if (apy(i,j+1,k).lt.1d0) then
                if (centy_x(i,j+1,k).le.0.d0) then
                  fracx = - centy_x(i,j+1,k)*nbr(-1,0,0)
                  if(centy_z(i,j+1,k).le. 0.0d0)then
                    fracz = - centy_z(i,j+1,k)*nbr(0,0,-1)
                    fyp = (1.d0-fracz)*(     fracx *fluxy(i-1,j+1,k  ,n)  + &
                    &             (1.d0-fracx)*fluxy(i  ,j+1,k  ,n)) + &
                    &      fracz *(     fracx *fluxy(i-1,j+1,k-1,n)  + &
                    &             (1.d0-fracx)*fluxy(i  ,j+1,k-1,n))
                  else
                    fracz =  centy_z(i,j+1,k)*nbr(0,0,1)
                    fyp = (1.d0-fracz)*(     fracx *fluxy(i-1,j+1,k  ,n)  + &
                    &             (1.d0-fracx)*fluxy(i  ,j+1,k  ,n)) + &
                    &      fracz *(     fracx *fluxy(i-1,j+1,k+1,n)  + &
                    &             (1.d0-fracx)*fluxy(i  ,j+1,k+1,n))
                  endif
                else
                  fracx =  centy_x(i,j+1,k)*nbr(1,0,0)
                  if(centy_z(i,j+1,k).le. 0.0d0)then
                    fracz = -centy_z(i,j+1,k)*nbr(0,0,-1)
                    fyp = (1.d0-fracz)*(     fracx *fluxy(i+1,j+1,k  ,n)  + &
                    &             (1.d0-fracx)*fluxy(i  ,j+1,k  ,n)) + &
                    &      fracz *(     fracx *fluxy(i+1,j+1,k-1,n)  + &
                    &             (1.d0-fracx)*fluxy(i  ,j+1,k-1,n))
                  else
                    fracz = centy_z(i,j+1,k)*nbr(0,0,1)
                    fyp = (1.d0-fracz)*(     fracx *fluxy(i+1,j+1,k  ,n)  + &
                    &             (1.d0-fracx)*fluxy(i  ,j+1,k  ,n)) + &
                    &      fracz *(     fracx *fluxy(i+1,j+1,k+1,n)  + &
                    &             (1.d0-fracx)*fluxy(i  ,j+1,k+1,n))
                  endif
                endif
              else
                fyp = fluxy(i,j+1,k,n)
              end if

              if (valid_cell) fctrdy(i,j+1,k,n) = fyp

              ! z-direction lo face
              if(apz(i,j,k).lt.1.d0)then
                if(centz_x(i,j,k).le. 0.0d0)then
                  fracx = - centz_x(i,j,k)*nbr(-1,0,0)
                  if(centz_y(i,j,k).le. 0.0d0)then
                    fracy = - centz_y(i,j,k)*nbr(0,-1,0)
                    fzm = (1.d0-fracy)*(     fracx *fluxz(i-1,j  ,k,n)  + &
                    &             (1.d0-fracx)*fluxz(i  ,j  ,k,n)) + &
                    &      fracy* (     fracx *fluxz(i-1,j-1,k,n)  + &
                    &             (1.d0-fracx)*fluxz(i  ,j-1,k,n))
                  else
                    fracy =  centz_y(i,j,k)*nbr(0,1,0)
                    fzm = (1.d0-fracy)*(     fracx *fluxz(i-1,j  ,k,n)  + &
                    &             (1.d0-fracx)*fluxz(i  ,j  ,k,n)) + &
                    &      fracy *(     fracx *fluxz(i-1,j+1,k,n)  + &
                    &             (1.d0-fracx)*fluxz(i  ,j+1,k,n))
                  endif
                else
                  fracx =  centz_x(i,j,k)*nbr(1,0,0)
                  if(centz_y(i,j,k).le. 0.0d0)then
                    fracy = -centz_y(i,j,k)*nbr(0,-1,0)
                    fzm = (1.d0-fracy)*(     fracx *fluxz(i+1,j  ,k,n)  + &
                    &             (1.d0-fracx)*fluxz(i  ,j  ,k,n)) + &
                    &      fracy *(     fracx *fluxz(i+1,j-1,k,n)  + &
                    &             (1.d0-fracx)*fluxz(i  ,j-1,k,n))
                  else
                    fracy = centz_y(i,j,k)*nbr(0,1,0)
                    fzm = (1.d0-fracy)*(     fracx *fluxz(i+1,j  ,k,n)+ &
                    &             (1.d0-fracx)*fluxz(i  ,j  ,k,n)) + &
                    &      fracy* (     fracx *fluxz(i+1,j+1,k,n)+ &
                    &             (1.d0-fracx)*fluxz(i  ,j+1,k,n))
                  endif
                endif
              else
                fzm = fluxz(i,j,k,n)
              endif

              if (valid_cell) fctrdz(i,j,k,n) = fzm

              ! z-direction hi face
              if(apz(i,j,k+1).lt.1.d0)then
                if(centz_x(i,j,k+1).le. 0.0d0)then
                  fracx = - centz_x(i,j,k+1)*nbr(-1,0,0)
                  if(centz_y(i,j,k+1).le. 0.0d0)then
                    fracy = - centz_y(i,j,k+1)*nbr(0,-1,0)
                    fzp = (1.d0-fracy)*(     fracx *fluxz(i-1,j  ,k+1,n)  + &
                    &             (1.d0-fracx)*fluxz(i  ,j  ,k+1,n)) + &
                    &      fracy* (     fracx *fluxz(i-1,j-1,k+1,n)  + &
                    &             (1.d0-fracx)*fluxz(i  ,j-1,k+1,n))
                  else
                    fracy =  centz_y(i,j,k+1)*nbr(0,1,0)
                    fzp = (1.d0-fracy)*(     fracx *fluxz(i-1,j  ,k+1,n)  + &
                    &             (1.d0-fracx)*fluxz(i  ,j  ,k+1,n)) + &
                    &      fracy *(     fracx *fluxz(i-1,j+1,k+1,n)  + &
                    &             (1.d0-fracx)*fluxz(i  ,j+1,k+1,n))
                  endif
                else
                  fracx =  centz_x(i,j,k+1)*nbr(1,0,0)
                  if(centz_y(i,j,k+1).le. 0.0d0)then
                    fracy = -centz_y(i,j,k+1)*nbr(0,-1,0)
                    fzp = (1.d0-fracy)*(     fracx *fluxz(i+1,j  ,k+1,n)  + &
                    &             (1.d0-fracx)*fluxz(i  ,j  ,k+1,n)) + &
                    &      fracy *(     fracx *fluxz(i+1,j-1,k+1,n)  + &
                    &             (1.d0-fracx)*fluxz(i  ,j-1,k+1,n))
                  else
                    fracy = centz_y(i,j,k+1)*nbr(0,1,0)
                    fzp = (1.d0-fracy)*(     fracx *fluxz(i+1,j  ,k+1,n)+ &
                    &             (1.d0-fracx)*fluxz(i  ,j  ,k+1,n)) + &
                    &      fracy* (     fracx *fluxz(i+1,j+1,k+1,n)+ &
                    &             (1.d0-fracx)*fluxz(i  ,j+1,k+1,n))
                  endif
                endif
              else
                fzp = fluxz(i,j,k+1,n)
              endif

              if (valid_cell) fctrdz(i,j,k+1,n) = fzp

              iwall = iwall + 1
              if (n .eq. 1) then
                call compute_hyp_wallflux(divhyp(:,iwall), i,j,k, q(i,j,k,qrho), &
                  q(i,j,k,qu), q(i,j,k,qv), q(i,j,k,qw), q(i,j,k,qp), &
                  apx(i,j,k), apx(i+1,j,k), &
                  apy(i,j,k), apy(i,j+1,k), &
                  apz(i,j,k), apz(i,j,k+1))
              end if

              divwn = divhyp(n,iwall)

              ! we assume dx == dy == dz
              divc(i,j,k) = -((apx(i+1,j,k)*fxp - apx(i,j,k)*fxm) * dxinv(1) &
                +          (apy(i,j+1,k)*fyp - apy(i,j,k)*fym) * dxinv(2) &
                +          (apz(i,j,k+1)*fzp - apz(i,j,k)*fzm) * dxinv(3) &
                +          divwn * dxinv(1)) / vfrac(i,j,k)
            end if
          end do

          if (n.eq.1) then
            ! use volfrac as weight
            do i = lo(1)-2, hi(1)+2
              rediswgt(i,j,k) = vfrac(i,j,k)
            end do
          end if
        end do
      end do

      optmp = 0.d0

      !
      ! Second, we compute delta M on (lo-1,hi+1)
      !
      do       k = lo(3)-1, hi(3)+1
        do    j = lo(2)-1, hi(2)+1
          do i = lo(1)-1, hi(1)+1
            if (is_single_valued_cell(cellflag(i,j,k))) then
              vtot = 0.d0
              divnc = 0.d0
              call get_neighbor_cells(cellflag(i,j,k),nbr)
              do kk = -1,1
                do jj = -1,1
                  do ii = -1,1
                    if ((ii.ne. 0 .or. jj.ne.0 .or. kk.ne. 0) .and. nbr(ii,jj,kk).eq.1) then
                      vtot = vtot + vfrac(i+ii,j+jj,k+kk)
                      divnc = divnc + vfrac(i+ii,j+jj,k+kk)*divc(i+ii,j+jj,k+kk)
                    end if
                  end do
                enddo
              enddo
              divnc = divnc / vtot
              optmp(i,j,k) = (1.d0-vfrac(i,j,k))*(divnc-divc(i,j,k))
              delm(i,j,k) = -vfrac(i,j,k)*optmp(i,j,k)
            else
              delm(i,j,k) = 0.d0
            end if
          end do
        end do
      end do


      !
      ! Third, redistribution
      !
      do       k = lo(3)-1, hi(3)+1
        do    j = lo(2)-1, hi(2)+1
          do i = lo(1)-1, hi(1)+1
            if (is_single_valued_cell(cellflag(i,j,k))) then
              wtot = 0.d0
              call get_neighbor_cells(cellflag(i,j,k),nbr)
              do kk = -1,1
                do jj = -1,1
                  do ii = -1,1
                    if ((ii.ne. 0 .or. jj.ne.0 .or. kk.ne. 0) .and. nbr(ii,jj,kk).eq.1) then
                      wtot = wtot + vfrac(i+ii,j+jj,k+kk)*rediswgt(i+ii,j+jj,k+kk)
                    end if
                  end do
                enddo
              enddo

              as_crse_crse_cell = .false.
              as_crse_covered_cell = .false.
              if (as_crse) then
                as_crse_crse_cell = is_inside(i,j,k,lo,hi) .and. &
                  rr_flag_crse(i,j,k) .eq. crse_fine_boundary_cell
                as_crse_covered_cell = rr_flag_crse(i,j,k) .eq. covered_by_fine
              end if

              as_fine_valid_cell = .false.  ! valid cells near box boundary
              as_fine_ghost_cell = .false.  ! ghost cells just outside valid region
              if (as_fine) then
                as_fine_valid_cell = is_inside(i,j,k,lo,hi)
                as_fine_ghost_cell = levmsk(i,j,k) .eq. 2 !levmsk_notcovered , not covered by other grids
              end if

              wtot = 1.d0/wtot
              do kk = -1,1
                do jj = -1,1
                  do ii = -1,1
                    if((ii.ne. 0 .or. jj.ne.0 .or. kk.ne. 0) .and. nbr(ii,jj,kk).eq.1) then

                      iii = i + ii
                      jjj = j + jj
                      kkk = k + kk

                      drho = delm(i,j,k)*wtot*rediswgt(iii,jjj,kkk)
                      optmp(iii,jjj,kkk) = optmp(iii,jjj,kkk) + drho

                      valid_dst_cell = is_inside(iii,jjj,kkk,lo,hi)

                      if (as_crse_crse_cell) then
                        if (rr_flag_crse(iii,jjj,kkk).eq.covered_by_fine &
                          .and. vfrac(i,j,k).gt.reredistribution_threshold) then
                          rr_drho_crse(i,j,k,n) = rr_drho_crse(i,j,k,n) &
                            + dt*drho*(vfrac(iii,jjj,kkk)/vfrac(i,j,k))
                        end if
                      end if

                      if (as_crse_covered_cell) then
                        if (valid_dst_cell) then
                          if (rr_flag_crse(iii,jjj,kkk).eq.crse_fine_boundary_cell &
                            .and. vfrac(iii,jjj,kkk).gt.reredistribution_threshold) then
                            ! the recipient is a crse/fine boundary cell
                            rr_drho_crse(iii,jjj,kkk,n) = rr_drho_crse(iii,jjj,kkk,n) &
                              - dt*drho
                          end if
                        end if
                      end if

                      if (as_fine_valid_cell) then
                        if (.not.valid_dst_cell) then
                          dm_as_fine(iii,jjj,kkk,n) = dm_as_fine(iii,jjj,kkk,n) &
                            + dt*drho*vfrac(iii,jjj,kkk)
                        end if
                      end if

                      if (as_fine_ghost_cell) then
                        if (valid_dst_cell) then
                          dm_as_fine(i,j,k,n) = dm_as_fine(i,j,k,n) &
                            - dt*drho*vfrac(iii,jjj,kkk)
                        end if
                      end if

                    endif
                  enddo
                enddo
              end do
            end if
          end do
        end do
      end do

      do       k = lo(3), hi(3)
        do    j = lo(2), hi(2)
          do i = lo(1), hi(1)
            ebdivop(i,j,k,n) = divc(i,j,k) + optmp(i,j,k)
          end do
        end do
      end do

    end do
    call amrex_deallocate(divhyp)
  end subroutine

  ! Compute F^{EB}
  subroutine compute_hyp_wallflux (divw, i,j,k, rho, u, v, w, p, &
    axm, axp, aym, ayp, azm, azp)
    ! gamma and analriem
    integer, intent(in) :: i,j,k
    real(rt), intent(in) :: rho, u, v, w, p, axm, axp, aym, ayp, azm, azp
    real(rt), intent(out) :: divw(5)

    real(rt) :: apnorm, apnorminv, anrmx, anrmy, anrmz, un
    real(rt) :: flux(5)

    apnorm = sqrt((axm-axp)**2 + (aym-ayp)**2 + (azm-azp)**2)

    if (apnorm .eq. 0.d0) then
      print *, "compute_hyp_wallflux: ", i,j,k, axm, axp, aym, ayp, azm, azp
      flush(6)
      call amrex_abort("compute_hyp_wallflux: we are in trouble.")
    end if

    apnorminv = 1.d0 / apnorm
    anrmx = (axm-axp) * apnorminv  ! pointing to the wall
    anrmy = (aym-ayp) * apnorminv
    anrmz = (azm-azp) * apnorminv

    ! perpendicular to EB surface
    un = u*anrmx + v*anrmy + w*anrmz

    !TODO: compute flux here
    call compute_eb_flux(rho, un, p, flux)

    divw = 0.d0
    divw(umx) = (axm-axp) * flux(2)
    divw(umy) = (aym-ayp) * flux(2)
    divw(umz) = (azm-azp) * flux(2)

  end subroutine compute_hyp_wallflux

  ! solve riemann problem on EB face
  !TODO: use exact riemann solver for this
  subroutine compute_eb_flux(rl, un, p, flux)
    real(rt), intent(in) :: rl, un, p
    real(rt), intent(inout) :: flux(qvar)

    real(rt) :: fp(qvar), fn(qvar)
    real(rt) :: cL, cR, ML, MR, Mp, tmp, Mn, tmpn
    real(rt) :: rhoL, rhoR, uL, uR, vL, vR, wL, wR, pL, pR

    real(rt) :: gm1 = gamma - 1.d0

    cL = sqrt(gamma * p/rl)
    cR = sqrt(gamma * p/rl)

    ML = un/cL
    MR = (-un)/cR

    fp = 0.d0
    fn = 0.d0

    if (ML >= 1.d0) then
      fp(1) = rl * un
      fp(2) = rl * un * un + p
      fp(3) = 0.d0
      fp(4) = 0.d0
      fp(5) = un * (gamma * p / gm1 + 0.5d0 * rl * (un * un))
    else if (abs(ML) < 1.0) then
      Mp = 0.25d0 * (1.d0 + ML) * (1.d0 + ML)
      tmp = rl * cL * Mp
      fp(1) = tmp
      fp(2) = tmp * (gm1 * un + 2.d0 * cL) / gamma
      fp(3) = 0.d0
      fp(4) = 0.d0
      fp(5) = tmp * ((gm1 * un + 2.d0 * cL)**2 * 0.5d0 / (gamma**2 - 1.d0))
    end if

    if (abs(MR) < 1.d0) then
      Mn = -0.25d0 * (MR - 1.d0) * (MR - 1.d0)
      tmpn = rl * cR * Mn
      fn(1) = tmpn
      fn(2) = tmpn * (gm1 * (-un) - 2.d0 * cR) / gamma
      fn(3) = 0.d0
      fn(4) = 0.d0
      fn(5) = tmpn * ((gm1 * (-un) - 2.d0 * cR)**2 * 0.5d0 / (gamma**2 - 1.d0))
    else if (MR <= -1.d0) then
      fn(1) = rl * (-un)
      fn(2) = rl * (-un) * (-un) + p
      fn(3) = 0.d0
      fn(4) = 0.d0
      fn(5) = (-un) * (gamma * p / gm1 + 0.5d0 * rl * ((-un) * (-un)))
    end if
    flux = fp + fn

  end subroutine compute_eb_flux

  pure function minmod(a, b) result(res)
    real(rt) , intent(in) :: a, b
    real(rt) :: res
    if (a*b> 0) then
      if (abs(a)>abs(b)) then
        res = b
      else
        res = a
      end if
    else
      res = 0.d0
    endif
  end function minmod

  ! for debug
  subroutine write_slice(filename, lo, hi, var)
    character(len=50) :: filename
    integer, intent(in) :: lo(3), hi(3)
    real(rt), intent(inout) :: var(lo(1):hi(1), lo(2):hi(2))

    integer :: i,j
    open(16, file = trim(filename))
    write(16, '(a)') 'variable = x, y, var'
    write(16, *) 'zone i = ', hi(1)+1-lo(1), ' j = ', hi(2)+1-lo(2)
    do j = lo(2), hi(2)
      do i = lo(1), hi(1)
        write(16, *) i, j, var(i,j)
      end do
    end do
    close(16)
  end subroutine
end module nc_ibm_module
