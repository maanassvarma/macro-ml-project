program bellman_backup
  implicit none
  integer, parameter :: Nk=120, Nz=7
  real(8), parameter :: alpha=0.33d0, beta=0.96d0, delta=0.08d0
  real(8) :: k_grid(Nk), z_grid(Nz)
  real(8) :: Vnew(Nk,Nz), EV(Nk,Nz)
  integer :: policy(Nk,Nz)
  integer :: ik, iz, j, r, reps
  real(8) :: k, z, y, kp, c, rhs, best
  integer :: arg
  real(8) :: ct0, ct1   ! CPU_TIME seconds

  ! grids
  do ik=1, Nk
     k_grid(ik) = 0.5d0 + (3.0d0 - 0.5d0) * dble(ik-1) / dble(Nk-1)
  end do
  z_grid = (/0.74d0, 0.84d0, 0.95d0, 1.00d0, 1.06d0, 1.18d0, 1.30d0/)
  EV = 0.0d0

  reps = 200
  call cpu_time(ct0)

  do r=1, reps
!$omp parallel do collapse(2) private(ik,iz,j,k,z,y,kp,c,rhs,best,arg) shared(Vnew,policy,k_grid,z_grid,EV)
     do iz=1, Nz
        do ik=1, Nk
           k = k_grid(ik);  z = z_grid(iz)
           y = z * k**alpha
           best = -1.0d18;  arg = 1
           do j=1, Nk
              kp = k_grid(j)
              c  = y + (1.0d0 - delta)*k - kp
              if (c > 0.0d0) then
                 rhs = log(c) + beta * EV(j, iz)
              else
                 rhs = -1.0d10
              end if
              if (rhs > best) then
                 best = rhs; arg = j
              end if
           end do
           Vnew(ik,iz)   = best
           policy(ik,iz) = arg
        end do
     end do
!$omp end parallel do
  end do

  call cpu_time(ct1)
  print *, 'Fortran Bellman backup completed. Example: policy(61,4)=', policy(61,4)
  print *, 'Fortran total time (reps=', reps, '): ', ct1 - ct0, ' s'
  print *, 'Fortran per-sweep time: ', (ct1 - ct0)/dble(reps), ' s'
end program bellman_backup
