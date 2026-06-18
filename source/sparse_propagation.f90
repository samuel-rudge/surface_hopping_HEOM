! ---------------------------------------------------------------------------
! 
!                     PROPAGATION ADOs ONE TIME STEP
!
! ---------------------------------------------------------------------------
!
! This Fortran subroutine is meant to be converted to a Python wrapper using f2py, for use 
! in the main Python code. The various subroutines detail the sparse nature of the HEOM.
!
! USAGE - RUN FROM TERMINAL (LINUX) TO CREATE PYTHON WRAPPER:
!       Check fortran compilers available in your platform:  f2py -c --help-fcompiler
!       For 'gfortran': f2py3 --fcompiler=gfortran -m sparse_propagation -c sparse_propagation.f90 --f90flags="-fopenmp -ffree-line-length-512 -I/opt/intel/oneapi/mkl/2025.1/include" --opt="-O3" -L/opt/intel/oneapi/mkl/latest/lib/intel64 -lmkl_gf_lp64 -lmkl_core -lmkl_gnu_thread -lpthread -lm -ldl  
!
! It uses parallel programming and the previously generated sparse representation of the HEOM to 
! propagate an initial density matrix in time, keeping track of both the time-dependent current
! and the density matrix itself.
!
! USAGE - RUN FROM MAIN PYTHON CODE ONCE WRAPPED:
!       rho_output = sparse_propagation.sparse_propagation(pair_info_row=pair_info_row_fil,pair_info_col=pair_info_col_fil,
!                                        pair_values=pair_values_one_x,dt=dt,rho_input=rho_input,&
!                                        max_expan_order=max_expan_order,nthreads_liouvillian=nthreads_liouvillian,npairs=npairs,&
!                                        nnz_elements=nnz_elemetns,rk_coeff=rk_coeff,rho_temp=rho_temp,rho_deriv=rho_temp)
!
! INPUTS:
!
!       nnz_elements                                Number of matrix elements in all ADOs that connect to the zeroth-tier ADO.
!                                                   This is essentially the length of the vectorized \boldsymbol{\rho} in the HEOM.
!       
!       npairs                                      Number of connected pairs of elements in HEOM. That is, the sparsity of L_HEOM.
!                                                   Does not change from vib. coordinate to vib. coordinate.
!
!       pair_info_row                               Array of size [npairs] containing rows of nonzero elements in sparse HEOM Liouvillian (COO format)
!
!       pair_info_col                               Corresponding array of size [npairs] containing columns of nonzero elements in sparse HEOM Liouvillian (COO format)
!
!       pair_values                                 Array of size [npairs] containing the value of this connection (i.e. the corresponding value of the nonzero element of L_HEOM). 
!                                                   This is what will change from vibrational coordinate to vibrational coordinate.
!
!       rho_input                                   Array of size [nnz_elements] containing the input \boldsymbol{\rho} for this timestep;
!                                                   that is, all nonzero elements of the vectorized ADOs at time t, ready to update
!                                                   to time t + dt.
!
!
!       dt                                          Scalar that specifies size of each timestep. Keep constant for now.
!
!       max_expan_order                             Scalar that determines the maximum order to go to in the approximate expansion of e^{L*dt}.
!                                                   So far, I have only implemented fourth-order Runge-Kutta, so must be 4. 
!
!       rk_coeff                                    Runge-Kutta coefficients
!
!       nthreads_liouvillian                        Available number of cores/threads to help in sparse matrix multiplication
!
!       rho_temp,rho_deriv                          Helper arrays of size [nnz_elements] that we use in the time-propagation. Can ignore.
!
! OUTPUTS:
!       rho_output                                  Array of size [nnz_elements] containing the output \boldsymbol{\rho} at time t + dt.
!

! include 'mkl_spblas.f90'
! include 'sparse_liouvillian_storage.f90'

subroutine sparse_one_step_propagation(pair_info_row,pair_info_col,pair_values,dt,rho_input,&
                                        max_expan_order,nthreads_liouvillian,npairs,&
                                        nnz_elements,rk_coeff,rho_output,rho_temp,rho_deriv)

    use mkl_spblas
    use sparse_matrix_storage

    implicit none

    integer, intent(in) :: max_expan_order,nnz_elements,npairs,nthreads_liouvillian
    double precision, intent(in), dimension(0:nnz_elements-1) :: rho_input
    double precision, intent(in), dimension(0:npairs-1) :: pair_values
    integer, intent(in), dimension(0:npairs-1) :: pair_info_row,pair_info_col
    double precision, intent(in) :: dt
    double precision, intent(in), dimension(0:max_expan_order-1) :: rk_coeff
    double precision, intent(inout), dimension(0:nnz_elements-1) :: rho_temp,rho_deriv

    double precision, intent(out), dimension(0:nnz_elements-1) :: rho_output

    ! integer :: ik_coo,
    integer :: ik_csr,itrl
    ! integer :: ik_destroy_coo, ik_destroy_csr

    ! TYPE(SPARSE_MATRIX_T) :: sparse_handle_coo,sparse_handle_csr
    ! TYPE(MATRIX_DESCR) :: descra
    ! descra % TYPE = SPARSE_MATRIX_TYPE_GENERAL

    if (.not. is_initialized) then
        call initialize_sparse_matrix(nnz_elements, npairs, pair_info_row, pair_info_col, pair_values)
    else
        call mkl_sparse_d_update_values(sparse_handle_csr,npairs,pair_info_row,pair_info_col,pair_values)
    endif

    call dcopy(nnz_elements,rho_input,1,rho_temp,1)
    call dcopy(nnz_elements,rho_input,1,rho_output,1)
    do itrl = 0,max_expan_order-1                                                               ! Loop through the Taylor series expansion of e^{L*dt}
        rho_deriv = 0.d0
        ik_csr = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,dt,sparse_handle_csr,&
                                    descra,rho_temp,1.0d0,rho_deriv)
        call daxpy(nnz_elements,rk_coeff(itrl),rho_deriv,1,rho_output,1)
        call dcopy(nnz_elements,rho_deriv,1,rho_temp,1)
    enddo                  
    
    ! ik_csr = mkl_sparse_destroy(sparse_handle_csr)
    ! ik_destroy_csr = mkl_sparse_destroy(sparse_handle_csr)
    ! if (ik_coo /= 0) then
    !     print *, "Error: Failed to destroy sparse matrix handle"
    ! end if
    ! if (ik_csr /= 0) then
    !     print *, "Error: Failed to destroy sparse matrix handle"
    ! end if
    ! ik_coo = 0
    ! ik_csr = 0
    ! call mkl_free_buffers()

end subroutine sparse_one_step_propagation

! subroutine return_quantum_observables()

! end subroutine return_quantum_observables()