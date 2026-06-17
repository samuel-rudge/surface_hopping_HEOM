! ---------------------------------------------------------------------------
! 
!    GENERATE VALUES AND INFORMATION OF EACH COUPLED PAIR OF ADO ELEMENTS 
!
! ---------------------------------------------------------------------------
!
! This Fortran subroutine is similar to the previous sparse_matrix_elements_a subroutine, which calculated
! the number of coupled pairs of ADO elements in the HEOM time-propagation. This subroutine goes further and 
! records information about each coupled pair: the value connecting them and their position in the hierarchy.
! For example, if we know that rho_{i}(a,b) = e^{L*dt}(a,c)*rho_{j}(c,b), then the ADO elements 
! rho_{i}(a,b) (ath row and bth column of ADO with index i) and rho_{j}(c,b) (cth row and bth column of ADO with
! index j) are connected via the ath row and cth column of e^{L*dt}. We need to know this value and these positions
! in order to propagate the HEOM. Note that it is necessary to run sparse_matrix_elements_a before 
! sparse_matrix_elements_b in order to work out the size of the arrays containing the information. Fortran cannot
! dynamically change the shape of arrays (it kind of can but it is slow) and it is much too slow to allocate an 
! array size larger than necessary and then cut it down at the end.
!
! USAGE - RUN IN ABOVE FORTRAN SUBROUTINE nnz(...):
!               pair_info,pair_values,ham_pair_info,gamma_values = sparsity.sparse_matrix_elements_b(ksiglm=KsigLm,tier_index=tier_index,
!                                un_ind=Un_Ind,index_minus=Index_Minus,index_plus=Index_Plus,d_ops=d_ops,ham=Ham,gamma_vec=gamma_vec,eta_vec=eta_vec,
!                                rho_sparsity=rho_sparsity,nnz_elements=nnz_elements,dim_rho=dim_rho,len_index_plus=len_index_plus,len_un_ind=len_un_ind,
!                                len_index_minus=len_un_ind,nmax=Nmax,nmodes=Nmodes,nel=Nel,nsign=Nsign,nleads=Nleads,npoles=Npoles)
!
! INPUTS:
!       ksiglm,tier_index,index_minus,              Same input parameters as in sparse_matrix_elements_a above
!       index_plus,ham,max_expan_order
!       dim_rho,len_index_plus,len_un_ind,
!       len_index_minus,nmax,nmodes,nel,d_ops
!       ham,rho_nonzeros,nnz_elements,
!       rho_sparsity
!
!       gamma_vec,eta_vec                           Arrays containing the exponents and coefficients of the bath-correlation function expansion
!
!       nsign,nleads,npoles                         nsign = 2 (+=0, -=1), nleads =  number of electronic leads, npoles = number of Pade poles
!
! OUTPUTS:
!       pair_info                                   Array of size [npairs,4] containing information about the pairs of coupled nonzero elements in the HEOM.
!                                                   Each row corresponds to a different coupled pair;
!                                                   Column 1 contains the nonzero index of the LHS ADO element, 
!                                                   Column 2 contains the nonzero index of the RHS ADO element,
!                                                   Column 3 contains 1 if the connection between these two ADOs requires a hermiticity relation, and 0 if not
!                                                   Column 4 contains ??? FINISH
!
!       pair_values                                 Array of size [1,npairs] containing the value of this connection (i.e. the corresponding element of e^{L*dt} in Liouville space)            
!
!       ham_pair_info                               The same as pair_info, but just for the coherent part containing the commutator with the Hamiltonian
!
!       gamma_values                                Array of size [1,nnz_elements] containing the sum over gamma values for each ADO in the HEOM

subroutine heom_liouvillian_one_x(pair_values_gamma,si_ham_row_info,si_ham_col_info,si_coupledown_row_info,&
                                si_coupledown_col_info,si_coupleup_row_info,si_coupleup_col_info,&
                                ham_loc_info,coupleup_loc_info,coupledown_loc_info,coupleup_conj_info,&
                                coupledown_conj_info,pair_values_coupleup_wout_el_lead_coupling,&
                                pair_values_coupledown_wout_el_lead_coupling,ham_x,el_lead_couplings_x,npairs,&
                                pair_values,nleads,nel,dim_rho)

    implicit none

    integer, intent(in) :: npairs,nleads,nel,dim_rho   
    double precision, intent(in), dimension(0:nleads-1,0:nel-1) :: el_lead_couplings_x
    complex*16, intent(in), dimension(0:dim_rho-1,0:dim_rho-1) :: ham_x
    double precision, intent(in), dimension(0:npairs-1) :: pair_values_gamma
    integer, intent(in), dimension(0:npairs-1) :: si_ham_row_info,si_ham_col_info
    integer, intent(in), dimension(0:npairs-1,0:1) :: si_coupledown_row_info,si_coupledown_col_info
    integer, intent(in), dimension(0:npairs-1,0:1) :: si_coupleup_row_info,si_coupleup_col_info
    integer, intent(in), dimension(0:npairs-1,0:1) :: ham_loc_info,coupleup_loc_info,coupledown_loc_info
    integer, intent(in), dimension(0:npairs-1,0:1) :: coupleup_conj_info,coupledown_conj_info
    complex*16, intent(in), dimension(0:npairs-1,0:1) :: pair_values_coupleup_wout_el_lead_coupling
    complex*16, intent(in), dimension(0:npairs-1,0:1) :: pair_values_coupledown_wout_el_lead_coupling

    double precision, intent(inout), dimension(0:npairs-1) :: pair_values

    integer :: itrpairs                                   ! Define other necessary variables
    integer :: si_real,si_imag
    complex*16, parameter :: ci=(0.d0,1.d0)                                                      ! Define imaginary number (=sqrt(-1))

    pair_values = pair_values_gamma
    do itrpairs = 0,npairs-1
        !!!! HAMILTONIAN SECTION !!!!
        if ((si_ham_row_info(itrpairs) == 1) .and. (si_ham_col_info(itrpairs) == 1)) then
            pair_values(itrpairs) = pair_values(itrpairs) + dble(-ci*ham_x(ham_loc_info(itrpairs,0),ham_loc_info(itrpairs,1)))
        elseif ((si_ham_row_info(itrpairs) == 1) .and. (si_ham_col_info(itrpairs) == 0)) then
            pair_values(itrpairs) = pair_values(itrpairs) - aimag(-ci*ham_x(ham_loc_info(itrpairs,0),ham_loc_info(itrpairs,1)))
        elseif ((si_ham_row_info(itrpairs) == 0) .and. (si_ham_col_info(itrpairs) == 1)) then
            pair_values(itrpairs) = pair_values(itrpairs) + aimag(-ci*ham_x(ham_loc_info(itrpairs,0),ham_loc_info(itrpairs,1)))        
        elseif ((si_ham_row_info(itrpairs) == 0) .and. (si_ham_col_info(itrpairs) == 0)) then
            pair_values(itrpairs) = pair_values(itrpairs) + dble(-ci*ham_x(ham_loc_info(itrpairs,0),ham_loc_info(itrpairs,1)))
        endif
        !!!! COUPLE DOWN SECTION !!!!
        if ((si_coupledown_row_info(itrpairs,0) == 0) .and. (si_coupledown_col_info(itrpairs,0) == 0)) then
            if (coupledown_conj_info(itrpairs,0) == 1) then
                pair_values(itrpairs) = pair_values(itrpairs) + dble(pair_values_coupledown_wout_el_lead_coupling(itrpairs,0)*&
                                    el_lead_couplings_x(coupledown_loc_info(itrpairs,0),coupledown_loc_info(itrpairs,1)))
            endif
        elseif ((si_coupledown_row_info(itrpairs,0) == 1) .and. (si_coupledown_col_info(itrpairs,0) == 0)) then
            if (coupledown_conj_info(itrpairs,0) == 1) then
                pair_values(itrpairs) = pair_values(itrpairs) + aimag(pair_values_coupledown_wout_el_lead_coupling(itrpairs,0)*&
                                    el_lead_couplings_x(coupledown_loc_info(itrpairs,0),coupledown_loc_info(itrpairs,1)))
            endif
        elseif ((si_coupledown_row_info(itrpairs,0) == 0) .and. (si_coupledown_col_info(itrpairs,0) == 1)) then
            if (coupledown_conj_info(itrpairs,0) == 1) then
                pair_values(itrpairs) = pair_values(itrpairs) + aimag(pair_values_coupledown_wout_el_lead_coupling(itrpairs,0)*&
                                    el_lead_couplings_x(coupledown_loc_info(itrpairs,0),coupledown_loc_info(itrpairs,1)))    
            endif
        elseif ((si_coupledown_row_info(itrpairs,0) == 1) .and. (si_coupledown_col_info(itrpairs,0) == 1)) then
            if (coupledown_conj_info(itrpairs,0) == 1) then
                pair_values(itrpairs) = pair_values(itrpairs) - dble(pair_values_coupledown_wout_el_lead_coupling(itrpairs,0)*&
                                    el_lead_couplings_x(coupledown_loc_info(itrpairs,0),coupledown_loc_info(itrpairs,1)))
            endif
        endif
        if ((si_coupledown_row_info(itrpairs,1) == 0) .and. (si_coupledown_col_info(itrpairs,1) == 0)) then
            if (coupledown_conj_info(itrpairs,1) == 1) then
                pair_values(itrpairs) = pair_values(itrpairs) + dble(pair_values_coupledown_wout_el_lead_coupling(itrpairs,1)*&
                                    el_lead_couplings_x(coupledown_loc_info(itrpairs,0),coupledown_loc_info(itrpairs,1)))
            endif
        elseif ((si_coupledown_row_info(itrpairs,1) == 1) .and. (si_coupledown_col_info(itrpairs,1) == 0)) then
            if (coupledown_conj_info(itrpairs,1) == 1) then
                pair_values(itrpairs) = pair_values(itrpairs) - aimag(pair_values_coupledown_wout_el_lead_coupling(itrpairs,1)*&
                                    el_lead_couplings_x(coupledown_loc_info(itrpairs,0),coupledown_loc_info(itrpairs,1)))
            endif
        elseif ((si_coupledown_row_info(itrpairs,1) == 0) .and. (si_coupledown_col_info(itrpairs,1) == 1)) then
            if (coupledown_conj_info(itrpairs,1) == 1) then
                pair_values(itrpairs) = pair_values(itrpairs) + aimag(pair_values_coupledown_wout_el_lead_coupling(itrpairs,1)*&
                                    el_lead_couplings_x(coupledown_loc_info(itrpairs,0),coupledown_loc_info(itrpairs,1)))    
            endif
        elseif ((si_coupledown_row_info(itrpairs,1) == 1) .and. (si_coupledown_col_info(itrpairs,1) == 1)) then
            if (coupledown_conj_info(itrpairs,1) == 1) then
                pair_values(itrpairs) = pair_values(itrpairs) + dble(pair_values_coupledown_wout_el_lead_coupling(itrpairs,1)*&
                                    el_lead_couplings_x(coupledown_loc_info(itrpairs,0),coupledown_loc_info(itrpairs,1)))
            endif
        endif
        !!!! COUPLE UP SECTION !!!!
        if ((si_coupleup_row_info(itrpairs,0) == 0) .and. (si_coupleup_col_info(itrpairs,0) == 0)) then
            if (coupleup_conj_info(itrpairs,0) == 1) then
                pair_values(itrpairs) = pair_values(itrpairs) + dble(pair_values_coupleup_wout_el_lead_coupling(itrpairs,0)*&
                                    el_lead_couplings_x(coupleup_loc_info(itrpairs,0),coupleup_loc_info(itrpairs,1)))
            endif
        elseif ((si_coupleup_row_info(itrpairs,0) == 1) .and. (si_coupleup_col_info(itrpairs,0) == 0)) then
            if (coupleup_conj_info(itrpairs,0) == 1) then
                pair_values(itrpairs) = pair_values(itrpairs) + aimag(pair_values_coupleup_wout_el_lead_coupling(itrpairs,0)*&
                                    el_lead_couplings_x(coupleup_loc_info(itrpairs,0),coupleup_loc_info(itrpairs,1)))
            endif
        elseif ((si_coupleup_row_info(itrpairs,0) == 0) .and. (si_coupleup_col_info(itrpairs,0) == 1)) then
            if (coupleup_conj_info(itrpairs,0) == 1) then
                pair_values(itrpairs) = pair_values(itrpairs) + aimag(pair_values_coupleup_wout_el_lead_coupling(itrpairs,0)*&
                                    el_lead_couplings_x(coupleup_loc_info(itrpairs,0),coupleup_loc_info(itrpairs,1)))   
            endif
        elseif ((si_coupleup_row_info(itrpairs,0) == 1) .and. (si_coupleup_col_info(itrpairs,0) == 1)) then
            if (coupleup_conj_info(itrpairs,0) == 1) then
                pair_values(itrpairs) = pair_values(itrpairs) - dble(pair_values_coupleup_wout_el_lead_coupling(itrpairs,0)*&
                                    el_lead_couplings_x(coupleup_loc_info(itrpairs,0),coupleup_loc_info(itrpairs,1)))
            endif
        endif
        if ((si_coupleup_row_info(itrpairs,1) == 0) .and. (si_coupleup_col_info(itrpairs,1) == 0)) then
            if (coupleup_conj_info(itrpairs,1) == 1) then
                pair_values(itrpairs) = pair_values(itrpairs) + dble(pair_values_coupleup_wout_el_lead_coupling(itrpairs,1)*&
                                    el_lead_couplings_x(coupleup_loc_info(itrpairs,0),coupleup_loc_info(itrpairs,1)))
            endif
        elseif ((si_coupleup_row_info(itrpairs,1) == 1) .and. (si_coupleup_col_info(itrpairs,1) == 0)) then
            if (coupleup_conj_info(itrpairs,1) == 1) then
                pair_values(itrpairs) = pair_values(itrpairs) - aimag(pair_values_coupleup_wout_el_lead_coupling(itrpairs,1)*&
                                    el_lead_couplings_x(coupleup_loc_info(itrpairs,0),coupleup_loc_info(itrpairs,1)))
            endif
        elseif ((si_coupleup_row_info(itrpairs,1) == 0) .and. (si_coupleup_col_info(itrpairs,1) == 1)) then
            if (coupleup_conj_info(itrpairs,1) == 1) then
                pair_values(itrpairs) = pair_values(itrpairs) + aimag(pair_values_coupleup_wout_el_lead_coupling(itrpairs,1)*&
                                    el_lead_couplings_x(coupleup_loc_info(itrpairs,0),coupleup_loc_info(itrpairs,1)))    
            endif
        elseif ((si_coupleup_row_info(itrpairs,1) == 1) .and. (si_coupleup_col_info(itrpairs,1) == 1)) then
            if (coupleup_conj_info(itrpairs,1) == 1) then
                pair_values(itrpairs) = pair_values(itrpairs) + dble(pair_values_coupleup_wout_el_lead_coupling(itrpairs,1)*&
                                    el_lead_couplings_x(coupleup_loc_info(itrpairs,0),coupleup_loc_info(itrpairs,1)))
            endif
        endif
    enddo

end subroutine heom_liouvillian_one_x