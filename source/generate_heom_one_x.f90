!-----------------------------------------------------------------------
!
!  SUBROUTINE: heom_liouvillian_one_x
!
!  PURPOSE:
!  --------
!  This routine evaluates the *numerical values* of all nonzero entries
!  of the sparse HEOM Liouvillian L_HEOM at a given nuclear coordinate x.
!
!  The sparsity pattern of L_HEOM (i.e. which matrix elements exist and
!  how they are connected) is assumed to be fixed and precomputed. This
!  routine only updates the *values* of those entries as a function of:
!
!      - the molecular electronic Hamiltonian at coordinate x
!      - the molecule–lead coupling strengths at coordinate x
!
!  All structural information (connectivity, index mapping, conjugation
!  flags, and decomposition coefficients) is held fixed and supplied as
!  input arrays.
!
!  This separation allows efficient evaluation of L_HEOM along a nuclear
!  trajectory without rebuilding the sparse structure at each step.
!
!
!  CALLING CONTEXT (PYTHON WRAPPER):
!  ---------------------------------
!  This subroutine is invoked from:
!
!      generate_quantum_heom_class.generate_quantum_heom
!          -> return_sparse_heom_one_x(...)
!
!  via:
!
!      generate_heom_one_x.heom_liouvillian_one_x(...)
!
!  It is used to update the precomputed sparse HEOM Liouvillian
!  for a specific nuclear configuration x during propagation.
!
!
!  INPUTS:
!  -------
!  npairs
!      Number of nonzero entries in the sparse Liouvillian.
!
!  nleads, nel, dim_rho
!      System dimensions (leads, electronic levels, Liouville space size).
!
!  ham_x(dim_rho, dim_rho)
!      Molecular electronic Hamiltonian evaluated at coordinate x.
!
!  el_lead_couplings_x(nleads, nel)
!      Molecule–lead coupling matrix evaluated at coordinate x.
!
!  pair_values_gamma(npairs)
!      Precomputed static contribution (bath / damping / baseline terms).
!
!  si_ham_row_info(npairs), si_ham_col_info(npairs)
!      Flags specifying whether Hamiltonian contribution enters real/imag
!      parts of each sparse element.
!
!  ham_loc_info(npairs,2)
!      Index mapping identifying which (i,j) element of ham_x contributes
!      to each sparse entry.
!
!  si_coupleup/down_*_info
!  couple*_loc_info
!  couple*_conj_info
!      Precomputed structural information controlling how coupling terms
!      enter each sparse element (direction, conjugation, placement).
!
!  pair_values_couple*_wout_el_lead_coupling(npairs,2)
!      Coupling prefactors excluding the x-dependent coupling strengths.
!
!
!  OUTPUT:
!  -------
!  pair_values(npairs)
!      Updated numerical values of all nonzero entries of the sparse
!      HEOM Liouvillian at coordinate x.
!
!
!  NUMERICAL ROLE:
!  ---------------
!  The Liouvillian is represented in sparse form as:
!
!      L_HEOM(x) ≡ { (row_k, col_k, value_k(x)) } for k = 1..npairs
!
!  This routine computes value_k(x) for fixed (row_k, col_k).
!
!
!  PERFORMANCE NOTE:
!  -----------------
!  Since only values change with x, this avoids recomputing:
!      - sparsity pattern
!      - index mappings
!      - ADO connectivity structure
!
!  making it suitable for repeated evaluation along trajectories.
!
!-----------------------------------------------------------------------


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