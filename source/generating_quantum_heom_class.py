# ---------------------------------------------------------------------
#
#                  GENERARTING QUANTUM HEOM INGREDIENTS
#
# ---------------------------------------------------------------------
#
# This Python class generates the molecular system (Hamiltonian, ann./cre. operators etc), the 
# decomposition of the BCF, the indices of the ADOs, and, finally, the HEOM superoperator at a 
# representative vibrational coordinate in a sparse format. It will also generate the ingredients 
# one needs to then generate the HEOM superoperator as a function of the vibrational coordinates. 
# Because implementing HEOM is a complex task, various parts are split into Python modules and Fortran
# subroutines. This code imports all modules and runs them in the correct order.
#
# Note that one must create the Python wrappers from the Fortran subroutines first (eta_gamma,sparsity,sparse_propagation)
# Alternatively, one could go into a python environment (type Python into command line) and run each line manually.
#
# There are no direct inputs, rather, one must first change the input_parameters.py and system.py file to reflect the problem you want to solve. These
# are imported automatically into this code. 
#
# USAGE - RUN IN PYTHON ENVIRONMENT:
#
#       import generating_quantum_heom_class
#       quantum_heom_ingredients_object = generating_quantum_heom_class.generate_quantum_heom(regenerate_info=True)
#
# OUTPUT -
#
#       molecular_system_ingredients = quantum_heom_ingredients_object.return_molecular_system_ingredients()
#       sparse_heom_ingredients = quantum_heom_ingredients_object.return_sparse_heom_ingredients()
#

#### IMPORT PYTHON MODULES ####

import gc,os,pickle,sys
import numpy as np
from time import perf_counter, asctime
from importlib import reload
# Import HEOM modules
from source.constants import * 
import sparsity
from source.input_parameters import *
import source.system as system
import source.Index_pm_filter as Index_pm_filter
import source.generating_sparsity_class as generating_sparsity_class
reload(generating_sparsity_class)
import source.eta_gamma_barycentric as eta_gamma_barycentric
import generate_heom_one_x

class generate_quantum_heom():

    def __init__(self,regenerate_info):

        self.regenerate_info = regenerate_info

        self.t_start_generation = asctime()                     
        if os.path.isfile('generating_quantum_heom_info.dat'):
            self.output_info = open('generating_quantum_heom_info.dat','a')
            self.output_info.write('Start HEOM generation at '+str(self.t_start_generation)+'\n')
            self.output_info.close()
        else:
            self.output_info = open('generating_quantum_heom_info.dat','w') 
            self.output_info.write('Start HEOM generation at '+str(self.t_start_generation)+'\n')
            self.output_info.close()
        self.t_start = perf_counter()
        
        self.generate_molecular_system()
        self.generate_bath_correlation_expansion()
        self.generate_ado_indices()
        self.generate_heom_in_sparse_representation_for_example_x()

    # ---------------------------------------------------------------
    #            DEFINE OPERATORS IN MOLECULAR HILBERT SPACE
    # ---------------------------------------------------------------

    def generate_molecular_system(self):

        self.system_output = system.system_operators(Single_El_Int,Double_El_Int,Nel,N_qu_vib_modes,El_Nuclear_Couplings_cl,
                                        max_occ_qu_vib_modes,dim_rho)

        self.d_ops,self.d,self.ddag,self.Fock_states,self.molham_func,self.molham_log,\
        self.d_ops_log,self.rho_0_log,self.identity_dim_rho,self.el_occ_op,\
        self.el_occ_op_log = self.system_output[0:11]
        if bool(N_qu_vib_modes):
            self.b_ops,self.b,self.bdag = self.system_output[11:14]
        
        sys.stdout.flush()
        self.t_end = perf_counter()                                                                                  
        self.output_info = open('generating_quantum_heom_info.dat','a')
        self.output_info.write("Elapsed time of molecular system generation: " + str(self.t_end-self.t_start) +'\n') 
        self.output_info.close()
        self.output_info = open('generating_quantum_heom_info.dat','a')
        self.t_start = self.t_end

    def return_molecular_system_ingredients(self):

        return self.system_output

    # ---------------------------------------------------------------
    #      BATH-CORRELATION EXPANSION - BARYCENTRIC AND PADE
    # ---------------------------------------------------------------

    def generate_bath_correlation_expansion(self):

        EtaGamma = eta_gamma_barycentric.bath_correlation_decomposition(Ncutoff,specwidth,Nsupport_points_barycentric,
                        Npoles_pade,symmetrized_fermi_specwidth,Temp,Nleads,Nsign,muvec,tol_Gamma_barycentric,
                        tol_fermi_symmetrized_barycentric,wbl_YN,analytic_spectral_function_decomposition,tol_F)
        self.eta_vec_barycentric,self.gamma_vec_barycentric = EtaGamma.barycentric_bath_correlation_expansion()
        self.eta_vec_pade,self.gamma_vec_pade = EtaGamma.pade_bath_correlation_expansion()

        if pole_choice == "pade":
            self.eta_vec = self.eta_vec_pade
            self.gamma_vec = self.gamma_vec_pade
        elif pole_choice == "barycentric":
            self.eta_vec = self.eta_vec_barycentric
            self.gamma_vec = self.gamma_vec_barycentric
        elif pole_choice == "prony":
            raise ValueError("Prony/MPM decomposition not yet implemented")
            # eta_vec = eta_vec_prony
            # gamma_vec = gamma_vec_prony
        else:
            raise ValueError("Choose an appropriate pole decomposition scheme: Options are pade or barycentric")

        if wbl_YN == 0:                                                                    # If not under the wide-band limit, they are assumed to have a Lorentzian density of states
            self.Npoles = len(self.eta_vec[0,0,:]) - 1
            self.Nmodes = (self.Npoles+1)*Nel*Nleads*Nsign                                           # Calculate number of modes outside of wide-band limit
        elif wbl_YN == 1:
            self.Npoles = len(self.eta_vec[0,0,:])
            self.Nmodes = self.Npoles*Nel*Nleads*Nsign         

        sys.stdout.flush()
        self.t_end = perf_counter()                                                                                      # Return value of performance counter at end of index generation
        self.output_info.write("Elapsed time of BCF decomposition: " + str(self.t_end-self.t_start) +'\n')                        # Write into simulation_info.txt the time taken to perform index generation
        self.output_info.close()
        self.output_info = open('generating_quantum_heom_info.dat','a')
        self.t_start = self.t_end

    # ---------------------------------------------------------------
    #                 INDEX GENERATION OF ADOs IN HEOM
    # ---------------------------------------------------------------

    def generate_ado_indices(self):

        Indices = Index_pm_filter.Hierarchy_index(Nmax,Nel,self.Npoles,Nleads,Nsign,self.Nmodes,wbl_YN)                              # Define object of Hierarchy_index class with HEOM parameters as input
        if wbl_YN == 0:
            if filtering_YN == 1:                                                                                   # Run filtering process if filtering_YN is true
                max_V_km = V_Km
                self.KsigLm_filtered,self.Un_Ind_filtered,self.Hier_ind_filtered,self.Index_minus_filtered,\
                self.Index_plus_filtered = \
                Indices.Print_Filtered_Ind_Info(tol,self.eta_vec,self.gamma_vec,max_V_km)
            else:
                self.KsigLm,self.Un_Ind,self.Hier_ind,self.Index_Minus,self.Index_Plus,\
                self.len_un_ind,self.len_index_plus,self.tier_index = Indices.Print_Ind_Info()
                                                                                                                    # Return index information from Indices object; see Index_pm_filter for details
        elif wbl_YN == 1:
            if filtering_YN == 1:                                                                                   # Run filtering process if filtering_YN is true
                max_V_km = V_Km # Define maximum coupling between leads and electronic levels in the system
                self.KsigLm_filtered,self.Ksig0m_filtered,self.Un_Ind_filtered,\
                self.Hier_ind_filtered,self.Index_minus_filtered,self.Index_plus_filtered = \
                Indices.Print_Filtered_Ind_Info(tol,self.eta_vec,self.gamma_vec,max_V_km)
            else:
                self.KsigLm,self.Ksig0m,self.Un_Ind,self.Hier_ind,self.Index_Minus,self.Index_Plus,
                self.len_un_ind,self.len_index_plus,self.tier_index = Indices.Print_Ind_Info()
                                                                                                                    # Return index information from Indices object; see Index_pm_filter for details

        sys.stdout.flush()
        self.t_end = perf_counter()                                                                                      # Return value of performance counter at end of index generation
        self.output_info.write("Elapsed time of indices generation: " + str(self.t_end-self.t_start) +'\n')                        # Write into simulation_info.txt the time taken to perform index generation
        self.t_start = self.t_end

    # ---------------------------------------------------------------
    #          TRANSFORMING HEOM TO SPARSE REPRESENTATION 
    # ---------------------------------------------------------------

    def generate_heom_in_sparse_representation_for_example_x(self):

        import source.generating_sparsity_class as generating_sparsity_class
        reload(generating_sparsity_class)

        x_vec_tester = np.array([10],dtype=float)
        sparsity_object = generating_sparsity_class.sparsity_heom_liouvillian(ksiglm=self.KsigLm,
            tier_index=self.tier_index,index_minus=self.Index_Minus,index_plus=self.Index_Plus,
            d_ops_comp=self.d_ops,d_ops_comp_log=self.d_ops_log,ham_log=self.molham_log,
            rho_0_log=self.rho_0_log,max_expan_order=max_expan_order,dim_rho=dim_rho,
            len_index_plus=self.len_index_plus,len_un_ind=self.len_un_ind,nmax=Nmax,nel=Nel,
            degenerate_levels=degenerate_levels,atol=atol,rtol=rtol,un_ind=self.Un_Ind,
            gamma_vec=self.gamma_vec,eta_vec=self.eta_vec,nsign=Nsign,nleads=Nleads,npoles=self.Npoles,
            molham_one_x=self.molham_func(dim_rho,self.d_ops,El_Nuclear_Couplings_cl,x_vec_tester),
            el_lead_couplings_one_x=el_lead_couplings_func(Nleads,Nel,V_Km,x_vec_tester))
        self.sparse_heom_ingredients = sparsity_object.save_sparse_heom()
        print("Sparse HEOM has been (re)generated")
        
        sys.stdout.flush()
        self.t_end = perf_counter()
        self.output_info.write("Elapsed time to transform to sparse representation: " + str(self.t_end-self.t_start) +'\n')
        self.t_end = asctime()                                                                                           
        self.output_info.write('The simulation ends at '+str(self.t_end)+'\n')
        self.output_info.close() 

    def return_sparse_heom_ingredients(self):

        # with open("sparsity_ingredients.p", "rb") as sparse_heom_ingredients_file:
        #     sparse_heom_ingredients = pickle.load(sparse_heom_ingredients_file)

        # sparse_heom_ingredients_file = open("sparsity_ingredients.p","rb")
        # sparse_heom_ingredients = pickle.load(sparse_heom_ingredients_file)
        # sparse_heom_ingredients_file.close()
        (self.pair_info_row_fil,self.pair_info_col_fil,self.pair_values_fil,self.npairs_fil,self.npairs_uf,
        self.nnz_elements_sparse_fil,self.nnz_elements_sparse_zeroth_tier_fil,self.row_old_indices,
        self.atol_vec,self.rtol_vec,self.rho_nonzeros_sparse,self.isreal_sparse,self.complex_coefficients,
        self.nnz_elements_zeroth_tier,self.rho_nonzeros,self.rho_sparsity,self.nnz_elements,
        self.is_connected_array,self.rho_out,self.pair_values_gamma_fil,self.si_ham_row_info_fil,
        self.si_ham_col_info_fil,self.si_coupledown_row_info_fil,self.si_coupledown_col_info_fil,
        self.si_coupleup_row_info_fil,self.si_coupleup_col_info_fil,
        self.ham_loc_info_fil,self.coupleup_loc_info_fil,
        self.coupledown_loc_info_fil,self.coupleup_conj_info_fil,self.coupledown_conj_info_fil,
        self.pair_values_coupleup_wout_el_lead_coupling_fil,
        self.pair_values_coupledown_wout_el_lead_coupling_fil,
        self.trace_cols,self.rhs_vector,self.sparse_trace_array,self.tier_ind,self.ksiglm,self.un_ind) = self.sparse_heom_ingredients[0:39]

        return self.sparse_heom_ingredients

    def return_sparse_heom_one_x(self,ham_this_x,el_lead_couplings_this_x,pair_values_one_x):

        generate_heom_one_x.heom_liouvillian_one_x(pair_values_gamma=self.pair_values_gamma_fil,
            si_ham_row_info=self.si_ham_row_info_fil,si_ham_col_info=self.si_ham_col_info_fil,
            si_coupledown_row_info=self.si_coupledown_row_info_fil,
            si_coupledown_col_info=self.si_coupledown_col_info_fil,
            si_coupleup_row_info=self.si_coupleup_row_info_fil,
            si_coupleup_col_info=self.si_coupleup_col_info_fil,
            ham_loc_info=self.ham_loc_info_fil,coupleup_loc_info=self.coupleup_loc_info_fil,
            coupledown_loc_info=self.coupledown_loc_info_fil,coupleup_conj_info=self.coupleup_conj_info_fil,
            coupledown_conj_info=self.coupledown_conj_info_fil,
            pair_values_coupleup_wout_el_lead_coupling=self.pair_values_coupleup_wout_el_lead_coupling_fil,
            pair_values_coupledown_wout_el_lead_coupling=self.pair_values_coupledown_wout_el_lead_coupling_fil,
            ham_x=ham_this_x,el_lead_couplings_x=el_lead_couplings_this_x,npairs=self.npairs_fil,
            pair_values=pair_values_one_x,nleads=Nleads,nel=Nel,dim_rho=dim_rho)

        return pair_values_one_x