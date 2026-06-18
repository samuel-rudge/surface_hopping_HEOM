import sparsity
import gc
import scipy.sparse as sparse
import numpy as np
import pickle

class sparsity_heom_liouvillian():                                                                    

    def __init__(self,ksiglm,tier_index,index_minus,index_plus,d_ops_comp,d_ops_comp_log,ham_log,
                    rho_0_log,max_expan_order,dim_rho,len_index_plus,len_un_ind,nmax,nel,
                    degenerate_levels,atol,rtol,un_ind,gamma_vec,eta_vec,nsign,nleads,
                    npoles,molham_one_x,el_lead_couplings_one_x):
       
        self.ksiglm = ksiglm
        self.tier_index = tier_index
        self.index_minus = index_minus
        self.index_plus = index_plus
        self.d_ops_comp = d_ops_comp
        self.d_ops_comp_log = d_ops_comp_log
        self.ham_log = ham_log
        self.rho_0_log = rho_0_log
        self.max_expan_order = max_expan_order
        self.dim_rho = dim_rho
        self.len_index_plus = len_index_plus
        self.len_un_ind = len_un_ind
        self.len_index_minus = len_un_ind
        self.nmodes = len(self.ksiglm)
        self.nmax = nmax
        self.nel = nel
        self.degenerate_levels = degenerate_levels
        self.atol = atol
        self.rtol = rtol
        self.un_ind = un_ind
        self.gamma_vec = gamma_vec
        self.eta_vec = eta_vec
        self.nsign = nsign
        self.nleads = nleads
        self.npoles = npoles
        self.molham_one_x = molham_one_x
        self.el_lead_couplings_one_x = el_lead_couplings_one_x

        self.find_nonzero_ado_elements()
        self.find_number_of_nonzero_elements_in_heom_liouvillian()
        self.generate_unfiltered_sparse_representation_of_heom_liouvillian()
        self.generate_filtered_sparse_representation_of_heom_liouvillian()
        self.generate_tr_rho_sparse_condition()

        # Calculates the total number of non-zero elements (nnz_elements) after propagating
        # by 1 timestep (i.e. what ADO elements are coupled to the initial condition and thus 
        # need to be retained during time-propagation), and what each 
        # non-zero element in the coupled set of ADOs is. E.g. if the element in the 0th 
        # row and second column of ADO with index = 47 is non-zero, then 
        # rho_sparsity(0,1,47) = how many non-zero elements are before this one + 1. If 
        # the element in the 0th row and second column of ADO with index = 47 is zero, then 
        # rho_sparsity(0,1,47) = 0. 

    def find_nonzero_ado_elements(self):
        
        self.rho_sparsity,self.nnz_elements,self.rho_out = \
            sparsity.nnz(ksiglm=self.ksiglm,tier_index=self.tier_index,index_minus=self.index_minus,
                        index_plus=self.index_plus,d_ops=self.d_ops_comp_log,ham=self.ham_log,
                        rho_0=self.rho_0_log,max_expan_order=self.max_expan_order,dim_rho=self.dim_rho,
                        len_index_plus=self.len_index_plus,len_un_ind=self.len_un_ind,
                        len_index_minus=self.len_un_ind,nmax=self.nmax,nmodes=self.nmodes,nel=self.nel,
                        degenerate_levels=self.degenerate_levels) 

    # Returns multiple objects. rho_nonzeros is an array containing a row for each non-zero 
    # element, with the columns containing the corresponding ADO, and row and column value. 
    # npairs is the number of non-zero pairs connecting elements of the ADOs, and  nhampairs 
    # is this for the Hamiltonian part only. pair_info is an array containing a row for each 
    # pair, with the columns containing the non-zero indices of the pair, whether the pair 
    # involves a conjugation of the density matrix, and the HEOM value of the pair. 
    # ham_pair_info is similar, except that it records the row and column instead of the HEOM 
    # value, as this is easily calculated later on in the time-propagation. gamma_values is 
    # self-evident.

    def find_number_of_nonzero_elements_in_heom_liouvillian(self):

        self.rho_nonzeros,self.npairs,self.n_indnz2_this_indnz1_max,self.n_indnz2_prev_indnz1_vec = \
                sparsity.sparse_matrix_elements_a(ksiglm=self.ksiglm,tier_index=self.tier_index,
                                index_minus=self.index_minus,index_plus=self.index_plus,
                                d_ops_log=self.d_ops_comp_log,ham_log=self.ham_log,
                                rho_sparsity=self.rho_sparsity,nnz_elements=self.nnz_elements,
                                dim_rho=self.dim_rho,len_index_plus=self.len_index_plus,
                                len_un_ind=self.len_un_ind,len_index_minus=self.len_un_ind,
                                nmax=self.nmax,nmodes=self.nmodes,nel=self.nel,
                                degenerate_levels=self.degenerate_levels)        

    # Returns the actual information about each pair of coupled elements. See sparsity.f90
    # for details about nnz, sparse_matrix_elements_a, and sparse_matrix_elements_b
    
    def generate_unfiltered_sparse_representation_of_heom_liouvillian(self):

        self.pair_info_row,self.pair_info_col,self.pair_values,self.pair_values_gamma,\
        self.si_ham_row_info,self.si_ham_col_info,self.si_coupledown_row_info,self.si_coupledown_col_info,\
        self.si_coupleup_row_info,self.si_coupleup_col_info,\
        self.ham_loc_info,self.coupleup_loc_info,self.coupledown_loc_info,self.coupleup_conj_info,self.coupledown_conj_info,\
        self.pair_values_coupleup_wout_el_lead_coupling,self.pair_values_coupledown_wout_el_lead_coupling = \
                sparsity.sparse_matrix_elements_b(ksiglm=self.ksiglm,tier_index=self.tier_index,un_ind=self.un_ind,
                    index_minus=self.index_minus,index_plus=self.index_plus,d_ops=self.d_ops_comp,gamma_vec=self.gamma_vec,
                    eta_vec=self.eta_vec,rho_sparsity=self.rho_sparsity,rho_nonzeros=self.rho_nonzeros,
                    nnz_elements=self.nnz_elements,dim_rho=self.dim_rho,len_index_plus=self.len_index_plus,
                    len_un_ind=self.len_un_ind,len_index_minus=self.len_un_ind,nmax=self.nmax,nmodes=self.nmodes,
                    nel=self.nel,nsign=self.nsign,npairs=self.npairs,nleads=self.nleads,npoles=self.npoles,
                    ham_log=self.ham_log,d_ops_log=self.d_ops_comp_log,degenerate_levels=self.degenerate_levels,
                    n_indnz2_this_indnz1_max=self.n_indnz2_this_indnz1_max,ham_x=self.molham_one_x,
                    n_indnz2_prev_indnz1_vec=self.n_indnz2_prev_indnz1_vec,el_lead_couplings_x=self.el_lead_couplings_one_x)

    def generate_filtered_sparse_representation_of_heom_liouvillian(self):

        self.nnz_elements_zeroth_tier = (self.rho_sparsity[:,:,0]!=-1).sum()                                                  # Define nnz elements in RDM
        npairs_uf = self.pair_info_row.shape[0]
        zero_couplings = []
        for itr_pair in range(npairs_uf):
            if (np.abs(self.pair_values[itr_pair]) <= 1e-14).all():
                zero_couplings.append(itr_pair)

        zero_couplings = np.array(zero_couplings,dtype=int)
        pair_values_uf = np.delete(self.pair_values,zero_couplings,axis=0)
        pair_info_row_uf = np.delete(self.pair_info_row,zero_couplings,axis=0)
        pair_info_col_uf = np.delete(self.pair_info_col,zero_couplings,axis=0)
        pair_values_gamma_uf = np.delete(self.pair_values_gamma,zero_couplings,axis=0)
        si_ham_row_info_uf = np.delete(self.si_ham_row_info,zero_couplings,axis=0)
        si_ham_col_info_uf = np.delete(self.si_ham_col_info,zero_couplings,axis=0)
        si_coupledown_row_info_uf = np.delete(self.si_coupledown_row_info,zero_couplings,axis=0)
        si_coupledown_col_info_uf = np.delete(self.si_coupledown_col_info,zero_couplings,axis=0)
        si_coupleup_row_info_uf = np.delete(self.si_coupleup_row_info,zero_couplings,axis=0)
        si_coupleup_col_info_uf = np.delete(self.si_coupleup_col_info,zero_couplings,axis=0)
        ham_loc_info_uf = np.delete(self.ham_loc_info,zero_couplings,axis=0)
        coupleup_loc_info_uf = np.delete(self.coupleup_loc_info,zero_couplings,axis=0)
        coupledown_loc_info_uf = np.delete(self.coupledown_loc_info,zero_couplings,axis=0)
        coupleup_conj_info_uf = np.delete(self.coupleup_conj_info,zero_couplings,axis=0)
        coupledown_conj_info_uf = np.delete(self.coupledown_conj_info,zero_couplings,axis=0)
        pair_values_coupleup_wout_el_lead_coupling_uf = np.delete(self.pair_values_coupleup_wout_el_lead_coupling,zero_couplings,axis=0)
        pair_values_coupledown_wout_el_lead_coupling_uf = np.delete(self.pair_values_coupledown_wout_el_lead_coupling,zero_couplings,axis=0)
        self.npairs_uf = pair_info_row_uf.shape[0]

        ### CHECK WHICH REAL + IMAG COMPONENTS ACTUALLY COUPLED ###

        nnz_elements_sparse_uf = 2*self.nnz_elements                # Number of sparse matrix elements coupled to transport (uf = unfiltered)
        self.logical_values_heom = np.ones(self.npairs_uf,dtype=int)
        sparse_bool_heom_generator = sparse.csc_matrix((self.logical_values_heom,(pair_info_row_uf,pair_info_col_uf)),
                                                        shape=(nnz_elements_sparse_uf,nnz_elements_sparse_uf),dtype=bool)
        is_connected_array_old = np.zeros((nnz_elements_sparse_uf),dtype=bool) 
        is_connected_array_old[0] = True
        no_change = False
        while (no_change == False):
            self.is_connected_array = is_connected_array_old | sparse_bool_heom_generator.dot(is_connected_array_old)
            no_change = np.all(self.is_connected_array == is_connected_array_old)
            is_connected_array_old = self.is_connected_array

        ### REMOVE THESE ROWS AND COLUMNS FROM L_HEOM AND RENUMBER ###

        self.pair_info_row_fil = np.copy(pair_info_row_uf)
        self.pair_info_col_fil = np.copy(pair_info_col_uf)
        self.pair_values_fil = np.copy(pair_values_uf)
        self.pair_values_gamma_fil = np.copy(pair_values_gamma_uf)
        self.si_ham_row_info_fil = np.copy(si_ham_row_info_uf)
        self.si_ham_col_info_fil = np.copy(si_ham_col_info_uf)
        self.si_coupledown_row_info_fil = np.copy(si_coupledown_row_info_uf)
        self.si_coupledown_col_info_fil = np.copy(si_coupledown_col_info_uf)
        self.si_coupleup_row_info_fil = np.copy(si_coupleup_row_info_uf)
        self.si_coupleup_col_info_fil = np.copy(si_coupleup_col_info_uf)
        self.ham_loc_info_fil = np.copy(ham_loc_info_uf)
        self.coupleup_loc_info_fil = np.copy(coupleup_loc_info_uf)
        self.coupledown_loc_info_fil = np.copy(coupledown_loc_info_uf)
        self.coupleup_conj_info_fil = np.copy(coupleup_conj_info_uf)
        self.coupledown_conj_info_fil = np.copy(coupledown_conj_info_uf)
        self.pair_values_coupleup_wout_el_lead_coupling_fil = np.copy(pair_values_coupleup_wout_el_lead_coupling_uf)
        self.pair_values_coupledown_wout_el_lead_coupling_fil = np.copy(pair_values_coupledown_wout_el_lead_coupling_uf)
        self.row_old_indices = np.arange(nnz_elements_sparse_uf)
        self.row_new_indices = np.arange(nnz_elements_sparse_uf)
        for itr_nnz in range(nnz_elements_sparse_uf):
            if (self.is_connected_array[itr_nnz] == False):
                pair_indices_row_above = np.where(pair_info_row_uf > itr_nnz)[0]
                pair_indices_col_above = np.where(pair_info_col_uf > itr_nnz)[0]
                self.pair_info_row_fil[pair_indices_row_above] -= 1
                self.pair_info_col_fil[pair_indices_col_above] -= 1
                self.row_new_indices[itr_nnz+1:] -= 1
                self.row_new_indices[itr_nnz] = -1

        pair_indices_remove = np.array([],dtype=int)
        for itr_nnz in range(nnz_elements_sparse_uf):
            if not self.is_connected_array[itr_nnz]:
                pair_indices_remove = np.concatenate((np.where(pair_info_row_uf == itr_nnz)[0],pair_indices_remove),axis=0)
                pair_indices_remove = np.concatenate((np.where(pair_info_col_uf == itr_nnz)[0],pair_indices_remove),axis=0)

        self.pair_values_fil = np.delete(self.pair_values_fil,pair_indices_remove,axis=0)
        self.pair_info_row_fil = np.delete(self.pair_info_row_fil,pair_indices_remove,axis=0)
        self.pair_info_col_fil = np.delete(self.pair_info_col_fil,pair_indices_remove,axis=0)
        self.pair_values_gamma_fil = np.delete(self.pair_values_gamma_fil,pair_indices_remove,axis=0)
        self.si_ham_row_info_fil = np.delete(self.si_ham_row_info_fil,pair_indices_remove,axis=0)
        self.si_ham_col_info_fil = np.delete(self.si_ham_col_info_fil,pair_indices_remove,axis=0)
        self.si_coupledown_row_info_fil = np.delete(self.si_coupledown_row_info_fil,pair_indices_remove,axis=0)
        self.si_coupledown_col_info_fil = np.delete(self.si_coupledown_col_info_fil,pair_indices_remove,axis=0)
        self.si_coupleup_row_info_fil = np.delete(self.si_coupleup_row_info_fil,pair_indices_remove,axis=0)
        self.si_coupleup_col_info_fil = np.delete(self.si_coupleup_col_info_fil,pair_indices_remove,axis=0)
        self.ham_loc_info_fil = np.delete(self.ham_loc_info_fil,pair_indices_remove,axis=0)
        self.coupleup_loc_info_fil = np.delete(self.coupleup_loc_info_fil,pair_indices_remove,axis=0)
        self.coupledown_loc_info_fil = np.delete(self.coupledown_loc_info_fil,pair_indices_remove,axis=0)
        self.coupleup_conj_info_fil = np.delete(self.coupleup_conj_info_fil,pair_indices_remove,axis=0)
        self.coupledown_conj_info_fil = np.delete(self.coupledown_conj_info_fil,pair_indices_remove,axis=0)
        self.pair_values_coupleup_wout_el_lead_coupling_fil = np.delete(self.pair_values_coupleup_wout_el_lead_coupling_fil,pair_indices_remove,axis=0)
        self.pair_values_coupledown_wout_el_lead_coupling_fil = np.delete(self.pair_values_coupledown_wout_el_lead_coupling_fil,pair_indices_remove,axis=0)
        self.npairs_fil = self.pair_info_row_fil.shape[0]
        self.nnz_elements_sparse_fil = self.is_connected_array.sum()
        self.nnz_elements_sparse_zeroth_tier_fil = self.is_connected_array[0:2*self.nnz_elements_zeroth_tier].sum()

        ### WORK OUT MAPPING FOR NEW SPARSITY ###

        self.complex_coefficients = np.array([1,1j],dtype=complex)
        self.complex_coefficients = np.tile(self.complex_coefficients,self.nnz_elements)
        self.complex_coefficients = self.complex_coefficients[self.is_connected_array]
        self.isreal_sparse = np.array([True,False],dtype=bool)
        self.isreal_sparse = np.tile(self.isreal_sparse,self.nnz_elements)
        self.isreal_sparse = self.isreal_sparse[self.is_connected_array]
        self.rho_nonzeros_sparse = np.repeat(self.rho_nonzeros,2,axis=0)
        self.rho_nonzeros_sparse = self.rho_nonzeros_sparse[self.is_connected_array]
        self.atol_vec = np.full((self.nnz_elements_sparse_fil,1),self.atol,dtype=float)                                                     # Specify adaptive Runge-Kutta constraints
        self.rtol_vec = np.full((self.nnz_elements_sparse_fil,1),self.rtol,dtype=float)
        self.rhs_vector = np.zeros(self.nnz_elements_sparse_fil,dtype=float) ; self.rhs_vector[0] = 1.0

    def generate_tr_rho_sparse_condition(self):

        trace_rows = np.zeros(self.dim_rho,dtype=int)
        trace_elements = 2*np.diag(self.rho_sparsity[:,:,0])
        self.trace_cols = np.zeros(self.dim_rho,dtype=int)
        trace_values = np.ones(self.dim_rho,dtype=float)
        for itr_element in np.arange(self.dim_rho):
            self.trace_cols[itr_element] = self.row_new_indices[self.row_old_indices == trace_elements[itr_element]]

        self.sparse_trace_array = sparse.csc_matrix((trace_values,(trace_rows,self.trace_cols)),
                                            shape=(self.nnz_elements_sparse_fil,self.nnz_elements_sparse_fil),dtype=float)


    def save_sparse_heom(self):

        sparsity_ingredients=[self.pair_info_row_fil,self.pair_info_col_fil,self.pair_values_fil,self.npairs_fil,self.npairs_uf,\
               self.nnz_elements_sparse_fil,self.nnz_elements_sparse_zeroth_tier_fil,self.row_old_indices,\
               self.atol_vec,self.rtol_vec,self.rho_nonzeros_sparse,self.isreal_sparse,self.complex_coefficients,\
               self.nnz_elements_zeroth_tier,self.rho_nonzeros,self.rho_sparsity,self.nnz_elements,\
               self.is_connected_array,self.rho_out,self.pair_values_gamma_fil,self.si_ham_row_info_fil,\
               self.si_ham_col_info_fil,self.si_coupledown_row_info_fil,self.si_coupledown_col_info_fil,\
               self.si_coupleup_row_info_fil,self.si_coupleup_col_info_fil,self.ham_loc_info_fil,\
               self.coupleup_loc_info_fil,self.coupledown_loc_info_fil,self.coupleup_conj_info_fil,\
               self.coupledown_conj_info_fil,self.pair_values_coupleup_wout_el_lead_coupling_fil,\
               self.pair_values_coupledown_wout_el_lead_coupling_fil,self.trace_cols,self.rhs_vector,self.sparse_trace_array,\
               self.tier_index,self.ksiglm,self.un_ind]
        
        pickle.dump(sparsity_ingredients,open("sparsity_ingredients.p","wb"))
        
        return sparsity_ingredients
        # for name in list(globals().keys()):
        #     if not name.startswith("_") and name not in ("gc", "pickle", "os"):  # keep what you need
        #         del globals()[name]

        # self.__dict__.clear()
        # gc.collect()