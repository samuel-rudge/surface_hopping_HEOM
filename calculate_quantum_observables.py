from input_parameters import *

class quantum_observables_class():

    def __init__(self,sparse_heom_ingredients,molecular_system_ingredients,
                projected_0,projected_1):

        self.d_ops = molecular_system_ingredients[0]
        self.nnz_elements_sparse_fil = sparse_heom_ingredients[5]
        self.rho_nonzeros_sparse = sparse_heom_ingredients[10]
        self.isreal_sparse = sparse_heom_ingredients[11]
        self.complex_coefficients = sparse_heom_ingredients[12]
        self.tier_index = sparse_heom_ingredients[36]
        self.ksiglm = sparse_heom_ingredients[37]
        self.un_ind = sparse_heom_ingredients[38]
        self.projected_0 = projected_0
        self.projected_1 = projected_1

        self.populations_this_x = np.zeros(dim_el,dtype=float)
        self.rho_mol_this_x = np.zeros((dim_rho,dim_rho),dtype=complex)
        self.current_this_x = np.zeros(Nleads,dtype=float)

    def return_quantum_observables_this_x(self,quantum_state_this_x,el_lead_couplings_this_x):

        self.rho_mol_this_x = np.zeros((dim_rho,dim_rho),dtype=complex)
        self.current_this_x = np.zeros(Nleads,dtype=float)
        self.partial_prob_current_0_to_1 = 0.0
        self.partial_prob_current_1_to_0 = 0.0
        for itrnz in range(self.nnz_elements_sparse_fil):
            indjn = self.rho_nonzeros_sparse[itrnz,0]
            itrn = self.tier_index[indjn]
            nrow = self.rho_nonzeros_sparse[itrnz,1]
            ncol = self.rho_nonzeros_sparse[itrnz,2]
            # print(nrow)
            # print(ncol)
            if (itrn == 0):
                self.rho_mol_this_x[nrow,ncol] += quantum_state_this_x[itrnz]*self.complex_coefficients[itrnz]
                # if (self.isreal_sparse[itrnz] and self.el_occ_log[ncol,nrow]):
                #     self.populations_this_x += self.el_occ[ncol,nrow]*steady_state_x[itrnz]
            elif (itrn == 1):
                jn = self.un_ind[indjn,0]
                leads_n = self.ksiglm[jn,0]                                                         
                sign_n = 1-self.ksiglm[jn,1]                                                         
                eldash_n = self.ksiglm[jn,3]
                if not self.isreal_sparse[itrnz]:
                    if not degenerate_levels:
                        for itrel in range(self.Nel):
                            self.current_this_x[leads_n] += ((-1.0)**sign_n)*2.0*el_lead_couplings_this_x[leads_n,eldash_n]*\
                                                self.d_ops[ncol,nrow,itrel,sign_n]*quantum_state_this_x[itrnz]
                    else:
                        self.current_this_x[leads_n] += ((-1.0)**sign_n)*2.0*el_lead_couplings_this_x[leads_n,eldash_n]*\
                                                self.d_ops[ncol,nrow,eldash_n,sign_n]*quantum_state_this_x[itrnz]
                    if (nrow == 0) & (ncol == 1):
                        # print("true")
                        self.partial_prob_current_0_to_1 += np.abs(el_lead_couplings_this_x[leads_n,eldash_n]*\
                                                                   quantum_state_this_x[itrnz])
                    elif (nrow == 1) & (ncol == 0):
                        # print("true")
                        self.partial_prob_current_1_to_0 += np.abs(el_lead_couplings_this_x[leads_n,eldash_n]*\
                                                                    quantum_state_this_x[itrnz])
                        self.partial_prob_current_0_to_1 += np.abs(el_lead_couplings_this_x[leads_n,eldash_n]*\
                                                                   quantum_state_this_x[itrnz])
            else:
                break
        # print(self.partial_prob_current_0_to_1)
        # print(self.partial_prob_current_1_to_0)
        self.populations_this_x = np.real(np.diag(self.rho_mol_this_x))
        return self.current_this_x,self.rho_mol_this_x,self.populations_this_x,\
               self.partial_prob_current_0_to_1,self.partial_prob_current_1_to_0
    
    def split_quantum_state_this_x(self,quantum_state_this_x):

        initial_state_10 = np.zeros(self.nnz_elements_sparse_fil,dtype=float)
        initial_state_01 = np.zeros(self.nnz_elements_sparse_fil,dtype=float)
        initial_state_10[self.projected_0] = quantum_state_this_x[self.projected_0]
        initial_state_01[self.projected_1] = quantum_state_this_x[self.projected_1]
        initial_state_10 = np.concatenate((np.array([1,0],dtype=float),quantum_state_this_x[2:]),axis=0)#*quantum_state_this_x[0]
        initial_state_01 = np.concatenate((np.array([0,1],dtype=float),quantum_state_this_x[2:]),axis=0)#*quantum_state_this_x[1]
        initial_state_list = [initial_state_10,initial_state_01]

        return initial_state_list