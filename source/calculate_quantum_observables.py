# ---------------------------------------------------------------------
#
#        QUANTUM OBSERVABLES FROM SPARSE HEOM STATE VECTOR
#
# ---------------------------------------------------------------------
#
# This module defines a post-processing / analysis layer for the SHEOM
# framework, responsible for extracting physical observables from the
# sparse HEOM state representation propagated during the dynamics.
#
# The HEOM state is stored as a sparse vector representation of the full
# hierarchical density operator, including both:
#
#   - the molecular reduced electronic density matrix
#   - auxiliary density operators (ADOs) encoding bath memory effects
#
# Observables are evaluated "on the fly" at a given nuclear coordinate x,
# by reconstructing physically meaningful quantities from the sparse state.
#
# ---------------------------------------------------------------------
#
# PHYSICAL ROLE IN SHEOM PIPELINE
#
# For each nuclear configuration x(t), the propagator produces:
#
#     quantum_state_this_x(t)  ≡  sparse HEOM state vector
#
# This object is not directly observable. This class maps it to:
#
#   - molecular electronic density matrix  ρ_mol(x)
#   - electronic populations               diag[ρ_mol(x)]
#   - lead currents                        I_leads(x)
#   - auxiliary diagnostic probabilities   (partial transition weights)
#
# The mapping explicitly resolves contributions from:
#   - physical density matrix blocks (tier 0)
#   - first-tier ADO contributions relevant for currents (tier 1)
#
# ---------------------------------------------------------------------
#
# OBSERVABLE DEFINITION STRATEGY
#
# The sparse HEOM representation encodes different operator sectors
# through index metadata:
#
#   - tier_index:
#         identifies whether a component belongs to
#         physical density matrix (tier 0) or ADOs (tier > 0)
#
#   - rho_nonzeros_sparse:
#         maps sparse vector entries → (row, column) matrix indices
#
#   - complex_coefficients:
#         reconstruction weights for physical density matrix elements
#
#   - ksiglm / un_ind:
#         encode lead, sign, and coupling-channel structure
#
# This class reconstructs observables by explicitly looping over
# nonzero HEOM components.
#
# ---------------------------------------------------------------------
#
# CURRENT OPERATOR
#
# Lead currents are computed from first-tier ADO contributions
# (tier_index == 1), which encode system–bath particle exchange.
#
# The current expression depends on:
#
#   - molecule–lead coupling V_km(x)
#   - fermionic operator matrix elements d_ops
#   - sign structure of creation/annihilation channels
#
# ---------------------------------------------------------------------
#
# USAGE IN SHEOM MAIN LOOP
#
# At each time step and for each trajectory:
#
#     observables = quantum_observables_class(
#         sparse_heom_ingredients,
#         molecular_system_ingredients,
#         projected_0,
#         projected_1
#     )
#
#     current, rho_mol, populations, p01, p10 = \
#         observables.return_quantum_observables_this_x(
#             quantum_state_this_x,
#             el_lead_couplings_this_x
#         )
#
# These outputs are then ensemble-averaged across trajectories.
#
# ---------------------------------------------------------------------
#
# NOTES ON APPROXIMATION LEVEL
#
# - No additional physical approximation is introduced here.
# - All observables are linear functionals of the HEOM state.
# - Reconstruction assumes correct sparsity encoding from the HEOM generator.
#
# - Current evaluation is restricted to first-tier contributions,
#   consistent with standard HEOM current expressions.
#
# ---------------------------------------------------------------------
#
# OUTPUT SUMMARY
#
# return_quantum_observables_this_x returns:
#
#   current_this_x:
#       Lead-resolved particle currents
#
#   rho_mol_this_x:
#       Reconstructed molecular reduced density matrix
#
#   populations_this_x:
#       Electronic populations (diagonal of rho_mol)
#
#   partial_prob_current_0_to_1:
#       Diagnostic transition weight (0 → 1 channel)
#
#   partial_prob_current_1_to_0:
#       Diagnostic transition weight (1 → 0 channel)
#
# ---------------------------------------------------------------------
#
# This module is intentionally low-level and explicit, reflecting the
# structure of the sparse HEOM representation without abstraction.
#
# It is designed for expert-level inspection and debugging of the
# quantum–classical coupling pipeline.
#
# ---------------------------------------------------------------------

from source.input_parameters import *

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