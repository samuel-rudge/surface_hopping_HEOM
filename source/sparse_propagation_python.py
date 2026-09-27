import numpy as np
from scipy.sparse import csr_matrix,coo_matrix

class sparse_propagator:

    def __init__(self, pair_info_row, pair_info_col, pair_values, nnz_elements):
        """
        Initialize the sparse matrix with given structure and initial values.
        """
        self.nnz_elements = nnz_elements
        coo = coo_matrix((pair_values, (pair_info_row, pair_info_col)),
                            shape=(nnz_elements, nnz_elements),dtype=np.float64)
        self.coo_sorted = coo.tocsr().tocoo()
        self.csr_sort_index = np.lexsort((self.coo_sorted.col, self.coo_sorted.row))  # match CSR .data order
        self.coo_sort_order = np.lexsort((pair_info_col, pair_info_row))
        # This is your fixed sparse matrix
        self.sparse_mat = self.coo_sorted.tocsr()
        # self.csr_data = self.sparse_mat
        # self.sparse_mat = coo_matrix((pair_values, (pair_info_row, pair_info_col)),
        #                             shape=(nnz_elements, nnz_elements),dtype=np.float64).tocsr()
        self.sparse_data = self.sparse_mat.data
        # self.sparse_mat = csr_matrix((pair_values, (pair_info_row, pair_info_col)),
        #                             shape=(nnz_elements, nnz_elements),dtype=np.float64)
    
    def return_sparse_matrix(self):

        return self.sparse_mat

    def update_values(self, new_pair_values):
        """
        Update only the nonzero values of the sparse matrix.
        """
        if len(new_pair_values) != len(self.sparse_mat.data):
            raise ValueError("New values length does not match sparse matrix nnz count.")
        
        self.sparse_data[:] = np.asarray(new_pair_values)[self.coo_sort_order][self.csr_sort_index]
        # coo_check = coo_matrix((np.asarray(new_pair_values)[self.coo_sort_order], (self.coo_sorted.row, self.coo_sorted.col)))
        # coo_ref = self.coo_sorted

        # if not np.allclose(coo_check.toarray(), coo_ref.toarray(),rtol=1e0, atol=1e0):
        #     print(" Structural mismatch in sparse matrix update!")
        # self.sparse_calculate = self.sparse_mat.toarray()

    def rho_derivative(self,rho_input,rho_deriv):
        """
        Calculate time-derivatve as d/dt \rho = L \rho
        """
        rho_deriv[:] = self.sparse_mat.dot(rho_input)
        return rho_deriv

    def propagate(self, dt, rho_input, max_expan_order, rk_coeff, rho_temp, rho_output, rho_deriv):
        """
        Perform one step propagation using the Runge-Kutta-like expansion.
        """
        
        rho_temp[:] = rho_input
        rho_output[:] = rho_input
        # rho_deriv[:] = 0.0

        for itrl in range(max_expan_order):
            # rho_deriv[:] = self.sparse_calculate.dot(rho_temp)*dt  # Efficient: write directly to rho_deriv
            rho_deriv[:] = self.sparse_mat.dot(rho_temp)  # Efficient: write directly to rho_deriv
            rho_deriv *= dt
            rho_output += rk_coeff[0,itrl] * rho_deriv
            rho_temp[:] = rho_deriv  # safe after output is updated

        return rho_output
