
module sparse_matrix_storage
    
    use mkl_spblas

    implicit none

    TYPE(SPARSE_MATRIX_T), save :: sparse_handle_csr !f2py ignore
    logical, save :: is_initialized = .false.
    TYPE(MATRIX_DESCR) :: descra !f2py ignore

contains

    subroutine initialize_sparse_matrix(nnz_elements, npairs, pair_info_row, pair_info_col, pair_values)
        
        integer, intent(in) :: nnz_elements, npairs
        integer, intent(in), dimension(0:npairs-1) :: pair_info_row, pair_info_col
        double precision, intent(in), dimension(0:npairs-1) :: pair_values

        TYPE(SPARSE_MATRIX_T) :: sparse_handle_coo
        integer :: ik_coo, ik_csr, ik_destroy_coo

        descra % TYPE = SPARSE_MATRIX_TYPE_GENERAL

        if (.not. is_initialized) then
            ik_coo = mkl_sparse_d_create_coo(sparse_handle_coo, SPARSE_INDEX_BASE_ZERO, nnz_elements, nnz_elements, &
                                             npairs, pair_info_row, pair_info_col, pair_values)
            if (ik_coo /= 0) print *, "Error creating COO matrix: ", ik_coo

            ik_csr = mkl_sparse_convert_csr(sparse_handle_coo, SPARSE_OPERATION_NON_TRANSPOSE, sparse_handle_csr)
            if (ik_csr /= 0) print *, "Error converting to CSR: ", ik_csr

            ik_destroy_coo = mkl_sparse_destroy(sparse_handle_coo)

            ik_csr = MKL_SPARSE_OPTIMIZE(sparse_handle_csr)
            if (ik_csr /= 0) print *, "Error optimizing CSR: ", ik_csr

            is_initialized = .true.
        endif
        
    end subroutine initialize_sparse_matrix

end module sparse_matrix_storage