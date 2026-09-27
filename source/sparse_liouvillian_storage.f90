
!-----------------------------------------------------------------------
!
!  MODULE: sparse_matrix_storage
!
!  PURPOSE:
!  --------
!  This module provides persistent storage and initialization utilities
!  for the sparse representation of the HEOM Liouvillian (or related
!  operators) using Intel MKL sparse BLAS data structures.
!
!  The sparse operator is initially constructed in COO format from
!  precomputed index/value data and then converted to CSR format for
!  efficient numerical linear algebra operations (e.g. matrix-vector
!  products during propagation).
!
!
!  ROLE IN WORKFLOW:
!  -----------------
!  This module is used downstream of:
!
!      generate_quantum_heom_class.py
!          -> return_sparse_heom_ingredients()
!          -> return_sparse_heom_one_x(...)
!
!  where:
!      - pair_info_row / pair_info_col define sparse structure
!      - pair_values define numerical entries (at a given x)
!
!  The resulting CSR matrix is used for repeated HEOM propagation steps
!  where efficient sparse matrix-vector multiplication is required.
!
!
!  DATA STRUCTURE:
!  ---------------
!  The sparse matrix is internally stored as:
!
!      sparse_handle_csr : MKL SPARSE_MATRIX_T
!
!  after conversion from a temporary COO representation.
!
!  The sparsity pattern is assumed fixed after initialization.
!
!
!  GLOBAL STATE:
!  ------------
!  is_initialized
!      Logical flag ensuring that sparse structure is built only once.
!
!  sparse_handle_csr
!      Persistent MKL CSR sparse matrix handle used in all subsequent
!      linear algebra operations.
!
!
!  INPUT FORMAT:
!  ------------
!  The initializer expects sparse data in COO-like form:
!
!      nnz_elements
!          Dimension of the square Liouvillian (rows = cols)
!
!      npairs
!          Number of nonzero entries in sparse representation
!
!      pair_info_row(npairs)
!      pair_info_col(npairs)
!          Row and column indices (0-based indexing)
!
!      pair_values(npairs)
!          Numerical values of sparse entries at a fixed nuclear
!          coordinate x (or reference configuration)
!
!
!  OUTPUT:
!  -------
!  sparse_handle_csr (module global)
!      Optimized MKL CSR sparse matrix handle ready for:
!
!          - sparse matrix-vector products
!          - time propagation routines
!
!
!  INITIALIZATION PROCEDURE:
!  ------------------------
!  Subroutine:
!
!      initialize_sparse_matrix(nnz_elements, npairs,
!                               pair_info_row, pair_info_col,
!                               pair_values)
!
!  Steps:
!      1. Create COO sparse matrix from input triplets
!      2. Convert COO → CSR format for MKL efficiency
!      3. Destroy temporary COO handle
!      4. Optimize CSR structure for repeated operations
!      5. Store result in persistent module variable
!
!
!  IMPORTANT ASSUMPTIONS:
!  ---------------------
!  - Sparsity pattern is fixed and does not change during propagation
!  - Only numerical values (not indices) vary with nuclear coordinate x
!  - Initialization is performed once before time propagation
!
!
!  PERFORMANCE NOTE:
!  -----------------
!  CSR conversion and MKL optimization are expensive operations and are
!  therefore executed only once at setup time. Subsequent dynamics rely
!  exclusively on the prebuilt CSR handle.
!
!-----------------------------------------------------------------------

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