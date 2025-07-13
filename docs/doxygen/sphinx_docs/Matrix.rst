***************
Matrix Overview
***************

The ``csalt++`` matrix module provides a flexible, type-safe, and format-aware framework
for working with two-dimensional numerical data in C++. It is designed for engineering
applications that require efficient, adaptive storage and manipulation of large matrices,
including dense and sparse formats.

The library is built around a polymorphic architecture with a shared base class, 
``MatrixBase<T>``, from which format-specific matrix types inherit. This design allows 
developers to write high-level algorithms without binding to a specific storage format.

Supported matrix types include:

* **DenseMatrix<T>** — Row-major storage of all matrix elements with optional initialization tracking
* **COOMatrix<T>** — Sparse storage using coordinate (triplet) format
* **CSRMatrix<T>** — Compressed Sparse Row format optimized for matrix-vector products and row slicing
* **Matrix<T>** - A wrapper class that selects the appropriate underlying class based on matrix sparsity

Key Features
============

* Format-specific classes with common interface support (e.g., ``get()``, ``set()``, ``rows()``, ``cols()``)
* SIMD-accelerated operations for supported types (e.g., ``float``, ``double``)
* Automatic bounds checking and initialization safety
* Support for element-wise and matrix multiplication
* Determinant and inverse computations (for dense matrices)
* Easy expansion for other storage formats (e.g., ELL, DIA, Block)

Use Cases
=========

The ``csalt++`` matrix types are ideal for:

* Solving large systems of equations in scientific computing
* Performing linear algebra operations in numerical simulations
* Representing sparsely populated matrices in PDE/FEM applications
* Dynamically switching between storage formats based on matrix sparsity

MatrixBase<T> 
=============
The ``MatrixBase<T>`` class is a template class that can be applied to any data 
type. This is an abstract base class providing the contract for the 
:ref:`dense_matrix`, :ref:`sparsecoo_matrix`, and :ref:`sparsecsr_matrix` classes.
While not generally intended for public use, it can be extended by users who wish
to implement custom matrix formats.

.. doxygenclass:: slt::MatrixBase
   :project: csalt++
   :members:
   :undoc-members:

.. _dense_matrix:

DenseMatrix<T>
==============

.. doxygenclass:: slt::DenseMatrix
   :project: csalt++

Constructors
------------

DenseMatrix(std::size_t, std::size_t, T)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenfunction:: slt::DenseMatrix::DenseMatrix(std::size_t, std::size_t, T)
   :project: csalt++

DenseMatrix(std::size_t, std::size_t)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenfunction:: slt::DenseMatrix::DenseMatrix(std::size_t, std::size_t)
   :project: csalt++

DenseMatrix(std:vector<std::vector<T>>& vec)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenfunction:: slt::DenseMatrix::DenseMatrix(const std::vector<std::vector<T>>&)
   :project: csalt++

DenseMatrix(const std::array<std::array<T, Cols>, Rows& arr)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenfunction:: slt::DenseMatrix::DenseMatrix(const std::array<std::array<T, Cols>, Rows>&)
   :project: csalt++

DenseMatrix(std::initializer_list<std::initializer_list<T>> init_list)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenfunction:: slt::DenseMatrix::DenseMatrix(std::initializer_list<std::initializer_list<T>>)
   :project: csalt++

DenseMatrix(const std::vector<T>& flat_data, std::size_t r, std::size_t c)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenfunction:: slt::DenseMatrix::DenseMatrix(const std::vector<T>&, std::size_t, std::size_t)
   :project: csalt++

.. cpp:function:: slt::DenseMatrix::DenseMatrix(std::vector<T>&& flat_data, std::size_t r, std::size_t c)

   Constructs a ``DenseMatrix<T>`` by moving a flat ``std::vector<T>`` in row-major order.

   Transfers ownership of the flat vector into the matrix, avoiding extra memory copies.
   The input vector must contain exactly ``r * c`` elements.

   :param flat_data: Rvalue reference to a ``std::vector<T>`` in row-major order.
   :param r: Number of rows.
   :param c: Number of columns.
   :throws std::invalid_argument: If ``flat_data.size() != r * c``.

   **Example**::

      std::vector<float> flat = {
          1.0f, 2.0f, 3.0f,
          4.0f, 5.0f, 6.0f
      };

      slt::DenseMatrix<float> mat(std::move(flat), 2, 3);

      // mat(0,0) == 1.0f
      // mat(1,2) == 6.0f
      // flat is now empty

DenseMatrix(const std::array<T, N>& arr, std::size_t r, std::size_t c)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenfunction:: slt::DenseMatrix::DenseMatrix(const std::array<T, N>&, std::size_t, std::size_t)
   :project: csalt++

DenseMatrix Copy Constructor 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenfunction:: slt::DenseMatrix::DenseMatrix(const DenseMatrix<T>&)
   :project: csalt++

.. cpp:function:: explicit slt::DenseMatrix::DenseMatrix(const slt::SparseCOOMatrix<T>& sparse)

   Constructs a ``DenseMatrix<T>`` from a ``SparseCOOMatrix<T>``.

   This initializes only the positions represented in the sparse triplet list as valid.  
   Remaining positions are left uninitialized (``is_initialized() == false``).

   Useful for converting sparse representations to full dense storage for further numerical operations.

   :param sparse: Source ``SparseCOOMatrix<T>`` to convert.

   **Example**::

      slt::SparseCOOMatrix<float> sparse(3, 3);
      sparse.set(0, 1, 5.0f);
      sparse.set(2, 2, 3.0f);

      slt::DenseMatrix<float> dense(sparse);

      EXPECT_FLOAT_EQ(dense(0, 1), 5.0f);
      EXPECT_FLOAT_EQ(dense(2, 2), 3.0f);
      EXPECT_FALSE(dense.is_initialized(0, 0));

.. cpp:function:: slt::DenseMatrix::DenseMatrix(const slt::SparseCSRMatrix<T>& csr)

   Constructs a dense matrix from a compressed sparse row (CSR) matrix.

   This constructor creates a new :cpp:class:`DenseMatrix<T>` object by copying
   all explicitly initialized (non-zero) elements from a
   :cpp:class:`SparseCSRMatrix<T>` input. Internally, it populates a dense
   row-major storage layout, initializing each non-zero element to its value from
   the CSR matrix, and marking the corresponding position as initialized.

   Uninitialized entries (i.e., those not explicitly stored in the CSR matrix)
   are set to ``T{}`` and marked as uninitialized. The total matrix dimensions
   are preserved.

   **Requirements:**

   - The template parameter ``T`` must be either ``float`` or ``double``.
   - The input matrix must have valid CSR indexing (monotonic, non-overlapping).

   :param csr: A reference to the input :cpp:class:`SparseCSRMatrix<T>` object.
   :type csr: const SparseCSRMatrix<T>&

   :raises std::bad_alloc: If memory allocation fails during construction.
   :raises std::out_of_range: If invalid CSR indexing leads to an invalid access.

   **Example:**

   .. code-block:: cpp

      slt::SparseCSRMatrix<float> csr(3, 3);
      // Assume csr is populated with valid values...

      slt::DenseMatrix<float> dense(csr);

      assert(dense.get(0, 1) == 1.5f);  // Example value
      assert(!dense.is_initialized(2, 2));  // Uninitialized position

   :note:
      The resulting dense matrix uses row-major layout with separate
      initialization tracking via a parallel ``std::vector<uint8_t>``.

.. cpp:function:: DenseMatrix<T>& DenseMatrix::operator=(const SparseCSRMatrix<T>& csr)

   Assignment operator that fills a :cpp:class:`DenseMatrix` using values from a
   :cpp:class:`SparseCSRMatrix`.

   This overload copies the contents of the sparse matrix into a full 2D dense
   array layout. Entries in the CSR matrix are directly mapped into their row/column
   positions, and all remaining entries are initialized to zero.

   :param csr: Input matrix in CSR format.
   :type csr: const SparseCSRMatrix<T>&
   :returns: A reference to the updated DenseMatrix.
   :rtype: DenseMatrix<T>&
   :throws: ``std::out_of_range`` if invalid indexing occurs internally.

   **Example:**

   .. code-block:: cpp

      slt::DenseMatrix<float> dense = {{1.0f, 2.0f}, {3.0f, 4.0f}};
      slt::SparseCSRMatrix<float> csr = dense;
      slt::DenseMatrix<float> newDense = csr;

DenseMatrix Move Constructor 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenfunction:: slt::DenseMatrix::DenseMatrix(DenseMatrix<T>&&)
   :project: csalt++

.. cpp:function:: explicit DenseMatrix(SparseCOOMatrix<T>&& sparse)

   Constructs a dense matrix by moving the contents of a sparse matrix
   in Coordinate List (COO) format.

   This constructor transfers all non-zero entries from the sparse matrix into
   their corresponding positions in the dense matrix. All remaining entries are
   set to zero. After the conversion, the sparse matrix is cleared.

   :param sparse: A sparse matrix (in COO format) whose contents will be moved into the dense matrix.
   :type sparse: SparseCOOMatrix<T>&&
   :raises std::out_of_range: If any row or column index is outside matrix bounds.

   .. note::

      This constructor assumes that ``SparseCOOMatrix<T>`` supports iteration
      using range-based for loops, and that each element yields ``row``, ``col``,
      and ``value`` members.

   **Example**

   .. code-block:: cpp

      SparseCOOMatrix<double> sparse(3, 3);
      sparse.insert(0, 1, 4.5);
      sparse.insert(2, 0, -1.2);

      DenseMatrix<double> dense(std::move(sparse));

      std::cout << dense.get(0, 1);  // Prints 4.5
      std::cout << dense.get(2, 0);  // Prints -1.2
      // dense.get(1, 1); would throw if uninitialized access is disallowed

.. cpp:function:: DenseMatrix(SparseCSRMatrix<T>&& csr)

   Move constructor that converts a ``SparseCSRMatrix<T>`` to a ``DenseMatrix<T>``.

   This constructor takes ownership of a sparse matrix and creates a dense
   matrix representation with the same dimensions. All initialized elements in
   the CSR matrix are transferred to their corresponding locations in the dense
   matrix. Any uninitialized entries remain zeroed in the dense output.

   :param csr: A sparse matrix in CSR format to be converted to dense form.
   :type csr: SparseCSRMatrix<T>&&
   :throws: ``std::out_of_range`` if index bounds are violated internally.
   :note: The CSR matrix is left in a logically empty state (rows and cols set to 0).

   **Example:**

   .. code-block:: cpp

      slt::SparseCSRMatrix<float> csr = create_sparse_matrix();
      slt::DenseMatrix<float> dense(std::move(csr));

DenseMatrix Identify Constructor 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenfunction:: slt::DenseMatrix::DenseMatrix(std::size_t)
   :project: csalt++

Operator Overloads 
------------------

operator=
~~~~~~~~~

.. doxygenfunction:: slt::DenseMatrix::operator=(const DenseMatrix<T>&)
   :project: csalt++

.. doxygenfunction:: slt::DenseMatrix::operator=(DenseMatrix<T>&&) noexcept
   :project: csalt++

.. cpp:function:: slt::DenseMatrix& slt::DenseMatrix::operator=(const slt::SparseCOOMatrix<T>& sparse)

   Assigns a ``SparseCOOMatrix<T>`` into an existing ``DenseMatrix<T>``.

   This clears the current DenseMatrix and fills it with values from the sparse matrix.  
   Only stored triplet entries are initialized; other positions remain uninitialized.

   :param sparse: Source ``SparseCOOMatrix<T>`` to assign from.
   :returns: Reference to ``*this``.

   **Example**::

      slt::SparseCOOMatrix<float> sparse(2, 2);
      sparse.set(1, 0, 4.5f);

      slt::DenseMatrix<float> dense(2, 2);
      dense = sparse;

      EXPECT_FLOAT_EQ(dense(1, 0), 4.5f);
      EXPECT_FALSE(dense.is_initialized(0, 0));

.. cpp:function:: DenseMatrix<T>& operator=(SparseCOOMatrix<T>&& sparse)

   Move-assigns a sparse COO matrix to a dense matrix by converting its non-zero
   entries into the dense layout. All values in the sparse matrix are transferred
   using move semantics, and the sparse matrix is cleared after assignment.

   If the dimensions of the sparse matrix differ from the target matrix,
   the dense matrix is resized. Otherwise, its internal contents are cleared
   and reused.

   :param sparse: A COO-format sparse matrix to be moved into this dense matrix.
   :type sparse: SparseCOOMatrix<T>&&
   :returns: Reference to the updated DenseMatrix object
   :rtype: DenseMatrix<T>&
   :raises: std::out_of_range if any row or column index is outside matrix bounds

   .. warning::

      This operation clears the input sparse matrix and leaves it in an
      uninitialized state.

   **Example**

   .. code-block:: cpp

      slt::SparseCOOMatrix<double> coo(2, 2);
      coo.set(0, 1, 3.14);
      coo.set(1, 0, -2.0);

      slt::DenseMatrix<double> mat(2, 2);
      mat = std::move(coo);  // Transfer and convert

      std::cout << mat.get(0, 1);  // Prints 3.14



operator()
~~~~~~~~~~

.. cpp:function:: T& DenseMatrix::operator()(std::size_t r, std::size_t c)

   Access or assign a value at the specified matrix index ``(r, c)``.

   This non-const overload allows users to assign a value to an element. If the
   element has not been previously initialized (tracked via the internal ``init`` vector),
   it will be marked as initialized. If already initialized, it acts as a regular update.

   Bounds checking is performed. If the index is out of range, ``std::out_of_range`` is thrown.

   :param r: Row index
   :param c: Column index
   :return: Reference to the value at the specified index
   :throws std::out_of_range: If the index is out of bounds

   **Example:**

   .. code-block:: cpp

      slt::DenseMatrix<float> mat(2, 3);
      mat(0, 1) = 4.2f;  // Initializes and sets the value
      mat(0, 1) = 5.0f;  // Updates existing value
      std::cout << mat(0, 1);  // Outputs: 5.0

.. cpp:function:: const T& DenseMatrix::operator()(std::size_t r, std::size_t c) const

   Read-only access to a matrix element at ``(r, c)``.

   This const overload allows read-only access to a matrix element.
   Throws a ``std::runtime_error`` if the element has not been initialized
   via ``set()``, ``operator()``, or ``update()``.

   Bounds checking is performed. If the index is out of range, ``std::out_of_range`` is thrown.

   :param r: Row index
   :param c: Column index
   :return: Const reference to the initialized value
   :throws std::runtime_error: If the element has not been initialized
   :throws std::out_of_range: If the index is out of bounds

   **Example:**

   .. code-block:: cpp

      slt::DenseMatrix<float> mat(2, 3);
      mat.set(1, 2, 8.5f);
      std::cout << mat(1, 2);  // Outputs: 8.5

      // mat(0, 0);  // Would throw std::runtime_error since it's uninitialized

operator+
~~~~~~~~~

.. doxygenfunction:: slt::DenseMatrix::operator+(const DenseMatrix& other) const 
   :project: csalt++

.. doxygenfunction:: slt::DenseMatrix::operator+(T scalar) const 
   :project: csalt++

operator-
~~~~~~~~~

.. doxygenfunction:: slt::DenseMatrix::operator-(const DenseMatrix& other) const 
   :project: csalt++

.. doxygenfunction:: slt::DenseMatrix::operator-(T scalar) const 
   :project: csalt++

operator*
~~~~~~~~~

.. doxygenfunction:: slt::DenseMatrix::operator*(const DenseMatrix& other) const 
   :project: csalt++

.. doxygenfunction:: slt::DenseMatrix::operator*(T scalar) const 
   :project: csalt++

operator/
~~~~~~~~~

.. doxygenfunction:: slt::DenseMatrix::operator/(T scalar) const 
   :project: csalt++

operator<<
~~~~~~~~~~

.. cpp:function:: template <typename T> std::ostream& operator<<(std::ostream& os, const slt::DenseMatrix<T>& mat)

   Stream output operator for ``DenseMatrix<T>``.

   Prints the contents of the matrix in row-major order:

   - Initialized values are printed numerically.
   - Uninitialized values are printed as "." (dot).

   :tparam T: Element type (``float`` or ``double``).
   :param os: Output stream (e.g., ``std::cout``).
   :param mat: Dense matrix to print.
   :returns: Reference to the output stream.

   **Example**::

      slt::DenseMatrix<float> mat(2, 2);
      mat.set(0, 0, 1.0f);
      mat.set(1, 1, 2.0f);

      std::cout << mat;

      // Output:
      // 1.0 .
      // .   2.0


Data Access Methods 
-------------------

size()
~~~~~~

.. doxygenfunction:: slt::DenseMatrix::size
   :project: csalt++

begin()
~~~~~~~

.. doxygenfunction:: slt::DenseMatrix::begin() const
   :project: csalt++

.. doxygenfunction:: slt::DenseMatrix::begin()
   :project: csalt++

end()
~~~~~

.. doxygenfunction:: slt::DenseMatrix::end() const
   :project: csalt++

.. doxygenfunction:: slt::DenseMatrix::end()
   :project: csalt++

init_ptr()
~~~~~~~~~~

.. doxygenfunction:: slt::DenseMatrix::init_ptr() const
   :project: csalt++

initialized_count() 
~~~~~~~~~~~~~~~~~~~

.. doxygenfunction:: slt::DenseMatrix::initialized_count() const
   :project: csalt++

is_initialized()
~~~~~~~~~~~~~~~~

.. doxygenfunction:: slt::DenseMatrix::is_initialized
   :project: csalt++

rows()
~~~~~~
.. cpp:function:: std::size_t slt::DenseMatrix::rows() const

   Returns the number of rows in the matrix.

   This function is inherited from the base class ``MatrixBase`` and applies to all matrix types. 
   It provides the number of rows allocated for the matrix, regardless of whether they are fully or partially initialized.

   :returns: Number of rows in the matrix.
   :rtype: std::size_t

   **Example**::

      slt::DenseMatrix<float> mat(3, 4);
      std::cout << "Rows: " << mat.rows();  // Outputs: 3

cols()
~~~~~~
.. cpp:function:: std::size_t slt::DenseMatrix::cols() const

   Returns the number of columns in the matrix.

   This function is inherited from the base class ``MatrixBase`` and applies to all matrix types. 
   It provides the number of columns allocated for the matrix, regardless of how many elements are initialized.

   :returns: Number of columns in the matrix.
   :rtype: std::size_t

   **Example**::

      slt::DenseMatrix<float> mat(3, 4);
      std::cout << "Columns: " << mat.cols();  // Outputs: 4

get()
~~~~~

.. doxygenfunction:: slt::DenseMatrix::get
   :project: csalt++

Operations 
----------

inverse 
~~~~~~~

.. doxygenfunction:: slt::DenseMatrix::inverse() const 

tranpose()
~~~~~~~~~~

.. doxygenfunction:: slt::DenseMatrix::transpose

.. _sparsecoo_matrix:

set()
~~~~~

.. doxygenfunction:: slt::DenseMatrix::set
   :project: csalt++

update()
~~~~~~~~

.. doxygenfunction:: slt::DenseMatrix::update
   :project: csalt++

remove()
~~~~~~~~

.. doxygenfunction:: slt::DenseMatrix::remove
   :project: csalt++

clone()
~~~~~~~

.. doxygenfunction:: slt::DenseMatrix::clone
   :project: csalt++

clear()
~~~~~~~

.. cpp:function:: void slt::DenseMatrix::clear()

   Clears all contents of the dense matrix and resets its dimensions to zero.

   This method removes all data values and resets the matrix's internal shape
   (row and column count) to zero. The matrix becomes uninitialized after this
   call and must be redefined before it can be safely accessed.

   :raises: None

   .. warning::

      After calling this method, any further access to matrix elements will result
      in undefined behavior or runtime errors unless the matrix is properly rebuilt.

   **Example**

   .. code-block:: cpp

      slt::DenseMatrix<float> mat(3, 3);
      mat.set(0, 0, 1.0f);
      mat.set(1, 1, 2.0f);
      
      mat.clear();  // All data and dimensions are reset

      EXPECT_EQ(mat.rows(), 0);
      EXPECT_EQ(mat.cols(), 0);


SparseCOOMatrix<T>
==================

.. doxygenclass:: slt::SparseCOOMatrix
   :project: csalt++

See also: :ref:`triplet_class`

Constructors
------------

Identity Constructor
~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: slt::SparseCOOMatrix::SparseCOOMatrix(std::size_t n)

   Constructs an ``n x n`` sparse identity matrix.

   Only the diagonal elements ``(i, i)`` are stored in the COO format with value ``1.0``.
   Off-diagonal entries are implicitly zero.

   The matrix is returned in sorted form (``fast_set = false``), ready for efficient queries.

   :param n: The number of rows and columns (must be square).
   :returns: A sparse identity matrix of size ``n x n``.

   **Example**::

      slt::SparseCOOMatrix<float> I(4);
      // I(0,0) == 1.0, I(1,1) == 1.0, I(2,2) == 1.0, I(3,3) == 1.0

SparseCOOMatrix(std::size_t, std::size_t, std::size_t)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenfunction:: slt::SparseCOOMatrix::SparseCOOMatrix(std::size_t, std::size_t, std::size_t)
   :project: csalt++

SparseCOOMatrix(DenseMatrix<T>, bool)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: explicit slt::SparseCOOMatrix::SparseCOOMatrix(const slt::DenseMatrix<T>& dense, bool accept_zeros = true)

   Constructs a sparse COO matrix from an existing dense matrix.

   All initialized, non-zero elements of the dense matrix are copied into the sparse matrix.  
   Uninitialized and exact-zero values are skipped. The result is automatically sorted  
   (``fast_set = false``) for optimized access.

   :param dense: The source DenseMatrix<T> to convert.
   :param accept_zeros: Accepts 0 if true, rejects them if false.  Defaulted to true

   **Example**::

      slt::DenseMatrix<float> dense({
          {1.0f, 0.0f},
          {0.0f, 2.5f}
      });

      slt::SparseCOOMatrix<float> sparse(dense);

      EXPECT_EQ(sparse.initialized_count(), 2);
      EXPECT_FLOAT_EQ(sparse.get(0, 0), 1.0f);
      EXPECT_FLOAT_EQ(sparse.get(1, 1), 2.5f);


std::vector<slt::Triplet<T>>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: slt::SparseCOOMatrix::SparseCOOMatrix(std::size_t r, std::size_t c, const std::vector<slt::Triplet<T>>& triplets)

   Constructs a ``SparseCOOMatrix<T>`` from a ``std::vector< Triplet<T> >``.

   The resulting matrix is initialized with the provided triplets.  
   The internal storage is automatically sorted in row-major order (row, then column),  
   and the matrix is ready for optimized access (``fast_set = false``).

   :param r: Number of rows in the matrix.
   :param c: Number of columns in the matrix.
   :param triplets: ``std::vector`` of triplet values to insert.

   **Example**::

      std::vector<slt::Triplet<float>> triplets = {
          {0, 0, 1.0f},
          {1, 2, 2.5f},
          {4, 4, 3.1f}
      };

      slt::SparseCOOMatrix<float> mat(5, 5, triplets);

.. cpp:function:: slt::SparseCOOMatrix::SparseCOOMatrix(std::size_t r, std::size_t c, std::vector<slt::Triplet<T>>&& triplets)

   Constructs a ``SparseCOOMatrix<T>`` from an rvalue ``std::vector< Triplet<T> >`` (move).

   Moves the contents of the given vector into the matrix to avoid unnecessary copying.
   This is the most efficient way to initialize a large sparse matrix from a temporary
   or intermediate vector of triplets. After the move, the input vector will be empty.

   The triplets are automatically sorted in row-major order (row first, then column),  
   and the matrix is ready for optimized access (``fast_set = false``).

   :param r: Number of rows in the matrix.
   :param c: Number of columns in the matrix.
   :param triplets: Rvalue reference to a vector of triplet values to move into the matrix.

   **Note:** The original vector passed in will be left empty after construction.

   **Example**::

      std::vector<slt::Triplet<float>> triplets = {
          {0, 0, 1.0f},
          {1, 2, 2.5f},
          {4, 4, 3.1f}
      };

      // Efficient move construction
      slt::SparseCOOMatrix<float> mat(5, 5, std::move(triplets));

      // After this, triplets.size() == 0

std::array<Triplet<T>, N>
~~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: template<std::size_t N> slt::SparseCOOMatrix<T>::SparseCOOMatrix(std::size_t r, std::size_t c, const std::array<slt::Triplet<T>, N>& triplets)

   Constructs a ``SparseCOOMatrix<T>`` from a fixed-size ``std::array< Triplet<T>, N >``.

   The matrix is initialized with the given triplets and automatically sorted in row-major order.

   :tparam N: Number of triplets.
   :param r: Number of rows in the matrix.
   :param c: Number of columns in the matrix.
   :param triplets: Array of triplet values to insert.

   **Example**::

      std::array<slt::Triplet<double>, 2> triplets = {
          slt::Triplet<double>(0, 1, 3.14),
          slt::Triplet<double>(2, 2, 2.71)
      };

      slt::SparseCOOMatrix<double> mat(3, 3, triplets);

C-style array
~~~~~~~~~~~~~

.. cpp:function:: slt::SparseCOOMatrix::SparseCOOMatrix(std::size_t r, std::size_t c, const slt::Triplet<T>* triplets, std::size_t count)

   Constructs a ``SparseCOOMatrix<T>`` from a C-style array of ``Triplet<T>``.

   Useful for integrating with legacy code or static data.  
   The matrix is sorted automatically after construction.

   :param r: Number of rows in the matrix.
   :param c: Number of columns in the matrix.
   :param triplets: Pointer to an array of ``Triplet<T>``.
   :param count: Number of triplets in the array.

   **Example**::

      slt::Triplet<float> triplets[] = {
          {0, 0, 1.0f},
          {2, 3, 4.5f}
      };

      slt::SparseCOOMatrix<float> mat(4, 4, triplets, 2);

Initializer list
~~~~~~~~~~~~~~~~

.. cpp:function:: slt::SparseCOOMatrix::SparseCOOMatrix(std::size_t r, std::size_t c, std::initializer_list<slt::Triplet<T>> init_list)

   Constructs a ``SparseCOOMatrix<T>`` from an initializer list of ``Triplet<T>``.

   Allows convenient inline initialization using brace-enclosed lists.  
   The matrix is sorted automatically after construction.

   :param r: Number of rows in the matrix.
   :param c: Number of columns in the matrix.
   :param init_list: List of triplets to insert.

   **Example**::

      slt::SparseCOOMatrix<float> mat(4, 4, {
          {0, 1, 1.5f},
          {2, 0, 3.0f},
          {3, 3, 2.0f}
      });

Copy constructor
~~~~~~~~~~~~~~~~

.. cpp:function:: slt::SparseCOOMatrix::SparseCOOMatrix(const slt::SparseCOOMatrix<T>& other)

   Constructs a new ``SparseCOOMatrix<T>`` as a deep copy of the provided matrix.

   All internal data structures (values, row/column indices, flags) are duplicated,  
   preserving the state of the original matrix while ensuring full independence.

   :param other: The ``SparseCOOMatrix<T>`` instance to copy.

   .. note:: This performs a deep copy. Changes to the new matrix will not affect the original.

   **Example**::

      slt::SparseCOOMatrix<float> mat(4, 4, {
          {0, 1, 1.5f},
          {2, 0, 3.0f},
          {3, 3, 2.0f}
      });

      slt::SparseCOOMatrix<float> new_mat(mat); // Copies mat to new_mat

.. cpp:function:: SparseCOOMatrix(const SparseCSRMatrix<T>& csr)

  Constructs a coordinate list (COO) sparse matrix from a CSR-format matrix.
  This conversion expands the compressed row storage into explicit triplets
  (row, column, value) for each non-zero element.

  :param csr: A constant reference to the source CSR matrix.
  :throws: std::bad_alloc if memory allocation fails.

  **Example**::

     slt::SparseCSRMatrix<float> csr = ...;
     slt::SparseCOOMatrix<float> coo(csr);


Move constructor
~~~~~~~~~~~~~~~~

.. cpp:function:: slt::SparseCOOMatrix::SparseCOOMatrix(slt::SparseCOOMatrix<T>&& other)

   Move constructor — constructs a new sparse matrix by transferring ownership of data  
   from another matrix.

   This performs a shallow move of internal vectors and resets the source matrix to default state.

   :param other: The matrix to move from. After the operation, ``other`` is empty.

   .. note:: The ``fast_set`` flag is also transferred and reset in the source.

   **Example**::

      slt::SparseCOOMatrix<float> mat(4, 4, {
          {0, 1, 1.5f},
          {2, 0, 3.0f},
          {3, 3, 2.0f}
      });

      slt::SparseCOOMatrix<float> new_mat(std::move(mat)); // Moves contents of mat to new_mat

.. cpp:function:: SparseCOOMatrix(DenseMatrix<T>&& dense, bool accept_zeros = true)

   Constructs a sparse COO matrix by moving data from a dense matrix.

   All initialized values in the dense matrix are transferred as triplets into the
   sparse COO matrix. If ``accept_zeros`` is set to ``false``, any explicitly
   initialized zeros are excluded from the resulting sparse matrix. The input
   dense matrix is cleared after the conversion.

   :param dense: The dense matrix to convert from.
   :type dense: DenseMatrix<T>&&
   :param accept_zeros: Whether to include explicitly zero-valued entries.
   :type accept_zeros: bool, default = true
   :raises: None

   .. warning::

      After construction, the dense matrix is cleared and should not be reused.

   **Example**

   .. code-block:: cpp

      slt::DenseMatrix<float> mat(2, 2);
      mat.set(0, 0, 1.0f);
      mat.set(1, 1, 0.0f);  // Initialized to zero

      slt::SparseCOOMatrix<float> coo(std::move(mat), false);

      // Only one triplet will exist since accept_zeros = false

.. cpp:function:: SparseCOOMatrix(SparseCSRMatrix<T>&& csr)

   Move constructor that converts a :cpp:class:`SparseCSRMatrix` into a :cpp:class:`SparseCOOMatrix`.

   This constructor transfers ownership of all initialized entries in the CSR matrix
   and re-expresses them in COO format using a vector of triplets. The resulting COO matrix
   preserves the row-wise ordering from the CSR layout.

   :param csr: A sparse matrix in CSR format (rvalue reference).
   :type csr: :cpp:expr:`SparseCSRMatrix<T>&&`

   :throws: ``std::bad_alloc`` if memory allocation fails during triplet construction.

   :note: After the move, the source CSR matrix is left in a logically empty state
          (i.e., rows and cols are set to 0 and internal storage is cleared).

   **Example**

   .. code-block:: cpp

      slt::DenseMatrix<float> dense = {
          {1.0f, 0.0f},
          {0.0f, 2.0f}
      };

      slt::SparseCSRMatrix<float> csr(dense);
      slt::SparseCOOMatrix<float> coo(std::move(csr));


Operator Overloads 
------------------

operator()
~~~~~~~~~~

.. cpp:function:: const T& SparseCOOMatrix::operator()(std::size_t r, std::size_t c) const

   Read-only access to a matrix element at ``(r, c)``.

   This const overload allows read-only access to a matrix element.
   Throws a ``std::runtime_error`` if the element has not been initialized
   via ``set()``, ``operator()``, or ``update()``.

   Bounds checking is performed. If the index is out of range, ``std::out_of_range`` is thrown.

   :param r: Row index
   :param c: Column index
   :return: Const reference to the initialized value
   :throws std::runtime_error: If the element has not been initialized
   :throws std::out_of_range: If the index is out of bounds

   **Example:**

   .. code-block:: cpp

      slt::SparseCOOMatrix<float> mat(2, 3);
      mat.set(1, 2, 8.5f);
      std::cout << mat(1, 2);  // Outputs: 8.5

      // mat(0, 0);  // Would throw std::runtime_error since it's uninitialized

operator+
~~~~~~~~~

.. cpp:function:: DenseMatrix<T> SparseCOOMatrix::operator+(const SparseCOOMatrix& other) const

   Adds two sparse matrices element-wise and returns the result as a dense matrix.

   Performs element-wise addition of two matrices in sparse COO format.  
   The result is returned as a ``DenseMatrix<T>`` to ensure full representation  
   of potential non-zero values in the output.

   Both matrices must have identical dimensions.  
   If either matrix contains a non-zero value at a given (row, col),  
   the result will include that value.

   Internally, values are added using a nested loop and a temporary dense buffer.  
   This operation is not optimized for SIMD or sparsity-aware acceleration,  
   but is functionally correct and safe.

   :param other: The sparse matrix to add.
   :return: A ``DenseMatrix<T>`` containing the result of the element-wise addition.
   :throws std::invalid_argument: if the matrix dimensions do not match.

   .. note::

      This implementation uses full dense representation for the result,  
      even if the result remains sparse.  
      Use a future ``to_sparse_sum()`` method if you want a sparse result.

   **Example**::

      slt::SparseCOOMatrix<float> A(2, 2, {
          {0, 0, 1.0f},
          {1, 1, 2.0f}
      });

      slt::SparseCOOMatrix<float> B(2, 2, {
          {0, 1, 3.0f},
          {1, 0, 4.0f}
      });

      slt::DenseMatrix<float> result = A + B;
      // result: [[1.0, 3.0], [4.0, 2.0]]

.. cpp:function:: SparseCOOMatrix SparseCOOMatrix::operator+(T scalar) const

   Adds a scalar to each non-zero element of the sparse matrix.

   Each stored value in the COO matrix has the scalar added to it.  
   This preserves the sparsity pattern — zero elements not explicitly stored remain unchanged.

   :param scalar: Scalar value to add.
   :return: A new ``SparseCOOMatrix<T>`` with updated values.

   **Example**::

      slt::SparseCOOMatrix<float> A(2, 2, {
          {0, 0, 1.0f},
          {1, 1, 2.0f}
      });

      auto result = A + 1.0f;
      // result: {{2.0f, 0.0f}, {0.0f, 3.0f}}

operator-
~~~~~~~~~

.. cpp:function:: DenseMatrix slt::SparseCOOMatrix::operator-(const slt::SparseCOOMatrix<T>& other) const

   Subtracts two sparse matrices element-wise and returns the result as a dense matrix.

   Performs element-wise subtraction of two matrices in sparse COO format. The result is returned  
   as a ``DenseMatrix<T>`` to ensure full representation of potential non-zero values in the output.

   Both matrices must have identical dimensions. If either matrix contains a non-zero value  
   at a given ``(row, col)`` index, the result will include that value.

   :param other: The sparse matrix to subtract.
   :return: A dense matrix containing the result of the element-wise subtraction.
   :throws std::invalid_argument: if the matrix dimensions do not match.

   .. note::

      This implementation uses full dense representation for the result, even if  
      the result remains sparse. A future ``to_sparse_difference()`` may provide a sparse result.

   **Example**::

      slt::SparseCOOMatrix<float> A(2, 2, {
          {0, 0, 1.0f},
          {1, 1, 2.0f}
      });

      slt::SparseCOOMatrix<float> B(2, 2, {
          {0, 1, 3.0f},
          {1, 0, 4.0f}
      });

      slt::DenseMatrix<float> result = A - B;
      // result: [[1.0, -3.0], [-4.0, 2.0]]

.. doxygenfunction:: slt::SparseCOOMatrix::operator-(T scalar) const 
   :project: csalt++

operator*
~~~~~~~~~

.. cpp:function:: slt::SparseCOOMatrix slt::SparseCOOMatrix::operator*(const SparseCOOMatrix<T>& other) const

   Performs element-wise (Hadamard) multiplication of two sparse matrices.

   Only non-zero entries present in both matrices at the same (row, col) position will appear in the result.

   :param other: The second sparse matrix to multiply with.
   :returns: A new ``SparseCOOMatrix<T>`` representing the element-wise product.
   :throws std::invalid_argument: if matrix dimensions do not match.

   **Example**::

      slt::SparseCOOMatrix<float> A(2, 2, {
          {0, 0, 1.0f},
          {0, 1, 2.0f}
      });

      slt::SparseCOOMatrix<float> B(2, 2, {
          {0, 0, 3.0f},
          {1, 1, 4.0f}
      });

      auto result = A * B;
      // result contains: (0,0) = 3.0

.. doxygenfunction:: slt::SparseCOOMatrix::operator*(T) const
   :project: csalt++

operator/ 
~~~~~~~~~

.. doxygenfunction:: slt::SparseCOOMatrix::operator/(T) const
   :project: csalt++

operator=
~~~~~~~~~

.. cpp:function:: SparseCOOMatrix& slt::SparseCOOMatrix::operator=(const SparseCOOMatrix<T>& other)

   Deep copy assignment operator.

   Copies all metadata and contents (rows, cols, triplet, etc.) from another SparseCOOMatrix.  
   The two matrices become fully independent after this operation.

   :param other: Source matrix to copy from.
   :return: Reference to this matrix.

   **Example**::

      slt::SparseCOOMatrix<float> A(3, 3, {
          {0, 0, 1.0f},
          {1, 1, 2.0f}
      });

      slt::SparseCOOMatrix<float> B(3, 3);
      B = A;

      assert(B == A);  // Deep copy — B is now equal to A

.. cpp:function:: SparseCOOMatrix& slt::SparseCOOMatrix::operator=(SparseCOOMatrix<T>&& other)

   Move assignment operator.

   Transfers resources from another SparseCOOMatrix, leaving the source in a valid but empty state.  
   Enables efficient transfer of large matrices without deep copying.

   :param other: Source matrix to move from.
   :return: Reference to this matrix.

   **Example**::

      slt::SparseCOOMatrix<float> A(3, 3, {
          {0, 0, 1.0f},
          {1, 1, 2.0f}
      });

      slt::SparseCOOMatrix<float> B(3, 3);

      B = std::move(A);

      assert(A.initialized_count() == 0);  // A is now empty
      assert(B(0, 0) == 1.0f);

.. cpp:function:: slt::SparseCOOMatrix& slt::SparseCOOMatrix::operator=(const slt::DenseMatrix<T>& dense)

   Assignment operator: replaces this ``SparseCOOMatrix<T>`` with the contents of a ``DenseMatrix<T>``.

   All initialized and non-zero elements from the dense matrix are copied into this sparse matrix.
   Zero and uninitialized values are skipped. Existing sparse data is cleared.

   The sparse matrix will match the shape of the input dense matrix after assignment.
   Triplets are sorted after assignment, and ``fast_set`` is set to ``false``.

   :param dense: The ``DenseMatrix<T>`` to assign from.
   :return: Reference to this ``SparseCOOMatrix<T>``.

   **Example**::

      slt::DenseMatrix<float> dense({
          {1.0f, 0.0f},
          {0.0f, 3.0f}
      });

      slt::SparseCOOMatrix<float> sparse(2, 2);
      sparse = dense;

      EXPECT_EQ(sparse.initialized_count(), 2);
      EXPECT_FLOAT_EQ(sparse.get(0, 0), 1.0f);
      EXPECT_FLOAT_EQ(sparse.get(1, 1), 3.0f);


.. cpp:function:: SparseCOOMatrix<T>& operator=(DenseMatrix<T>&& dense)

   Move-assigns a dense matrix to this sparse COO matrix.

   This operation converts a dense matrix into a sparse matrix by transferring
   all explicitly initialized values into the COO triplet format. The dense matrix
   is cleared after the conversion to avoid data duplication. Zero values are
   included if they were initialized in the source matrix.

   :param dense: The dense matrix to move and convert.
   :type dense: DenseMatrix<T>&&
   :returns: Reference to the updated SparseCOOMatrix
   :rtype: SparseCOOMatrix<T>&
   :raises: None

   .. note::

      All values marked as initialized in the dense matrix are transferred.
      This includes explicitly set zero values.

   **Example**

   .. code-block:: cpp

      slt::DenseMatrix<float> mat(2, 2);
      mat.set(0, 0, 1.0f);
      mat.set(1, 1, 0.0f);  // Explicit zero

      slt::SparseCOOMatrix<float> coo;
      coo = std::move(mat);

      EXPECT_EQ(coo.size(), 2);  // Includes explicit zero

.. cpp:function:: SparseCOOMatrix<T>& operator=(const SparseCSRMatrix<T>& csr)

   Assignment operator that converts a ``SparseCSRMatrix<T>`` to a
   ``SparseCOOMatrix<T>``.

   This operator extracts all initialized elements from the CSR matrix and
   populates the COO matrix as a list of triplets. Any previous data in the
   COO matrix is discarded.

   :param csr: A reference to the CSR matrix to assign from.
   :type csr: ``SparseCSRMatrix<T> const &``
   :returns: Reference to the modified ``SparseCOOMatrix<T>``
   :throws: ``std::bad_alloc`` if memory allocation for triplets fails.

   **Example:**

   .. code-block:: cpp

      slt::DenseMatrix<double> dense(2, 2);
      slt::SparseCSRMatrix<double> csr(dense);
      slt::SparseCOOMatrix<double> coo = csr;

.. cpp:function:: DenseMatrix<T>& DenseMatrix::operator=(SparseCSRMatrix<T>&& csr)

   **Move-assignment operator** that consumes a *rvalue* :cpp:`SparseCSRMatrix`
   and rewrites this matrix into dense form.

   :param csr: Sparse matrix in CSR format (rvalue reference).
   :type  csr: ``SparseCSRMatrix<T>&&``
   :returns: Reference to *this* dense matrix.
   :rtype: ``DenseMatrix<T>&``
   :throws std::bad_alloc: If internal buffers must grow and allocation fails.

   Internally the operator allocates a contiguous row-major buffer, then
   scatters every initialized entry of *csr* into its `(row,col)` slot.  
   After transfer, the source CSR matrix is reset to an empty (but valid) state.

   **Example**

   .. code-block:: cpp

      slt::SparseCSRMatrix<double> csr = build_sparse();
      slt::DenseMatrix<double>     A;       // empty
      A = std::move(csr);                   // csr cleared; A now dense


operator<<
~~~~~~~~~~

.. cpp:function:: std::ostream& operator<<(std::ostream& os, const slt::SparseCOOMatrix<T>& mat)

   Outputs the contents of the ``SparseCOOMatrix<T>`` in **triplet format**:

   .. code-block:: text

      SparseCOOMatrix<float> (rows x cols), nonzeros = N
      (row, col) = value
      (row, col) = value
      ...

   Each stored triplet is printed on a separate line.

   - In ``fast_set`` mode, the output order is insertion order.  
   - In finalized mode, the triplets are sorted by (row, col).

   :param os: The output stream (usually ``std::cout`` or a file stream)
   :param mat: The sparse COO matrix to print
   :return: A reference to the output stream (allows chaining)

   **Example**::

      slt::SparseCOOMatrix<float> mat(3, 3);
      mat.set(0, 0, 1.0f);
      mat.set(2, 1, 5.0f);
      mat.finalize();

      std::cout << mat << std::endl;

   **Example output**::

      SparseCOOMatrix<float> (3 x 3), nonzeros = 2
      (0, 0) = 1.0
      (2, 1) = 5.0

Data Access Methods 
-------------------

size()
~~~~~~

.. doxygenfunction:: slt::SparseCOOMatrix::size
   :project: csalt++

initialized_count()
~~~~~~~~~~~~~~~~~~~

.. doxygenfunction:: slt::SparseCOOMatrix::initialized_count() const
   :project: csalt++

begin() 
~~~~~~~

.. doxygenfunction:: slt::SparseCOOMatrix::begin()
   :project: csalt++

.. doxygenfunction:: slt::SparseCOOMatrix::begin() const
   :project: csalt++

end()
~~~~~

.. doxygenfunction:: slt::SparseCOOMatrix::end()
   :project: csalt++

.. doxygenfunction:: slt::SparseCOOMatrix::end() const
   :project: csalt++

is_initialized()
~~~~~~~~~~~~~~~~

.. doxygenfunction:: slt::SparseCOOMatrix::is_initialized(std::size_t, std::size_t) const
   :project: csalt++

rows()
~~~~~~
.. cpp:function:: std::size_t slt::SparseCOOMatrix::rows() const

   Returns the number of rows in the sparse matrix.

   This function reports the total number of allocated rows, regardless of whether all rows contain nonzero entries.  
   It is inherited from ``MatrixBase`` and available in all matrix types.

   :returns: Number of rows in the matrix.
   :rtype: std::size_t

   **Example**::

      slt::SparseCOOMatrix<float> mat(5, 3);
      std::cout << "Rows: " << mat.rows();  // Outputs: 5

cols()
~~~~~~
.. cpp:function:: std::size_t slt::SparseCOOMatrix::cols() const

   Returns the number of columns in the sparse matrix.

   This function reports the total number of allocated columns, regardless of whether all columns contain data.  
   It is inherited from ``MatrixBase`` and implemented consistently across all matrix types.

   :returns: Number of columns in the matrix.
   :rtype: std::size_t

   **Example**::

      slt::SparseCOOMatrix<float> mat(5, 3);
      std::cout << "Columns: " << mat.cols();  // Outputs: 3

get()
~~~~~

.. doxygenfunction:: slt::SparseCOOMatrix::get(std::size_t, std::size_t) const
   :project: csalt++

Operations 
----------

set()
~~~~~

.. cpp:function:: void slt::SparseCOOMatrix::set(std::size_t r, std::size_t c, T value)

   Sets a value in the matrix at the given ``(row, column)`` index.

   - If ``fast_set == true``, the value is appended (no duplicate check, O(1) insertion).  
     You must call ``finalize()`` before reliable queries or retrievals.
   - If ``fast_set == false``, a binary search is performed and the value is inserted at the correct position.
     Duplicate insertion will throw an exception.

   :param r: Row index of the element (0-based).
   :param c: Column index of the element (0-based).
   :param value: Value to insert.

   :throws std::out_of_range: if indices are out of bounds.
   :throws std::runtime_error: if element already exists (only when ``fast_set == false``).

   **Example**::

      slt::SparseCOOMatrix<float> mat(3, 3);
      mat.set(1, 2, 4.5f);
      mat.finalize();
      float val = mat.get(1, 2);  // Returns 4.5f

update()
~~~~~~~~

.. cpp:function:: void slt::SparseCOOMatrix::update(std::size_t r, std::size_t c, T value)

   Updates an existing value in the matrix at the specified ``(row, column)`` position.

   - If ``fast_set == true``, performs a linear search to locate the element.
   - If ``fast_set == false``, performs a binary search (requires ``finalize()``).

   If the element is not present, throws an exception.  
   You must use ``set()`` to insert the element before updating.

   :param r: Row index of the element (0-based).
   :param c: Column index of the element (0-based).
   :param value: New value to assign.

   :throws std::out_of_range: if indices are out of bounds.
   :throws std::runtime_error: if the element does not exist.

   **Example**::

      slt::SparseCOOMatrix<float> mat(4, 4);
      mat.set(2, 2, 10.0f);
      mat.finalize();

      mat.update(2, 2, 20.0f);
      float val = mat.get(2, 2);  // Returns 20.0f

      // mat.update(1, 1, 5.0f);  // Would throw std::runtime_error

finalize()
~~~~~~~~~~

.. doxygenfunction:: slt::SparseCOOMatrix::finalize
   :project: csalt++

clone()
~~~~~~~

.. doxygenfunction:: slt::SparseCOOMatrix::clone
   :project: csalt++

transpose()
~~~~~~~~~~~

.. cpp:function:: void slt::SparseCOOMatrix::transpose()

   Transposes the sparse matrix **in place**.

   Swaps the row and column indices of each stored element and updates
   the matrix dimensions:
   
   - New rows = old columns  
   - New columns = old rows

   If the matrix is in fast-insert mode (``fast_set == true``), the transpose
   preserves fast-insert mode.

   If the matrix is in retrieval-optimized mode (``fast_set == false``), the
   triplet vector is re-sorted after the row/column swap.

   **Example**::

      slt::SparseCOOMatrix<float> mat(2, 3, {
          {0, 1, 5.0f},
          {1, 2, 3.0f}
      });

      mat.transpose();

      // Now mat has shape (3, 2)
      // and mat.get(1, 0) == 5.0f
      // and mat.get(2, 1) == 3.0f

inverse()
~~~~~~~~~

.. cpp:function:: slt::DenseMatrix slt::SparseCOOMatrix::inverse() const

   Computes the matrix inverse of this SparseCOOMatrix as a dense matrix.

   This method returns a ``DenseMatrix<T>`` containing the inverse of the sparse matrix.  
   Internally, the sparse matrix is first converted to a dense format, and then a dense matrix inversion  
   algorithm (such as Gauss-Jordan or LU decomposition) is used.

   The result is always dense, because in general, the inverse of a sparse matrix is not sparse.

   :returns: DenseMatrix<T> containing the matrix inverse.
   :raises: ``std::invalid_argument`` if the matrix is not square.  
            ``std::runtime_error`` if the matrix is singular (non-invertible).

   .. note:: The inverse of a sparse matrix is generally dense — expect higher memory usage.

   **Example**::

      slt::SparseCOOMatrix<float> A(2, 2, {
          {0, 0, 4.0f},
          {0, 1, 7.0f},
          {1, 0, 2.0f},
          {1, 1, 6.0f}
      });

      slt::DenseMatrix<float> A_inv = A.inverse();

remove()
~~~~~~~~

.. cpp:function:: void slt::SparseCOOMatrix::remove(std::size_t r, std::size_t c)

   Removes an element at the specified ``(row, column)`` position from the sparse matrix.

   If an entry with matching ``(row, col)`` exists, it is erased from the internal triplet vector.  
   If no such entry exists, the method does nothing — it is safe to call even if the entry is missing.

   In ``fast_set`` mode (unsorted triplet vector), this performs a linear search ``O(n)``.  
   In finalized mode (``fast_set == false``), this performs a binary search ``O(log n)``.

   :param r: Row index of the element to remove.
   :param c: Column index of the element to remove.
   :raises std::out_of_range: If the row or column index is invalid (out of matrix bounds).

   **Example**::

      slt::SparseCOOMatrix<float> mat(3, 3);
      mat.set(1, 2, 5.0f);
      mat.finalize();

      mat.remove(1, 2);  // (1,2) no longer exists

      EXPECT_THROW(mat.get(1, 2), std::runtime_error);


.. _sparsecsr_matrix:

clear()
~~~~~~~

.. cpp:function:: slt::SparseCOOMatrix::clear()

   Clears all entries from the COO matrix and resets its shape to zero rows and columns.

   This method removes all stored triplets and sets the internal row and column
   dimensions to zero. It is considered a destructive operation that completely
   resets the matrix.

   :raises: None

   .. warning::

      After calling this method, the matrix is uninitialized and must be resized
      or reconstructed before reuse.

   **Example**

   .. code-block:: cpp

      slt::SparseCOOMatrix<float> coo(3, 3);
      coo.set(0, 0, 1.0f);
      coo.set(1, 2, -2.5f);

      coo.clear();  // All data and shape are reset

      EXPECT_EQ(coo.rows(), 0);
      EXPECT_EQ(coo.cols(), 0);


SparseCSRMatrix<T>
==================

Constructors
------------

Build from DenseMatrix<T>
~~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: explicit SparseCSRMatrix(const DenseMatrix<T>& dense, bool accept_zeros = true)

   Constructs a ``SparseCSRMatrix`` from a given dense matrix by retaining initialized values,
   and optionally skipping explicit zeros.

   :tparam T: Data type of the matrix entries (must be ``float`` or ``double``).
   :param dense: Source dense matrix to convert.
   :param accept_zeros: If true (default), stores explicitly initialized zero values.
                        If false, skips values equal to ``T{}``.

   Only values that have been initialized in the dense matrix are included.
   The matrix structure (rows and columns) is preserved during conversion.

   Example:

   .. code-block:: cpp

      slt::DenseMatrix<float> dense = {
          {1.0f, 0.0f},
          {0.0f, 2.0f}
      };

      // Accept zeros
      slt::SparseCSRMatrix<float> csr1(dense);

      // Skip zeros
      slt::SparseCSRMatrix<float> csr2(dense, false);

.. cpp:function:: SparseCSRMatrix(DenseMatrix<T>&& dense, bool accept_zeros = true)

   Constructs a SparseCSRMatrix by moving from a DenseMatrix and converting it into CSR format.

   :tparam T: Must be ``float`` or ``double``.
   :param dense: Rvalue reference to a ``DenseMatrix<T>`` that holds the source data.
   :param accept_zeros: If ``true`` (default), explicitly zero-valued entries are included.
                        If ``false``, zero-valued but initialized entries are ignored.
   :throws: ``std::bad_alloc`` if memory allocation fails during construction.

   This constructor inspects each initialized entry in the input DenseMatrix and inserts valid
   elements into the internal CSR structure: ``data``, ``col_indices``, and ``row_indices``.
   After construction, the input matrix is left in a valid but cleared state.

   **Example**

   .. code-block:: cpp

      slt::DenseMatrix<float> dense = {
          {1.0f, 0.0f},
          {0.0f, 2.0f}
      };

      // Convert to sparse format, ignoring zeros
      slt::SparseCSRMatrix<float> csr(std::move(dense), false);

Build from SparseCOOMatrix<T>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: explicit SparseCSRMatrix(const SparseCOOMatrix<T>& coo)

   Constructs a ``SparseCSRMatrix`` from a ``SparseCOOMatrix`` by converting
   the triplet-based representation (row, col, value) into compressed row storage.

   :param coo: The source matrix in COO format
   :type coo: SparseCOOMatrix<T>

   All non-zero elements are preserved, and row compression is applied to
   build the CSR structure.

   **Example:**

   .. code-block:: cpp

      slt::SparseCOOMatrix<float> coo(3, 3);
      coo.set(0, 0, 1.0f);
      coo.set(1, 2, 2.5f);
      coo.set(2, 1, 3.0f);

      slt::SparseCSRMatrix<float> csr(coo);

.. cpp:function:: SparseCSRMatrix(SparseCOOMatrix<T>&& coo)

   Constructs a SparseCSRMatrix by moving data from a SparseCOOMatrix.

   This constructor converts a COO-formatted sparse matrix to a CSR representation
   by transferring ownership of its data and reformatting it internally.

   :param coo: Rvalue reference to the source SparseCOOMatrix.
   :throws: ``std::bad_alloc`` if memory allocation fails.
   :note: After the move, the source matrix is cleared and left in a valid but empty state.

   **Example**

   .. code-block:: cpp

      slt::SparseCOOMatrix<float> coo(3, 3);
      coo.set(0, 0, 1.0f);
      coo.set(1, 2, 2.5f);
      coo.set(2, 1, 3.0f);

      slt::SparseCSRMatrix<float> csr(std::move(coo));


Build from SparseCSRMatrix<T>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: SparseCSRMatrix(const SparseCSRMatrix<T>& other)

   Constructs a new :cpp:class:`SparseCSRMatrix` as a deep copy of another.

   This copy constructor duplicates all internal data members, including
   the non-zero values, column indices, and row pointers. It ensures
   the resulting matrix is a distinct object with no shared memory.

   :param other: The sparse matrix to copy from.
   :type other: const SparseCSRMatrix<T>&
   :tparam T: Must be either ``float`` or ``double``.

   :note: Modifications to the copied matrix do not affect the original.

   **Example**

   .. code-block:: cpp

      slt::SparseCSRMatrix<float> csr1 = ...;
      slt::SparseCSRMatrix<float> csr2(csr1);  // Deep copy

.. cpp:function:: SparseCSRMatrix(SparseCSRMatrix<T>** other)

      Move constructor for :cpp:class:`SparseCSRMatrix`.

      Transfers ownership of all internal resources (data, column indices, and row indices)
      from the source matrix into a new matrix instance. This constructor avoids deep copying
      and is especially useful for handling temporary objects or optimizing container operations.

      After the move, the source matrix (`other`) is left in a valid but empty state:
      its number of rows and columns is reset to zero.

      :param other: Matrix to move from.

      **Example**::

         slt::SparseCSRMatrix<float> mat1(5, 5);
         mat1.set(0, 0, 3.14f);

         // Transfer data from mat1 to mat2
         slt::SparseCSRMatrix<float> mat2(std::move(mat1));

Identiy Matrix 
~~~~~~~~~~~~~~

.. cpp:function:: SparseCSRMatrix(std::size_t size)

   Constructs an identity matrix in :cpp:class:`SparseCSRMatrix` format.

   This creates a square matrix of dimension ``size x size`` with 1.0 on the diagonal
   and zero elsewhere. The matrix is stored in CSR format for efficient row-wise access.

   :param size: Number of rows and columns (i.e., the size of the square identity matrix).
   :type size: std::size_t
   :throws: ``std::bad_alloc`` if memory allocation fails.

   **Example:**

   .. code-block:: cpp

      slt::SparseCSRMatrix<float> identity(4);
      // Identity matrix of size 4x4 with float precision


Operator Overloads 
------------------

operator=
~~~~~~~~~

.. cpp:function:: SparseCSRMatrix<T>& operator=(const SparseCSRMatrix<T>& other)

   Copy assignment operator for :cpp:class:`SparseCSRMatrix`.

   Performs a deep copy of all contents, including non-zero values, column indices,
   and row pointers. The dimensions of the target matrix are updated to match those
   of the source matrix.

   :param other: The matrix to copy from.
   :type other: const :cpp:class:`SparseCSRMatrix<T>` &
   :return: Reference to the updated :cpp:class:`SparseCSRMatrix` instance.
   :rtype: :cpp:class:`SparseCSRMatrix<T>` &

   :throws: ``std::bad_alloc`` if memory allocation fails.

   **Example:**

   .. code-block:: cpp

      slt::SparseCSRMatrix<float> matA(5);
      slt::SparseCSRMatrix<float> matB = matA;

.. cpp:function:: SparseCSRMatrix &operator=(SparseCSRMatrix &&other) noexcept

   Move assignment operator.

   Transfers ownership of all internal data structures from the given
   :cpp:class:`SparseCSRMatrix` to this matrix. After the operation, the source matrix
   is left in a cleared state with `rows_` and `cols_` set to zero.

   :param other: The matrix to move from.
   :type other: ``SparseCSRMatrix&&``
   :return: Reference to this matrix.
   :rtype: ``SparseCSRMatrix&``

   **Example:**

   .. code-block:: cpp

      slt::SparseCSRMatrix<float> A = build_matrix();
      slt::SparseCSRMatrix<float> B = std::move(A);

operator()
~~~~~~~~~~

operator+
~~~~~~~~~

operator-
~~~~~~~~~

operator*
~~~~~~~~~

operator/
~~~~~~~~~

operator<<
~~~~~~~~~~

Data Access Methods 
-------------------

Operations 
----------

Global Operators 
================

Addition 
--------

Scalar + DenseMatrix 
~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: template<typename T> DenseMatrix<T> operator+(T scalar, const DenseMatrix<T>& matrix)

   Adds a scalar value to each initialized element of a DenseMatrix.

   :param scalar: Scalar value to be added.
   :param matrix: Target matrix. Only initialized elements will be affected.
   :returns: A new DenseMatrix with each element equal to `matrix(i, j) + scalar`.

   :throws std::runtime_error: If an element in `matrix` is accessed without being initialized.

   **Example**::

      slt::DenseMatrix<float> A(2, 2);
      A.set(0, 0, 1.0f);
      A.set(1, 1, 2.0f);

      auto B = 3.0f + A;

      // B(0, 0) == 4.0, B(1, 1) == 5.0

Scalar + SparseCOOMatrix 
~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: slt::SparseCOOMatrix slt::operator+(T scalar, const slt::SparseCOOMatrix<T>& matrix)

   Adds a scalar to each non-zero element of a ``SparseCOOMatrix<T>``.

   This operator allows symmetric scalar addition: ``scalar + matrix``.  
   The operation is equivalent to ``matrix + scalar`` and preserves the sparsity pattern:  
   only stored elements are modified. Unstored zero elements remain unaffected.

   :param scalar: Scalar value to add.
   :param matrix: Sparse matrix to operate on.
   :returns: A new ``SparseCOOMatrix<T>`` with updated values.

   **Example**::

      slt::SparseCOOMatrix<float> A(2, 2, {
          {0, 0, 2.0f},
          {1, 1, 5.0f}
      });

      auto result = 3.0f + A;

      // result.get(0, 0) == 5.0f
      // result.get(1, 1) == 8.0f

DenseMatrix + SparseCOOMatrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: slt::DenseMatrix slt::operator+(const slt::DenseMatrix<T>& dense, const slt::SparseCOOMatrix<T>& sparse)

   Adds a ``DenseMatrix<T>`` and a ``SparseCOOMatrix<T>`` element-wise.

   Returns a new ``DenseMatrix<T>`` where each non-zero value in the sparse matrix is added to the corresponding entry in the dense matrix.  
   The result is fully initialized and will have the same dimensions as the input matrices.

   :tparam T: The type of matrix elements (must be ``float`` or ``double``).
   :param dense: The dense matrix operand.
   :param sparse: The sparse COO matrix operand.
   :returns: A new ``DenseMatrix<T>`` containing the result.
   :throws std::invalid_argument: If the input matrices do not have the same shape.

   **Example**::

      slt::DenseMatrix<float> A(2, 2);
      A.set(0, 0, 1.0f);
      A.set(1, 1, 2.0f);

      slt::SparseCOOMatrix<float> B(2, 2);
      B.set(0, 1, 3.0f);

      slt::DenseMatrix<float> C = A + B;

      // C(0, 0) == 1.0
      // C(0, 1) == 3.0
      // C(1, 1) == 2.0

SparseCOOMatrix + DenseMatrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: slt::DenseMatrix slt::operator+(const slt::SparseCOOMatrix<T>& sparse, const slt::DenseMatrix<T>& dense)

   Adds a ``SparseCOOMatrix<T>`` and a ``DenseMatrix<T>`` element-wise.

   Returns a new ``DenseMatrix<T>`` where each non-zero value in the sparse matrix is added to the corresponding entry in the dense matrix.  
   The result is fully initialized and will have the same dimensions as the input matrices.

   :tparam T: The type of matrix elements (must be ``float`` or ``double``).
   :param sparse: The sparse COO matrix operand.
   :param dense: The dense matrix operand.
   :returns: A new ``DenseMatrix<T>`` containing the result.
   :throws std::invalid_argument: If the input matrices do not have the same shape.

   **Example**::

      slt::DenseMatrix<float> A(2, 2);
      A.set(0, 0, 1.0f);
      A.set(1, 1, 2.0f);

      slt::SparseCOOMatrix<float> B(2, 2);
      B.set(0, 1, 3.0f);

      slt::DenseMatrix<float> C = A + B;

      // C(0, 0) == 1.0
      // C(0, 1) == 3.0
      // C(1, 1) == 2.0


Subtraction
-----------

Scalar - DenseMatrix 
~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: template<typename T> DenseMatrix<T> operator-(T scalar, const DenseMatrix<T>& matrix)

   Subtracts each initialized element of a DenseMatrix from a scalar.

   :param scalar: The scalar value to subtract from.
   :param matrix: The DenseMatrix whose elements are subtracted.
   :returns: A new DenseMatrix where each initialized element is `scalar - matrix(i, j)`.

   :throws std::runtime_error: If `matrix(i, j)` is accessed and uninitialized.

   **Example**::

      slt::DenseMatrix<float> A(2, 2);
      A.set(0, 0, 1.0f);
      A.set(1, 1, 2.0f);

      auto B = 5.0f - A;

      // B(0, 0) == 4.0, B(1, 1) == 3.0

scalar - SparseCOOMatrix
~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: slt::SparseCOOMatrix slt::operator-(T scalar, const slt::SparseCOOMatrix<T>& matrix)

   Subtracts each non-zero element of a ``SparseCOOMatrix<T>`` from a scalar value.

   Creates a new sparse matrix where each stored element is computed as ``scalar - value``.  
   Unstored zero elements remain zero and are not explicitly added to the result.

   This operation preserves the sparsity pattern of the original matrix.

   :param scalar: The scalar value to subtract each matrix element from.
   :param matrix: The input ``SparseCOOMatrix<T>``.
   :returns: A new ``SparseCOOMatrix<T>`` with updated values.
   :throws std::invalid_argument: If the matrix is improperly initialized.

   **Example**::

      slt::SparseCOOMatrix<float> A(2, 2);
      A.set(0, 0, 3.0f);
      A.set(1, 1, 1.0f);

      auto B = 5.0f - A;

      // B.get(0, 0) == 2.0f
      // B.get(1, 1) == 4.0f

SparseCOOMatrix - DenseMatrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: slt::DenseMatrix slt::operator-(const slt::SparseCOOMatrix<T>& sparse, const slt::DenseMatrix<T>& dense)

   Subtracts a ``DenseMatrix<T>`` from a ``SparseCOOMatrix<T>`` and returns a ``DenseMatrix<T>``.

   Computes element-wise: ``result(i,j) = sparse(i,j) - dense(i,j)``.  
   The result is stored as a fully initialized dense matrix to capture all entries,  
   including those with implicit zeros in the sparse matrix.

   SIMD acceleration is used for the negation of the dense matrix if available.

   :tparam T: Floating-point type (``float`` or ``double``).
   :param sparse: The sparse matrix operand.
   :param dense: The dense matrix operand.
   :returns: ``DenseMatrix<T>`` with the subtraction result.
   :throws std::invalid_argument: If matrix dimensions do not match.

   **Example**::

      slt::SparseCOOMatrix<float> A(2, 2);
      A.set(0, 0, 1.0f);
      A.set(1, 1, 2.0f);

      slt::DenseMatrix<float> B(2, 2);
      B.set(0, 0, 5.0f);
      B.set(0, 1, 6.0f);
      B.set(1, 0, 7.0f);
      B.set(1, 1, 8.0f);

      slt::DenseMatrix<float> C = A - B;

      // C == {{-4.0f, -6.0f}, {-7.0f, -6.0f}};

DenseMatrix - SparseCOOMatrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: slt::DenseMatrix slt::operator-(const slt::DenseMatrix<T>& dense, const slt::SparseCOOMatrix<T>& sparse)

   Subtracts a ``SparseCOOMatrix<T>`` from a ``DenseMatrix<T>`` and returns a ``DenseMatrix<T>``.

   Computes element-wise: ``result(i,j) = dense(i,j) - sparse(i,j)``.  
   The result preserves the original dense structure and includes any corrections from the sparse matrix.

   SIMD acceleration is used to copy the dense matrix where available.

   :tparam T: Floating-point type (``float`` or ``double``).
   :param dense: The dense matrix operand.
   :param sparse: The sparse matrix operand.
   :returns: ``DenseMatrix<T>`` with the subtraction result.
   :throws std::invalid_argument: If matrix dimensions do not match.

   **Example**::

      slt::DenseMatrix<float> A(2, 2);
      A.set(0, 0, 5.0f);
      A.set(0, 1, 6.0f);
      A.set(1, 0, 7.0f);
      A.set(1, 1, 8.0f);

      slt::SparseCOOMatrix<float> B(2, 2);
      B.set(0, 0, 1.0f);
      B.set(1, 1, 2.0f);

      slt::DenseMatrix<float> C = A - B;

      // C == {{4.0f, 6.0f}, {7.0f, 6.0f}};

Multiplication 
--------------

Scalar * DenseMatrix 
~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: template<typename T> DenseMatrix<T> operator*(T scalar, const DenseMatrix<T>& matrix)

   Multiplies each initialized element of a DenseMatrix by a scalar.

   :param scalar: The scalar multiplier.
   :param matrix: The DenseMatrix whose values are to be scaled.
   :returns: A new DenseMatrix where each initialized element is `matrix(i, j) * scalar`.

   :throws std::runtime_error: If an uninitialized element is accessed.

   **Example**::

      slt::DenseMatrix<float> A(2, 2);
      A.set(0, 0, 3.0f);
      A.set(1, 1, 2.0f);

      auto B = 2.0f * A;

      // B(0, 0) == 6.0, B(1, 1) == 4.0

scalar * SparseCOOMatrix
~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: slt::SparseCOOMatrix slt::operator*(T scalar, const slt::SparseCOOMatrix<T>& matrix)

   Multiplies every non-zero element of a ``SparseCOOMatrix<T>`` by a scalar value.

   Creates a new sparse matrix where each stored value is multiplied by the given scalar.  
   Zero elements remain zero and are not explicitly added to the result.

   Internally, this operator delegates to the member ``SparseCOOMatrix::operator*(T scalar)``.

   :param scalar: The scalar value to multiply each matrix element by.
   :param matrix: The input ``SparseCOOMatrix<T>``.
   :returns: A new ``SparseCOOMatrix<T>`` with updated values.

   **Example**::

      slt::SparseCOOMatrix<float> A(2, 2);
      A.set(0, 0, 2.0f);
      A.set(1, 1, 4.0f);

      auto B = 3.0f * A;

      // B.get(0, 0) == 6.0f
      // B.get(1, 1) == 12.0f

DenseMatrix * SparseCOOMatrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: template <typename T> DenseMatrix<T> slt::operator*(const DenseMatrix<T>& dense, const SparseCOOMatrix<T>& sparse)

   Performs element-wise multiplication of a ``DenseMatrix`` and a ``SparseCOOMatrix``.

   Returns a new ``DenseMatrix<T>`` where each element is the product of corresponding entries
   in the dense and sparse matrices. Multiplication is performed only at non-zero positions
   of the sparse matrix — implicit zero entries are skipped.

   The result is fully initialized. Positions in the dense matrix that do not correspond to
   a non-zero entry in the sparse matrix are set to zero.

   :param dense: The dense matrix operand.
   :param sparse: The sparse COO matrix operand.
   :return: A new ``DenseMatrix<T>`` with the element-wise product.
   :throws std::invalid_argument: If the matrix dimensions do not match.

   **Example**::

      slt::DenseMatrix<float> A(2, 2);
      A.set(0, 0, 1.0f);
      A.set(0, 1, 2.0f);
      A.set(1, 0, 3.0f);
      A.set(1, 1, 4.0f);

      slt::SparseCOOMatrix<float> B(2, 2);
      B.set(0, 1, 5.0f);
      B.set(1, 0, 6.0f);

      slt::DenseMatrix<float> C = A * B;

      // C(0, 0) == 0.0f
      // C(0, 1) == 10.0f
      // C(1, 0) == 18.0f
      // C(1, 1) == 0.0f

SparseCOOMatrix * DenseMatrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: template <typename T> DenseMatrix<T> slt::operator*(const SparseCOOMatrix<T>& sparse, const DenseMatrix<T>& dense)

   Performs element-wise multiplication of a ``SparseCOOMatrix`` and a ``DenseMatrix``.

   This function is equivalent to ``dense * sparse`` and reuses that implementation.
   The multiplication is commutative in this case — only positions with non-zero entries
   in the sparse matrix are affected.

   :param sparse: The sparse COO matrix operand.
   :param dense: The dense matrix operand.
   :return: A new ``DenseMatrix<T>`` with the element-wise product.
   :throws std::invalid_argument: If the matrix dimensions do not match.

   **Example**::

      slt::SparseCOOMatrix<float> A(2, 2);
      A.set(0, 1, 2.0f);
      A.set(1, 0, 4.0f);

      slt::DenseMatrix<float> B(2, 2);
      B.set(0, 0, 10.0f);
      B.set(0, 1, 20.0f);
      B.set(1, 0, 30.0f);
      B.set(1, 1, 40.0f);

      slt::DenseMatrix<float> C = A * B;

      // C(0, 1) == 40.0f
      // C(1, 0) == 120.0f
      // Other entries == 0.0f


Matrix Muultiplication 
----------------------

DenseMatrix * DenseMatrix 
~~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: template<typename T> DenseMatrix<T> mat_mul(const DenseMatrix<T>& A, const DenseMatrix<T>& B)

   Performs matrix multiplication between two DenseMatrix objects.
   **NOTE:** This function results in a true matrix multiplication and not an
   element wise multiplication.

   :param A: Left-hand matrix operand of size M × N.
   :param B: Right-hand matrix operand of size N × P.
   :returns: A new DenseMatrix of size M × P, representing the matrix product A * B.

   :throws std::invalid_argument: If the number of columns in A does not match the number of rows in B.
   :throws std::runtime_error: If any required element in A or B is uninitialized.

   **Example**::

      slt::DenseMatrix<float> A({
         {1.0f, 2.0f},
         {3.0f, 4.0f}
      });

      slt::DenseMatrix<float> B({
         {5.0f, 6.0f},
         {7.0f, 8.0f}
      });

      auto C = mat_mul(A, B);

      // C(0, 0) == 1*5 + 2*7 == 19
      // C(0, 1) == 1*6 + 2*8 == 22

SparseCOOMatrix * SparseCOOMatrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: template <typename T> DenseMatrix<T> slt::mat_mul(const SparseCOOMatrix<T>& A, const SparseCOOMatrix<T>& B)

   Performs sparse matrix multiplication: result = ``A * B``.

   Multiplies two sparse COO matrices ``A`` and ``B``, returning the result  
   as a ``DenseMatrix<T>``. The function avoids unnecessary dense conversion  
   and uses a hash-based lookup internally.

   The algorithm computes the dot product of row ``i`` of ``A`` with column ``j`` of ``B``  
   for all rows and columns in the result matrix.

   The result is always returned as a full dense matrix, even if the original inputs are sparse.

   .. note::
      This implementation is not SIMD accelerated. Future optimization could use ``CSR`` or ``CSC``.

   :tparam T: Element type (``float`` or ``double``).
   :param A: Left-hand operand (SparseCOOMatrix).
   :param B: Right-hand operand (SparseCOOMatrix).
   :return: ``DenseMatrix<T>`` result of ``A * B``.
   :throws std::invalid_argument: If dimensions are incompatible for multiplication.

   **Example**::

      slt::SparseCOOMatrix<float> A(2, 3);
      slt::SparseCOOMatrix<float> B(3, 2);

      A.set(0, 1, 4.0f);
      A.set(1, 2, 5.0f);

      B.set(1, 0, 2.0f);
      B.set(2, 1, 3.0f);

      auto C = slt::mat_mul(A, B);

      EXPECT_FLOAT_EQ(C(0, 0), 8.0f);   // 4 * 2
      EXPECT_FLOAT_EQ(C(1, 1), 15.0f);  // 5 * 3

SparseCOOMatrix * DenseMatrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: template<typename T> DenseMatrix<T> slt::mat_mul(const SparseCOOMatrix<T>& A, const DenseMatrix<T>& B)

   Multiplies a ``SparseCOOMatrix<T>`` by a ``DenseMatrix<T>`` (A × B), producing a ``DenseMatrix<T>``.

   The result is computed as:

   .. math::

      result(i,j) = \sum_k A(i,k) * B(k,j)

   :param A: The sparse matrix (SparseCOOMatrix<T>), size (m × n).
   :param B: The dense matrix (DenseMatrix<T>), size (n × p).
   :return: A dense matrix result (m × p).

   :throws std::invalid_argument: If A.cols() != B.rows().

   **Example**::

      slt::SparseCOOMatrix<float> A(2, 3);
      A.set(0, 1, 4.0f);
      A.set(1, 2, 5.0f);

      slt::DenseMatrix<float> B({
          {1.0f, 2.0f},
          {3.0f, 4.0f},
          {5.0f, 6.0f}
      });

      auto C = mat_mul(A, B);

      EXPECT_FLOAT_EQ(C(0, 0), 12.0f);  // 4.0 * 3.0
      EXPECT_FLOAT_EQ(C(1, 1), 30.0f);  // 5.0 * 6.0

DenseMatrix * SparseCOOMatrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: template<typename T> DenseMatrix<T> slt::mat_mul(const DenseMatrix<T>& A, const SparseCOOMatrix<T>& B)

   Multiplies a ``DenseMatrix<T>`` by a ``SparseCOOMatrix<T>`` (A × B), producing a ``DenseMatrix<T>``.

   The result is computed as:

   .. math::

      result(i,j) = \sum_k A(i,k) * B(k,j)

   :param A: The dense matrix (DenseMatrix<T>), size (m × n).
   :param B: The sparse matrix (SparseCOOMatrix<T>), size (n × p).
   :return: A dense matrix result (m × p).

   :throws std::invalid_argument: If A.cols() != B.rows().

   **Example**::

      slt::DenseMatrix<float> A({
          {1.0f, 2.0f},
          {3.0f, 4.0f}
      });

      slt::SparseCOOMatrix<float> B(2, 3);
      B.set(0, 0, 5.0f);
      B.set(1, 2, 6.0f);

      auto C = mat_mul(A, B);

      EXPECT_FLOAT_EQ(C(0, 0), 5.0f);   // 1.0 * 5.0
      EXPECT_FLOAT_EQ(C(0, 2), 12.0f);  // 2.0 * 6.0


.. _triplet_class:

Triplet<T>
==========

.. cpp:class:: template <typename T> slt::Triplet<T>

   Represents a single non-zero entry in a sparse COO matrix.

   Stores the **row index**, **column index**, and **value** for a sparse matrix element.
   Supports sorting and equality comparison based on (row, col) order.

   :tparam T: Must be either ``float`` or ``double``.

Constructors 
------------

Default Constructor
~~~~~~~~~~~~~~~~~~~

.. doxygenfunction:: slt::Triplet::Triplet()
   :project: csalt++

Parameterized Constructor
~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenfunction:: slt::Triplet::Triplet(std::size_t, std::size_t, T)
   :project: csalt++

Copy Constructor 
~~~~~~~~~~~~~~~~

.. doxygenfunction:: slt::Triplet::Triplet(const Triplet&)
   :project: csalt++

Move Constructor 
~~~~~~~~~~~~~~~~

.. doxygenfunction:: slt::Triplet::Triplet(Triplet&&) nooexcept
   :project: csalt++

Operators 
---------

operator()=
~~~~~~~~~~~

.. cpp:function:: Triplet& slt::Triplet::operator=(const Triplet& other)

   Copy-assigns another Triplet.

   :param other: The Triplet to copy from.
   :return: Reference to the current object.

   Example::

      slt::Triplet<double> t1(1, 2, 3.0);
      slt::Triplet<double> t2;
      t2 = t1;
      assert(t2.equals(t1));

.. cpp:function:: Triplet& slt::Triplet::operator=(Triplet&& other)

   Move-assigns another Triplet.

   :param other: The Triplet to move from.
   :return: Reference to the current object.

   Example::

      slt::Triplet<float> t1(1, 2, 3.0f);
      slt::Triplet<float> t2;
      t2 = std::move(t1);

operator()==
~~~~~~~~~~~~

.. doxygenfunction:: slt::Triplet::operator==(const Triplet&) const
   :project: csalt++

operator()<
~~~~~~~~~~~

.. doxygenfunction:: slt::Triplet::operator<(const Triplet&) const
   :project: csalt++

equals()
~~~~~~~~

.. doxygenfunction:: slt::Triplet::equals(const Triplet&) const
   :project: csalt++
