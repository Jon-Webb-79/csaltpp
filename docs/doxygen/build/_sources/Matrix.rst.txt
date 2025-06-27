**************
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

DenseMatrix(const SparseCOOMatrix<T>& sparse)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


DenseMatrix Copy Constructor 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenfunction:: slt::DenseMatrix::DenseMatrix(const DenseMatrix<T>&)
   :project: csalt++

DenseMatrix Move Constructor 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenfunction:: slt::DenseMatrix::DenseMatrix(DenseMatrix<T>&&)
   :project: csalt++

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

SparseCSRMatrix<T>
==================

Constructors
------------

Operator Overloads 
------------------

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
