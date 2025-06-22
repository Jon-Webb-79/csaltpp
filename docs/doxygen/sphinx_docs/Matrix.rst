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

DenseMatrix(const std::array<T, N>& arr, std::size_t r, std::size_t c)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenfunction:: slt::DenseMatrix::DenseMatrix(const std::array<T, N>&, std::size_t, std::size_t)
   :project: csalt++

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

nonzero_count() 
~~~~~~~~~~~~~~~

.. doxygenfunction:: slt::DenseMatrix::nonzero_count() const
   :project: csalt++

is_initialized()
~~~~~~~~~~~~~~~~

.. doxygenfunction:: slt::DenseMatrix::is_initialized
   :project: csalt++

rows()
~~~~~~

.. doxygenfunction:: slt::DenseMatrix::rows
   :project: csalt++

cols()
~~~~~~

.. doxygenfunction:: slt::DenseMatrix::cols
   :project: csalt++

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

print()
~~~~~~~

.. doxygenfunction:: slt::DenseMatrix::print
   :project: csalt++

SparseCOOMatrix<T>
==================

.. doxygenclass:: slt::SparseCOOMatrix
   :project: csalt++

See also: :ref:`triplet_class`

Constructors
------------

SparseCOOMatrix(std::size_t, std::size_t, std::size_t)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenfunction:: slt::SparseCOOMatrix::SparseCOOMatrix(std::size_t, std::size_t, std::size_t)
   :project: csalt++

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

      assert(A.nonzero_count() == 0);  // A is now empty
      assert(B(0, 0) == 1.0f);

Data Access Methods 
-------------------

size()
~~~~~~

.. doxygenfunction:: slt::SparseCOOMatrix::size
   :project: csalt++

nonzero_count()
~~~~~~~~~~~~~~~

.. doxygenfunction:: slt::SparseCOOMatrix::nonzero_count() const
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

.. doxygenfunction:: slt::SparseCOOMatrix::rows() const
   :project: csalt++

cols()
~~~~~~

.. doxygenfunction:: slt::SparseCOOMatrix::cols() const
   :project: csalt++

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

Matrix Muultiplication 
----------------------

DenseMatrix * DenseMatrix 
~~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: template<typename T> DenseMatrix<T> mat_mul(const DenseMatrix<T>& A, const DenseMatrix<T>& B)

   Performs matrix multiplication between two DenseMatrix objects.

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
