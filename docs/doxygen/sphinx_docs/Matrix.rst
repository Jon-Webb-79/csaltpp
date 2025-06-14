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

DenseMatrix<T>
==============

The ``DenseMatrix<T>`` class provides a high-performance, row-major layout for matrix elements,
suitable for use in tightly coupled linear algebra operations. It supports both single and 
double precision values (``float`` or ``double``), with internal tracking of which elements 
have been initialized. This allows for safe manipulation of partially constructed matrices,
and helps prevent unintentional reads of uninitialized memory.

``DenseMatrix`` is optimized for numerical applications where full or mostly-full storage is expected.
SIMD acceleration is applied automatically when supported by the underlying hardware for the type ``T``.

Constructors
------------

DenseMatrix provides multiple constructors for initializing matrices in various ways,
including fixed-size dimensions, nested containers, initializer lists, and flat storage.

Fixed Dimensions
~~~~~~~~~~~~~~~~

.. cpp:function:: DenseMatrix(std::size_t rows, std::size_t cols, T value)

   Constructs a matrix of the given dimensions and initializes all values to ``value``.

   :param rows: Number of rows in the matrix.
   :param cols: Number of columns in the matrix.
   :param value: Initial value for all elements.

   :throws std::invalid_argument: If either dimension is zero.

   Example::

      slt::DenseMatrix<float> B(2, 4, 1.0f);     // 2x4 matrix filled with 1.0

.. cpp:function:: DenseMatrix(std::size_t rows, std::size_t cols)

   Constructs a matrix of the given dimensions without initializing elements

   :param rows: Number of rows in the matrix.
   :param cols: Number of columns in the matrix.

   :throws std::invalid_argument: If either dimension is zero.

   Example::

      slt::DenseMatrix<double> A(3, 3);          // 3x3 zero matrix

2D std::vector
~~~~~~~~~~~~~~

.. cpp:function:: DenseMatrix(const std::vector<std::vector<T>>& data)

   Constructs a matrix from a 2D ``std::vector``. All inner vectors must be the same length.

   :param data: A 2D vector containing the matrix values.

   :throws std::invalid_argument: If the outer vector is empty or inner vectors have unequal lengths.

   Example::

      std::vector<std::vector<double>> values = {
          {1.0, 2.0},
          {3.0, 4.0}
      };
      slt::DenseMatrix<double> M(values);

std::array of std::array
~~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: template<std::size_t R, std::size_t C>DenseMatrix(const std::array<std::array<T, C>, R>& data)

   Constructs a matrix from a fixed-size 2D ``std::array``.

   :param data: A statically sized 2D array representing the matrix contents.

   Example::

      std::array<std::array<float, 2>, 2> arr = {{
          {1.0f, 2.0f},
          {3.0f, 4.0f}
      }};
      slt::DenseMatrix<float> A(arr);

Initializer List
~~~~~~~~~~~~~~~~

.. cpp:function:: DenseMatrix(std::initializer_list<std::initializer_list<T>> init_list)

   Constructs a matrix from a nested initializer list. All rows must have the same number of columns.

   :param init_list: A nested initializer list.

   :throws std::invalid_argument: If the outer list is empty or inner lists have unequal lengths.

   Example::

      slt::DenseMatrix<double> A = {
          {1.0, 2.0},
          {3.0, 4.0}
      };

Flat Data Vector
~~~~~~~~~~~~~~~~

.. cpp:function:: DenseMatrix(const std::vector<T>& flat_data, std::size_t rows, std::size_t cols)

   Constructs a matrix from a flat data vector with explicit dimensions.

   :param flat_data: A flat vector containing matrix elements in row-major order.
   :param rows: Number of rows.
   :param cols: Number of columns.

   :throws std::invalid_argument: If ``flat_data.size() != rows * cols`` or any dimension is zero.

   Example::

      std::vector<double> flat = {1.0, 2.0, 3.0, 4.0};
      slt::DenseMatrix<double> A(flat, 2, 2);  // Creates 2x2 matrix

Copy Constructor
~~~~~~~~~~~~~~~~

.. cpp:function:: DenseMatrix(const DenseMatrix<T>& other)

   Constructs a deep copy of an existing ``DenseMatrix``.

   This constructor allocates new memory for the internal storage and copies all element values and initialization flags from ``other``.

   :param other: Source matrix to copy
   :throws std::bad_alloc: If memory allocation fails

   Example::

      slt::DenseMatrix<float> A({1.0f, 2.0f, 3.0f, 4.0f}, 2, 2);
      slt::DenseMatrix<float> B(A);  // Deep copy of A

      // B is now a separate matrix with the same content as A
      REQUIRE(B(1, 0) == 3.0f);

Move Constructor
~~~~~~~~~~~~~~~~

.. cpp:function:: DenseMatrix(DenseMatrix<T>&& other) noexcept

   Constructs a new ``DenseMatrix`` by transferring the resources from ``other``.

   The move constructor avoids deep copying and instead takes ownership of the internal buffers from ``other``, which is left in a valid but unspecified state.

   :param other: Source matrix to move from
   :post: ``other`` is left empty and should not be used further

   Example::

      slt::DenseMatrix<float> A({5.0f, 6.0f, 7.0f, 8.0f}, 2, 2);
      slt::DenseMatrix<float> B(std::move(A));  // Move A into B

      // B now owns the data originally in A
      REQUIRE(B(0, 1) == 6.0f);

Core Methods
------------

get()
~~~~~

.. cpp:function:: T DenseMatrix::get(std::size_t row, std::size_t col) const

   Returns the value at the specified (row, col) index.

   :param row: Zero-based row index
   :param col: Zero-based column index
   :return: Value of the matrix element

   :throws std::out_of_range: If the index is outside the matrix bounds.
   :throws std::runtime_error: If the element is uninitialized.

   Example::

      slt::DenseMatrix<double> A(2, 2, 0.0);
      A.set(0, 1, 3.14);
      double val = A.get(0, 1);  // val == 3.14

set()
~~~~~

.. cpp:function:: void DenseMatrix::set(std::size_t row, std::size_t col, T value)

   Sets the value at the specified (row, col) index only if the element
   is uninitialized.

   :param row: Zero-based row index
   :param col: Zero-based column index
   :param value: Value to assign

   :throws std::out_of_range: If the index is outside the matrix bounds.
   :throws std::runtime_error: If the element has already been initialized.

   Example::

      slt::DenseMatrix<float> B(2, 2);
      B.set(1, 0, 42.0f);

update()
~~~~~~~~

.. cpp:function:: void DenseMatrix::update(std::size_t row, std::size_t col, T value)

   Updates the value of an already-initialized element.

   :param row: Zero-based row index
   :param col: Zero-based column index
   :param value: New value to assign

   :throws std::out_of_range: If the index is outside the matrix bounds.
   :throws std::runtime_error: If the element is uninitialized.

   Example::

      slt::DenseMatrix<double> C(2, 2);
      C.set(0, 0, 1.23);
      C.update(0, 0, 4.56);  // Updates value

rows()
~~~~~~~

.. cpp:function:: std::size_t DenseMatrix::rows() const

   Returns the number of rows in the matrix.

   :return: Number of rows

   Example::

      slt::DenseMatrix<float> D(3, 4);
      std::size_t r = D.rows();  // r == 3

cols()
~~~~~~~

.. cpp:function:: std::size_t DenseMatrix::cols() const

   Returns the number of columns in the matrix.

   :return: Number of columns

   Example::

      slt::DenseMatrix<float> D(3, 4);
      std::size_t c = D.cols();  // c == 4

transpose()
~~~~~~~~~~~

.. cpp:function:: void DenseMatrix::transpose()

   Performs an in-place transposition of the matrix. Converts an (r x c)
   matrix into a (c x r) matrix.

   Example::

      slt::DenseMatrix<double> A = {
          {1.0, 2.0},
          {3.0, 4.0}
      };
      A.transpose();
      // A is now:
      // [1.0, 3.0]
      // [2.0, 4.0]

inverse()
~~~~~~~~~

.. cpp:function:: DenseMatrix<T> DenseMatrix::inverse() const

   Computes and returns the inverse of the matrix using Gauss-Jordan elimination.

   :return: A new ``DenseMatrix<T>`` representing the inverse

   :throws std::invalid_argument: If the matrix is not square.
   :throws std::runtime_error: If the matrix is singular or not invertible.

   Example::

      slt::DenseMatrix<double> A = {
          {4.0, 7.0},
          {2.0, 6.0}
      };
      slt::DenseMatrix<double> invA = A.inverse();

size()
~~~~~~
.. cpp:function:: std::size_t DenseMatrix::size() const

   Returns the total number of elements in the matrix (i.e., ``rows() * cols()``).

   :return: Total number of matrix elements.

   Example::

      slt::DenseMatrix<float> A(3, 4);
      std::size_t total = A.size();  // total == 12

data_ptr()
~~~~~~~~~~
.. cpp:function:: const T* DenseMatrix::data_ptr() const

   Returns a raw pointer to the underlying data buffer (read-only).

   :return: Pointer to the start of the matrix data (row-major order).

.. cpp:function:: T* DenseMatrix::data_ptr()

   Returns a mutable raw pointer to the underlying data buffer.

   :return: Mutable pointer to the start of the matrix data (row-major order).

   Example::

      slt::DenseMatrix<float> A(2, 2, 1.0f);
      const float* ptr = A.data_ptr();        // Access read-only values
      float* modifiable = A.data_ptr();       // Modify values directly

init_ptr()
~~~~~~~~~~
.. cpp:function:: const uint8_t* DenseMatrix::init_ptr() const

   Returns a raw pointer to the internal initialization tracking buffer (read-only).
   Each element is 1 if the corresponding matrix element has been initialized, and 0 otherwise.

   :return: Pointer to initialization flags for each matrix entry.

.. cpp:function:: uint8_t* DenseMatrix::init_ptr()

   Returns a mutable pointer to the internal initialization tracking buffer.

   :return: Mutable pointer to initialization flags for each matrix entry.

   Example::

      slt::DenseMatrix<double> A(2, 2);
      A.set(0, 0, 5.0);
      const uint8_t* flags = A.init_ptr();
      assert(flags[0] == 1);  // Element (0,0) is initialized

nonzero_count()
~~~~~~~~~~~~~~~
.. cpp:function:: std::size_t DenseMatrix::nonzero_count() const

   Returns the number of initialized (non-zero) elements in the matrix.
   This is equivalent to counting how many elements are marked as initialized.

   :return: Number of initialized entries.

   Example::

      slt::DenseMatrix<float> A(2, 2);
      A.set(0, 0, 1.0f);
      A.set(1, 1, 2.0f);
      std::size_t count = A.nonzero_count();  // count == 2


Operators
---------

Copy Assignment Operator
~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: DenseMatrix<T>& operator=(const DenseMatrix<T>& other)

   Overwrites the contents of the current matrix with a deep copy of ``other``.

   Allocates new memory and copies all values and initialization flags. Existing data is discarded.

   :param other: Source matrix to copy
   :returns: Reference to the updated matrix
   :throws std::bad_alloc: If memory allocation fails

   Example::

      slt::DenseMatrix<float> A({1.0f, 2.0f, 3.0f, 4.0f}, 2, 2);
      slt::DenseMatrix<float> B;
      B = A;  // Deep copy from A into B

      REQUIRE(B(0, 1) == 2.0f);

Move Assignment Operator
~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: DenseMatrix<T>& operator=(DenseMatrix<T>&& other) noexcept

   Transfers ownership of all resources from ``other`` to the current matrix.

   The existing data is discarded and replaced by the moved content. ``other`` is left in a valid but unspecified state.

   :param other: Source matrix to move from
   :returns: Reference to the updated matrix
   :post: ``other`` is cleared and should not be used
   :note: No memory is copied; only ownership is transferred

   Example::

      slt::DenseMatrix<float> A({5.0f, 6.0f, 7.0f, 8.0f}, 2, 2);
      slt::DenseMatrix<float> B;
      B = std::move(A);  // Transfer ownership from A to B

      REQUIRE(B(1, 0) == 7.0f);

Element Access Operator
~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: T operator()(std::size_t row, std::size_t col) const

   Returns the value at the specified row and column index. Equivalent to calling ``get(row, col)``.

   :param row: Zero-based row index
   :param col: Zero-based column index
   :returns: Element value
   :throws std::out_of_range: If index is out of bounds
   :throws std::runtime_error: If the element is uninitialized

   Example::

      slt::DenseMatrix<double> A(2, 2);
      A.set(0, 1, 3.14);
      double x = A(0, 1);  // x == 3.14

Equality Operator
~~~~~~~~~~~~~~~~~

.. cpp:function:: bool operator==(const DenseMatrix<T>& other) const

   Compares two matrices for equality. Returns true if both matrices have the same dimensions,
   and all initialized elements are equal.

   :param other: Matrix to compare
   :returns: True if matrices are equal, false otherwise

   Example::

      slt::DenseMatrix<float> A = {{1.0f, 2.0f}, {3.0f, 4.0f}};
      slt::DenseMatrix<float> B = {{1.0f, 2.0f}, {3.0f, 4.0f}};
      assert(A == B);

Matrix Multiplication Operator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: DenseMatrix<T> operator*(const DenseMatrix<T>& rhs) const

   Performs matrix multiplication with another ``DenseMatrix<T>``. Throws an exception
   if the dimensions are incompatible.

   :param rhs: Right-hand side matrix
   :returns: Resulting matrix
   :throws std::invalid_argument: If inner dimensions do not match

   Example::

      slt::DenseMatrix<float> A = {{1.0f, 2.0f}, {3.0f, 4.0f}};
      slt::DenseMatrix<float> B = {{2.0f, 0.0f}, {1.0f, 2.0f}};
      auto C = A * B;

Subtraction Operator
~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: DenseMatrix<T> operator-(const DenseMatrix<T>& rhs) const

   Performs element-wise subtraction between two matrices. All elements must be initialized.

   :param rhs: Matrix to subtract
   :returns: Resulting matrix
   :throws std::invalid_argument: If dimensions do not match

   Example::

      slt::DenseMatrix<double> A = {{5.0, 6.0}, {7.0, 8.0}};
      slt::DenseMatrix<double> B = {{1.0, 2.0}, {3.0, 4.0}};
      auto C = A - B;

Division Operator
~~~~~~~~~~~~~~~~~

.. cpp:function:: DenseMatrix<T> operator/(const DenseMatrix<T>& rhs) const

   Performs element-wise division. All elements must be initialized. Division by zero will throw.

   :param rhs: Divisor matrix
   :returns: Resulting matrix
   :throws std::invalid_argument: If dimensions do not match
   :throws std::domain_error: If division by zero is attempted

   Example::

      slt::DenseMatrix<double> A = {{10.0, 20.0}, {30.0, 40.0}};
      slt::DenseMatrix<double> B = {{2.0, 4.0}, {5.0, 8.0}};
      auto C = A / B;

Addition Operator 
~~~~~~~~~~~~~~~~~

.. cpp:function:: DenseMatrix<T> operator+(DenseMatrix<T>& rhs)

   Returns a matrix that is an element wise addition between the two matrices 

   :param rhs: A DenseMatrix
   :returns: Resulting matrix

   Example::

      slt::DenseMatrix<double> A = {{1.0, 2.0}, {3.0, 4.0}};
      slt::DenseMatrix<double> B = {5.0, 6.0}, {7.0, 8.0}};
      auto C = A + B;

Scalar Multiplication
~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: DenseMatrix<T> operator*(T scalar) const

   Returns a new matrix where each element is multiplied by a scalar.

   :param scalar: Scalar multiplier
   :returns: Resulting matrix

   Example::

      slt::DenseMatrix<double> A = {{1.0, 2.0}, {3.0, 4.0}};
      auto B = A * 2.0;

Scalar Division
~~~~~~~~~~~~~~~

.. cpp:function:: DenseMatrix<T> operator/(T scalar) const

   Returns a new matrix where each element is divided by a scalar.

   :param scalar: Scalar divisor
   :returns: Resulting matrix
   :throws std::domain_error: If scalar is zero

   Example::

      slt::DenseMatrix<float> A = {{2.0f, 4.0f}, {6.0f, 8.0f}};
      auto B = A / 2.0f;

Scalar Addition
~~~~~~~~~~~~~~~

.. cpp:function:: DenseMatrix<T> operator+(T scalar) const

   Returns a new matrix where each initialized element is incremented by the scalar value.

   :param scalar: Scalar value to add
   :returns: Resulting matrix
   :throws std::runtime_error: If any element is uninitialized

   Example::

      slt::DenseMatrix<double> A = {{1.0, 2.0}, {3.0, 4.0}};
      auto B = A + 10.0;
      // B is {{11.0, 12.0}, {13.0, 14.0}}

Scalar Subtraction
~~~~~~~~~~~~~~~~~~

.. cpp:function:: DenseMatrix<T> operator-(T scalar) const

   Returns a new matrix where each initialized element is decremented by the scalar value.

   :param scalar: Scalar value to subtract
   :returns: Resulting matrix
   :throws std::runtime_error: If any element is uninitialized

   Example::

      slt::DenseMatrix<float> A = {{5.0f, 6.0f}, {7.0f, 8.0f}};
      auto B = A - 2.0f;
      // B is {{3.0f, 4.0f}, {5.0f, 6.0f}}

Dense Matrix Addition
~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: template<typename T> DenseMatrix<T> operator+(const DenseMatrix<T>& dense, const SparseCOOMatrix<T>& sparse)

   Returns a new dense matrix that is the element-wise sum of a dense and sparse matrix.

   :param dense: A DenseMatrix object
   :param sparse: A SparseCOOMatrix object
   :returns: Resulting DenseMatrix
   :throws std::invalid_argument: If matrix dimensions do not match

   Example::

      slt::DenseMatrix<float> A = {{5.0f, 6.0f}, {7.0f, 8.0f}};
      slt::SparseCOOMatrix<float> B = {{9.0f, 10.0f}, {11.0f, 12.0f}};
      slt::DenseMatrix<float> C = A + B;
      // C is {{14.0f, 16.0f}, {18.0f, 20.0f}}

Global Operators
----------------

Stream Output
~~~~~~~~~~~~~

.. cpp:function:: std::ostream& operator<<(std::ostream& os, const DenseMatrix<T>& mat)

   Outputs the matrix to the provided output stream using its internal `print` method.

   :param os: Output stream
   :param mat: Matrix to print
   :returns: The output stream

   Example::

      slt::DenseMatrix<float> A = {{1.0f, 2.0f}, {3.0f, 4.0f}};
      std::cout << A << std::endl;

   Output::

      1.0 2.0 
      3.0 4.0

Scalar + Matrix
~~~~~~~~~~~~~~~

.. cpp:function:: DenseMatrix<T> operator+(T scalar, const DenseMatrix<T>& matrix)

   Adds a scalar to each initialized element of the matrix (scalar on the left-hand side).
   Equivalent to `matrix + scalar`.

   :param scalar: Scalar value to add
   :param matrix: Matrix operand
   :returns: Resulting matrix
   :throws std::runtime_error: If any element is uninitialized

   Example::

      slt::DenseMatrix<double> A = {{1.0, 2.0}, {3.0, 4.0}};
      auto B = 10.0 + A;

Scalar - Matrix
~~~~~~~~~~~~~~~

.. cpp:function:: DenseMatrix<T> operator-(T scalar, const DenseMatrix<T>& matrix)

   Subtracts each initialized element of the matrix from the scalar.

   :param scalar: Scalar value
   :param matrix: Matrix to subtract from scalar
   :returns: Resulting matrix
   :throws std::runtime_error: If any element is uninitialized

   Example::

      slt::DenseMatrix<double> A = {{1.0, 2.0}, {3.0, 4.0}};
      auto B = 10.0 - A;  // B contains {9.0, 8.0}, {7.0, 6.0}

Scalar * Matrix
~~~~~~~~~~~~~~~

.. cpp:function:: DenseMatrix<T> operator*(T scalar, const DenseMatrix<T>& matrix)

   Multiplies each initialized matrix element by the scalar. Equivalent to `matrix * scalar`.

   :param scalar: Scalar multiplier
   :param matrix: Matrix operand
   :returns: Resulting matrix
   :throws std::runtime_error: If any element is uninitialized

   Example::

      slt::DenseMatrix<float> A = {{1.0f, 2.0f}, {3.0f, 4.0f}};
      auto B = 2.0f * A;  // B is {{2.0f, 4.0f}, {6.0f, 8.0f}}

Matrix Multiplication
~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: template<typename T> DenseMatrix<T> mat_mul(const DenseMatrix<T>& A, const DenseMatrix<T>& B)

   Performs matrix multiplication between two ``DenseMatrix<T>`` instances. This function computes the dot product 
   of rows from the left-hand side matrix with columns of the right-hand side matrix. SIMD acceleration is applied 
   internally to speed up dot product calculations when available.

   The resulting matrix has dimensions :math:`(m \times n)` where:

   - ``A`` is an :math:`(m \times k)` matrix
   - ``B`` is a :math:`(k \times n)` matrix

   The mathematical operation performed is:

   .. math::

      C_{i,j} = \sum_{l=1}^{k} A_{i,l} \cdot B_{l,j}

   :param A: Left-hand side matrix of shape (m × k)
   :param B: Right-hand side matrix of shape (k × n)
   :returns: Resultant matrix of shape (m × n)
   :throws std::invalid_argument: If inner dimensions (A.cols != B.rows) do not match
   :throws std::runtime_error: If uninitialized elements are accessed during multiplication

   ----

   **Example (float)**

   .. code-block:: cpp

      #include "dense_matrix.hpp"

      slt::DenseMatrix<float> A = {
         {1.0f, 2.0f, 3.0f},
         {4.0f, 5.0f, 6.0f}
      };

      slt::DenseMatrix<float> B = {
         {7.0f, 8.0f},
         {9.0f, 10.0f},
         {11.0f, 12.0f}
      };

      slt::DenseMatrix<float> C = slt::mat_mul(A, B);
      C.print();

   **Output**::

      58.0 64.0
      139.0 154.0

   ----

   **Example (double)**

   .. code-block:: cpp

      slt::DenseMatrix<double> A = {
         {2.0, 0.0},
         {1.0, 3.0}
      };

      slt::DenseMatrix<double> B = {
         {1.0, 2.0},
         {4.0, 5.0}
      };

      slt::DenseMatrix<double> C = slt::mat_mul(A, B);
      C.print();

   **Output**::

      2.0 4.0
      13.0 17.0

SparseCOOMatrix<T>
==================

The ``SparseCOOMatrix<T>`` class provides a memory-efficient representation of sparse matrices
using the Coordinate List (COO) format. It stores non-zero values along with their corresponding
row and column indices, making it ideal for matrices with a high proportion of zero entries.

This class supports both ``float`` and ``double`` element types. Internally, it maintains
fast and final insertion modes to optimize construction versus access patterns. Arithmetic operations
with other sparse and dense matrices are supported, as well as scalar operations.

Constructors
------------

SparseCOOMatrix provides several constructors for different initialization scenarios, including
fixed dimensions, nested containers, fixed-size arrays, and initializer lists.

Fixed Dimensions
~~~~~~~~~~~~~~~~

.. cpp:function:: SparseCOOMatrix(std::size_t rows, std::size_t cols, bool fastInsert = true)

   Constructs an empty sparse matrix with the given number of rows and columns. No elements are initialized.
   If ``fastInsert`` is true (default), insertions will be performed in append-only mode for efficiency.
   Call :cpp:func:`finalize()` to enable fast retrievals (via binary search) after bulk construction.

   :param rows: Number of rows in the matrix.
   :param cols: Number of columns in the matrix.
   :param fastInsert: Enables fast insertion mode if true (default: true).
   :throws std::invalid_argument: If either dimension is zero.

   Example::

      slt::SparseCOOMatrix<float> mat(4, 5);            // 4x5 sparse matrix in fast insert mode
      mat.set(1, 2, 3.5f);                               // Insert non-zero element
      mat.finalize();                                    // Sort and finalize for access

2D std::vector
~~~~~~~~~~~~~~

.. cpp:function:: SparseCOOMatrix(const std::vector<std::vector<T>>& data, bool fastInsert = true)

   Constructs a sparse matrix from a 2D ``std::vector`` by inserting all non-zero values. All rows
   must have the same number of columns. If ``fastInsert`` is true (default), the matrix is optimized
   for bulk construction and must be finalized before access operations like :cpp:func:`get()`.

   :param data: A 2D vector representing the matrix.
   :param fastInsert: Enables fast insertion mode if true (default: true).
   :throws std::invalid_argument: If rows have inconsistent lengths.

   Example::

      std::vector<std::vector<float>> data = {
          {0.0f, 2.5f},
          {3.0f, 0.0f}
      };
      slt::SparseCOOMatrix<float> mat(data);
      mat.finalize();  // Recommended before calls to get() or update()

2D std::array
~~~~~~~~~~~~~

.. cpp:function:: SparseCOOMatrix(const std::array<std::array<T, C>, R>& data, bool fastInsert = true)

   Constructs a sparse matrix from a fixed-size 2D ``std::array`` by inserting all non-zero values.
   If ``fastInsert`` is true (default), entries are added in append mode and require a call to
   :cpp:func:`finalize()` before using binary search operations.

   :tparam R: Number of rows (deduced from the array).
   :tparam C: Number of columns (deduced from the array).
   :param data: A statically sized 2D array.
   :param fastInsert: Enables fast insertion mode if true (default: true).

   Example::

      std::array<std::array<float, 2>, 2> arr = {{
          {0.0f, 4.5f},
          {1.2f, 0.0f}
      }};
      slt::SparseCOOMatrix<float> mat(arr);
      mat.finalize();

Initializer List
~~~~~~~~~~~~~~~~

.. cpp:function:: SparseCOOMatrix(std::initializer_list<std::initializer_list<T>> init_list, bool fastInsert = true)

   Constructs a sparse matrix from a nested initializer list. All rows must be the same length.
   Only non-zero values are inserted. If ``fastInsert`` is true (default), you must call
   :cpp:func:`finalize()` before retrieval operations.

   :param init_list: Nested initializer list representing matrix data.
   :param fastInsert: Enables fast insertion mode if true (default: true).
   :throws std::invalid_argument: If inner lists have unequal lengths.

   Example::

      slt::SparseCOOMatrix<float> mat = {
          {0.0f, 3.0f},
          {4.0f, 0.0f}
      };
      mat.finalize();

Flat Storage Constructor
~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: SparseCOOMatrix(const std::vector<T>& flat_data, std::size_t r, std::size_t c, bool fastInsert = true)

   Constructs a sparse matrix from a flat 1D vector in row-major order.
   Only non-zero elements from the vector are stored. If ``fastInsert`` is enabled,
   the entries are appended efficiently and require a call to :cpp:func:`finalize()`.

   :param flat_data: A flat vector of values in row-major order.
   :param r: Number of rows in the matrix.
   :param c: Number of columns in the matrix.
   :param fastInsert: Enables fast insertion mode (default: true).

   :throws std::invalid_argument: If the size of the vector does not match ``r * c``.

   Example::

      std::vector<double> flat = {
          0.0, 3.0,
          1.5, 0.0
      };
      slt::SparseCOOMatrix<double> mat(flat, 2, 2);
      mat.finalize();

Copy Constructor 
~~~~~~~~~~~~~~~~

.. cpp:function:: SparseCOOMatrix(const SparseCOOMatrix<T>& other)

   Copy constructor for ``SparseCOOMatrix``.

   Creates a deep copy of another sparse matrix in coordinate (COO) format.
   This includes copying the row indices, column indices, data values,
   and the internal insertion optimization flag. The new matrix is
   completely independent from the original.

   :param other: The source sparse matrix to copy.
   :type other: const SparseCOOMatrix<T>&

   **Example:**

   .. code-block:: cpp

      SparseCOOMatrix<float> A = {{1.0f, 0.0f}, {0.0f, 2.0f}};
      SparseCOOMatrix<float> B(A);  // Deep copy of A

Move Constructor 
~~~~~~~~~~~~~~~~
.. cpp:function:: SparseCOOMatrix(SparseCOOMatrix<T>&& other) noexcept

   Move constructor for the :cpp:class:`SparseCOOMatrix`.

   Transfers ownership of the matrix contents from another instance, avoiding deep copies.
   This constructor moves the internal data structures (`row`, `col`, `data`) from the
   source matrix and resets the source to an empty, valid state.

   :param other: The matrix to move from. After the move, `other` is valid but empty.

   .. note::
      This constructor is noexcept and provides efficient transfer of ownership
      for temporary objects or explicit `std::move` usage.

   **Example:**

   .. code-block:: cpp

      slt::SparseCOOMatrix<float> original = {{1.0f, 0.0f}, {0.0f, 2.0f}};
      slt::SparseCOOMatrix<float> moved_to = std::move(original);
      // `moved_to` now owns the matrix data; `original` is empty


Core Methods
------------

get()
~~~~~

.. cpp:function:: T SparseCOOMatrix::get(std::size_t row, std::size_t col) const

   Returns the value at the specified (row, col) index.

   Performs a linear search if the matrix is in fast insertion mode; otherwise, a binary search is used after finalization.

   :param row: Zero-based row index
   :param col: Zero-based column index
   :return: Value of the matrix element
   :throws std::out_of_range: If the index is outside the matrix bounds.
   :throws std::runtime_error: If the element is uninitialized (i.e., not present in storage).

   Example::

      slt::SparseCOOMatrix<double> A(2, 2);
      A.set(0, 1, 3.14);
      A.finalize();
      double val = A.get(0, 1);  // val == 3.14

set()
~~~~~

.. cpp:function:: void SparseCOOMatrix::set(std::size_t row, std::size_t col, T value)

   Inserts a value at the specified (row, col) position.

   - If ``fast_set`` is true, the value is appended with no duplicate checks.
   - If ``fast_set`` is false, values are inserted in sorted order with uniqueness enforcement.

   :param row: Zero-based row index
   :param col: Zero-based column index
   :param value: Value to insert
   :throws std::out_of_range: If the index is out of bounds.
   :throws std::runtime_error: If inserting into a sorted matrix where the value already exists.

   Example::

      slt::SparseCOOMatrix<float> B(2, 2);
      B.set(1, 0, 42.0f);

update()
~~~~~~~~

.. cpp:function:: void SparseCOOMatrix::update(std::size_t row, std::size_t col, T value)

   Updates the value of an already-initialized element.

   Requires finalized state (i.e., fast insertion mode must be disabled).

   :param row: Zero-based row index
   :param col: Zero-based column index
   :param value: New value to assign
   :throws std::out_of_range: If the index is invalid.
   :throws std::runtime_error: If the element has not been inserted using ``set()``.

   Example::

      slt::SparseCOOMatrix<double> C(2, 2, false);
      C.set(0, 0, 1.23);
      C.update(0, 0, 4.56);  // Updates value

rows()
~~~~~~~

.. cpp:function:: std::size_t SparseCOOMatrix::rows() const

   Returns the number of rows in the sparse matrix.

   :return: Row dimension

   Example::

      slt::SparseCOOMatrix<float> D(3, 4);
      std::size_t r = D.rows();  // r == 3

cols()
~~~~~~~

.. cpp:function:: std::size_t SparseCOOMatrix::cols() const

   Returns the number of columns in the sparse matrix.

   :return: Column dimension

   Example::

      slt::SparseCOOMatrix<float> D(3, 4);
      std::size_t c = D.cols();  // c == 4

is_initialized()
~~~~~~~~~~~~~~~~

.. cpp:function:: bool SparseCOOMatrix::is_initialized(std::size_t row, std::size_t col) const

   Checks whether the specified (row, col) element has been explicitly set.

   Performs a linear search in fast insertion mode; otherwise, binary search is used in finalized mode.

   :param row: Zero-based row index
   :param col: Zero-based column index
   :return: ``true`` if the element has been initialized, otherwise ``false``
   :throws std::out_of_range: If the index is outside the matrix bounds.

   Example::

      slt::SparseCOOMatrix<double> A(3, 3);
      A.set(0, 1, 2.5);
      bool found = A.is_initialized(0, 1);  // true
      bool missing = A.is_initialized(2, 2);  // false

finalize()
~~~~~~~~~~

.. cpp:function:: void SparseCOOMatrix::finalize()

   Sorts the internal coordinate arrays (by row-major order) and eliminates duplicate entries.

   This method must be called before performing operations like `update()` or before enabling fast access modes.
   After finalization, all operations assume sorted data for efficient lookups.

   :throws std::runtime_error: If duplicates are found when `fast_set` is false and unique entries are expected.

   Example::

      slt::SparseCOOMatrix<float> B(2, 2);
      B.set(0, 0, 1.0f);
      B.set(1, 1, 2.0f);
      B.finalize();  // Enables optimized operations

nonzero_count()
~~~~~~~~~~~~~~~

.. cpp:function:: std::size_t SparseCOOMatrix::nonzero_count() const

   Returns the number of explicitly stored (non-zero) entries in the sparse matrix.

   :return: Number of non-zero elements

   Example::

      slt::SparseCOOMatrix<double> C(4, 4);
      C.set(0, 0, 1.0);
      C.set(2, 3, 5.0);
      std::size_t count = C.nonzero_count();  // count == 2

Operator Overloads
------------------

Equality Operator 
~~~~~~~~~~~~~~~~~
.. cpp:function:: bool operator==(const SparseCOOMatrix<T>& other) const

   Compares two sparse COO matrices for equality.

   This operator returns ``true`` if both matrices have the same dimensions and 
   identical non-zero entries at the same positions. The comparison accounts 
   for floating-point imprecision (when ``T`` is a floating-point type) using an absolute difference threshold.

   :param other: The matrix to compare against
   :return: ``true`` if matrices are equal; otherwise, ``false``
   :throws std::runtime_error: If either matrix is not finalized

   Example::

      slt::SparseCOOMatrix<float> A(2, 2, false);
      A.set(0, 0, 1.0f);
      A.set(1, 1, 2.0f);
      A.finalize();

      slt::SparseCOOMatrix<float> B(2, 2, false);
      B.set(1, 1, 2.0f);
      B.set(0, 0, 1.0f);
      B.finalize();

      bool equal = (A == B);  // equal == true

operator= (copy)
~~~~~~~~~~~~~~~~~

.. cpp:function:: SparseCOOMatrix<T>& operator=(const SparseCOOMatrix<T>& other)

   Performs a deep copy of another ``SparseCOOMatrix`` into this instance.

   All dimensions, insertion mode, and internal coordinate storage (row, col, data) are duplicated.
   The two matrices will be fully independent after the operation.

   :param other: Source matrix to copy
   :return: Reference to this matrix
   :throws: None

   Example::

      slt::SparseCOOMatrix<float> A(2, 2);
      A.set(0, 1, 3.14f);
      A.finalize();

      slt::SparseCOOMatrix<float> B(2, 2);
      B = A;  // Deep copy

operator= (move)
~~~~~~~~~~~~~~~~~

.. cpp:function:: SparseCOOMatrix<T>& operator=(SparseCOOMatrix<T>&& other) noexcept

   Transfers ownership of resources from another ``SparseCOOMatrix`` into this one.

   After the operation, the ``other`` matrix is left in a valid but empty state.

   :param other: Rvalue reference to the source matrix
   :return: Reference to this matrix
   :throws: None

   Example::

      slt::SparseCOOMatrix<float> A(2, 2);
      A.set(1, 0, 42.0f);
      A.finalize();

      slt::SparseCOOMatrix<float> B;
      B = std::move(A);  // Transfer ownership


Function Call Operator
~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: T operator()(std::size_t row, std::size_t col) const

   Accesses the element at the specified position using function call syntax.

   This is equivalent to calling ``get(row, col)`` and performs the same bounds and initialization checks.

   :param row: Zero-based row index
   :param col: Zero-based column index
   :return: Value of the matrix element
   :throws std::out_of_range: If the index is out of bounds.
   :throws std::runtime_error: If the element is uninitialized.

   Example::

      slt::SparseCOOMatrix<float> A(3, 3);
      A.set(1, 2, 9.81f);
      A.finalize();
      float value = A(1, 2);  // value == 9.81

Addition: Sparse + Sparse
~~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: DenseMatrix<T> operator+(const SparseCOOMatrix<T>& other) const

   Adds two sparse matrices element-wise.

   Returns a ``DenseMatrix`` containing the element-wise sum. This ensures correct representation
   of overlapping non-zero entries.

   :param other: The other sparse matrix operand.
   :return: A fully dense matrix representing the sum.
   :throws std::invalid_argument: If matrix dimensions do not match.

   Example::

      slt::SparseCOOMatrix<float> A = {{1.0f, 0.0f}, {0.0f, 2.0f}};
      slt::SparseCOOMatrix<float> B = {{0.0f, 3.0f}, {4.0f, 0.0f}};
      slt::DenseMatrix<float> C = A + B;

Addition: Sparse + Dense
~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: DenseMatrix<T> operator+(const DenseMatrix<T>& other) const

   Adds a sparse matrix to a dense matrix element-wise.

   The result is a new ``DenseMatrix``. If supported, SIMD acceleration is applied to the dense operand
   before sparse additions are performed.

   :param other: The dense matrix operand.
   :return: A dense matrix containing the sum.
   :throws std::invalid_argument: If matrix dimensions do not match.

   Example::

      slt::DenseMatrix<double> dense(2, 2, 1.0);
      slt::SparseCOOMatrix<double> sparse(2, 2);
      sparse.set(1, 1, 3.0);
      sparse.finalize();
      auto result = sparse + dense;

Addition: Sparse + Scalar
~~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: SparseCOOMatrix<T> operator+(T scalar) const

   Adds a scalar value to each non-zero element in the sparse matrix.

   The result is another sparse matrix with the same structure but updated values.

   :param scalar: The scalar to add.
   :return: A new ``SparseCOOMatrix`` with updated values.

   Example::

      slt::SparseCOOMatrix<float> A = {{0.0f, 2.0f}, {0.0f, 0.0f}};
      auto B = A + 1.0f;

Addition: Scalar + Sparse (Friend)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: template<typename T>SparseCOOMatrix<T> operator+(T scalar, const SparseCOOMatrix<T>& matrix)

   Adds a scalar to each non-zero element of a sparse matrix (commutative overload).

   This is the friend equivalent of ``matrix + scalar`` and uses the member function internally.

   :param scalar: The scalar value to add.
   :param matrix: The sparse matrix.
   :return: A new ``SparseCOOMatrix`` with scalar applied.

   Example::

      slt::SparseCOOMatrix<float> A = {{0.0f, 2.0f}, {1.0f, 0.0f}};
      auto B = 1.0f + A;

Subtraction: Sparse + Sparse
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: DenseMatrix<T> SparseCOOMatrix<T>::operator-(const SparseCOOMatrix<T>& other) const

   Performs element-wise subtraction of two sparse COO matrices and returns the result as a dense matrix.

   Both input matrices must have the same dimensions. Values from `other` are subtracted from the calling matrix.
   If either matrix contains a non-zero value at a given index, the result includes it in dense form.

   :param other: The sparse matrix to subtract.
   :returns: A DenseMatrix representing the result of A - B
   :throws std::invalid_argument: If matrix dimensions do not match.

   Example::

      slt::SparseCOOMatrix<float> A(2, 2);
      A.set(0, 0, 1.0f);
      A.set(1, 1, 2.0f);

      slt::SparseCOOMatrix<float> B(2, 2);
      B.set(0, 1, 3.0f);
      B.set(1, 0, 4.0f);

      slt::DenseMatrix<float> C = A - B;
      // Result: C(0,0)=1.0, C(0,1)=-3.0, C(1,0)=-4.0, C(1,1)=2.0

Subtraction: Sparse + Scalar
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: SparseCOOMatrix<T> SparseCOOMatrix<T>::operator-(T scalar) const

   Subtracts a scalar from every stored (non-zero) element in the sparse COO matrix. Uninitialized elements remain untouched.

   This operation preserves the sparsity structure, only modifying explicitly stored entries.

   :param scalar: Scalar value to subtract.
   :returns: A new SparseCOOMatrix with updated values.

   Example::

      slt::SparseCOOMatrix<float> A(2, 2);
      A.set(0, 0, 1.0f);
      A.set(1, 1, 2.0f);

      auto result = A - 1.0f;
      // result: (0,0)=0.0, (1,1)=1.0

Subtraction: Scalar + Sparse 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: template<typename T> SparseCOOMatrix<T> operator-(T scalar, const SparseCOOMatrix<T>& matrix)

   Returns a new sparse matrix whose elements are the result of subtracting each non-zero entry 
   in the input matrix from the scalar. That is, each stored value becomes ``scalar - value``.

   This preserves the sparsity pattern of the original matrix. Only initialized (non-zero) elements 
   are modified and included in the result.

   :param scalar: The scalar to subtract each matrix value from.
   :param matrix: The input sparse matrix.
   :returns: A SparseCOOMatrix containing ``scalar - value`` for each non-zero element.
   :throws std::invalid_argument: If the matrix is improperly initialized.

   Example::

      slt::SparseCOOMatrix<float> A(2, 2);
      A.set(0, 0, 3.0f);
      A.set(1, 1, 1.0f);

      auto B = 5.0f - A;
      // B.get(0, 0) == 2.0f
      // B.get(1, 1) == 4.0f

Subtraction: Sparse - Dense
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: DenseMatrix<T> operator-(const SparseCOOMatrix<T>& sparse, const DenseMatrix<T>& dense)

   Computes the element-wise difference between a sparse matrix and a dense matrix, 
   returning the result as a new dense matrix.

   This operation returns ``result(i,j) = sparse(i,j) - dense(i,j)`` for all 
   ``(i,j)``. Elements from the sparse matrix are subtracted from the corresponding 
   values in the dense matrix.

   :param sparse: A ``SparseCOOMatrix<T>`` representing the left-hand side of the subtraction.
   :param dense: A ``DenseMatrix<T>`` representing the right-hand side of the subtraction.
   :returns: A new ``DenseMatrix<T>`` with the computed result.
   :throws std::invalid_argument: If matrix dimensions do not match.

   Example::

      slt::SparseCOOMatrix<float> A(2, 2);
      A.set(0, 0, 2.0f);
      A.set(1, 1, 3.0f);

      slt::DenseMatrix<float> B(2, 2, 1.0f);
      slt::DenseMatrix<float> C = A - B;

      // C(0, 0) == 1.0, C(1, 1) == 2.0, other values == -1.0

Subtraction: Dense - Sparse
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. cpp:function:: DenseMatrix<T> operator-(const DenseMatrix<T>& dense, const SparseCOOMatrix<T>& sparse)

   Computes the element-wise difference between a dense matrix and a sparse matrix, 
   returning the result as a new dense matrix.

   This operation returns ``result(i,j) = dense(i,j) - sparse(i,j)`` for all 
   ``(i,j)``. Sparse values are subtracted from the initialized dense matrix values.

   :param dense: A ``DenseMatrix<T>`` representing the left-hand side of the subtraction.
   :param sparse: A ``SparseCOOMatrix<T>`` representing the right-hand side of the subtraction.
   :returns: A new ``DenseMatrix<T>`` containing the result.
   :throws std::invalid_argument: If matrix dimensions do not match.

   Example::

      slt::DenseMatrix<float> A(2, 2, 1.0f);

      slt::SparseCOOMatrix<float> B(2, 2);
      B.set(0, 0, 0.5f);
      B.set(1, 1, 2.0f);

      slt::DenseMatrix<float> C = A - B;

      // C(0, 0) == 0.5, C(1, 1) == -1.0, other values == 1.0

