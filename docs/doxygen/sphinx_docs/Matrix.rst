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

.. cpp:function:: template<std::size_t R, std::size_t C>
                  DenseMatrix(const std::array<std::array<T, C>, R>& data)

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

Core Methods
------------

get()
~~~~~

.. cpp:function:: T get(std::size_t row, std::size_t col) const

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

.. cpp:function:: void set(std::size_t row, std::size_t col, T value)

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

.. cpp:function:: void update(std::size_t row, std::size_t col, T value)

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

.. cpp:function:: std::size_t rows() const

   Returns the number of rows in the matrix.

   :return: Number of rows

   Example::

      slt::DenseMatrix<float> D(3, 4);
      std::size_t r = D.rows();  // r == 3

cols()
~~~~~~~

.. cpp:function:: std::size_t cols() const

   Returns the number of columns in the matrix.

   :return: Number of columns

   Example::

      slt::DenseMatrix<float> D(3, 4);
      std::size_t c = D.cols();  // c == 4

transpose()
~~~~~~~~~~~

.. cpp:function:: void transpose()

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

.. cpp:function:: DenseMatrix<T> inverse() const

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
.. cpp:function:: std::size_t size() const

   Returns the total number of elements in the matrix (i.e., ``rows() * cols()``).

   :return: Total number of matrix elements.

   Example::

      slt::DenseMatrix<float> A(3, 4);
      std::size_t total = A.size();  // total == 12

data_ptr()
~~~~~~~~~~
.. cpp:function:: const T* data_ptr() const

   Returns a raw pointer to the underlying data buffer (read-only).

   :return: Pointer to the start of the matrix data (row-major order).

.. cpp:function:: T* data_ptr()

   Returns a mutable raw pointer to the underlying data buffer.

   :return: Mutable pointer to the start of the matrix data (row-major order).

   Example::

      slt::DenseMatrix<float> A(2, 2, 1.0f);
      const float* ptr = A.data_ptr();        // Access read-only values
      float* modifiable = A.data_ptr();       // Modify values directly

init_ptr()
~~~~~~~~~~
.. cpp:function:: const uint8_t* init_ptr() const

   Returns a raw pointer to the internal initialization tracking buffer (read-only).
   Each element is 1 if the corresponding matrix element has been initialized, and 0 otherwise.

   :return: Pointer to initialization flags for each matrix entry.

.. cpp:function:: uint8_t* init_ptr()

   Returns a mutable pointer to the internal initialization tracking buffer.

   :return: Mutable pointer to initialization flags for each matrix entry.

   Example::

      slt::DenseMatrix<double> A(2, 2);
      A.set(0, 0, 5.0);
      const uint8_t* flags = A.init_ptr();
      assert(flags[0] == 1);  // Element (0,0) is initialized

nonzero_count()
~~~~~~~~~~~~~~~
.. cpp:function:: std::size_t nonzero_count() const

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

The ``DenseMatrix<T>`` class overloads common operators to support arithmetic and comparison operations
in a type-safe, intuitive manner. These include element access, assignment, equality, and matrix multiplication.

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

Assignment Operator
~~~~~~~~~~~~~~~~~~~

.. cpp:function:: DenseMatrix<T>& operator=(const DenseMatrix<T>& other)

   Assigns the contents of another matrix to this one. Performs a deep copy.

   :param other: Matrix to copy from
   :returns: Reference to this matrix

   Example::

      slt::DenseMatrix<double> A(2, 2, 1.0);
      slt::DenseMatrix<double> B = A;  // Uses copy constructor
      B = A;  // Uses assignment operator

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

