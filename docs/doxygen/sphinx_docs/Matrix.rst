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

.. cpp:function:: DenseMatrix(std::size_t rows, std::size_t cols, T value = T{})

   Constructs a matrix of the given dimensions and initializes all values to ``value``.
   If ``value`` is not provided, the elements default to zero. Elements are marked as initialized
   if ``value != T{}``.

   :param rows: Number of rows in the matrix.
   :param cols: Number of columns in the matrix.
   :param value: Initial value for all elements (optional).

   :throws std::invalid_argument: If either dimension is zero.

   Example::

      slt::DenseMatrix<double> A(3, 3);          // 3x3 zero matrix
      slt::DenseMatrix<float> B(2, 4, 1.0f);     // 2x4 matrix filled with 1.0

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

