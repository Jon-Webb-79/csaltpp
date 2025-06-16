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

data_ptr()
~~~~~~~~~~

.. doxygenfunction:: slt::DenseMatrix::data_ptr()
   :project: csalt++

.. doxygenfunction:: slt::DenseMatrix::data_ptr() const
   :project: csalt++

init_ptr()
~~~~~~~~~~

.. doxygenfunction:: slt::DenseMatrix::init_ptr() const
   :project: csalt++

nonzero_count() 
~~~~~~~~~~~~~~~

.. doxygenfunction:: slt::DenseMatrix::nonzero_count() const

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

clonse()
~~~~~~~~

.. doxygenfunction:: slt::DenseMatrix::clone
   :project: csalt++

print()
~~~~~~~

.. doxygenfunction:: slt::DenseMatrix::print
   :project: csalt++

SparseCOOMatrix<T>
==================

Constructors
------------

Operator Overloads 
------------------

Data Access Methods 
-------------------

Operations 
----------

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

