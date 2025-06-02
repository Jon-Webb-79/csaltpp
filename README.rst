CSalt++
*******
The `csalt++` project is a modern C++ library designed to support flexible, 
efficient, and safe numerical computing with matrices and vectors. It builds 
upon the ideas developed in the original C version of CSalt, enhancing them with 
templates, operator overloading, and runtime format adaptation.

This library targets performance-critical applications such as scientific 
simulations, engineering solvers, and adaptive mesh computations where matrix 
structure can vary dramatically.

Why CSalt++?
############

C++ offers many improvements over C, but working with numerical data still 
involves challenges:

* Dynamic matrix manipulation is verbose and error-prone with raw arrays
* Standard containers lack built-in numerical semantics (e.g., transposition, inversion)
* No adaptive matrix types that convert formats based on sparsity or structure
* SIMD acceleration is not automatic
* Type safety and dimensionality checking must be implemented manually

CSalt++ addresses these issues by offering:

* Strongly typed dense matrices with optional SIMD support for float/double
* Planned sparse formats (COO, CSR) with runtime format switching
* Format-agnostic interface via `Matrix<T>` manager class
* Safe access and modification with uninitialized memory tracking
* Operator overloading for scalar and element-wise arithmetic
* Support for inversion, transposition, and numerical manipulation

Core Features
#############

Dense Matrix
------------
* Template-based for `float` or `double`
* Row-major layout
* Internal tracking of initialized elements to prevent invalid reads
* SIMD-accelerated operations if compiled with `-march=native`, `-mavx`, or `-msse`
* Operator overloading for:

  - `+`, `-`, `*`, `/` with scalars
  - `+`, `-`, `*` element-wise with other matrices
  - Matrix transposition
  - Matrix inversion (for square matrices)

Sparse Formats (Planned)
-------------------------
* **COO** (Coordinate List)
* **CSR** (Compressed Sparse Row)
* Seamless conversion between formats based on runtime sparsity thresholds
* Efficient memory layout for high-performance iterative solvers
* Format-agnostic API

Matrix Manager (Planned)
-------------------------
* Single front-end type `Matrix<T>` that wraps `DenseMatrix<T>`, `COOMatrix<T>`, or `CSRMatrix<T>`
* Chooses best format at runtime
* Allows automatic promotion/demotion of types as sparsity changes
* Dispatches operations to correct internal type safely

