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

Typical Use Cases
#################

* Engineering calculations (FEM, CFD, PDEs)
* Adaptive data structures for large numerical grids
* Real-time simulation or optimization
* Any scenario where matrix sparsity evolves during computation

Getting Started
###############

Clone the repository:

.. code-block:: bash

    git clone https://github.com/Jon-Webb-79/csaltcpp.git
    cd csalt

CMake Build Instructions
------------------------

**Debug build (with tests):**

.. code-block:: bash

    cd scripts/bash
    ./debug.sh

**Static build (no tests):**

.. code-block:: bash

    cd scripts/bash
    ./static.sh

**Install system-wide (optional):**

.. code-block:: bash

    sudo ./install.sh

Run Unit Tests
--------------

.. code-block:: bash

    cd build/debug
    ./unit_tests

You may optionally run under `valgrind` (Linux only):

.. code-block:: bash

    valgrind ./unit_tests

Dependencies
############

Required:

* C++ compiler supporting C++17 (tested with GCC 14.2.1 and Clang 16.0.6)
* CMake ≥ 3.31.3
* CMocka (for unit tests)

Optional:

* valgrind (memory leak detection)
* Python 3.10+ and Sphinx (for documentation)

Development & Contribution
##########################

This library is modular and extensible. Contributions are welcome!

1. Fork the repo and create a branch
2. Write or update code
3. Add tests in the `test` directory
4. Ensure tests pass under `debug` mode
5. Update or add Sphinx docstrings
6. Submit a pull request

Documentation
=============

Build the documentation using Sphinx:

.. code-block:: bash

    cd docs/doxygen
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    make html

Documentation is also hosted online:

TBD

License
#######

CSalt++ is provided under the MIT License. See the `LICENSE` file for details.
