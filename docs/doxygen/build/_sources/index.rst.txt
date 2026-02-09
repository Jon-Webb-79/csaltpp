.. Core Utilities documentation master file, created by
   sphinx-quickstart
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CSalt++ documentation!
=================================
The `csalt++` project is a modern C++ library designed for safety-critical and 
high-performance applications requiring MISRA C++ compliance. It provides 
comprehensive error handling and efficient numerical computing with matrices and 
vectors, with optional compile-time exclusion of dynamic allocation for embedded 
and safety-critical systems.

This library builds upon the ideas developed in the original C version of CSalt, 
enhancing them with templates, operator overloading, and runtime format adaptation, 
while maintaining compliance with MISRA C++ guidelines through careful design 
choices including optional static-only compilation, fixed-size buffers, and 
predictable behavior.

CSalt++ targets performance-critical applications such as automotive systems, 
aerospace software, medical devices, scientific simulations, and engineering solvers 
where both safety and performance are paramount. It provides SIMD-accelerated 
numerical operations with support for x86 instruction sets (AVX2, AVX-512, SSE2, 
SSE3, SSE4.1) and ARM architectures (NEON, SVE, SVE2).

Why CSalt++
###########

C++ offers many improvements over C, but working with numerical data still 
involves challenges:

* On the fly dynamic memory allocation which drives time complexity
* Standard exception handling violates MISRA C++ rules (dynamic allocation, non-deterministic behavior)
* Most C++ libraries cannot disable dynamic allocation for safety-critical use

CSalt++ addresses these issues by offering:

* Custom allocators for dynamic memory management
* MISRA C++ compliant error hierarchy with zero dynamic allocation
* `Expected<T>` for deterministic, exception-free error handling
* `STATIC_ONLY` flag to exclude all dynamic allocation at compile time
* Dual-mode design: full features with dynamic allocation OR static-only for certification
* Predictable, auditable behavior suitable for certification

Core Features 
#############

Error Classes 
-------------
* **STL-independent exception hierarchy** - Alternative to standard library exceptions with no dynamic allocation
* **Three-level taxonomy** - Base Error → 9 categories → 40+ specific types (e.g., Error → ArgumentError → NullPointerError)
* **Fixed-size messages** - All errors use 256-byte stack buffers, ensuring predictable memory usage
* **Default messages** - Each error type has a predefined message that can be overridden via constructor
* **Message composition** - Helper methods for prepending/appending context to standard messages
* **Dual-use design** - Works with traditional ``throw``/``catch`` or modern ``Expected<T>`` pattern
* **MISRA compliant** - Available in both standard and ``STATIC_ONLY`` modes
* **Type-safe handling** - Catch specific errors (``DivByZeroError``), categories (``MathError``), or base (``Error``)

Expected Class 
--------------
* **Errors as values** - Type-safe representation of computations that may succeed (``T``) or fail (``Error``)
* **Explicit error handling** - Forces callers to check for errors before accessing values, preventing forgotten checks
* **Zero overhead** - No exception unwinding, no dynamic allocation, deterministic performance
* **Stack-based storage** - ``sizeof(Expected<T>) = sizeof(bool) + sizeof(T) + sizeof(Error)``
* **Simple API** - ``setValue()``/``setError()`` to construct, ``hasValue()``/``hasError()`` to query, ``value()``/``error()`` to access
* **Safe defaults** - ``valueOr()`` method provides fallback values without checking
* **Bool conversion** - Use in conditionals: ``if (result) { /* success */ }``
* **MISRA compliant** - Fully compliant in both standard and ``STATIC_ONLY`` modes
* **Recommended pattern** - Preferred over exceptions for safety-critical and real-time code

Typical Use Cases
#################

* Embedded software
* Engineering calculations (FEM, CFD, PDEs)
* Adaptive data structures for large numerical grids
* Real-time simulation or optimization

.. toctree::
   :maxdepth: 1
   :caption: Modules:

    Utilities <Utilities>
    Smart Pointers <Pointers>
    Error <Error>
    Allocator <Allocator>
    String <String>
    
Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Getting Started
###############

Clone the repository:

.. code-block:: bash

    git clone https://github.com/Jon-Webb-79/csaltcpp.git
    cd csalt++

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
#############

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
