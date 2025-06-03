**************
SIMD Utilities 
**************

This section documents the SIMD trait templates and operation specializations used to accelerate
matrix operations for supported data types (`float` and `double`). These utilities are primarily
used internally by matrix classes but may be of general interest to advanced users.

SIMD Traits
===========

.. cpp:class:: template<typename T> simd_traits

   A generic trait structure to determine SIMD support for a given type.

   :tparam T: The data type to query for SIMD support

   .. cpp:member:: static constexpr bool supported

      Indicates whether SIMD is supported for the given type.

   .. cpp:member:: static constexpr std::size_t width

      Number of elements processed per SIMD register.

.. cpp:class:: template<> simd_traits<float>

   Specialization of `simd_traits` for `float`. Uses compile-time flags to determine SIMD capability.

   :var supported: ``true`` if `SIMD_WIDTH_FLOAT > 1`
   :var width: SIMD vector width in elements

.. cpp:class:: template<> simd_traits<double>

   Specialization of `simd_traits` for `double`.

   :var supported: ``true`` if `SIMD_WIDTH_DOUBLE > 1`
   :var width: SIMD vector width in elements

SIMD Operations
===============

.. cpp:class:: template<typename T> simd_ops

   Base template class for SIMD operations on type `T`. Specialized for supported types.

.. cpp:class:: template<> simd_ops<float>

   Provides AVX/SSE/fallback implementations of basic arithmetic operations on float arrays.

   .. cpp:function:: static void add(const float* a, const float* b, float* result, std::size_t size)

      Performs element-wise addition of two float arrays.

   .. cpp:function:: static void sub(const float* a, const float* b, float* result, std::size_t size)

      Performs element-wise subtraction of two float arrays.

   .. cpp:function:: static void add_scalar(const float* a, float scalar, float* result, std::size_t size)

      Adds a scalar to each element of a float array.

   .. cpp:function:: static void sub_scalar(const float* a, float scalar, float* result, std::size_t size)

      Subtracts a scalar from each element of a float array.

   .. cpp:function:: static void mul(const float* a, const float* b, float* result, std::size_t size)

      Performs element-wise multiplication of two float arrays.

   .. cpp:function:: static void mul_scalar(const float* a, float scalar, float* result, std::size_t size)

      Multiplies each float array element by a scalar.

   .. cpp:function:: static void div_scalar(const float* a, float scalar, float* result, std::size_t size)

      Divides each float array element by a scalar.

.. cpp:class:: template<> simd_ops<double>

   Provides AVX/SSE2/fallback implementations of basic arithmetic operations on double arrays.

   .. cpp:function:: static void add(const double* a, const double* b, double* result, std::size_t size)

      Performs element-wise addition of two double arrays.

   .. cpp:function:: static void sub(const double* a, const double* b, double* result, std::size_t size)

      Performs element-wise subtraction of two double arrays.

   .. cpp:function:: static void add_scalar(const double* a, double scalar, double* result, std::size_t size)

      Adds a scalar to each element of a double array.

   .. cpp:function:: static void sub_scalar(const double* a, double scalar, double* result, std::size_t size)

      Subtracts a scalar from each element of a double array.

   .. cpp:function:: static void mul(const double* a, const double* b, double* result, std::size_t size)

      Performs element-wise multiplication of two double arrays.

   .. cpp:function:: static void mul_scalar(const double* a, double scalar, double* result, std::size_t size)

      Multiplies each double array element by a scalar.

   .. cpp:function:: static void div_scalar(const double* a, double scalar, double* result, std::size_t size)

      Divides each double array element by a scalar.

