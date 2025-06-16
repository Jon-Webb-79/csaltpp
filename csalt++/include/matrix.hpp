// ================================================================================
// ================================================================================
// - File:    matrix.hpp
// - Purpose: Describe the file purpose here
//
// Source Metadata
// - Author:  Jonathan A. Webb
// - Date:    May 31, 2025
// - Version: 1.0
// - Copyright: Copyright 2022, Jon Webb Inc.
// ================================================================================
// ================================================================================
// Include modules here

#ifndef DENSE_MATRIX_HPP
#define DENSE_MATRIX_HPP

#include <vector>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <algorithm>
#include <iomanip>
#include <cassert>
#include <numeric>
#include <cmath>
#include <type_traits>


#ifdef __AVX2__
        #include <immintrin.h>
        #define SIMD_WIDTH_FLOAT 8
        #define SIMD_WIDTH_DOUBLE 4
#elif defined(__SSE2__)
        #include <emmintrin.h>
        #define SIMD_WIDTH_FLOAT 4
        #define SIMD_WIDTH_DOUBLE 2
#else
        #define SIMD_WIDTH_FLOAT 1
        #define SIMD_WIDTH_DOUBLE 1
#endif
// ================================================================================ 
// ================================================================================ 

    namespace slt {
        /// @brief SIMD trait template to determine SIMD capabilities for a given type.
        /// 
        /// This generic template assumes no SIMD support. Specializations
        /// for specific types (e.g., `float`, `double`) provide actual SIMD support info.
        /// 
        /// @tparam T The data type to query for SIMD support.
        // SIMD traits
        template<typename T>
        struct simd_traits {
            static constexpr bool supported = false;
            static constexpr std::size_t width = 1;
        };
// -------------------------------------------------------------------------------- 

        /// @brief SIMD traits specialization for `float`.
        ///
        /// Provides SIMD capability and vector width information for `float` types,
        /// based on compile-time SIMD availability (e.g., SSE, AVX).
        template<>
        struct simd_traits<float> {
            static constexpr bool supported = SIMD_WIDTH_FLOAT > 1;
            static constexpr std::size_t width = SIMD_WIDTH_FLOAT;
        };
// -------------------------------------------------------------------------------- 

        /// @brief SIMD traits specialization for `double`.
        ///
        /// Provides SIMD capability and vector width information for `double` types,
        /// based on compile-time SIMD availability (e.g., SSE2, AVX).
        template<>
        struct simd_traits<double> {
            static constexpr bool supported = SIMD_WIDTH_DOUBLE > 1;
            static constexpr std::size_t width = SIMD_WIDTH_DOUBLE;
        };
// -------------------------------------------------------------------------------- 

        // SIMD operations
        template<typename T> struct simd_ops;

// -------------------------------------------------------------------------------- 
       
        /**
         * @brief SIMD-accelerated operations for float arrays.
         *
         * This specialization of simd_ops provides AVX, SSE, or fallback implementations
         * of basic arithmetic operations for `float` arrays. Used to accelerate matrix operations.
         */
        template<>
        struct simd_ops<float> {
            /**
             * @brief Adds two float arrays element-wise.
             *
             * Performs `result[i] = a[i] + b[i]` for all elements. Uses AVX or SSE where available.
             *
             * @param a Pointer to the first input array.
             * @param b Pointer to the second input array.
             * @param result Pointer to the output array.
             * @param size Number of elements in the arrays.
             */
            static void add(const float* a, const float* b, float* result, std::size_t size) {
#if defined(__AVX2__)
                std::size_t end = size / 8 * 8;
                for (std::size_t i = 0; i < end; i += 8) {
                    __m256 va = _mm256_loadu_ps(&a[i]);
                    __m256 vb = _mm256_loadu_ps(&b[i]);
                    __m256 vr = _mm256_add_ps(va, vb);
                    _mm256_storeu_ps(&result[i], vr);
                }
#elif defined(__SSE2__)
                std::size_t end = size / 4 * 4;
                for (std::size_t i = 0; i < end; i += 4) {
                    __m128 va = _mm_loadu_ps(&a[i]);
                    __m128 vb = _mm_loadu_ps(&b[i]);
                    __m128 vr = _mm_add_ps(va, vb);
                    _mm_storeu_ps(&result[i], vr);
                }
#else
                std::size_t end = 0;
#endif
                for (std::size_t i = end; i < size; ++i)
                    result[i] = a[i] + b[i];
            }
// -------------------------------------------------------------------------------- 

            /**
             * @brief Subtracts two float arrays element-wise.
             *
             * Performs `result[i] = a[i] - b[i]` for all elements. Uses AVX or SSE where available.
             *
             * @param a Pointer to the first input array.
             * @param b Pointer to the second input array.
             * @param result Pointer to the output array.
             * @param size Number of elements in the arrays.
             */
            static void sub(const float* a, const float* b, float* result, std::size_t size) {
#if defined(__AVX2__)
                std::size_t end = size / 8 * 8;
                for (std::size_t i = 0; i < end; i += 8) {
                    __m256 va = _mm256_loadu_ps(&a[i]);
                    __m256 vb = _mm256_loadu_ps(&b[i]);
                    __m256 vr = _mm256_sub_ps(va, vb);
                    _mm256_storeu_ps(&result[i], vr);
                }
#elif defined(__SSE2__)
                std::size_t end = size / 4 * 4;
                for (std::size_t i = 0; i < end; i += 4) {
                    __m128 va = _mm_loadu_ps(&a[i]);
                    __m128 vb = _mm_loadu_ps(&b[i]);
                    __m128 vr = _mm_sub_ps(va, vb);
                    _mm_storeu_ps(&result[i], vr);
                }
#else
                std::size_t end = 0;
#endif
                for (std::size_t i = end; i < size; ++i)
                    result[i] = a[i] - b[i];
            }
// -------------------------------------------------------------------------------- 

            /**
             * @brief Adds a scalar to each element of a float array.
             *
             * Performs `result[i] = a[i] + scalar` for all elements. SIMD-accelerated where available.
             *
             * @param a Pointer to the input array.
             * @param scalar Scalar value to add.
             * @param result Pointer to the output array.
             * @param size Number of elements in the array.
             */
            static void add_scalar(const float* a, float scalar, float* result, std::size_t size) {
#if defined(__AVX2__)
                __m256 vscalar = _mm256_set1_ps(scalar);
                std::size_t end = size / 8 * 8;
                for (std::size_t i = 0; i < end; i += 8) {
                    __m256 va = _mm256_loadu_ps(&a[i]);
                    __m256 vr = _mm256_add_ps(va, vscalar);
                    _mm256_storeu_ps(&result[i], vr);
                }
#elif defined(__SSE2__)
                __m128 vscalar = _mm_set1_ps(scalar);
                std::size_t end = size / 4 * 4;
                for (std::size_t i = 0; i < end; i += 4) {
                    __m128 va = _mm_loadu_ps(&a[i]);
                    __m128 vr = _mm_add_ps(va, vscalar);
                    _mm_storeu_ps(&result[i], vr);
                }
#else
                std::size_t end = 0;
#endif
                for (std::size_t i = end; i < size; ++i)
                    result[i] = a[i] + scalar;
            }
// -------------------------------------------------------------------------------- 

            /**
             * @brief Subtracts a scalar from each element of a float array.
             *
             * Performs `result[i] = a[i] - scalar` for all elements. SIMD-accelerated where available.
             *
             * @param a Pointer to the input array.
             * @param scalar Scalar value to subtract.
             * @param result Pointer to the output array.
             * @param size Number of elements in the array.
             */
            static void sub_scalar(const float* a, float scalar, float* result, std::size_t size) {
#if defined(__AVX2__)
                __m256 vscalar = _mm256_set1_ps(scalar);
                std::size_t end = size / 8 * 8;
                for (std::size_t i = 0; i < end; i += 8) {
                    __m256 va = _mm256_loadu_ps(&a[i]);
                    __m256 vr = _mm256_sub_ps(va, vscalar);
                    _mm256_storeu_ps(&result[i], vr);
                }
#elif defined(__SSE2__)
                __m128 vscalar = _mm_set1_ps(scalar);
                std::size_t end = size / 4 * 4;
                for (std::size_t i = 0; i < end; i += 4) {
                    __m128 va = _mm_loadu_ps(&a[i]);
                    __m128 vr = _mm_sub_ps(va, vscalar);
                    _mm_storeu_ps(&result[i], vr);
                }
#else
                std::size_t end = 0;
#endif
                for (std::size_t i = end; i < size; ++i)
                    result[i] = a[i] - scalar;
            }
// -------------------------------------------------------------------------------- 

            /**
             * @brief Performs SIMD-accelerated element-wise multiplication of two float arrays.
             *
             * Multiplies each corresponding element of arrays `a` and `b`, storing the result in `result`.
             * Uses AVX or SSE where available, and falls back to scalar processing otherwise.
             *
             * @param a       Pointer to the first input array.
             * @param b       Pointer to the second input array.
             * @param result  Pointer to the output array.
             * @param size    Number of elements to process.
             */
            static void mul(const float* a, const float* b, float* result, std::size_t size) {
#if defined(__AVX2__)
                std::size_t end = size / 8 * 8;
                for (std::size_t i = 0; i < end; i += 8) {
                    __m256 va = _mm256_loadu_ps(&a[i]);
                    __m256 vb = _mm256_loadu_ps(&b[i]);
                    __m256 vr = _mm256_mul_ps(va, vb);
                    _mm256_storeu_ps(&result[i], vr);
                }
#elif defined(__SSE2__)
                std::size_t end = size / 4 * 4;
                for (std::size_t i = 0; i < end; i += 4) {
                    __m128 va = _mm_loadu_ps(&a[i]);
                    __m128 vb = _mm_loadu_ps(&b[i]);
                    __m128 vr = _mm_mul_ps(va, vb);
                    _mm_storeu_ps(&result[i], vr);
                }
#else
                std::size_t end = 0;
#endif
                for (std::size_t i = end; i < size; ++i)
                    result[i] = a[i] * b[i];
            }
// -------------------------------------------------------------------------------- 

            /**
             * @brief Performs SIMD-accelerated scalar multiplication on a float array.
             *
             * Multiplies each element in array `a` by the scalar `scalar`, storing the result in `result`.
             * Uses AVX or SSE where available, and falls back to scalar processing otherwise.
             *
             * @param a       Pointer to the input array.
             * @param scalar  Scalar value to multiply each element by.
             * @param result  Pointer to the output array.
             * @param size    Number of elements to process.
             */
            static void mul_scalar(const float* a, float scalar, float* result, std::size_t size) {
#if defined(__AVX2__)
                __m256 vscalar = _mm256_set1_ps(scalar);
                std::size_t end = size / 8 * 8;
                for (std::size_t i = 0; i < end; i += 8) {
                    __m256 va = _mm256_loadu_ps(&a[i]);
                    __m256 vr = _mm256_mul_ps(va, vscalar);
                    _mm256_storeu_ps(&result[i], vr);
                }
#elif defined(__SSE2__)
                __m128 vscalar = _mm_set1_ps(scalar);
                std::size_t end = size / 4 * 4;
                for (std::size_t i = 0; i < end; i += 4) {
                    __m128 va = _mm_loadu_ps(&a[i]);
                    __m128 vr = _mm_mul_ps(va, vscalar);
                    _mm_storeu_ps(&result[i], vr);
                }
#else
                std::size_t end = 0;
#endif
                for (std::size_t i = end; i < size; ++i)
                    result[i] = a[i] * scalar;
            }
// -------------------------------------------------------------------------------- 

            /**
             * @brief Divides each element of a float array by a scalar.
             *
             * Performs `result[i] = a[i] / scalar` for all elements. Utilizes AVX or SSE
             * SIMD instructions for hardware-accelerated performance where available.
             *
             * @param a Pointer to the input float array.
             * @param scalar The float scalar divisor.
             * @param result Pointer to the output float array.
             * @param size Number of elements in the array.
             *
             * @note Division by zero is undefined and may result in NaN or Inf.
             */
            static void div_scalar(const float* a, float scalar, float* result, std::size_t size) {
#if defined(__AVX2__)
                __m256 vscalar = _mm256_set1_ps(scalar);
                std::size_t end = size / 8 * 8;
                for (std::size_t i = 0; i < end; i += 8) {
                    __m256 va = _mm256_loadu_ps(&a[i]);
                    __m256 vr = _mm256_div_ps(va, vscalar);
                    _mm256_storeu_ps(&result[i], vr);
                }
#elif defined(__SSE2__)
                __m128 vscalar = _mm_set1_ps(scalar);
                std::size_t end = size / 4 * 4;
                for (std::size_t i = 0; i < end; i += 4) {
                    __m128 va = _mm_loadu_ps(&a[i]);
                    __m128 vr = _mm_div_ps(va, vscalar);
                    _mm_storeu_ps(&result[i], vr);
                }
#else
                std::size_t end = 0;
#endif
                for (std::size_t i = end; i < size; ++i)
                    result[i] = a[i] / scalar;
            }
// -------------------------------------------------------------------------------- 

            static void copy(const float* src, float* dst, std::size_t size) {
#if defined(__AVX2__)
                std::size_t end = size / 8 * 8;
                for (std::size_t i = 0; i < end; i += 8) {
                    __m256 v = _mm256_loadu_ps(&src[i]);
                    _mm256_storeu_ps(&dst[i], v);
                }
#elif defined(__SSE2__)
                std::size_t end = size / 4 * 4;
                for (std::size_t i = 0; i < end; i += 4) {
                    __m128 v = _mm_loadu_ps(&src[i]);
                    _mm_storeu_ps(&dst[i], v);
                }
#else
                std::size_t end = 0;
#endif
                for (std::size_t i = end; i < size; ++i)
                    dst[i] = src[i];
            }
// -------------------------------------------------------------------------------- 


        };
// -------------------------------------------------------------------------------- 

        /**
         * @brief SIMD-accelerated operations for double arrays.
         *
         * This specialization of simd_ops provides AVX, SSE2, or fallback implementations
         * of basic arithmetic operations for `double` arrays. Used to accelerate matrix operations.
         */
        template<>
        struct simd_ops<double> {

            /**
             * @brief Adds two double arrays element-wise.
             *
             * Performs `result[i] = a[i] + b[i]` for all elements. Uses AVX or SSE2 where available.
             *
             * @param a Pointer to the first input array.
             * @param b Pointer to the second input array.
             * @param result Pointer to the output array.
             * @param size Number of elements in the arrays.
             */
            static void add(const double* a, const double* b, double* result, std::size_t size) {
#if defined(__AVX2__)
                std::size_t end = size / 4 * 4;
                for (std::size_t i = 0; i < end; i += 4) {
                    __m256d va = _mm256_loadu_pd(&a[i]);
                    __m256d vb = _mm256_loadu_pd(&b[i]);
                    __m256d vr = _mm256_add_pd(va, vb);
                    _mm256_storeu_pd(&result[i], vr);
                }
#elif defined(__SSE2__)
                std::size_t end = size / 2 * 2;
                for (std::size_t i = 0; i < end; i += 2) {
                    __m128d va = _mm_loadu_pd(&a[i]);
                    __m128d vb = _mm_loadu_pd(&b[i]);
                    __m128d vr = _mm_add_pd(va, vb);
                    _mm_storeu_pd(&result[i], vr);
                }
#else
                std::size_t end = 0;
#endif
                for (std::size_t i = end; i < size; ++i)
                    result[i] = a[i] + b[i];
            }
// -------------------------------------------------------------------------------- 

            /**
             * @brief Subtracts two double arrays element-wise.
             *
             * Performs `result[i] = a[i] - b[i]` for all elements. Uses AVX or SSE2 where available.
             *
             * @param a Pointer to the first input array.
             * @param b Pointer to the second input array.
             * @param result Pointer to the output array.
             * @param size Number of elements in the arrays.
             */ 
            static void sub(const double* a, const double* b, double* result, std::size_t size) {
#if defined(__AVX2__)
                std::size_t end = size / 4 * 4;
                for (std::size_t i = 0; i < end; i += 4) {
                    __m256d va = _mm256_loadu_pd(&a[i]);
                    __m256d vb = _mm256_loadu_pd(&b[i]);
                    __m256d vr = _mm256_sub_pd(va, vb);
                    _mm256_storeu_pd(&result[i], vr);
                }
#elif defined(__SSE2__)
                std::size_t end = size / 2 * 2;
                for (std::size_t i = 0; i < end; i += 2) {
                    __m128d va = _mm_loadu_pd(&a[i]);
                    __m128d vb = _mm_loadu_pd(&b[i]);
                    __m128d vr = _mm_sub_pd(va, vb);
                    _mm_storeu_pd(&result[i], vr);
                }
#else
                std::size_t end = 0;
#endif
                for (std::size_t i = end; i < size; ++i)
                    result[i] = a[i] - b[i];
            }
// -------------------------------------------------------------------------------- 

            /**
             * @brief Adds a scalar to each element of a double array.
             *
             * Performs `result[i] = a[i] + scalar` for all elements. SIMD-accelerated where available.
             *
             * @param a Pointer to the input array.
             * @param scalar Scalar value to add.
             * @param result Pointer to the output array.
             * @param size Number of elements in the array.
             */
            static void add_scalar(const double* a, double scalar, double* result, std::size_t size) {
#if defined(__AVX2__)
                __m256d vscalar = _mm256_set1_pd(scalar);
                std::size_t end = size / 4 * 4;
                for (std::size_t i = 0; i < end; i += 4) {
                    __m256d va = _mm256_loadu_pd(&a[i]);
                    __m256d vr = _mm256_add_pd(va, vscalar);
                    _mm256_storeu_pd(&result[i], vr);
                }
#elif defined(__SSE2__)
                __m128d vscalar = _mm_set1_pd(scalar);
                std::size_t end = size / 2 * 2;
                for (std::size_t i = 0; i < end; i += 2) {
                    __m128d va = _mm_loadu_pd(&a[i]);
                    __m128d vr = _mm_add_pd(va, vscalar);
                    _mm_storeu_pd(&result[i], vr);
                }
#else
                std::size_t end = 0;
#endif
                for (std::size_t i = end; i < size; ++i)
                    result[i] = a[i] + scalar;
            }
// -------------------------------------------------------------------------------- 

            /**
             * @brief Subtracts a scalar from each element of a double array.
             *
             * Performs `result[i] = a[i] - scalar` for all elements. SIMD-accelerated where available.
             *
             * @param a Pointer to the input array.
             * @param scalar Scalar value to subtract.
             * @param result Pointer to the output array.
             * @param size Number of elements in the array.
             */
            static void sub_scalar(const double* a, double scalar, double* result, std::size_t size) {
#if defined(__AVX2__)
                __m256d vscalar = _mm256_set1_pd(scalar);
                std::size_t end = size / 4 * 4;
                for (std::size_t i = 0; i < end; i += 4) {
                    __m256d va = _mm256_loadu_pd(&a[i]);
                    __m256d vr = _mm256_sub_pd(va, vscalar);
                    _mm256_storeu_pd(&result[i], vr);
                }
#elif defined(__SSE2__)
                __m128d vscalar = _mm_set1_pd(scalar);
                std::size_t end = size / 2 * 2;
                for (std::size_t i = 0; i < end; i += 2) {
                    __m128d va = _mm_loadu_pd(&a[i]);
                    __m128d vr = _mm_sub_pd(va, vscalar);
                    _mm_storeu_pd(&result[i], vr);
                }
#else
                std::size_t end = 0;
#endif
                for (std::size_t i = end; i < size; ++i)
                    result[i] = a[i] - scalar;
            }
// -------------------------------------------------------------------------------- 

            /**
             * @brief Performs SIMD-accelerated element-wise multiplication of two double arrays.
             *
             * Multiplies each corresponding element of arrays `a` and `b`, storing the result in `result`.
             * Uses AVX or SSE where available, and falls back to scalar processing otherwise.
             *
             * @param a       Pointer to the first input array.
             * @param b       Pointer to the second input array.
             * @param result  Pointer to the output array.
             * @param size    Number of elements to process.
             */
            static void mul(const double* a, const double* b, double* result, std::size_t size) {
#if defined(__AVX2__)
                std::size_t end = size / 4 * 4;
                for (std::size_t i = 0; i < end; i += 4) {
                    __m256d va = _mm256_loadu_pd(&a[i]);
                    __m256d vb = _mm256_loadu_pd(&b[i]);
                    __m256d vr = _mm256_mul_pd(va, vb);
                    _mm256_storeu_pd(&result[i], vr);
                }
#elif defined(__SSE2__)
                std::size_t end = size / 2 * 2;
                for (std::size_t i = 0; i < end; i += 2) {
                    __m128d va = _mm_loadu_pd(&a[i]);
                    __m128d vb = _mm_loadu_pd(&b[i]);
                    __m128d vr = _mm_mul_pd(va, vb);
                    _mm_storeu_pd(&result[i], vr);
                }
#else
                std::size_t end = 0;
#endif
                for (std::size_t i = end; i < size; ++i)
                    result[i] = a[i] * b[i];
            }
// -------------------------------------------------------------------------------- 

            /**
             * @brief Performs SIMD-accelerated scalar multiplication on a double array.
             *
             * Multiplies each element in array `a` by the scalar `scalar`, storing the result in `result`.
             * Uses AVX or SSE where available, and falls back to scalar processing otherwise.
             *
             * @param a       Pointer to the input array.
             * @param scalar  Scalar value to multiply each element by.
             * @param result  Pointer to the output array.
             * @param size    Number of elements to process.
             */
            static void mul_scalar(const double* a, double scalar, double* result, std::size_t size) {
#if defined(__AVX2__)
                __m256d vscalar = _mm256_set1_pd(scalar);
                std::size_t end = size / 4 * 4;
                for (std::size_t i = 0; i < end; i += 4) {
                    __m256d va = _mm256_loadu_pd(&a[i]);
                    __m256d vr = _mm256_mul_pd(va, vscalar);
                    _mm256_storeu_pd(&result[i], vr);
                }
#elif defined(__SSE2__)
                __m128d vscalar = _mm_set1_pd(scalar);
                std::size_t end = size / 2 * 2;
                for (std::size_t i = 0; i < end; i += 2) {
                    __m128d va = _mm_loadu_pd(&a[i]);
                    __m128d vr = _mm_mul_pd(va, vscalar);
                    _mm_storeu_pd(&result[i], vr);
                }
#else
                std::size_t end = 0;
#endif
                for (std::size_t i = end; i < size; ++i)
                    result[i] = a[i] * scalar;
            }
// -------------------------------------------------------------------------------- 

            /**
             * @brief Divides each element of a double array by a scalar.
             *
             * Performs `result[i] = a[i] / scalar` for all elements. Utilizes AVX or SSE
             * SIMD instructions for hardware-accelerated performance where available.
             *
             * @param a Pointer to the input double array.
             * @param scalar The double scalar divisor.
             * @param result Pointer to the output double array.
             * @param size Number of elements in the array.
             *
             * @note Division by zero is undefined and may result in NaN or Inf.
             */
            static void div_scalar(const double* a, double scalar, double* result, std::size_t size) {
#if defined(__AVX2__)
                __m256d vscalar = _mm256_set1_pd(scalar);
                std::size_t end = size / 4 * 4;
                for (std::size_t i = 0; i < end; i += 4) {
                    __m256d va = _mm256_loadu_pd(&a[i]);
                    __m256d vr = _mm256_div_pd(va, vscalar);
                    _mm256_storeu_pd(&result[i], vr);
                }
#elif defined(__SSE2__)
                __m128d vscalar = _mm_set1_pd(scalar);
                std::size_t end = size / 2 * 2;
                for (std::size_t i = 0; i < end; i += 2) {
                    __m128d va = _mm_loadu_pd(&a[i]);
                    __m128d vr = _mm_div_pd(va, vscalar);
                    _mm_storeu_pd(&result[i], vr);
                }
#else
                std::size_t end = 0;
#endif
                for (std::size_t i = end; i < size; ++i)
                    result[i] = a[i] / scalar;
            }
// -------------------------------------------------------------------------------- 

            static void copy(const double* src, double* dst, std::size_t size) {
#if defined(__AVX2__)
                std::size_t end = size / 4 * 4;
                for (std::size_t i = 0; i < end; i += 4) {
                    __m256d v = _mm256_loadu_pd(&src[i]);
                    _mm256_storeu_pd(&dst[i], v);
                }
#elif defined(__SSE2__)
                std::size_t end = size / 2 * 2;
                for (std::size_t i = 0; i < end; i += 2) {
                    __m128d v = _mm_loadu_pd(&src[i]);
                    _mm_storeu_pd(&dst[i], v);
                }
#else
                std::size_t end = 0;
#endif
                for (std::size_t i = end; i < size; ++i)
                    dst[i] = src[i];
            }
        };
// ================================================================================ 
// ================================================================================ 

    /**
     * @brief Abstract base class for matrix types.
     *
     * Provides a uniform interface for different matrix implementations (e.g., dense, sparse),
     * supporting essential matrix operations such as element access, mutation, and cloning.
     *
     * @tparam T The numeric type of the matrix elements (e.g., float, double).
     */
    template<typename T>
    class MatrixBase {
    public:
        /**
         * @brief Virtual destructor for safe polymorphic deletion.
         */
        virtual ~MatrixBase() = default;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Returns the number of rows in the matrix.
         *
         * @return Number of rows.
         */
        virtual std::size_t rows() const = 0;
// -------------------------------------------------------------------------------- 
    
        /**
         * @brief Returns the number of columns in the matrix.
         *
         * @return Number of columns.
         */
        virtual std::size_t cols() const = 0;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Retrieves the value at a specific matrix coordinate.
         *
         * @param row Zero-based row index.
         * @param col Zero-based column index.
         * @return Value at the specified location.
         */
        virtual T get(std::size_t row, std::size_t col) const = 0;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Sets the value at a specific matrix coordinate.
         *
         * @param row Zero-based row index.
         * @param col Zero-based column index.
         * @param value Value to assign.
         */
        virtual void set(std::size_t row, std::size_t col, T value) = 0;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Creates a polymorphic copy of the matrix object.
         *
         * Useful for cloning objects when only a base class pointer/reference is available.
         *
         * @return A std::unique_ptr to a new MatrixBase-derived object with the same contents.
         */
        virtual std::unique_ptr<MatrixBase<T>> clone() const = 0;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Determines if a row column pair is initialized 
         *
         * @return true if initialized false otherwise
         */
        virtual bool is_initialized(std::size_t row, std::size_t col) const = 0;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Returns the number of initialized elements in the matrix 
         *
         * @return The number of initialized elements in the matrix
         */
        virtual std::size_t nonzero_count() const = 0;
// -------------------------------------------------------------------------------- 

        /**
         * @brief The total size of the matrix 
         *
         * @return The number of columns multiplied by the number of rows
         */
        virtual std::size_t size() const = 0;
    };
// ================================================================================ 
// ================================================================================ 
    // Dense matrix class

    /**
     * @brief A dense matrix implementation supporting float or double values.
     *
     * Stores matrix elements in a contiguous 1D vector using row-major order.
     * Supports basic arithmetic operations, element access, cloning, and transposition.
     *
     * @tparam T Must be float or double. Enforced via static assertion.
     */
    template<typename T>
    class DenseMatrix : public MatrixBase<T> {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                      "DenseMatrix only supports float or double");

    private:
        std::vector<T> data; ///< Flat row-major storage of matrix elements. 
        std::vector<uint8_t> init; ///< A vector containin a binary representation of array initialization.
        std::size_t rows_ = 0; ///< Number of rows 
        std::size_t cols_ = 0; ///< Number of cols 
    // ================================================================================ 
    public:
        /**
         * @brief The total size of the matrix 
         *
         * @return The number of rows multiplied by the number of columns, 0 if not initialized
         */
        std::size_t size() const override {return rows_ * cols_;}
// -------------------------------------------------------------------------------- 

        /**
         * @brief Pointer to the first value within the data array.
         *
         * Returns a pointer to the beginning of the matrix's internal data array,
         * stored in row-major order. This is useful for passing the data to low-level
         * numerical libraries or performing custom SIMD operations.
         *
         * @return A const pointer to the beginning of the matrix data in contiguous memory. Returns nullptr if not initialized
         *
         * @code
         * #include <iostream>
         * #include "matrix.hpp" // assuming DenseMatrix is defined here
         *
         * int main() {
         *     slt::DenseMatrix<float> mat(2, 3, 1.5f);
         *     const float* ptr = mat.data_ptr();
         *
         *     for (std::size_t i = 0; i < mat.size(); ++i)
         *         std::cout << ptr[i] << " ";
         *     std::cout << std::endl;
         *     return 0;
         * }
         * @endcode
         *
         * Output:
         * @code
         * 1.5 1.5 1.5 1.5 1.5 1.5
         * @endcode
         */
        const T* data_ptr() const {return data.data();}
// -------------------------------------------------------------------------------- 

        /**
         * @brief Pointer to the first value within the data array.
         *
         * Returns a pointer to the beginning of the matrix's internal data array,
         * stored in row-major order. This is useful for passing the data to low-level
         * numerical libraries or performing custom SIMD operations.
         *
         * @return A pointer to the beginning of the matrix data in contiguous memory.  Returns nullptr if not initialized
         *
         * @code
         * #include <iostream>
         * #include "matrix.hpp" // assuming DenseMatrix is defined here
         *
         * int main() {
         *     slt::DenseMatrix<float> mat(2, 3, 1.5f);
         *     float* ptr = mat.data_ptr();
         *
         *     for (std::size_t i = 0; i < mat.size(); ++i)
         *         std::cout << ptr[i] << " ";
         *     std::cout << std::endl;
         *     return 0;
         * }
         * @endcode
         *
         * Output:
         * @code
         * 1.5 1.5 1.5 1.5 1.5 1.5
         * @endcode
         */
        T* data_ptr() {return data.data();}
// -------------------------------------------------------------------------------- 

        /**
         * @brief Pointer to the first value within the init array.
         *
         * Returns a pointer to the beginning of the matrix's internal init array,
         * stored in row-major order. This is may be useful for debugging issues if initialized data is showing as uninitialized
         *
         * @return A pointer to the beginning of the matrix data in contiguous memory.  Returns nullptr if not initialized
         *
         * @code
         * #include <iostream>
         * #include <stdint>
         * #include "matrix.hpp" // assuming DenseMatrix is defined here
         *
         * int main() {
         *     slt::DenseMatrix<float> mat(2, 3);
         *     mat.set(0, 0, 1.0f);
         *     mat.set(0, 1, 2.0f);
         *     uint8_t* ptr = mat.init_ptr();
         *
         *     for (std::size_t i = 0; i < mat.size(); ++i)
         *         std::cout << ptr[i] << " ";
         *     std::cout << std::endl;
         *     return 0;
         * }
         * @endcode
         *
         * Output:
         * @code
         * 1, 1, 0, 0, 0, 0
         * @endcode
         */
        const uint8_t* init_ptr() const {return init.data();}
        uint8_t* init_ptr() {return init.data();}
// -------------------------------------------------------------------------------- 

        /**
         * @brief Returns the number of initialized elements in the matrix.
         *
         * This function scans the internal `init` vector and returns the number
         * of elements that have been explicitly initialized. This allows tracking
         * sparse-style usage in a dense matrix implementation.
         *
         * @return The number of initialized elements in the matrix.
         *
         * @code
         * #include <iostream>
         * #include "matrix.hpp"  // assuming DenseMatrix is defined here
         *
         * int main() {
         *     slt::DenseMatrix<float> mat(2, 3);
         *     mat.set(0, 0, 3.14f);
         *     mat.set(1, 1, 2.71f);
         *     std::cout << "Initialized elements: " << mat.nonzero_count() << std::endl;
         *     return 0;
         * }
         * @endcode
         *
         * Output:
         * @code
         * Initialized elements: 2
         * @endcode
         */
        std::size_t nonzero_count() const override {
            if (init.empty()) return 0;
            return std::count(init.begin(), init.end(), static_cast<uint8_t>(1));
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Constructs a matrix with given dimensions and fills it with a specified value.
         *
         * This constructor initializes all elements of the matrix to a given value
         * and marks them as initialized.
         *
         * @tparam T Numeric data type of matrix elements.  Must be either `float` or `double`.
         * @param r Number of rows.
         * @param c Number of columns.
         * @param value The value to assign to each matrix element.
         *
         * @code
         * #include <iostream>
         * #include "matrix.hpp"  // assuming DenseMatrix is defined here
         *
         * int main() {
         *     slt::DenseMatrix<float> mat(2, 3, 5.0f);
         *     const float* ptr = mat.data_ptr();
         *
         *     for (std::size_t i = 0; i < mat.size(); ++i)
         *         std::cout << ptr[i] << " ";
         *     std::cout << std::endl;
         *     return 0;
         * }
         * @endcode
         *
         * Output:
         * @code
         * 5 5 5 5 5 5
         * @endcode
         */
        DenseMatrix(std::size_t r, std::size_t c, T value)
            : data(r * c, value), init(r * c, 1), rows_(r), cols_(c) {}
// -------------------------------------------------------------------------------- 

        /**
         * @brief Constructs a matrix with given dimensions and zero-initializes all elements.
         *
         * This constructor sets all values in the matrix to zero and marks them as uninitialized.
         * It is typically used when data will be populated later via `set()` or similar methods.
         *
         * @param r Number of rows.
         * @param c Number of columns.
         *
         * @code
         * #include <iostream>
         * #include "matrix.hpp"  // assuming DenseMatrix is defined here
         *
         * int main() {
         *     slt::DenseMatrix<float> mat(2, 3);
         *     mat.set(0, 1, 42.0f);
         *     mat.set(1, 2, 7.0f);
         *
         *     const float* ptr = mat.data_ptr();
         *     for (std::size_t i = 0; i < mat.size(); ++i)
         *         std::cout << ptr[i] << " ";
         *     std::cout << std::endl;
         *
         *     std::cout << "Initialized count: " << mat.nonzero_count() << std::endl;
         *     return 0;
         * }
         * @endcode
         *
         * Output:
         * @code
         * 0 42 0 0 0 7
         * Initialized count: 2
         * @endcode
         */
        DenseMatrix(std::size_t r, std::size_t c)
            : data(r * c, 0), init(r * c, 0), rows_(r), cols_(c) {}
// -------------------------------------------------------------------------------- 

        /**
         * @brief Constructs a matrix from a nested std::vector of values.
         *
         * Initializes the matrix with the contents of a row-major nested `std::vector`.
         * All rows must have the same number of columns, otherwise an exception is thrown.
         *
         * @tparam T Numeric data type of matrix elements.  Must be either `float` or `double`.
         * @param vec A 2D vector representing matrix data in row-major order.
         * @throws std::invalid_argument if rows have inconsistent sizes.
         *
         * @code
         * std::vector<std::vector<float>> values = {
         *     {1.0f, 2.0f},
         *     {3.0f, 4.0f}
         * };
         * slt::DenseMatrix<float> mat(values);
         * std::cout << mat.get(1, 0); // Output: 3.0
         * @endcode
         */
        DenseMatrix(const std::vector<std::vector<T>>& vec) {
            rows_ = vec.size();
            cols_ = rows_ ? vec[0].size() : 0;
            data.resize(rows_ * cols_);
            init.resize(rows_ * cols_, 1);
            for (std::size_t i = 0; i < rows_; ++i) {
                if (vec[i].size() != cols_)
                    throw std::invalid_argument("All rows must have the same number of columns");
                for (std::size_t j = 0; j < cols_; ++j)
                    data[i * cols_ + j] = vec[i][j];
            }
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Constructs a matrix from a fixed-size std::array of std::array values.
         *
         * This constructor allows initializing a matrix from a compile-time known
         * 2D `std::array` layout. The matrix is fully initialized.
         *
         * @tparam Rows Number of rows (inferred at compile time)
         * @tparam Cols Number of columns (inferred at compile time)
         * @tparam T Numeric data type of matrix elements.  Must be either `float` or `double`.


         * @param arr A 2D array containing matrix values in row-major order.
         
         *
         * @code
         * std::array<std::array<double, 2>, 2> arr = {{
         *     {1.1, 1.2},
         *     {2.1, 2.2}
         * }};
         * slt::DenseMatrix<double> mat(arr);
         * std::cout << mat.get(0, 1); // Output: 1.2
         * @endcode
         */
        template<std::size_t Rows, std::size_t Cols>
        DenseMatrix(const std::array<std::array<T, Cols>, Rows>& arr)
            : data(Rows * Cols), init(Rows * Cols, 1), rows_(Rows), cols_(Cols) {
            for (std::size_t i = 0; i < Rows; ++i)
                for (std::size_t j = 0; j < Cols; ++j)
                    data[i * Cols + j] = arr[i][j];
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Constructs a matrix from an initializer list of initializer lists.
         *
         * Enables matrix initialization using brace-enclosed values in row-major order.
         * All rows must be of equal length or an exception will be thrown.
         *
         * @param init_list Nested initializer list representing the matrix contents.
         * @throws std::invalid_argument if any row has inconsistent size.
         *
         * @code
         * #include "matrix.hpp"
         * #include <iostream>
         *
         * int main() {
         *     slt::DenseMatrix<float> mat = {
         *         {1.0f, 2.0f},
         *         {3.0f, 4.0f}
         *     };
         *     std::cout << mat.get(1, 0); // Output: 3.0
         *     return 0;
         * }
         * @endcode
         */
        DenseMatrix(std::initializer_list<std::initializer_list<T>> init_list) {
            rows_ = init_list.size();
            cols_ = rows_ ? init_list.begin()->size() : 0;
            data.reserve(rows_ * cols_);
            init.reserve(rows_ * cols_);
            for (const auto& row : init_list) {
                if (row.size() != cols_)
                    throw std::invalid_argument("All rows must have the same number of columns");
                data.insert(data.end(), row.begin(), row.end());
                init.insert(init.end(), row.size(), 1);
            }
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Constructs a matrix from a flat data vector with explicit dimensions.
         *
         * The data must be laid out in row-major order, and the vector must have exactly
         * `rows * cols` elements. All elements are marked initialized.
         *
         * @param flat_data Flat vector of values in row-major order.
         * @param r Number of rows.
         * @param c Number of columns.
         * @throws std::invalid_argument if the data size does not match r * c.
         *
         * @code
         * #include "matrix.hpp"
         * #include <iostream>
         * #include <vector>
         *
         * int main() {
         *     std::vector<double> data = {1.0, 2.0, 3.0, 4.0};
         *     slt::DenseMatrix<double> mat(data, 2, 2);
         *     std::cout << mat.get(0, 1); // Output: 2.0
         *     return 0;
         * }
         * @endcode
         */
        DenseMatrix(const std::vector<T>& flat_data, std::size_t r, std::size_t c)
            : data(flat_data), init(flat_data.size(), 1), rows_(r), cols_(c) {
            if (flat_data.size() != r * c)
                throw std::invalid_argument("Flat data size does not match matrix dimensions");
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Constructs a matrix from a flat std::array with specified dimensions.
         *
         * The flat array must be laid out in row-major order and its size must exactly
         * match `rows * cols`. All elements are marked initialized.
         *
         * @tparam N Size of the flat std::array.
         * @param arr Flat array containing matrix data in row-major order.
         * @param r Number of rows.
         * @param c Number of columns.
         * @throws std::invalid_argument if N does not match r * c.
         *
         * @code
         * #include "matrix.hpp"
         * #include <array>
         * #include <iostream>
         *
         * int main() {
         *     std::array<float, 6> arr = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
         *     slt::DenseMatrix<float> mat(arr, 2, 3);
         *     std::cout << mat.get(1, 2); // Output: 6.0
         *     return 0;
         * }
         * @endcode
         */
        template<std::size_t N>
        DenseMatrix(const std::array<T, N>& arr, std::size_t r, std::size_t c)
            : data(arr.begin(), arr.end()), init(N, 1), rows_(r), cols_(c) {
            if (N != r * c)
                throw std::invalid_argument("Flat array size does not match matrix dimensions");
        }

// --------------------------------------------------------------------------------

        /**
         * @brief Copy constructor for DenseMatrix.
         *
         * Creates a deep copy of another DenseMatrix, duplicating its internal data and
         * initialization state. The resulting matrix is independent of the original.
         *
         * @param other The DenseMatrix instance to copy.
         *
         * @code
         * slt::DenseMatrix<double> A(2, 2, 1.0);
         * slt::DenseMatrix<double> B(A);  // B is a deep copy of A
         * std::cout << B.get(0, 0);        // Output: 1.0
         * @endcode
         */
        DenseMatrix(const DenseMatrix<T>& other)
            : MatrixBase<T>(),
              data(other.data),
              init(other.init),
              rows_(other.rows_),
              cols_(other.cols_) {}

// -------------------------------------------------------------------------------- 

        /**
         * @brief Move constructor for DenseMatrix.
         *
         * Transfers ownership of the internal data from another DenseMatrix. This is a
         * lightweight operation that avoids deep copying, and the original matrix is
         * left in a valid but empty state.
         *
         * @param other The DenseMatrix to move from. It will be reset to a zero-sized state.
         *
         * @code
         * slt::DenseMatrix<float> A(3, 3, 2.0f);
         * slt::DenseMatrix<float> B = std::move(A);  // B takes ownership of A's data
         * std::cout << B.get(2, 2);                  // Output: 2.0
         * std::cout << A.size();                     // Output: 0
         * @endcode
         */
        DenseMatrix(DenseMatrix<T>&& other) noexcept
            : MatrixBase<T>(),
              data(std::move(other.data)),
              init(std::move(other.init)),
              rows_(std::exchange(other.rows_, 0)),
              cols_(std::exchange(other.cols_, 0)) {}
// -------------------------------------------------------------------------------- 

        /**
         * @brief Constructs a square identity matrix of size n x n.
         *
         * This constructor initializes a square matrix with 1s on the main diagonal and
         * 0s elsewhere. It marks all diagonal entries as initialized.
         *
         * @param n Size of the identity matrix (rows and columns).
         *
         * @throws std::invalid_argument if n is zero.
         */
        explicit DenseMatrix(std::size_t n) : data(n * n, 0), init(n * n, 0), rows_(n), cols_(n) {
            if (n == 0)
                throw std::invalid_argument("Size of identity matrix must be greater than zero");

            for (std::size_t i = 0; i < n; ++i) {
                std::size_t idx = i * n + i;
                data[idx] = static_cast<T>(1);
                init[idx] = 1;
            }
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Access or assign a value at the specified matrix index (r, c).
         *
         * This non-const overload allows users to assign a value to an element. If the
         * element has not been previously initialized (tracked via the internal `init` vector),
         * it will be marked as initialized. If already initialized, it acts as a regular update.
         *
         * Bounds checking is performed; if the index is out of range, std::out_of_range is thrown.
         *
         * @param r Row index
         * @param c Column index
         * @return Reference to the value at the specified index
         *
         * @code
         * slt::DenseMatrix<float> mat(2, 3);
         * mat(0, 1) = 4.2f;  // Initializes and sets the value
         * mat(0, 1) = 5.0f;  // Updates existing value
         * std::cout << mat(0, 1);  // Outputs: 5.0
         * @endcode
         */
        T& operator()(std::size_t r, std::size_t c) {
            if (r >= rows_ || c >= cols_)
                throw std::out_of_range("Matrix index out of bounds");

            std::size_t idx = r * cols_ + c;

            // If value is not initialized, we assume this is the first assignment
            if (!init[idx])
                init[idx] = 1;

            return data[idx];
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Read-only access to a matrix element at (r, c).
         *
         * This const overload allows read-only access to a matrix element.
         * Throws a std::runtime_error if the element has not been initialized via `set()`,
         * `operator()`, or `update()`.
         *
         * Bounds checking is performed; if the index is out of range, std::out_of_range is thrown.
         *
         * @param r Row index
         * @param c Column index
         * @return Const reference to the initialized value at (r, c)
         *
         * @code
         * slt::DenseMatrix<float> mat(2, 3);
         * mat.set(1, 2, 8.5f);
         * std::cout << mat(1, 2);  // Outputs: 8.5
         *
         * // mat(0, 0); // Would throw std::runtime_error since it's uninitialized
         * @endcode
         */
        const T& operator()(std::size_t r, std::size_t c) const {
            if (r >= rows_ || c >= cols_)
                throw std::out_of_range("Matrix index out of bounds");

            std::size_t idx = r * cols_ + c;

            if (!init[idx])
                throw std::runtime_error("Attempted to access uninitialized matrix value");

            return data[idx];
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Copy assignment operator for DenseMatrix.
         *
         * Copies the contents of another matrix, including data values,
         * initialization status, and dimensions.
         *
         * @param other The matrix to copy from.
         * @return Reference to the current matrix after copy.
         *
         * @code
         * slt::DenseMatrix<double> A(2, 2, 1.0);
         * slt::DenseMatrix<double> B = A;  // uses copy constructor
         * slt::DenseMatrix<double> C;
         * C = A;  // uses copy assignment
         * @endcode
         */
        DenseMatrix<T>& operator=(const DenseMatrix<T>& other) {
            if (this != &other) {
                data = other.data;
                init = other.init;
                rows_ = other.rows_;
                cols_ = other.cols_;
            }
            return *this;
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Move assignment operator for DenseMatrix.
         *
         * Transfers ownership of the resources from another matrix to this one.
         * After the move, the source matrix is left in a valid but unspecified state
         * (typically zero dimensions and empty internal buffers).
         *
         * This is useful for efficient reassignment of temporary matrices without
         * deep copying data.
         *
         * @param other The source matrix to move from (rvalue reference).
         * @return Reference to this matrix after move assignment.
         *
         * @code
         * #include <iostream>
         * #include <utility>  // For std::move
         * #include "matrix.hpp"
         *
         * int main() {
         *     slt::DenseMatrix<float> mat1(2, 2, 3.0f);
         *     slt::DenseMatrix<float> mat2;
         *
         *     mat2 = std::move(mat1);  // Efficient resource transfer
         *
         *     std::cout << mat2(0, 0);  // Outputs: 3.0
         *
         *     // mat1 is now in a valid but empty state
         *     std::cout << "Size after move: " << mat1.size();  // Outputs: 0
         *     return 0;
         * }
         * @endcode
         */ 
        DenseMatrix<T>& operator=(DenseMatrix<T>&& other) noexcept {
            if (this != &other) {
                data = std::move(other.data);
                init = std::move(other.init);
                rows_ = std::exchange(other.rows_, 0);
                cols_ = std::exchange(other.cols_, 0);
            }
            return *this;
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Adds two DenseMatrix objects element-wise.
         *
         * Performs element-wise addition between two matrices of equal dimensions.
         * If SIMD is supported, the addition is vectorized for performance.
         *
         * The returned matrix will be fully initialized.
         *
         * @param other The matrix to add.
         * @return A new DenseMatrix containing the sum of the current matrix and `other`.
         *
         * @throws std::invalid_argument if matrix dimensions do not match.
         *
         * @code
         * slt::DenseMatrix<float> A(2, 2, 1.0f);
         * slt::DenseMatrix<float> B(2, 2, 2.0f);
         * slt::DenseMatrix<float> C = A + B;
         * std::cout << C(0, 0);  // Outputs: 3.0
         * @endcode
         */
        DenseMatrix operator+(const DenseMatrix& other) const {
            if (rows_ != other.rows_ || cols_ != other.cols_)
                throw std::invalid_argument("Matrix dimensions must match for addition");

            DenseMatrix result(rows_, cols_);
            if constexpr (simd_traits<T>::supported) {
                simd_ops<T>::add(data.data(), other.data.data(), result.data.data(), data.size());
                std::fill(result.init.begin(), result.init.end(), 1);  // ← this line is crucial
            } else {
                for (std::size_t i = 0; i < data.size(); ++i) {
                    result.data[i] = data[i] + other.data[i];
                    result.init[i] = 1;  // ← also crucial
                }
            }
            return result;
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Adds a scalar to all elements of the matrix.
         *
         * Each element of the matrix is incremented by the given scalar value.
         * The result is stored in a new DenseMatrix that is fully initialized.
         * SIMD acceleration is used if supported by the platform.
         *
         * @param scalar The value to add to each element.
         * @return A new DenseMatrix containing the result of the scalar addition.
         *
         * @code
         * slt::DenseMatrix<float> A(2, 2, 1.0f);
         * slt::DenseMatrix<float> B = A + 3.0f;
         * std::cout << B(0, 0);  // Outputs: 4.0
         * @endcode
         */
        DenseMatrix operator+(T scalar) const {
            DenseMatrix result(rows_, cols_);
            if constexpr (simd_traits<T>::supported) {
                simd_ops<T>::add_scalar(data.data(), scalar, result.data.data(), data.size());
                std::fill(result.init.begin(), result.init.end(), 1);
            } else {
                for (std::size_t i = 0; i < data.size(); ++i) {
                    result.data[i] = data[i] + scalar;
                    result.init[i] = 1;
                }
            }
            return result;
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Element-wise matrix subtraction.
         *
         * Subtracts another DenseMatrix from this matrix element-wise. Both matrices
         * must have the same shape, otherwise an exception is thrown. If SIMD is supported,
         * subtraction is performed using optimized SIMD instructions.
         *
         * @param other Matrix to subtract from this matrix.
         * @return A new DenseMatrix representing the result of the subtraction.
         *
         * @throws std::invalid_argument if matrix dimensions do not match.
         *
         * @code
         * slt::DenseMatrix<float> A(2, 2, 4.0f);
         * slt::DenseMatrix<float> B(2, 2, 1.0f);
         * auto C = A - B;
         * // C now contains all 3.0 values
         * @endcode
         */
        DenseMatrix operator-(const DenseMatrix& other) const {
            if (rows_ != other.rows_ || cols_ != other.cols_)
                throw std::invalid_argument("Matrix dimensions must match for subtraction");

            DenseMatrix result(rows_, cols_);
            if constexpr (simd_traits<T>::supported) {
                simd_ops<T>::sub(data.data(), other.data.data(), result.data.data(), data.size());
                std::fill(result.init.begin(), result.init.end(), 1);  // Mark all entries as initialized
            } else {
                for (std::size_t i = 0; i < data.size(); ++i) {
                    result.data[i] = data[i] - other.data[i];
                    result.init[i] = 1;  // Mark entry as initialized
                }
            }
            return result;
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Scalar subtraction from all elements of the matrix.
         *
         * Subtracts a scalar value from each element in the matrix. If SIMD is available,
         * it uses optimized instructions for faster execution.
         *
         * @param scalar Value to subtract from each element.
         * @return A new DenseMatrix containing the result.
         *
         * @code
         * slt::DenseMatrix<double> mat(2, 2, 5.0);
         * auto result = mat - 2.0;
         * // result contains all 3.0 values
         * @endcode
         */
        DenseMatrix operator-(T scalar) const {
            DenseMatrix result(rows_, cols_);
            if constexpr (simd_traits<T>::supported) {
                simd_ops<T>::sub_scalar(data.data(), scalar, result.data.data(), data.size());
                std::fill(result.init.begin(), result.init.end(), 1);  // Mark as initialized
            } else {
                for (std::size_t i = 0; i < data.size(); ++i) {
                    result.data[i] = data[i] - scalar;
                    result.init[i] = 1;  // Mark each element as initialized
                }
            }
            return result;
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Element-wise matrix multiplication.
         *
         * Multiplies this matrix element-wise with another matrix of the same dimensions.
         * Throws an exception if the matrices differ in shape. Uses SIMD acceleration if supported.
         *
         * @param other The matrix to multiply with.
         * @return A new DenseMatrix containing the element-wise product.
         *
         * @throws std::invalid_argument if matrix dimensions do not match.
         *
         * @code
         * slt::DenseMatrix<float> A(2, 2);
         * A.set(0, 0, 2.0f); A.set(0, 1, 3.0f);
         * A.set(1, 0, 4.0f); A.set(1, 1, 5.0f);
         *
         * slt::DenseMatrix<float> B(2, 2, 2.0f);  // filled with 2.0
         * auto C = A * B;
         * std::cout << C(0, 0);  // Outputs: 4.0
         * @endcode
         */
        DenseMatrix operator*(const DenseMatrix& other) const {
            if (rows_ != other.rows_ || cols_ != other.cols_)
                throw std::invalid_argument("Matrix dimensions must match for element-wise multiplication");

            DenseMatrix result(rows_, cols_);
            if constexpr (simd_traits<T>::supported) {
                simd_ops<T>::mul(data.data(), other.data.data(), result.data.data(), data.size());
                std::fill(result.init.begin(), result.init.end(), 1);
            } else {
                for (std::size_t i = 0; i < data.size(); ++i) {
                    result.data[i] = data[i] * other.data[i];
                    result.init[i] = 1;
                }
            }
            return result;
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Multiply all matrix elements by a scalar.
         *
         * Returns a new matrix with each element multiplied by the provided scalar value.
         * This operation uses SIMD acceleration if available.
         *
         * @param scalar Value to multiply each matrix element by.
         * @return A new DenseMatrix containing the scaled values.
         *
         * @code
         * slt::DenseMatrix<double> A(2, 2, 3.0);
         * auto B = A * 2.0;
         * std::cout << B(1, 1);  // Outputs: 6.0
         * @endcode
         */
        DenseMatrix operator*(T scalar) const {
            DenseMatrix result(rows_, cols_);
            if constexpr (simd_traits<T>::supported) {
                simd_ops<T>::mul_scalar(data.data(), scalar, result.data.data(), data.size());
                std::fill(result.init.begin(), result.init.end(), 1);
            } else {
                for (std::size_t i = 0; i < data.size(); ++i) {
                    result.data[i] = data[i] * scalar;
                    result.init[i] = 1;
                }
            }
            return result;
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Divide all matrix elements by a scalar.
         *
         * Returns a new matrix with each element divided by the given scalar value.
         * Uses SIMD acceleration if available. Division by zero is explicitly checked
         * and will throw an exception if detected.
         *
         * @param scalar The scalar divisor.
         * @return A new DenseMatrix with scaled-down values.
         *
         * @throws std::invalid_argument if scalar is zero.
         *
         * @code
         * slt::DenseMatrix<float> A(2, 2, 8.0f);
         * auto B = A / 2.0f;
         * std::cout << B(0, 0);  // Outputs: 4.0
         * @endcode
         */
        DenseMatrix operator/(T scalar) const {
            if (scalar == T{}) throw std::invalid_argument("Division by zero");

            DenseMatrix result(rows_, cols_);
            if constexpr (simd_traits<T>::supported) {
                simd_ops<T>::div_scalar(data.data(), scalar, result.data.data(), data.size());
                std::fill(result.init.begin(), result.init.end(), 1);
            } else {
                for (std::size_t i = 0; i < data.size(); ++i) {
                    result.data[i] = data[i] / scalar;
                    result.init[i] = 1;
                }
            }
            return result;
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Transposes the matrix in-place.
         *
         * This function swaps the rows and columns of the matrix, modifying it directly.
         * It is only applicable to dense matrices and is performed without creating a new object.
         *
         * The `init` state of all elements is preserved.
         *
         * @code
         * slt::DenseMatrix<float> mat({
         *     {1.0f, 2.0f},
         *     {3.0f, 4.0f}
         * });
         *
         * mat.transpose();
         *
         * std::cout << mat(0, 1);  // Outputs: 3.0
         * std::cout << mat(1, 0);  // Outputs: 2.0
         * @endcode
         */
        void transpose() {
            std::vector<T> new_data(data.size());
            for (std::size_t i = 0; i < rows_; ++i) {
                for (std::size_t j = 0; j < cols_; ++j) {
                    new_data[j * rows_ + i] = data[i * cols_ + j];
                }
            }
            data.swap(new_data);
            std::swap(rows_, cols_);
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Computes and returns the inverse of a square matrix.
         *
         * This function implements Gauss-Jordan elimination with partial pivoting.
         * It throws if the matrix is not square or is singular (i.e., non-invertible).
         *
         * All elements of the result are marked as initialized.
         *
         * @return A new DenseMatrix object containing the inverse.
         *
         * @throws std::invalid_argument if the matrix is not square.
         * @throws std::runtime_error if the matrix is singular and cannot be inverted.
         *
         * @code
         * slt::DenseMatrix<double> mat({
         *     {4.0, 7.0},
         *     {2.0, 6.0}
         * });
         *
         * auto inv = mat.inverse();
         * std::cout << inv(0, 0);  // Outputs approximately 0.6
         * std::cout << inv(1, 1);  // Outputs approximately 0.4
         * @endcode
         */
        DenseMatrix<T> inverse() const {
            if (rows_ != cols_)
                throw std::invalid_argument("Only square matrices can be inverted");

            const std::size_t n = rows_;
            DenseMatrix<T> A(*this);
            DenseMatrix<T> I(n, n, T{});
            for (std::size_t i = 0; i < n; ++i)
                I.update(i, i, T{1});  // Identity matrix

            for (std::size_t i = 0; i < n; ++i) {
                // Pivot selection (partial pivoting)
                std::size_t pivot = i;
                T max_val = std::abs(A.get(i, i));
                for (std::size_t j = i + 1; j < n; ++j) {
                    T val = std::abs(A.get(j, i));
                    if (val > max_val) {
                        max_val = val;
                        pivot = j;
                    }
                }

                if (max_val == T{})
                    throw std::runtime_error("Matrix is singular and cannot be inverted");

                // Swap rows (corrected)
                if (pivot != i) {
                    for (std::size_t k = 0; k < n; ++k) {
                        std::swap(A.data[i * n + k], A.data[pivot * n + k]);
                        std::swap(A.init[i * n + k], A.init[pivot * n + k]);
                        std::swap(I.data[i * n + k], I.data[pivot * n + k]);
                        std::swap(I.init[i * n + k], I.init[pivot * n + k]);
                    }
                }

                // Normalize pivot row
                T pivot_val = A.data[i * n + i];
                for (std::size_t k = 0; k < n; ++k) {
                    A.data[i * n + k] /= pivot_val;
                    A.init[i * n + k] = 1;

                    I.data[i * n + k] /= pivot_val;
                    I.init[i * n + k] = 1;
                }

                // Eliminate other rows
                for (std::size_t j = 0; j < n; ++j) {
                    if (j == i) continue;
                    T factor = A.data[j * n + i];
                    for (std::size_t k = 0; k < n; ++k) {
                        A.data[j * n + k] -= factor * A.data[i * n + k];
                        A.init[j * n + k] = 1;

                        I.data[j * n + k] -= factor * I.data[i * n + k];
                        I.init[j * n + k] = 1;
                    }
                }
            }

            return I;
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Returns the number of rows in the matrix.
         *
         * This function provides access to the total number of rows stored in the matrix.
         *
         * @return Number of rows in the matrix.
         *
         * @code
         * slt::DenseMatrix<float> mat(3, 5);
         * std::cout << mat.rows();  // Outputs: 3
         * @endcode
         */
        std::size_t rows() const override { return rows_; }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Returns the number of columns in the matrix.
         *
         * This function provides access to the total number of columns stored in the matrix.
         *
         * @return Number of columns in the matrix.
         *
         * @code
         * slt::DenseMatrix<float> mat(3, 5);
         * std::cout << mat.cols();  // Outputs: 5
         * @endcode
         */
        std::size_t cols() const override { return cols_; }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Retrieves a copy of the value at the specified matrix index.
         *
         * This method allows read-only access to an individual matrix element.
         * If the index is out of bounds or the element is uninitialized, an exception is thrown.
         *
         * @param row Row index
         * @param col Column index
         * @return Value at the specified index
         *
         * @throws std::out_of_range if `row` or `col` is outside the matrix bounds.
         * @throws std::runtime_error if the element at the given index is uninitialized.
         *
         * @code
         * slt::DenseMatrix<double> mat(3, 3);
         * mat.set(1, 1, 42.0);
         * std::cout << mat.get(1, 1);  // Outputs: 42.0
         *
         * // mat.get(0, 0);  // Would throw std::runtime_error
         * @endcode
         */
        T get(std::size_t row, std::size_t col) const override {
            if (row >= rows_ || col >= cols_)
                throw std::out_of_range("Index out of range");
            
            std::size_t idx = row * cols_ + col;
            if (!init[idx])
                throw std::runtime_error("Accessing uninitialized matrix element");
            
            return data[idx];
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Sets the value at the given matrix index, marking it as initialized.
         *
         * This function assigns a value to the matrix at position (row, col), but only
         * if the element is currently uninitialized. If the element is already initialized,
         * it throws an exception. Use `update()` instead to modify existing values.
         *
         * @param row Row index
         * @param col Column index
         * @param value Value to assign
         *
         * @throws std::out_of_range if the index is outside the matrix bounds
         * @throws std::runtime_error if the element is already initialized
         *
         * @code
         * slt::DenseMatrix<float> mat(3, 3);
         * mat.set(1, 2, 9.5f);  // Initializes and sets value
         *
         * // mat.set(1, 2, 4.3f);  // Throws std::runtime_error
         * @endcode
         */
        void set(std::size_t row, std::size_t col, T value) override {
            if (row >= rows_ || col >= cols_)
                throw std::out_of_range("Index out of range");

            std::size_t idx = row * cols_ + col;
            if (init[idx])
                throw std::runtime_error("Cannot set value: element already initialized. Use update instead.");

            data[idx] = value;
            init[idx] = 1;
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Removes a value from the matrix by clearing its data and initialization flag.
         *
         * This function resets the element at (row, col) to the default value of type `T` and
         * marks it as uninitialized. If the value was not initialized to begin with, an error is thrown.
         *
         * @param row Row index
         * @param col Column index
         *
         * @throws std::out_of_range if the index is invalid
         * @throws std::runtime_error if the element was not initialized
         *
         * @code
         * slt::DenseMatrix<float> mat(2, 2);
         * mat.set(0, 1, 3.14f);
         * mat.remove(0, 1);  // Successfully removes
         *
         * // mat.remove(0, 1);  // Throws std::runtime_error
         * @endcode
         */
        void remove(std::size_t row, std::size_t col) {
            if (row >= rows_ || col >= cols_)
                throw std::out_of_range("Index out of range");
            std::size_t idx = row * cols_ + col;
            if (!init[idx]) 
                throw std::runtime_error("Cannot remove value: element not initialized.");
            data[idx] = T{0};      // Reset value
            init[idx] = 0;         // Mark uninitialized
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Updates the value at the given matrix index, assuming it is already initialized.
         *
         * This method allows modifying the value of an element that has already been initialized.
         * It does not change the initialization state. Use `set()` if the value is uninitialized.
         *
         * @param row Row index
         * @param col Column index
         * @param value New value to assign
         *
         * @throws std::out_of_range if the index is out of bounds
         * @throws std::runtime_error if the target element is uninitialized
         *
         * @code
         * slt::DenseMatrix<int> mat(3, 3);
         * mat.set(2, 1, 5);
         * mat.update(2, 1, 10);  // Replaces value
         *
         * // mat.update(0, 0, 1);  // Throws std::runtime_error if not previously set
         * @endcode
         */
        void update(std::size_t row, std::size_t col, T value) {
            if (row >= rows_ || col >= cols_)
                throw std::out_of_range("Update failed: index out of bounds");

            std::size_t index = row * cols_ + col;

            if (init[index] == 0)
                throw std::runtime_error("Update failed: value not initialized");

            data[index] = value;
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Creates a deep copy of the current matrix instance.
         *
         * This method returns a new `DenseMatrix` object that is a deep copy of the current
         * matrix. The returned pointer is cast to the base class `MatrixBase<T>` and stored
         * in a `std::unique_ptr` for memory-safe polymorphic use.
         *
         * @return A `std::unique_ptr` to a new copy of this matrix.
         *
         * @code
         * std::unique_ptr<MatrixBase<float>> original = std::make_unique<slt::DenseMatrix<float>>(2, 2);
         * std::unique_ptr<MatrixBase<float>> copy = original->clone();
         * @endcode
         */
        std::unique_ptr<MatrixBase<T>> clone() const override {
            return std::make_unique<DenseMatrix>(*this);
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Prints the contents of the matrix to the given output stream.
         *
         * Displays the matrix in a human-readable 2D format. Each element is printed
         * with fixed width spacing for readability. The default stream is `std::cout`, 
         * but any `std::ostream` can be passed (e.g., `std::ostringstream` for testing).
         *
         * Only initialized elements are printed. Uninitialized elements accessed through
         * this method may trigger runtime exceptions if bounds or initialization are violated.
         *
         * @param os The output stream to print to (defaults to `std::cout`)
         *
         * @code
         * slt::DenseMatrix<int> mat(2, 3);
         * mat.set(0, 0, 1);
         * mat.set(0, 1, 2);
         * mat.set(0, 2, 3);
         * mat.set(1, 0, 4);
         * mat.set(1, 1, 5);
         * mat.set(1, 2, 6);
         * mat.print();
         * @endcode
         *
         * Output:
         * @verbatim
         *          1          2          3 
         *          4          5          6 
         * @endverbatim
         */
        void print(std::ostream& os = std::cout) const {
            for (std::size_t i = 0; i < rows_; ++i) {
                for (std::size_t j = 0; j < cols_; ++j) {
                    os << std::setw(10) << operator()(i, j) << " ";
                }
                os << '\n';
            }
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Checks whether a specific matrix element has been initialized.
         *
         * Returns `true` if the element at the specified row and column has been initialized
         * using `set()`, `update()`, or the assignment operator. Otherwise, returns `false`.
         *
         * @param row Row index
         * @param col Column index
         * @return `true` if the element is initialized, `false` otherwise
         *
         * @throws std::out_of_range if the index is outside the matrix bounds
         *
         * @code
         * slt::DenseMatrix<float> mat(3, 3);
         * mat.set(1, 1, 2.0f);
         * bool check = mat.is_initialized(1, 1);  // true
         * @endcode
         */
        bool is_initialized(std::size_t row, std::size_t col) const override {
            if (row >= rows_ || col >= cols_)
                throw std::out_of_range("Index out of range");
            return init[row * cols_ + col] != 0;
        }
    };
// ================================================================================ 

    /**
     * @brief Stream output operator for DenseMatrix.
     *
     * @tparam T Matrix element type.
     * @param os Output stream.
     * @param mat Matrix to print.
     * @return Output stream.
     */
    template<typename T>
    std::ostream& operator<<(std::ostream& os, const DenseMatrix<T>& mat) {
        mat.print(os);
        return os;
    }
// -------------------------------------------------------------------------------- 

    /**
     * @brief Adds a scalar to each initialized element of a matrix.
     *
     * This overload enables `scalar + matrix` syntax by forwarding the operation
     * to the existing `matrix + scalar` member operator. Only initialized elements
     * are updated in the result; uninitialized elements remain uninitialized.
     *
     * @tparam T The type of the matrix elements.
     * @param scalar The scalar value to add.
     * @param matrix The DenseMatrix to which the scalar is added.
     * @return A new DenseMatrix<T> with `scalar + matrix(i,j)` for each initialized element.
     *
     * @code
     * slt::DenseMatrix<float> A(2, 2);
     * A.set(0, 0, 1.0f);
     * A.set(0, 1, 2.0f);
     * A.set(1, 0, 3.0f);
     * A.set(1, 1, 4.0f);
     *
     * slt::DenseMatrix<float> B = 10.0f + A;
     * B.print();
     * @endcode
     *
     * Output:
     * @verbatim
     *         11         12
     *         13         14
     * @endverbatim
     */ 
    template<typename T>
    DenseMatrix<T> operator+(T scalar, const DenseMatrix<T>& matrix) {
        return matrix + scalar;  // Leverage existing member operator+
    }
// -------------------------------------------------------------------------------- 

    /**
     * @brief Subtracts each element of a matrix from a scalar.
     *
     * This function computes the difference between a scalar and each initialized element
     * in the given matrix. Only initialized elements contribute to the result; uninitialized
     * elements remain uninitialized in the result.
     *
     * @tparam T The element type of the matrix.
     * @param scalar The scalar value to subtract from.
     * @param matrix The matrix whose values will be subtracted from the scalar.
     * @return A new DenseMatrix<T> where each initialized element is `scalar - matrix(i,j)`
     *
     * @code
     * slt::DenseMatrix<float> A(2, 2);
     * A.set(0, 0, 1.0f);
     * A.set(0, 1, 2.0f);
     * A.set(1, 0, 3.0f);
     * A.set(1, 1, 4.0f);
     *
     * slt::DenseMatrix<float> B = 10.0f - A;
     * B.print();
     * @endcode
     *
     * Output:
     * @verbatim
     *          9          8 
     *          7          6 
     * @endverbatim
     */
    template<typename T>
    slt::DenseMatrix<T> operator-(T scalar, const slt::DenseMatrix<T>& matrix) {
        slt::DenseMatrix<T> result(matrix.rows(), matrix.cols());
        for (std::size_t i = 0; i < matrix.rows(); ++i) {
            for (std::size_t j = 0; j < matrix.cols(); ++j) {
                if (matrix.is_initialized(i, j)) {
                    result.set(i, j, scalar - matrix.get(i, j));
                }
                // If not initialized, skip—result stays uninitialized
            }
        }
        return result;
    }
// -------------------------------------------------------------------------------- 

    /**
     * @brief Multiplies each element of a matrix by a scalar.
     *
     * Performs element-wise multiplication between a scalar and all initialized values
     * of the matrix. This overload allows `scalar * matrix` in addition to `matrix * scalar`.
     *
     * @tparam T The type of the matrix elements.
     * @param scalar The scalar multiplier.
     * @param matrix The matrix whose values will be multiplied.
     * @return A new DenseMatrix<T> with each element equal to `scalar * matrix(i,j)`
     *
     * @code
     * slt::DenseMatrix<int> A(2, 2);
     * A.set(0, 0, 1);
     * A.set(0, 1, 2);
     * A.set(1, 0, 3);
     * A.set(1, 1, 4);
     *
     * slt::DenseMatrix<int> B = 5 * A;
     * B.print();
     * @endcode
     *
     * Output:
     * @verbatim
     *          5         10 
     *         15         20 
     * @endverbatim
     */
    template<typename T>
    DenseMatrix<T> operator*(T scalar, const DenseMatrix<T>& matrix) {
        return matrix * scalar;  // Leverage member function
    }
// -------------------------------------------------------------------------------- 

    template<typename T>
    T dot(const T* a, const T* b, std::size_t size) {
        static_assert(std::is_same<T, float>::value || std::is_same<T, double>::value,
                      "dot<T>: only float and double are supported");

        if (!a || !b) throw std::invalid_argument("Null pointer passed to dot product");

        std::vector<T> temp(size);
        simd_ops<T>::mul(a, b, temp.data(), size);

        // Reduce result (scalar fallback)
        T sum = static_cast<T>(0);
        for (std::size_t i = 0; i < size; ++i)
            sum += temp[i];

        return sum;
    }
// -------------------------------------------------------------------------------- 

    // 2. std::vector overload
    template<typename T>
    T dot(const std::vector<T>& a, const std::vector<T>& b) {
        if (a.size() != b.size())
            throw std::invalid_argument("Vector sizes must match for dot product.");
        return dot(a.data(), b.data(), a.size());
    }
// -------------------------------------------------------------------------------- 

    // 3. std::array overload
    template<typename T, std::size_t N>
    T dot(const std::array<T, N>& a, const std::array<T, N>& b) {
        return dot(a.data(), b.data(), N);
    }
// -------------------------------------------------------------------------------- 

    template<typename T>
    inline void cross(const T* a, const T* b, T* result) {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "T must be float or double");
        assert(a && b && result);

        result[0] = a[1] * b[2] - a[2] * b[1];
        result[1] = a[2] * b[0] - a[0] * b[2];
        result[2] = a[0] * b[1] - a[1] * b[0];
    }
// -------------------------------------------------------------------------------- 

    template<typename T>
    inline std::array<T, 3> cross(const std::array<T, 3>& a, const std::array<T, 3>& b) {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "T must be float or double");

        return {
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        };
    }
// -------------------------------------------------------------------------------- 

    template<typename T>
    inline std::vector<T> cross(const std::vector<T>& a, const std::vector<T>& b) {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "T must be float or double");

        assert(a.size() == 3 && b.size() == 3);
        return {
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        };
    }
// -------------------------------------------------------------------------------- 

    /**
     * @brief Perform standard matrix multiplication (A × B) for dense matrices.
     *
     * This function multiplies two dense matrices A and B, producing a new matrix C.
     * It supports only `float` and `double` data types (enforced via static_assert).
     * 
     * Each element of the resulting matrix is computed as the dot product of a row of A and
     * a column of B. All accessed values must be initialized in A and B.
     *
     * @tparam T The numeric type of the matrix (must be float or double)
     * @param A The left matrix operand (dimensions: m × n)
     * @param B The right matrix operand (dimensions: n × p)
     * @return Resulting matrix C of dimensions m × p
     *
     * @throws std::invalid_argument if the number of columns in A does not match
     *         the number of rows in B.
     *
     * @code
     * slt::DenseMatrix<float> A({
     *     {1.0f, 2.0f},
     *     {3.0f, 4.0f}
     * });
     *
     * slt::DenseMatrix<float> B({
     *     {5.0f, 6.0f},
     *     {7.0f, 8.0f}
     * });
     *
     * auto C = mat_mul(A, B);
     * C.print();  // Output:
     *             //      19      22
     *             //      43      50
     * @endcode
     */
    template<typename T>
    DenseMatrix<T> mat_mul(const DenseMatrix<T>& A, const DenseMatrix<T>& B) {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                      "mat_mul only supports float or double types.");

        const std::size_t A_rows = A.rows();
        const std::size_t A_cols = A.cols();
        const std::size_t B_rows = B.rows();
        const std::size_t B_cols = B.cols();

        if (A_cols != B_rows) {
            throw std::invalid_argument("Matrix dimensions are incompatible for multiplication.");
        }

        DenseMatrix<T> result(A_rows, B_cols);

        for (std::size_t i = 0; i < A_rows; ++i) {
            for (std::size_t j = 0; j < B_cols; ++j) {
                // Extract row i of A and column j of B
                std::vector<T> row(A_cols);
                std::vector<T> col(A_cols);
                for (std::size_t k = 0; k < A_cols; ++k) {
                    row[k] = A.get(i, k);
                    col[k] = B.get(k, j);
                }
                result.set(i, j, dot(row, col));
            }
        }

        return result;
    }
// ================================================================================ 
// ================================================================================ 

    /**
     * @class SparseCOOMatrix
     * @brief A sparse matrix implementation using the Coordinate List (COO) format.
     *
     * This class stores non-zero elements of a matrix using three parallel vectors:
     * one for row indices, one for column indices, and one for the data values.
     * It supports two operational modes:
     * 
     * - **Fast Insertion Mode (`fast_set = true`)**: Allows fast, unordered appends of
     *   (row, column, value) triplets. This mode is ideal for incremental construction
     *   of the matrix but requires a call to `finalize()` before performing reliable
     *   access or update operations.
     * 
     * - **Finalized Mode (`fast_set = false`)**: Ensures the internal storage is sorted
     *   lexicographically by (row, column). Enables efficient binary search for
     *   `get()`, `update()`, and `is_initialized()` methods.
     *
     * The class conforms to a polymorphic base class `MatrixBase<T>`, allowing it to be
     * used in a generic matrix interface with other matrix types such as dense or CSR.
     *
     * @tparam T The numeric type stored in the matrix (must be `float` or `double`).
     *
     * Example usage:
     * @code
     * SparseCOOMatrix<float> mat(3, 3);
     * mat.set(0, 0, 1.0f);
     * mat.set(2, 1, 5.0f);
     * mat.finalize();
     * float value = mat.get(2, 1);  // returns 5.0
     * @endcode
     */
    template<typename T>
    class SparseCOOMatrix : public MatrixBase<T> {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                      "DenseMatrix only supports float or double");
    private:
        std::vector<T> data; ///< Flat row-major storage of matrix elements. 
        std::size_t rows_ = 0; ///< Number of Matrix rows.
        std::size_t cols_ = 0; ///< Number of Matrix columns.

        // COO Specific data
        std::vector<std::size_t> row; ///< A vector containing row indices
        std::vector<std::size_t> col; ///< A vector containing column indices
        bool fast_set = true;  ///< true if vectors are optimized for insertation, false if optimized for retrieval

        // Comparator for sorting and binary search
        struct COOComparator {
            bool operator()(std::pair<std::size_t, std::size_t> a,
                            std::pair<std::size_t, std::size_t> b) const {
                return a.first < b.first || (a.first == b.first && a.second < b.second);
            }
        };
// ================================================================================ 

    public:
        /**
         * @brief The total size of the matrix 
         *
         * @return The number of rows multiplied by the number of columns, 0 if not initialized
         */
        std::size_t size() const override {return rows_ * cols_;} 
// -------------------------------------------------------------------------------- 

        /**
         * @brief Returns the number of explicitly stored non-zero elements.
         *
         * This function returns the number of entries stored in the sparse COO matrix,
         * which corresponds to the number of non-zero elements it currently tracks. 
         * Unlike a dense matrix, uninitialized values are implicitly zero and are not stored.
         *
         * @return The number of stored non-zero elements.
         *
         * @code
         * #include <iostream>
         * #include "matrix.hpp"
         *
         * int main() {
         *     slt::SparseCOOMatrix<float> mat({
         *         {1.0f, 0.0f},
         *         {0.0f, 2.0f}
         *     });
         *     std::cout << "Non-zero elements: " << mat.nonzero_count() << std::endl;
         *     return 0;
         * }
         * @endcode
         *
         * Output:
         * @code
         * Non-zero elements: 2
         * @endcode
         */
        std::size_t nonzero_count() const override {return data.size();} 
// -------------------------------------------------------------------------------- 

        /**
         * @brief Returns the row index of the i-th non-zero element.
         *
         * This accessor returns the row position associated with the i-th stored
         * non-zero value in the sparse COO matrix. Bounds checking is performed.
         *
         * @param i Index of the non-zero element.
         * @return Row index corresponding to the i-th element.
         * @throws std::out_of_range if the index is outside the valid range.
         *
         * @code
         * slt::SparseCOOMatrix<float> mat({
         *     {1.0f, 0.0f},
         *     {0.0f, 2.0f}
         * });
         * std::size_t r = mat.row_index(1);  // Returns row index of second non-zero
         * @endcode
         */
        std::size_t row_index(std::size_t i) const {
            if (i >= row.size()) throw std::out_of_range("Row index out of range");
            return row[i];
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Returns the column index of the i-th non-zero element.
         *
         * This accessor returns the column position associated with the i-th stored
         * non-zero value in the sparse COO matrix. Bounds checking is performed.
         *
         * @param i Index of the non-zero element.
         * @return Column index corresponding to the i-th element.
         * @throws std::out_of_range if the index is outside the valid range.
         *
         * @code
         * slt::SparseCOOMatrix<float> mat({
         *     {1.0f, 0.0f},
         *     {0.0f, 2.0f}
         * });
         * std::size_t c = mat.col_index(1);  // Returns column index of second non-zero
         * @endcode
         */
        std::size_t col_index(std::size_t i) const {
            if (i >= col.size()) throw std::out_of_range("Column index out of range");
            return col[i];
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Returns the value of the i-th non-zero element.
         *
         * This accessor returns the stored numerical value of the i-th non-zero entry
         * in the sparse COO matrix. Bounds checking is performed.
         *
         * @param i Index of the non-zero element.
         * @return The value of the i-th non-zero element.
         * @throws std::out_of_range if the index is outside the valid range.
         *
         * @code
         * slt::SparseCOOMatrix<float> mat({
         *     {1.0f, 0.0f},
         *     {0.0f, 2.0f}
         * });
         * float v = mat.value_index(1);  // Returns 2.0f
         * @endcode
         */
        T value_index(std::size_t i) const {
            if (i >= data.size()) throw std::out_of_range("Value index out of range");
            return data[i];
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Constructs an empty sparse COO matrix with given dimensions.
         *
         * Initializes internal storage with a small reserved capacity and sets the
         * fast insertion mode according to `fastInsert`.
         *
         * @param r Number of rows in the matrix.
         * @param c Number of columns in the matrix.
         * @param fastInsert If true, enables fast insertion mode (default: true).
         */
        explicit SparseCOOMatrix(std::size_t r, std::size_t c, bool fastInsert = true)
            : rows_(r), cols_(c), fast_set(fastInsert) {
            row.reserve(8);
            col.reserve(8);
            data.reserve(8);
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Constructs a sparse COO matrix from a 2D std::vector.
         *
         * Non-zero elements from the input vector are inserted into the sparse matrix.
         * If `fastInsert` is true, elements are inserted in append mode and `finalize()`
         * must be called manually before sorted access (e.g., get()).
         *
         * @param vec A 2D vector representing the matrix.
         * @param fastInsert Enables fast insertion mode if true (default: true).
         * @throws std::invalid_argument if rows have inconsistent lengths.
         */
        SparseCOOMatrix(const std::vector<std::vector<T>>& vec, bool fastInsert = true)
            : rows_(vec.size()), cols_(vec.empty() ? 0 : vec[0].size()), fast_set(fastInsert) {
            for (std::size_t i = 0; i < rows_; ++i) {
                if (vec[i].size() != cols_)
                    throw std::invalid_argument("All rows must have the same number of columns");

                for (std::size_t j = 0; j < cols_; ++j) {
                    if (vec[i][j] != T{})  // Skip zeros
                        this->set(i, j, vec[i][j]);
                }
            }
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Constructs a sparse COO matrix from a fixed-size std::array.
         *
         * Non-zero values in the 2D array are inserted into the matrix.
         *
         * @tparam Rows Number of rows in the static array.
         * @tparam Cols Number of columns in the static array.
         * @param arr Fixed-size array of values to initialize the matrix.
         * @param fastInsert Enables fast insertion mode if true (default: true).
         */
        template<std::size_t Rows, std::size_t Cols>
        explicit SparseCOOMatrix(const std::array<std::array<T, Cols>, Rows>& arr, bool fastInsert = true)
            : rows_(Rows), cols_(Cols), fast_set(fastInsert) {
            for (std::size_t i = 0; i < Rows; ++i) {
                for (std::size_t j = 0; j < Cols; ++j) {
                    if (arr[i][j] != T{})  // Skip zeros
                        this->set(i, j, arr[i][j]);
                }
            }
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Constructs a sparse COO matrix from a nested initializer list.
         *
         * This allows convenient initialization using brace-enclosed lists. Only
         * non-zero elements are stored. Rows must be of consistent length.
         *
         * @param initList Nested initializer list (e.g., `{{1, 0}, {0, 2}}`).
         * @param fastInsert Enables fast insertion mode if true (default: true).
         * @throws std::invalid_argument if inner lists have inconsistent sizes.
         */
        SparseCOOMatrix(std::initializer_list<std::initializer_list<T>> initList, bool fastInsert = true)
            : rows_(initList.size()), cols_(initList.begin()->size()), fast_set(fastInsert) {
            std::size_t i = 0;
            for (const auto& rowList : initList) {
                if (rowList.size() != cols_)
                    throw std::invalid_argument("All rows must have the same number of columns");
                std::size_t j = 0;
                for (const T& val : rowList) {
                    if (val != T{})
                        this->set(i, j, val);
                    ++j;
                }
                ++i;
            }
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Constructs a sparse COO matrix from a flat vector in row-major order.
         *
         * Only non-zero values are inserted into the matrix. The resulting matrix will
         * include only the explicitly stored entries. If `fastInsert` is true, entries
         * are added in append mode and require `finalize()` before sorted access.
         *
         * @param flat_data A 1D vector in row-major order.
         * @param r Number of rows in the matrix.
         * @param c Number of columns in the matrix.
         * @param fastInsert Enables fast insertion mode if true (default: true).
         * @throws std::invalid_argument if the size of flat_data != r * c.
         */
        SparseCOOMatrix(const std::vector<T>& flat_data, std::size_t r, std::size_t c, bool fastInsert = true)
            : rows_(r), cols_(c), fast_set(fastInsert) {
            if (flat_data.size() != r * c)
                throw std::invalid_argument("Flat data size does not match matrix dimensions");

            for (std::size_t i = 0; i < r; ++i) {
                for (std::size_t j = 0; j < c; ++j) {
                    T val = flat_data[i * c + j];
                    if (val != T{})
                        this->set(i, j, val);
                }
            }
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Copy constructor for SparseCOOMatrix.
         *
         * Constructs a new SparseCOOMatrix as a deep copy of the provided matrix.
         * All internal data structures (values, row/column indices, flags) are duplicated,
         * preserving the state of the original matrix while ensuring full independence.
         *
         * @param other The SparseCOOMatrix instance to copy.
         *
         * @note This performs a deep copy. Changes to the new matrix will not affect the original.
         */
        SparseCOOMatrix(const SparseCOOMatrix<T>& other)
            : MatrixBase<T>(),
              data(other.data),
              rows_(other.rows_),
              cols_(other.cols_),
              row(other.row),
              col(other.col),
              fast_set(other.fast_set){}
// -------------------------------------------------------------------------------- 

        /**
         * @brief Move constructor for SparseCOOMatrix.
         *
         * Constructs a new sparse matrix by transferring ownership of data from another
         * matrix. This constructor performs a shallow move of internal vectors and resets
         * the source matrix to a default, empty state.
         *
         * This is more efficient than the copy constructor, as it avoids deep copying of
         * data and instead reuses existing memory buffers. After the move, the source matrix
         * is left in a valid but unspecified state (typically empty).
         *
         * @param other The matrix to move from. After the operation, `other` is empty.
         *
         * @note The `fast_set` flag is also transferred and reset in the source.
         */
        SparseCOOMatrix(SparseCOOMatrix<T>&& other) noexcept
            : MatrixBase<T>(),
              data(std::move(other.data)),
              rows_(std::exchange(other.rows_, 0)),
              cols_(std::exchange(other.cols_, 0)),
              row(std::move(other.row)),
              col(std::move(other.col)),
              fast_set(std::exchange(other.fast_set, true)) {}
// -------------------------------------------------------------------------------- 

        /**
         * @brief Accesses a matrix element (read-only).
         *
         * Retrieves the value at the specified row and column.
         * If the element has not been set, throws an exception.
         *
         * @param r Row index (zero-based).
         * @param c Column index (zero-based).
         * @return The value at the given position.
         * @throws std::out_of_range if indices are out of bounds.
         * @throws std::runtime_error if the element is uninitialized.
         */
        T operator()(std::size_t r, std::size_t c) const {
            return this->get(r, c);
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Compares two sparse COO matrices for equality.
         *
         * Determines whether this matrix and the given matrix are equal by checking:
         * - Matrix dimensions (rows and columns),
         * - Number of non-zero entries,
         * - Each corresponding (row, col, value) triple.
         *
         * If both matrices are finalized (i.e., `fast_set == false`), a direct comparison
         * of the underlying storage is used. If either matrix is not finalized,
         * both matrices are converted to dense format and compared entry-wise.
         *
         * For floating-point types, approximate comparison is used with a tolerance of `1e-6`.
         *
         * @param other The matrix to compare against.
         * @return true if the matrices are equal, false otherwise.
         *
         * @note Equality is defined structurally and numerically. Two matrices with
         *       the same non-zero values but in different insertion orders will still
         *       compare equal if finalized.
         *
         * @example
         * @code
         * slt::SparseCOOMatrix<float> A(2, 2);
         * A.set(0, 0, 1.0f);
         * A.set(1, 1, 2.0f);
         * A.finalize();
         *
         * slt::SparseCOOMatrix<float> B(2, 2);
         * B.set(0, 0, 1.0f);
         * B.set(1, 1, 2.0f);
         * B.finalize();
         *
         * assert(A == B);  // true
         * @endcode
         */
        bool operator==(const SparseCOOMatrix<T>& other) const {
            if (rows_ != other.rows_ || cols_ != other.cols_)
                return false;

            if (row != other.row || col != other.col || data.size() != other.data.size())
                return false;

            for (std::size_t i = 0; i < data.size(); ++i) {
                if constexpr (std::is_floating_point_v<T>) {
                    if (std::fabs(data[i] - other.data[i]) > 1e-6)
                        return false;
                } else {
                    if (data[i] != other.data[i])
                        return false;
                }
            }

            return true;
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Deep copy assignment operator.
         *
         * Copies all metadata and contents (rows, cols, data, etc.) from another
         * SparseCOOMatrix. The two matrices become fully independent.
         *
         * @param other Source matrix to copy from.
         * @return Reference to this matrix.
         */
        SparseCOOMatrix<T>& operator=(const SparseCOOMatrix<T>& other) {
            if (this != &other) {
                rows_ = other.rows_;
                cols_ = other.cols_;
                fast_set = other.fast_set;

                row = other.row;
                col = other.col;
                data = other.data;
            }
            return *this;
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Move assignment operator.
         *
         * Transfers resources from another SparseCOOMatrix, leaving the source in a
         * valid but empty state. Enables efficient transfer of large matrices.
         *
         * @param other Source matrix to move from.
         * @return Reference to this matrix.
         */
        SparseCOOMatrix<T>& operator=(SparseCOOMatrix<T>&& other) noexcept {
            if (this != &other) {
                rows_ = std::exchange(other.rows_, 0);
                cols_ = std::exchange(other.cols_, 0);
                fast_set = std::exchange(other.fast_set, true);

                row = std::move(other.row);
                col = std::move(other.col);
                data = std::move(other.data);
            }
            return *this;
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Adds two sparse matrices element-wise and returns the result as a dense matrix.
         *
         * Performs element-wise addition of two matrices in sparse COO format. The result is returned
         * as a `DenseMatrix<T>` to ensure full representation of potential non-zero values in the output.
         * 
         * Both matrices must have identical dimensions. If either matrix contains a non-zero value
         * at a given (row, col) index, the result will include that value. Internally, values are
         * added using a nested loop and temporary dense buffer. This operation is not optimized
         * for SIMD or sparsity-aware acceleration but is functionally correct and safe.
         *
         * @param other The sparse matrix to add.
         * @return A dense matrix containing the result of the element-wise addition.
         * @throws std::invalid_argument if the matrix dimensions do not match.
         *
         * @note This implementation uses full dense representation for the result, even if the
         *       result remains sparse. Use a future `to_sparse_sum()` method if you want a sparse result.
         *
         * @example
         * @code
         * SparseCOOMatrix<float> A = {{1.0f, 0.0f}, {0.0f, 2.0f}};
         * SparseCOOMatrix<float> B = {{0.0f, 3.0f}, {4.0f, 0.0f}};
         * DenseMatrix<float> result = A + B;
         * // result: [[1.0, 3.0], [4.0, 2.0]]
         * @endcode
         */
        DenseMatrix<T> operator+(const SparseCOOMatrix<T>& other) const {
            if (rows_ != other.rows_ || cols_ != other.cols_)
                throw std::invalid_argument("Matrix dimensions must match for addition");

            DenseMatrix<T> result(rows_, cols_);

            // Add all elements from this sparse matrix
            for (std::size_t i = 0; i < data.size(); ++i)
                result.set(row[i], col[i], data[i]);

            // Add all elements from the other sparse matrix
            for (std::size_t i = 0; i < other.data.size(); ++i) {
                std::size_t r = other.row[i];
                std::size_t c = other.col[i];
                if (result.is_initialized(r, c))
                    result.update(r, c, result(r, c) + other.data[i]);
                else
                    result.set(r, c, other.data[i]);
            }

            return result;
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Adds a scalar to each non-zero element of the sparse matrix.
         *
         * Each stored value in the COO matrix has the scalar added to it. This preserves
         * the sparsity pattern; zero elements not explicitly stored remain unchanged.
         *
         * @param scalar Scalar value to add.
         * @return A new `SparseCOOMatrix` with updated values.
         *
         * @example
         * @code
         * SparseCOOMatrix<float> A = {{1.0f, 0.0f}, {0.0f, 2.0f}};
         * auto result = A + 1.0f;
         * // result: {{2.0f, 0.0f}, {0.0f, 3.0f}};
         * @endcode
         */
        SparseCOOMatrix operator+(T scalar) const {
            SparseCOOMatrix result(*this);
            for (auto& val : result.data) {
                val += scalar;
            }
            return result;
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Subtracts two sparse matrices element-wise and returns the result as a dense matrix.
         *
         * Performs element-wise subtraction of two matrices in sparse COO format. The result is returned
         * as a `DenseMatrix<T>` to ensure full representation of potential non-zero values in the output.
         * 
         * Both matrices must have identical dimensions. If either matrix contains a non-zero value
         * at a given (row, col) index, the result will include that value. Internally, values are
         * added using a nested loop and temporary dense buffer. This operation is not optimized
         * for SIMD or sparsity-aware acceleration but is functionally correct and safe.
         *
         * @param other The sparse matrix to add.
         * @return A dense matrix containing the result of the element-wise addition.
         * @throws std::invalid_argument if the matrix dimensions do not match.
         *
         * @note This implementation uses full dense representation for the result, even if the
         *       result remains sparse. Use a future `to_sparse_sum()` method if you want a sparse result.
         *
         * @example
         * @code
         * SparseCOOMatrix<float> A = {{1.0f, 0.0f}, {0.0f, 2.0f}};
         * SparseCOOMatrix<float> B = {{0.0f, 3.0f}, {4.0f, 0.0f}};
         * DenseMatrix<float> result = A - B;
         * // result: [[1.0, -3.0], [-4.0, 2.0]]
         * @endcode
         */
        DenseMatrix<T> operator-(const SparseCOOMatrix<T>& other) const {
            if (rows_ != other.rows_ || cols_ != other.cols_)
                throw std::invalid_argument("Matrix dimensions must match for subtraction");

            DenseMatrix<T> result(rows_, cols_);

            // Add all elements from this sparse matrix
            for (std::size_t i = 0; i < data.size(); ++i)
                result.set(row[i], col[i], data[i]);

            // Subtract all elements from the other sparse matrix
            for (std::size_t i = 0; i < other.data.size(); ++i) {
                std::size_t r = other.row[i];
                std::size_t c = other.col[i];
                if (result.is_initialized(r, c))
                    result.update(r, c, result(r, c) - other.data[i]);
                else
                    result.set(r, c, -other.data[i]);  // <-- Fix: negate value
            }

            return result;
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Subtracts a scalar to each non-zero element of the sparse matrix.
         *
         * Each stored value in the COO matrix has the scalar added to it. This preserves
         * the sparsity pattern; zero elements not explicitly stored remain unchanged.
         *
         * @param scalar Scalar value to add.
         * @return A new `SparseCOOMatrix` with updated values.
         *
         * @example
         * @code
         * SparseCOOMatrix<float> A = {{1.0f, 0.0f}, {0.0f, 2.0f}};
         * auto result = A - 1.0f;
         * // result: {{0.0f, -1.0f}, {-1.0f, 1.0f}};
         * @endcode
         */
        SparseCOOMatrix operator-(T scalar) const {
            SparseCOOMatrix result(*this);
            for (auto& val : result.data) {
                val -= scalar;
            }
            return result;
        }
// -------------------------------------------------------------------------------- 
    /**
     * @brief Performs element-wise multiplication of two sparse matrices.
     *
     * Multiplies corresponding non-zero elements where both matrices store a value.
     * Returns a new SparseCOOMatrix containing only the overlapping non-zero positions.
     *
     * @param other The other SparseCOOMatrix to multiply with.
     * @return SparseCOOMatrix containing the element-wise product.
     * @throws std::invalid_argument if the matrix dimensions do not match.
     */
    SparseCOOMatrix operator*(const SparseCOOMatrix& other) const {
        if (rows_ != other.rows_ || cols_ != other.cols_)
            throw std::invalid_argument("Matrix dimensions must match for element-wise multiplication");

        SparseCOOMatrix<T> result(rows_, cols_);

        for (std::size_t i = 0; i < data.size(); ++i) {
            std::size_t r = row[i];
            std::size_t c = col[i];

            if (other.is_initialized(r, c)) {
                T product = data[i] * other.get(r, c);
                result.set(r, c, product);
            }
        }

        return result;
    }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Returns the number of rows in the matrix.
         *
         * This function provides the total number of rows defined in the matrix,
         * regardless of how many entries are explicitly stored or initialized.
         *
         * @return The total number of rows.
         *
         * @code
         * slt::SparseCOOMatrix<float> mat(3, 4);
         * std::size_t r = mat.rows();  // Returns 3
         * @endcode
         */
        std::size_t rows() const override { return rows_; }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Returns the number of columns in the matrix.
         *
         * This function returns the total number of columns allocated for the matrix,
         * which includes all column indices regardless of whether they contain non-zero entries.
         *
         * @return The total number of columns.
         *
         * @code
         * slt::SparseCOOMatrix<float> mat(3, 4);
         * std::size_t c = mat.cols();  // Returns 4
         * @endcode
         */
        std::size_t cols() const override { return cols_; }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Retrieves the value at the specified matrix location.
         *
         * Returns the value stored at the given row and column index in the sparse matrix.
         * If the element has not been explicitly initialized (i.e., not stored in the COO format),
         * the function throws a runtime exception.
         *
         * The function uses a linear search if the matrix was constructed with `fast_set == true`,
         * and a binary search if the entries are sorted (`fast_set == false`), allowing efficient
         * retrieval in both cases depending on construction strategy.
         *
         * @param r Row index of the target element.
         * @param c Column index of the target element.
         * @return The value at the specified matrix location.
         *
         * @throws std::out_of_range If the provided row or column index is outside the matrix bounds.
         * @throws std::runtime_error If the specified element is uninitialized and thus not stored.
         *
         * @code
         * slt::SparseCOOMatrix<float> mat({
         *     {1.0f, 0.0f},
         *     {0.0f, 3.0f}
         * });
         * float val = mat.get(1, 1);  // Returns 3.0f
         * float missing = mat.get(0, 1);  // Throws runtime_error
         * @endcode
         */
        T get(std::size_t r, std::size_t c) const override {
            if (r >= rows_ || c >= cols_)
                throw std::out_of_range("Index out of bounds");

            if (fast_set) {
                // Linear search (O(n)) for unsorted data
                for (std::size_t i = 0; i < data.size(); ++i) {
                    if (row[i] == r && col[i] == c)
                        return data[i];
                }
                throw std::runtime_error("Accessing uninitialized matrix element");
            }

            // Binary search (O(log n)) for sorted data
            std::size_t left = 0;
            std::size_t right = data.size();

            while (left < right) {
                std::size_t mid = left + (right - left) / 2;
                if (row[mid] == r && col[mid] == c) {
                    return data[mid];
                } else if (row[mid] < r || (row[mid] == r && col[mid] < c)) {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }

            throw std::runtime_error("Accessing uninitialized matrix element");
        }
// -------------------------------------------------------------------------------- 
    
        /**
         * @brief Creates a polymorphic deep copy of this matrix.
         *
         * @return Unique pointer to the copied matrix.
         */
        std::unique_ptr<MatrixBase<T>> clone() const override {
            return std::make_unique<SparseCOOMatrix>(*this);
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Sets a value in the matrix at the given (row, column) index.
         *
         * If `fast_set` is true, the value is appended without checking for duplicates
         * or maintaining order (O(1) insertion). This is efficient for bulk construction
         * but requires calling `finalize()` before reliable queries.
         *
         * If `fast_set` is false, the method performs a binary search and inserts the
         * value at the correct sorted position. Duplicate insertions will throw.
         *
         * @param r Row index of the element.
         * @param c Column index of the element.
         * @param value Value to insert.
         * @return True on successful insertion.
         * @throws std::out_of_range if indices are invalid.
         * @throws std::runtime_error if value already exists and `fast_set` is false.
         */
        void set(std::size_t r, std::size_t c, T value) override {
            if (r >= rows_ || c >= cols_)
                throw std::out_of_range("Index out of bounds");

            if (fast_set) {
                // Fast insert: append without duplicate checks
                row.push_back(r);
                col.push_back(c);
                data.push_back(value);
                return;
            }

            // Construct index pairs to search safely
            std::vector<std::pair<std::size_t, std::size_t>> indices;
            indices.reserve(row.size());
            for (std::size_t i = 0; i < row.size(); ++i) {
                indices.emplace_back(row[i], col[i]);
            }

            auto target = std::make_pair(r, c);
            auto it = std::lower_bound(indices.begin(), indices.end(), target);

            std::size_t index = std::distance(indices.begin(), it);

            if (index < indices.size() && indices[index] == target) {
                throw std::runtime_error("Value already set. Use update() instead.");
            }

            row.insert(row.begin() + index, r);
            col.insert(col.begin() + index, c);
            data.insert(data.begin() + index, value);
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Updates an existing value in the matrix at (row, column).
         *
         * Performs a binary search for the target index. If the element is found,
         * the value is updated in-place. If the element does not exist, an exception
         * is thrown (you must call `set()` first).
         *
         * This method requires `finalize()` to have been called if the matrix was
         * initially constructed in fast insertion mode.
         *
         * @param r Row index of the element.
         * @param c Column index of the element.
         * @param value New value to assign to the existing element.
         * @return True on successful update.
         * @throws std::out_of_range if indices are invalid.
         * @throws std::runtime_error if the element is not already set.
         */
        void update(std::size_t r, std::size_t c, T value) {
            if (r >= rows_ || c >= cols_)
                throw std::out_of_range("Index out of bounds");

            // Step 1: Binary search for the index
            auto it = std::lower_bound(
                row.begin(), row.end(),
                std::make_pair(r, c),
                [this](size_t i, const std::pair<size_t, size_t>& target) {
                    return std::pair<size_t, size_t>{row[i], col[i]} < target;
                });

            std::size_t index = std::distance(row.begin(), it);

            // Step 2: Must already exist
            if (index >= row.size() || row[index] != r || col[index] != c) {
                throw std::runtime_error("Element not set yet. Use set() first.");
            }

            data[index] = value;
            return;
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Checks if the element at the specified row and column is initialized (non-zero).
         *
         * This function determines whether a value has been explicitly assigned to the given
         * row and column in the sparse matrix. It supports two modes:
         *
         * - **Fast set mode (`fast_set = true`)**: Performs a linear search through the unsorted COO entries.
         * - **Sorted mode (`fast_set = false`)**: Performs a binary search assuming the entries are sorted
         *   by row-major order (i.e., row first, then column).
         *
         * This is useful for determining if a matrix entry is actively stored (i.e., not a structural zero).
         *
         * @param r Row index of the element.
         * @param c Column index of the element.
         * @return `true` if the element is explicitly initialized (non-zero); otherwise `false`.
         * @throws std::out_of_range if the given row or column index is outside the matrix bounds.
         *
         * @code
         * slt::SparseCOOMatrix<float> mat({
         *     {1.0f, 0.0f},
         *     {0.0f, 2.0f}
         * });
         *
         * bool found = mat.is_initialized(1, 1);  // Returns true
         * bool empty = mat.is_initialized(0, 1);  // Returns false
         * @endcode
         */
        bool is_initialized(std::size_t r, std::size_t c) const override {
            if (r >= rows_ || c >= cols_)
                throw std::out_of_range("Index out of range");

            if (fast_set) {
                // Linear search for unsorted data
                for (std::size_t i = 0; i < data.size(); ++i) {
                    if (row[i] == r && col[i] == c)
                        return true;
                }
                return false;
            } else {
                // Binary search for sorted data
                std::size_t left = 0;
                std::size_t right = data.size();

                while (left < right) {
                    std::size_t mid = left + (right - left) / 2;
                    if (row[mid] == r && col[mid] == c)
                        return true;
                    else if (row[mid] < r || (row[mid] == r && col[mid] < c))
                        left = mid + 1;
                    else
                        right = mid;
                }

                return false;
            }
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Finalizes the internal COO representation for querying.
         *
         * This method is required after using fast insertion mode (`fast_set = true`)
         * to sort the (row, column, value) triplets into lexicographic order. Once
         * finalized, efficient binary search and reliable get/update/is_initialized
         * operations are enabled.
         *
         * This method performs a stable sort and disables fast insertion mode.
         */
        void finalize() {
            if (!fast_set) return;

            std::vector<std::size_t> indices(data.size());
            std::iota(indices.begin(), indices.end(), 0);

            std::stable_sort(indices.begin(), indices.end(),
                [this](std::size_t a, std::size_t b) {
                    return std::tie(row[a], col[a]) < std::tie(row[b], col[b]);
                });

            std::vector<std::size_t> sorted_row, sorted_col;
            std::vector<T> sorted_data;
            sorted_row.reserve(row.size());
            sorted_col.reserve(col.size());
            sorted_data.reserve(data.size());

            for (std::size_t idx : indices) {
                sorted_row.push_back(row[idx]);
                sorted_col.push_back(col[idx]);
                sorted_data.push_back(data[idx]);
            }

            row = std::move(sorted_row);
            col = std::move(sorted_col);
            data = std::move(sorted_data);

            fast_set = false;
        }
// --------------------------------------------------------------------------------

        /**
         * @brief Returns whether the matrix is in fast insertion mode.
         *
         * This method reports the current status of the `fast_set` flag.
         * When true, the matrix is in fast insertion mode—entries can be appended
         * quickly without maintaining order or checking for duplicates. When false,
         * the matrix is in finalized mode and supports efficient retrieval operations
         * (e.g., via binary search).
         *
         * @return True if the matrix is in fast insertion mode; false if finalized.
         */
        bool set_fast() const {
            return fast_set;
        }
    };
// ================================================================================ 
// ================================================================================ 
// SparseCOOMatrix friend functions 

    /**
     * @brief Adds a scalar to each non-zero element of the sparse matrix (scalar + matrix).
     *
     * Symmetric to `matrix + scalar`. Adds `scalar` to each stored value in the sparse matrix.
     * The result maintains the same sparsity pattern as the original.
     *
     * @tparam T Element type.
     * @param scalar Scalar value to add.
     * @param matrix Sparse COO matrix.
     * @return A new `SparseCOOMatrix<T>` with scalar added to each stored element.
     */
    template<typename T>
    SparseCOOMatrix<T> operator+(T scalar, const SparseCOOMatrix<T>& matrix) {
        return matrix + scalar;  // Reuse member operator+
    }
// -------------------------------------------------------------------------------- 

    /**
     * @brief Adds a DenseMatrix and a SparseCOOMatrix element-wise.
     *
     * This function returns a new DenseMatrix that represents the element-wise
     * sum of the input dense and sparse matrices. All initialized values in
     * the sparse matrix are added to the corresponding entries in the dense matrix.
     *
     * The result is fully initialized regardless of which elements were modified by the sparse matrix.
     *
     * @tparam T Type of matrix elements (must be float or double).
     * @param dense The dense matrix operand.
     * @param sparse The sparse COO matrix operand.
     * @return A new DenseMatrix<T> containing the result of the addition.
     * @throws std::invalid_argument If the input matrices do not have the same shape.
     *
     * Example:
     * @code
     * DenseMatrix<float> A(2, 2);
     * A.set(0, 0, 1.0f);
     * A.set(1, 1, 2.0f);
     *
     * SparseCOOMatrix<float> B(2, 2);
     * B.set(0, 1, 3.0f);
     *
     * DenseMatrix<float> C = A + B;
     * // C(0, 0) == 1.0
     * // C(0, 1) == 3.0
     * // C(1, 1) == 2.0
     * @endcode
     */
    template<typename T>
    DenseMatrix<T> operator+(const DenseMatrix<T>& dense, const SparseCOOMatrix<T>& sparse) {
        if (dense.rows() != sparse.rows() || dense.cols() != sparse.cols())
            throw std::invalid_argument("Matrix dimensions must match for addition");

        DenseMatrix<T> result(dense.rows(), dense.cols());

        // Copy dense matrix data to result
        if constexpr (simd_traits<T>::supported) {
            simd_ops<T>::copy(dense.data_ptr(), result.data_ptr(), dense.size());
        } else {
            for (std::size_t i = 0; i < dense.size(); ++i)
                result.data_ptr()[i] = dense.data_ptr()[i];
        }

        // Mark all entries as initialized
        std::fill(result.init_ptr(), result.init_ptr() + result.size(), 1);

        // Add sparse values
        for (std::size_t i = 0; i < sparse.nonzero_count(); ++i) {
            std::size_t r = sparse.row_index(i);
            std::size_t c = sparse.col_index(i);
            result.update(r, c, result(r, c) + sparse.value(i));
        }

        return result;
    }
// -------------------------------------------------------------------------------- 

/**
     * @brief Adds a DenseMatrix and a SparseCOOMatrix element-wise.
     *
     * This function returns a new DenseMatrix that represents the element-wise
     * sum of the input dense and sparse matrices. All initialized values in
     * the sparse matrix are added to the corresponding entries in the dense matrix.
     *
     * The result is fully initialized regardless of which elements were modified by the sparse matrix.
     *
     * @tparam T Type of matrix elements (must be float or double).
     * @param sparse The sparse COO matrix operand.
     * @param dense The dense matrix operand.
     * @return A new DenseMatrix<T> containing the result of the addition.
     * @throws std::invalid_argument If the input matrices do not have the same shape.
     *
     * Example:
     * @code
     * DenseMatrix<float> A(2, 2);
     * A.set(0, 0, 1.0f);
     * A.set(1, 1, 2.0f);
     *
     * SparseCOOMatrix<float> B(2, 2);
     * B.set(0, 1, 3.0f);
     *
     * DenseMatrix<float> C = A + B;
     * // C(0, 0) == 1.0
     * // C(0, 1) == 3.0
     * // C(1, 1) == 2.0
     * @endcode
     */
    template<typename T>
    DenseMatrix<T> operator+(const SparseCOOMatrix<T>& sparse, const DenseMatrix<T>& dense) {
        if (dense.rows() != sparse.rows() || dense.cols() != sparse.cols())
            throw std::invalid_argument("Matrix dimensions must match for addition");

        DenseMatrix<T> result(dense.rows(), dense.cols());

        // Copy dense matrix data to result
        if constexpr (simd_traits<T>::supported) {
            simd_ops<T>::copy(dense.data_ptr(), result.data_ptr(), dense.size());
        } else {
            for (std::size_t i = 0; i < dense.size(); ++i)
                result.data_ptr()[i] = dense.data_ptr()[i];
        }

        // Mark all entries as initialized
        std::fill(result.init_ptr(), result.init_ptr() + result.size(), 1);

        // Add sparse values
        for (std::size_t i = 0; i < sparse.nonzero_count(); ++i) {
            std::size_t r = sparse.row_index(i);
            std::size_t c = sparse.col_index(i);
            result.update(r, c, result(r, c) + sparse.value(i));
        }

        return result;
    }
// -------------------------------------------------------------------------------- 

    /**
     * @brief Subtracts each non-zero element of a SparseCOOMatrix from a scalar value.
     *
     * Creates a new SparseCOOMatrix where each stored element is the result of
     * subtracting the matrix element from the given scalar (i.e., `scalar - value`).
     * Unstored zero elements remain zero and are not added to the result.
     *
     * This operation preserves the sparsity pattern of the original matrix.
     *
     * @param scalar The scalar value to subtract each matrix element from.
     * @param matrix The input SparseCOOMatrix.
     * @return A new SparseCOOMatrix with updated values.
     *
     * @throws std::invalid_argument if the matrix is improperly initialized.
     *
     * @example
     * @code
     * SparseCOOMatrix<float> A(2, 2);
     * A.set(0, 0, 3.0f);
     * A.set(1, 1, 1.0f);
     *
     * SparseCOOMatrix<float> B = 5.0f - A;
     * // B.get(0, 0) == 2.0f, B.get(1, 1) == 4.0f
     * @endcode
     */
    template<typename T>
    SparseCOOMatrix<T> operator-(T scalar, const SparseCOOMatrix<T>& matrix) {
        SparseCOOMatrix<T> result(matrix.rows(), matrix.cols());

        for (std::size_t i = 0; i < matrix.nonzero_count(); ++i) {
            std::size_t r = matrix.row_index(i);
            std::size_t c = matrix.col_index(i);
            T val = scalar - matrix.value(i);
            result.set(r, c, val);
        }

        return result;
    }
// -------------------------------------------------------------------------------- 

    /**
     * @brief Subtracts a dense matrix from a sparse matrix and returns the result as a dense matrix.
     *
     * Performs element-wise subtraction: result(i,j) = sparse(i,j) - dense(i,j). The result is 
     * stored in a DenseMatrix<T> to account for all positions, including those with implicit zeros 
     * in the sparse matrix. This function assumes that both matrices have the same dimensions.
     *
     * Internally, this negates all values in the dense matrix and adds the sparse values.
     * SIMD acceleration is used if available.
     *
     * @tparam T Floating-point type (float or double)
     * @param sparse The left-hand operand, a sparse matrix
     * @param dense The right-hand operand, a dense matrix
     * @return DenseMatrix<T> containing the result of the subtraction
     * @throws std::invalid_argument if matrix dimensions do not match
     *
     * @example
     * @code
     * slt::SparseCOOMatrix<float> A = {{1.0f, 0.0f}, {0.0f, 2.0f}};
     * slt::DenseMatrix<float> B = {{5.0f, 6.0f}, {7.0f, 8.0f}};
     * slt::DenseMatrix<float> C = A - B;
     * // C == {{-4.0f, -6.0f}, {-7.0f, -6.0f}};
     * @endcode
     */
    template<typename T>
    DenseMatrix<T> operator-(const SparseCOOMatrix<T>& sparse, const DenseMatrix<T>& dense) {
        if (sparse.rows() != dense.rows() || sparse.cols() != dense.cols())
            throw std::invalid_argument("Matrix dimensions must match for subtraction");

        DenseMatrix<T> result(dense.rows(), dense.cols());

        // Negate dense matrix and store in result
        if constexpr (simd_traits<T>::supported) {
            simd_ops<T>::mul_scalar(dense.data_ptr(), static_cast<T>(-1), result.data_ptr(), dense.size());
        } else {
            for (std::size_t i = 0; i < dense.size(); ++i)
                result.data_ptr()[i] = -dense.data_ptr()[i];
        }

        // Mark all entries as initialized
        std::fill(result.init_ptr(), result.init_ptr() + result.size(), 1);

        // Add sparse matrix values to result
        for (std::size_t i = 0; i < sparse.nonzero_count(); ++i) {
            std::size_t r = sparse.row_index(i);
            std::size_t c = sparse.col_index(i);
            result.update(r, c, result(r, c) + sparse.value(i));
        }

        return result;
    }
// -------------------------------------------------------------------------------- 

    /**
     * @brief Subtracts a sparse matrix from a dense matrix and returns the result as a dense matrix.
     *
     * Performs element-wise subtraction: result(i,j) = dense(i,j) - sparse(i,j). The result is 
     * stored in a DenseMatrix<T> to preserve all values. Zero entries in the sparse matrix do not 
     * affect the result.
     *
     * The function requires that both matrices have matching dimensions. SIMD is used to accelerate
     * the copy phase where supported.
     *
     * @tparam T Floating-point type (float or double)
     * @param dense The left-hand operand, a dense matrix
     * @param sparse The right-hand operand, a sparse matrix
     * @return DenseMatrix<T> representing the subtraction result
     * @throws std::invalid_argument if matrix dimensions do not match
     *
     * @example
     * @code
     * slt::DenseMatrix<float> A = {{5.0f, 6.0f}, {7.0f, 8.0f}};
     * slt::SparseCOOMatrix<float> B = {{1.0f, 0.0f}, {0.0f, 2.0f}};
     * slt::DenseMatrix<float> C = A - B;
     * // C == {{4.0f, 6.0f}, {7.0f, 6.0f}};
     * @endcode
     */
    template<typename T>
    DenseMatrix<T> operator-(const DenseMatrix<T>& dense, const SparseCOOMatrix<T>& sparse) {
        if (sparse.rows() != dense.rows() || sparse.cols() != dense.cols()) {
            throw std::invalid_argument("Matrix dimensions must match for subtraction");
        }

        DenseMatrix<T> result(dense.rows(), dense.cols());

        // Copy dense values
        if constexpr (simd_traits<T>::supported) {
            simd_ops<T>::copy(dense.data_ptr(), result.data_ptr(), dense.size());
        } else {
            for (std::size_t i = 0; i < dense.size(); ++i) {
                result.data_ptr()[i] = dense.data_ptr()[i];
            }
        }

        // Mark all entries as initialized
        std::fill(result.init_ptr(), result.init_ptr() + result.size(), 1);

        // Subtract sparse values
        for (std::size_t i = 0; i < sparse.nonzero_count(); ++i) {
            std::size_t r = sparse.row_index(i);
            std::size_t c = sparse.col_index(i);
            result.update(r, c, result(r, c) - sparse.value(i));
        }

        return result;
    }
} // namespace slt
// ================================================================================ 
// ================================================================================ 
#endif /* MATRIX_HPP */
// ================================================================================
// ================================================================================
// eof
