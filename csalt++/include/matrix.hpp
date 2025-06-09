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
        std::vector<uint8_t> init;
        std::size_t rows_, cols_; ///< Matrix dimensions. 
        std::size_t size() const {return rows_ * cols_;}
    // ================================================================================ 
    public:
        /**
         * @brief Constructs a matrix with given dimensions and initial value.
         *
         * @param r Number of rows.
         * @param c Number of columns.
         * @param value Initial value for all elements (defaults to zero).
         */
        DenseMatrix(std::size_t r, std::size_t c, T value)
            : data(r * c, value), init(r * c, 1), rows_(r), cols_(c) {}
// -------------------------------------------------------------------------------- 

        /**
         * @brief Constructs a matrix with given dimensions and initial value.
         *
         * @param r Number of rows.
         * @param c Number of columns.
         */
        DenseMatrix(std::size_t r, std::size_t c)
            : data(r * c, 0), init(r * c, 0), rows_(r), cols_(c) {}
// -------------------------------------------------------------------------------- 

        /**
         * @brief Constructs a matrix from a 2D vector.
         *
         * All inner vectors must have the same length.
         *
         * @param vec 2D vector representing the matrix.
         * @throws std::invalid_argument if inner vectors are uneven in length.
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
         * @brief Constructs a matrix from a fixed-size std::array of arrays.
         *
         * @param arr A static 2D array representing the matrix.
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
         * All inner initializer lists must be the same length.
         *
         * @param init_list Nested initializer list representing the matrix.
         * @throws std::invalid_argument if rows have inconsistent lengths.
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
         * @brief Constructs a matrix from flat row-major data and dimensions.
         *
         * @param flat_data Vector containing r * c elements.
         * @param r Number of rows.
         * @param c Number of columns.
         * @throws std::invalid_argument if flat_data size doesn't match r * c.
         */
        DenseMatrix(const std::vector<T>& flat_data, std::size_t r, std::size_t c)
            : data(flat_data), init(flat_data.size(), 1), rows_(r), cols_(c) {
            if (flat_data.size() != r * c)
                throw std::invalid_argument("Flat data size does not match matrix dimensions");
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Accesses a matrix element (modifiable).
         *
         * @param r Row index (zero-based).
         * @param c Column index (zero-based).
         * @return Reference to the element at (r, c).
         * @throws std::out_of_range if r or c is out of bounds.
         */
        T& operator()(std::size_t r, std::size_t c) {
            if (r >= rows_ || c >= cols_)
                throw std::out_of_range("Index out of range");
            return data[r * cols_ + c];
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Accesses a matrix element (read-only).
         *
         * @param r Row index (zero-based).
         * @param c Column index (zero-based).
         * @return Const reference to the element at (r, c).
         * @throws std::out_of_range if r or c is out of bounds.
         */
        const T& operator()(std::size_t r, std::size_t c) const {
            if (r >= rows_ || c >= cols_)
                throw std::out_of_range("Index out of range");
            return data[r * cols_ + c];
        }
//  -------------------------------------------------------------------------------- 

        /**
         * @brief Adds two matrices element-wise.
         *
         * @param other Matrix to add.
         * @return Resulting matrix.
         * @throws std::invalid_argument if dimensions don't match.
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
         * @brief Subtracts another matrix element-wise.
         *
         * @param other Matrix to subtract.
         * @return Resulting matrix.
         * @throws std::invalid_argument if dimensions don't match.
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
         * @brief Adds a scalar to every matrix element.
         *
         * @param scalar Value to add.
         * @return Resulting matrix.
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
         * @brief Subtracts a scalar from every matrix element.
         *
         * @param scalar Value to subtract.
         * @return Resulting matrix.
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
         * @brief Performs element-wise multiplication between two matrices.
         *
         * Computes a new matrix where each element is the product of the corresponding elements
         * from this matrix and the `other` matrix. Throws if the matrix dimensions do not match.
         *
         * SIMD acceleration is used if supported for the element type.
         *
         * @param other The matrix to multiply element-wise with.
         * @return A new DenseMatrix containing the element-wise product.
         * @throws std::invalid_argument if matrix dimensions do not match.
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
         * @brief Performs scalar multiplication on the matrix.
         *
         * Returns a new matrix where each element is the product of the corresponding matrix
         * element and the provided scalar.
         *
         * SIMD acceleration is used if supported for the element type.
         *
         * @param scalar The scalar value to multiply with.
         * @return A new DenseMatrix containing the result of scalar multiplication.
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
         * @brief Divides each element of the matrix by a scalar.
         *
         * Performs element-wise division: `result(i, j) = this(i, j) / scalar` for all
         * initialized elements of the matrix. Uses SIMD acceleration where supported.
         *
         * @throws std::invalid_argument if scalar is zero.
         *
         * @return A new DenseMatrix<T> containing the result of the division. All elements
         *         in the result are marked as initialized.
         *
         * @note Division by zero is explicitly checked and throws an exception.
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
         * Swaps rows and columns.
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
         * @brief Computes and returns the inverse of the matrix.
         *
         * Uses Gauss-Jordan elimination with partial pivoting to compute the inverse
         * of the matrix. Throws an exception if the matrix is not square or is singular.
         *
         * @return A new DenseMatrix<T> representing the inverse of this matrix.
         * @throws std::invalid_argument if the matrix is not square.
         * @throws std::runtime_error if the matrix is singular or non-invertible.
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
         */
        std::size_t rows() const override { return rows_; }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Returns the number of columns in the matrix.
         */
        std::size_t cols() const override { return cols_; }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Gets a matrix element.
         *
         * Implements the MatrixBase interface.
         *
         * @param row Row index.
         * @param col Column index.
         * @return Value at (row, col).
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
         * @brief Sets a matrix element.
         *
         * Implements the MatrixBase interface.
         *
         * @param row Row index.
         * @param col Column index.
         * @param value New value to assign.
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
         * @brief Removes an element from the matrix by resetting its value and marking it uninitialized.
         *
         * This function sets the value at the specified (row, col) to zero and marks it as uninitialized.
         * Attempting to remove an element that is not initialized will throw a runtime error.
         *
         * @param row Row index
         * @param col Column index
         *
         * @throws std::out_of_range if the row or column index is out of bounds
         * @throws std::runtime_error if the element is not initialized
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
         * @brief Updates the value of an already-initialized matrix element.
         *
         * This function modifies the value at the specified (row, col) index
         * **only if** the element has already been initialized using `set()`
         * or a constructor that populates the matrix (e.g., with a 2D vector).
         * 
         * If the element has not been initialized, this function throws a
         * `std::runtime_error` to avoid accidental overwrites of uninitialized data.
         *
         * @param row The zero-based row index of the element.
         * @param col The zero-based column index of the element.
         * @param value The new value to assign to the matrix element.
         *
         * @throws std::out_of_range if the row or column is out of bounds.
         * @throws std::runtime_error if the element at (row, col) has not been initialized.
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
         * @brief Creates a polymorphic deep copy of this matrix.
         *
         * @return Unique pointer to the copied matrix.
         */
        std::unique_ptr<MatrixBase<T>> clone() const override {
            return std::make_unique<DenseMatrix>(*this);
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Prints the matrix to an output stream.
         *
         * @param os Output stream (defaults to std::cout).
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
         * @brief Check whether the specified matrix element has been initialized.
         *
         * This method returns true if the element at the specified (row, col)
         * index has been initialized via `set()` or a constructor, and false otherwise.
         *
         * @param row Row index of the matrix element.
         * @param col Column index of the matrix element.
         * @return true if initialized, false otherwise.
         * @throws std::out_of_range if the index is invalid.
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
     * @brief Adds a scalar to each matrix element (scalar + matrix).
     *
     * @tparam T Matrix element type.
     * @param scalar Scalar value.
     * @param matrix Matrix operand.
     * @return Resulting matrix.
     */
    template<typename T>
    DenseMatrix<T> operator+(T scalar, const DenseMatrix<T>& matrix) {
        return matrix + scalar;  // Leverage existing member operator+
    }
// -------------------------------------------------------------------------------- 

    /**
     * @brief Subtracts each matrix element from a scalar (scalar - matrix).
     *
     * @tparam T Matrix element type.
     * @param scalar Scalar value.
     * @param matrix Matrix operand.
     * @return Resulting matrix.
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
     * @brief Global operator for scalar * matrix multiplication.
     *
     * Enables the syntax `scalar * matrix` for scalar multiplication, delegating
     * to the matrix's member `operator*`.
     *
     * @param scalar The scalar multiplier.
     * @param matrix The matrix operand.
     * @return A new DenseMatrix containing the result.
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

    template<typename T>
    class SparseCOOMatrix : public MatrixBase<T> {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                      "DenseMatrix only supports float or double");
    private:
        std::vector<T> data; ///< Flat row-major storage of matrix elements. 
        std::size_t rows_, cols_; ///< Matrix dimensions. 
        std::size_t size() const {return rows_ * cols_;}  ///< Returns the total number of elments in the matrix

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
        * @brief Returns the number of rows in the matrix.
        */
        std::size_t rows() const override { return rows_; }
// -------------------------------------------------------------------------------- 

        /**
        * @brief Returns the number of columns in the matrix.
        */
        std::size_t cols() const override { return cols_; }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Gets a matrix element.
         *
         * Implements the MatrixBase interface.
         *
         * @param row Row index.
         * @param col Column index.
         * @return Value at (row, col).
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
                // Fast insert: append to end without checking for duplicates
                row.push_back(r);
                col.push_back(c);
                data.push_back(value);
                return;
            }

            // Sorted insert with binary search
            auto it = std::lower_bound(
                row.begin(), row.end(),
                std::make_pair(r, c),
                [this](std::size_t i, const std::pair<std::size_t, std::size_t>& target) {
                    return std::pair<std::size_t, std::size_t>{row[i], col[i]} < target;
                });

            std::size_t index = std::distance(row.begin(), it);

            if (index < row.size() && row[index] == r && col[index] == c) {
                throw std::runtime_error("Value already set. Use update() instead.");
            }

            row.insert(row.begin() + index, r);
            col.insert(col.begin() + index, c);
            data.insert(data.begin() + index, value);
            return;
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
         * @brief Checks whether an element at (row, column) has been initialized.
         *
         * Searches for the target index using binary search. The result is reliable
         * only if `finalize()` has been called after using fast insertion mode.
         *
         * @param r Row index of the element.
         * @param c Column index of the element.
         * @return True if the element has been initialized (i.e., set or updated).
         * @throws std::out_of_range if indices are invalid.
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

        bool set_optimized() const {
            return fast_set;
        }
    };
 } // namespace slt
// ================================================================================ 
// ================================================================================ 
#endif /* MATRIX_HPP */
// ================================================================================
// ================================================================================
// eof
