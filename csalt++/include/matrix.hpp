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
        std::vector<uint8_t> init; ///< Flat row-major storage of matrix initialization elements.
        std::size_t rows_, cols_; ///< Matrix dimensions. 
    // ================================================================================ 
    public:
        /**
         * @brief Constructs a matrix with given dimensions and initial value.
         *
         * @param r Number of rows.
         * @param c Number of columns.
         * @param value Initial value for all elements (defaults to zero).
         */
        DenseMatrix(std::size_t r, std::size_t c, T value = T{})
            : data(r * c, value), init(r * c, value != T{}), rows_(r), cols_(c) {}
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
        bool is_initialized(std::size_t row, std::size_t col) const {
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
} // namespace slt
// ================================================================================ 
// ================================================================================ 
#endif /* MATRIX_HPP */
// ================================================================================
// ================================================================================
// eof
