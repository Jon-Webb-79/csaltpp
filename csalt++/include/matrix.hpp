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

#include "simd_sse2_float.inl"
#include "simd_sse2_double.inl"
#include "simd_avx2_float.inl"
#include "simd_avx2_double.inl"


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

// ================================================================================ 

        template<>
        struct simd_ops<float> {
            static void add(const float* a, const float* b, float* result, std::size_t size) {
            #if defined(__AVX2__)
                slt::simd_add_f32_avx2(a, b, result, size);
            #elif defined(__SSE2__)
                slt::simd_add_f32_sse2(a, b, result, size);
            #elif defined(__SSE3__)
                slt::simd_add_f32_sse3(a, b, result, size);
            #else
                slt::simd_add_f32_scalar(a, b, result, size);
            #endif
            }
// -------------------------------------------------------------------------------- 

            static void sub(const float* a, const float* b, float* result, std::size_t size) {
            #if defined(__AVX2__)
                slt::simd_sub_f32_avx2(a, b, result, size);
            #elif defined(__SSE2__)
                slt::simd_sub_f32_sse2(a, b, result, size);
            #elif defined(__SSE3__)
                slt::simd_sub_f32_sse3(a, b, result, size);
            #else
                slt::simd_sub_f32_scalar(a, b, result, size);
            #endif
            }
// -------------------------------------------------------------------------------- 

            static void add_scalar(const float* a, float scalar, float* result, std::size_t size) {
            #if defined(__AVX2__)
                slt::simd_add_scalar_f32_avx2(a, scalar, result, size);
            #elif defined(__SSE2__)
                slt::simd_add_scalar_f32_sse2(a, scalar, result, size);
            #elif defined(__SSE3__)
                slt::simd_add_sclar_f32_sse3(a, scalar, result, size);
            #else
                slt::simd_add_scalar_f32_scalar(a, scalar, result, size);
            #endif
            }
// -------------------------------------------------------------------------------- 

            static void sub_scalar(const float* a, float scalar, float* result, std::size_t size) {
            #if defined(__AVX2__)
                slt::simd_sub_scalar_f32_avx2(a, scalar, result, size);
            #elif defined(__SSE2__)
                slt::simd_sub_scalar_f32_sse2(a, scalar, result, size);
            #elif defined(__SSE3__)
                slt::simd_sub_scalar_f32_sse3(a, scalar, result, size);
            #else
                slt::simd_sub_scalar_f32_scalar(a, scalar, result, size);
            #endif
            }
// -------------------------------------------------------------------------------- 

            static void mul(const float* a, const float* b, float* result, std::size_t size) {
            #if defined(__AVX2__)
                slt::simd_mul_f32_avx2(a, b, result, size);
            #elif defined(__SSE2__)
                slt::simd_mul_f32_sse2(a, b, result, size);
            #elif defined(__SSE3__)
                slt::simd_mul_f32_sse3(a, b, result, size);
            #else
                slt::simd_mul_f32_scalar(a, b, result, size);
            #endif
            }
// -------------------------------------------------------------------------------- 

            static void mul_scalar(const float* a, float scalar, float* result, std::size_t size) {
            #if defined(__AVX2__)
                slt::simd_mul_scalar_f32_avx2(a, scalar, result, size);
            #elif defined(__SSE2__)
                slt::simd_mul_scalar_f32_sse2(a, scalar, result, size);
            #elif defined(__SSE3__)
                slt::simd_mul_scalar_f32_sse3(a, scalar, result, size);
            #else
                slt::simd_mul_scalar_f32_scalar(a, scalar, result, size);
            #endif
            }
// -------------------------------------------------------------------------------- 

            static void div_scalar(const float* a, float scalar, float* result, std::size_t size) {
            #if defined(__AVX2__)
                slt::simd_div_scalar_f32_avx2(a, scalar, result, size);
            #elif defined(__SSE2__)
                slt::simd_div_scalar_f32_sse2(a, scalar, result, size);
            #elif defined(__SSE3__)
                slt::simd_div_scalar_f32_sse3(a, scalar, result, size);
            #else
                slt::simd_div_scalar_f32_scalar(a, scalar, result, size);
            #endif
            }
// -------------------------------------------------------------------------------- 

            static void copy(const float* src, float* dst, std::size_t size) {
            #if defined(__AVX2__)
                slt::simd_copy_f32_avx2(src, dst, size);
            #elif defined(__SSE2__)
                slt::simd_copy_f32_sse2(src, dst, size);
            #elif defined(__SSE3__)
                slt::simd_copy_f32_sse3(src, dst, size);
            #else
                slt::simd_copy_f32_scalar(src, dst, size);
            #endif
            }
// -------------------------------------------------------------------------------- 

            static float magnitude_squared(const float* data, std::size_t size) {
            #if defined(__AVX2__)
                return slt::simd_magnitude_squared_f32_avx2(data, size);
            #elif defined(__SSE2__)
                return slt::simd_magnitude_squared_f32_sse2(data, size);
            #elif defined(__SSE3__)
                return slt::simd_magnitude_squared_f32_sse3(data, size);
            #else
                return slt::simd_magnitude_squared_f32_scalar(data, size);
            #endif
            }
        };
// ================================================================================ 


        template<>
        struct simd_ops<double> {
            static void add(const double* a, const double* b, double* result, std::size_t size) {
            #if defined(__AVX2__)
                slt::simd_add_f64_avx2(a, b, result, size);
            #elif defined(__SSE2__)
                slt::simd_add_f64_sse2(a, b, result, size);
            #elif defined(__SSE3__)
                slt::simd_add_f64_sse3(a, b, result, size);
            #else
                slt::simd_add_f64_scalar(a, b, result, size);
            #endif
            }
// -------------------------------------------------------------------------------- 

            static void sub(const double* a, const double* b, double* result, std::size_t size) {
            #if defined(__AVX2__)
                slt::simd_sub_f64_avx2(a, b, result, size);
            #elif defined(__SSE2__)
                slt::simd_sub_f64_sse2(a, b, result, size);
            #elif defined(__SSE3__)
                slt::simd_sub_f64_sse3(a, b, result, size);
            #else
                slt::simd_sub_f64_scalar(a, b, result, size);
            #endif
            }
// -------------------------------------------------------------------------------- 

            static void add_scalar(const double* a, double scalar, double* result, std::size_t size) {
            #if defined(__AVX2__)
                slt::simd_add_scalar_f64_avx2(a, scalar, result, size);
            #elif defined(__SSE2__)
                slt::simd_add_scalar_f64_sse2(a, scalar, result, size);
            #elif defined(__SSE3__)
                slt:simd_add_scalar_f64_sse3(a, scalar, result, size);
            #else
                slt::simd_add_scalar_f64_scalar(a, scalar, result, size);
            #endif
            }
// -------------------------------------------------------------------------------- 

            static void sub_scalar(const double* a, double scalar, double* result, std::size_t size) {
            #if defined(__AVX2__)
                slt::simd_sub_scalar_f64_avx2(a, scalar, result, size);
            #elif defined(__SSE2__)
                slt::simd_sub_scalar_f64_sse2(a, scalar, result, size);
            #elif defined(__SSE3__)
                slt::simd_sub_scalar_f64_sse3(a, scalar, result, size);
            #else
                slt::simd_sub_scalar_f64_scalar(a, scalar, result, size);
            #endif
            }
// -------------------------------------------------------------------------------- 

            static void mul(const double* a, const double* b, double* result, std::size_t size) {
            #if defined(__AVX2__)
                slt::simd_mul_f64_avx2(a, b, result, size);
            #elif defined(__SSE2__)
                slt::simd_mul_f64_sse2(a, b, result, size);
            #elif defined(__SSE3__)
                slt::simd_mul_f64_sse3(a, b, result, size);
            #else
                slt::simd_mul_f64_scalar(a, b, result, size);
            #endif
            }
// -------------------------------------------------------------------------------- 

            static void mul_scalar(const double* a, double scalar, double* result, std::size_t size) {
            #if defined(__AVX2__)
                slt::simd_mul_scalar_f64_avx2(a, scalar, result, size);
            #elif defined(__SSE2__)
                slt::simd_mul_scalar_f64_sse2(a, scalar, result, size);
            #elif defined(__SSE3__)
                slt::simd_mul_scalar_f64_sse3(a, scalar, result, size);
            #else
                slt::simd_mul_scalar_f64_scalar(a, scalar, result, size);
            #endif
            }
// -------------------------------------------------------------------------------- 

            static void div_scalar(const double* a, double scalar, double* result, std::size_t size) {
            #if defined(__AVX2__)
                slt::simd_div_scalar_f64_avx2(a, scalar, result, size);
            #elif defined(__SSE2__)
                slt::simd_div_scalar_f64_sse2(a, scalar, result, size);
            #elif defined(__SSE3__)
                slt::simd_div_scalar_f64_sse3(a, scalar, result, size);
            #else
                slt::simd_div_scalar_f64_scalar(a, scalar, result, size);
            #endif
            }
// -------------------------------------------------------------------------------- 

            static void copy(const double* src, double* dst, std::size_t size) {
            #if defined(__AVX2__)
                slt::simd_copy_f64_avx2(src, dst, size);
            #elif defined(__SSE2__)
                slt::simd_copy_f64_sse2(src, dst, size);
            #elif defined(__SSE3__)
                slt::simd_copy_f64_sse3(src, dst, size);
            #else
                slt::simd_copy_f64_scalar(src, dst, size);
            #endif
            }
// -------------------------------------------------------------------------------- 

            static double magnitude_squared(const double* data, std::size_t size) {
            #if defined(__AVX2__)
                return slt::simd_magnitude_squared_f64_avx2(data, size);
            #elif defined(__SSE2__)
                return slt::simd_magnitude_squared_f64_sse2(data, size);
            #elif defined(__SSE3__)
                return slt::simd_magnitude_squared_f64_sse2(data, size);
            #else
                return slt::simd_magnitude_squared_f64_scalar(data, size);
            #endif
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
    protected:
        std::size_t rows_ = 0;
        std::size_t cols_ = 0;
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
        std::size_t rows() const { return rows_; }
// -------------------------------------------------------------------------------- 
    
        /**
         * @brief Returns the number of columns in the matrix.
         *
         * @return Number of columns.
         */
        std::size_t cols() const { return cols_; }
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
        virtual std::size_t initialized_count() const = 0;
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
// FORWARD DECLARATIONS 

    template<typename T>
    class SparseCOOMatrix;
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
    // ================================================================================ 
    public:
        /**
         * @brief The total size of the matrix 
         *
         * @return The number of rows multiplied by the number of columns, 0 if not initialized
         */
        std::size_t size() const override {return this->rows_ * this->cols_;}
// -------------------------------------------------------------------------------- 

        /**
         * @brief Returns an iterator to the beginning of the data array.
         *
         * This allows iteration over all elements of the dense matrix in row-major order,
         * using range-based for loops or STL algorithms.
         *
         * @return Iterator to the first value in the matrix.
         *
         * Example:
         * @code
         * slt::DenseMatrix<float> mat(2, 3, 1.0f);
         *
         * for (auto it = mat.begin(); it != mat.end(); ++it) {
         *     std::cout << *it << " ";
         * }
         * // Output: 1 1 1 1 1 1
         * @endcode
         */
        auto begin() const { return data.begin(); }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Returns a mutable pointer to the beginning of matrix data.
         *
         * This overload allows modification of the matrix contents. It is useful for
         * optimized operations such as SIMD accelerated copy, assignment, or transformations.
         *
         * Example:
         * @code
         * slt::DenseMatrix<float> mat(3, 3, 1.0f);
         * auto* ptr = mat.begin();
         * ptr[0] = 42.0f;  // modifies element (0,0)
         * @endcode
         *
         * @return Pointer to the first element (float* or double* depending on T)
         */
        auto begin() { return data.end(); }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Returns an iterator to one-past-the-end of the data array.
         *
         * Used for iteration over the dense matrix in row-major order using range-based for loops or STL algorithms.
         *
         * @return Iterator to one past the last value.
         *
         * Example:
         * @code
         * slt::DenseMatrix<float> mat(2, 3, 2.0f);
         *
         * for (const auto& value : mat) {
         *     std::cout << value << " ";
         * }
         * // Output: 2 2 2 2 2 2
         * @endcode
         */
        auto end() const { return data.end(); }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Returns a mutable pointer to the end of matrix data.
         *
         * This overload allows modification of matrix contents in algorithms that operate
         * over a range (e.g., std::fill, std::copy, SIMD-accelerated loops).
         *
         * Example:
         * @code
         * slt::DenseMatrix<float> mat(2, 2);
         * std::fill(mat.begin(), mat.end(), 3.14f);  // fills all elements with 3.14
         * @endcode
         *
         * @return Pointer one-past-the-last element (float* or double* depending on T)
         */
        auto end()   { return data.end(); }
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
// -------------------------------------------------------------------------------- 

        uint8_t* init_ptr() {return init.data();}
// -------------------------------------------------------------------------------- 

        // Purpusefully not documenting to protect from bad use of this function used internally 
        const T* data_ptr() const {return data.data();}
// -------------------------------------------------------------------------------- 

        // Purpusefully not documenting to protect from bad use of this function used internally
        T* data_ptr() {return data.data();}
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
         *     std::cout << "Initialized elements: " << mat.initialized_count() << std::endl;
         *     return 0;
         * }
         * @endcode
         *
         * Output:
         * @code
         * Initialized elements: 2
         * @endcode
         */
        std::size_t initialized_count() const override {
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
            : data(r * c, value), init(r * c, 1) {
                this->rows_ = r;
                this->cols_ = c;
            }
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
         *     std::cout << "Initialized count: " << mat.initialized_count() << std::endl;
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
            : data(r * c, 0), init(r * c, 0) {
            this->rows_ = r;
            this->cols_ = c;
        }
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
            this->rows_ = vec.size();
            this->cols_ = this->rows_ ? vec[0].size() : 0;
            data.resize(this->rows_ * this->cols_);
            init.resize(this->rows_ * this->cols_, 1);
            for (std::size_t i = 0; i < this->rows_; ++i) {
                if (vec[i].size() != this->cols_)
                    throw std::invalid_argument("All rows must have the same number of columns");
                for (std::size_t j = 0; j < this->cols_; ++j)
                    data[i * this->cols_ + j] = vec[i][j];
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
            : data(Rows * Cols), init(Rows * Cols, 1){
            this->rows_ = Rows;
            this->cols_ = Cols;
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
            this->rows_ = init_list.size();
            this->cols_ = this->rows_ ? init_list.begin()->size() : 0;
            data.reserve(this->rows_ * this->cols_);
            init.reserve(this->rows_ * this->cols_);
            for (const auto& row : init_list) {
                if (row.size() != this->cols_)
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
            : data(flat_data), init(flat_data.size(), 1) {
            this->rows_ = r;
            this->cols_ = c;
            if (flat_data.size() != r * c)
                throw std::invalid_argument("Flat data size does not match matrix dimensions");
        }
// -------------------------------------------------------------------------------- 

        DenseMatrix(std::vector<T>&& flat_data, std::size_t r, std::size_t c)
            : data(std::move(flat_data)), init(data.size(), 1) {
            this->rows_ = r;
            this->cols_ = c;
            if (data.size() != r * c)
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
            : data(arr.begin(), arr.end()), init(N, 1) {
            this->rows_ = r;
            this->cols_ = c;
            if (N != r * c)
                throw std::invalid_argument("Flat array size does not match matrix dimensions");
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Constructs a DenseMatrix from a SparseCOOMatrix.
         *
         * Initializes this DenseMatrix by copying all non-zero values from the given sparse matrix.
         * Only the entries present in the sparse triplet vector are marked initialized.
         * Remaining entries are set to zero and marked uninitialized.
         *
         * This allows accurate use of `is_initialized()` and prevents uninitialized memory access.
         *
         * @param sparse The source sparse COO matrix to convert.
         *
         * @example
         * @code
         * slt::SparseCOOMatrix<float> sparse(3, 3);
         * sparse.set(0, 1, 5.0f);
         * sparse.set(2, 2, 3.0f);
         *
         * slt::DenseMatrix<float> dense(sparse);
         *
         * EXPECT_EQ(dense(0, 1), 5.0f);
         * EXPECT_EQ(dense(2, 2), 3.0f);
         * EXPECT_FALSE(dense.is_initialized(0, 0));  // Empty entry
         * @endcode
         */
        explicit DenseMatrix(const SparseCOOMatrix<T>& sparse)
            : data(sparse.rows() * sparse.cols(), T{}),
              init(sparse.rows() * sparse.cols(), 0)  // All entries initialized
        {
            this->rows_ = sparse.rows();
            this->cols_ = sparse.cols();
            for (const auto& t : sparse) {
                std::size_t idx = t.row * this->cols_ + t.col;
                data[idx] = t.value;
                init[idx] = 1;
            }
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
              init(other.init) {
            this->rows_ = other.rows_;
            this->cols_ = other.cols_;
        }
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
              init(std::move(other.init)) {
            this->rows_ = std::exchange(other.rows_, 0);
            this->cols_ = std::exchange(other.cols_, 0);
        }
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
        explicit DenseMatrix(std::size_t n) : data(n * n, 0), init(n * n, 1) {
            if (n == 0)
                throw std::invalid_argument("Size of identity matrix must be greater than zero");

            this->rows_ = n;
            this->cols_ = n;
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
            if (r >= this->rows_ || c >= this->cols_)
                throw std::out_of_range("Matrix index out of bounds");

            std::size_t idx = r * this->cols_ + c;

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
            if (r >= this->rows_ || c >= this->cols_)
                throw std::out_of_range("Matrix index out of bounds");

            std::size_t idx = r * this->cols_ + c;

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
                this->rows_ = other.rows_;
                this->cols_ = other.cols_;
            }
            return *this;
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Assigns the contents of a SparseCOOMatrix to this DenseMatrix.
         *
         * This operator clears the DenseMatrix and copies the non-zero elements from the given sparse matrix.
         * The `init` flags are updated to reflect only the initialized positions from the sparse source.
         *
         * Existing DenseMatrix data is resized to match the sparse matrix shape.
         *
         * @param sparse The source SparseCOOMatrix to assign from.
         * @return Reference to this DenseMatrix.
         *
         * @example
         * @code
         * slt::SparseCOOMatrix<float> sparse(2, 2);
         * sparse.set(1, 0, 4.5f);
         *
         * slt::DenseMatrix<float> dense(2, 2);
         * dense = sparse;
         *
         * EXPECT_FLOAT_EQ(dense(1, 0), 4.5f);
         * EXPECT_FALSE(dense.is_initialized(0, 0));
         * @endcode
         */
        DenseMatrix<T>& operator=(const SparseCOOMatrix<T>& sparse) {
            if (this->rows_ != sparse.rows() || this->cols_ != sparse.cols()) {
                this->rows_ = sparse.rows();
                this->cols_ = sparse.cols();
                data.resize(this->rows_ * this->cols_, T{});
                init.resize(this->rows_ * this->cols_, 1);  // Mark all initialized
            } else {
                std::fill(data.begin(), data.end(), T{});
                std::fill(init.begin(), init.end(), 0);
            }

            for (const auto& t : sparse) {
                std::size_t idx = t.row * this->cols_ + t.col;
                data[idx] = t.value;
                init[idx] = 1;
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
                this->rows_ = std::exchange(other.rows_, 0);
                this->cols_ = std::exchange(other.cols_, 0);
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
            if (this->rows_ != other.rows_ || this->cols_ != other.cols_)
                throw std::invalid_argument("Matrix dimensions must match for addition");

            DenseMatrix result(this->rows_, this->cols_);
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
            DenseMatrix result(this->rows_, this->cols_);
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
            if (this->rows_ != other.rows_ || this->cols_ != other.cols_)
                throw std::invalid_argument("Matrix dimensions must match for subtraction");

            DenseMatrix result(this->rows_, this->cols_);
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
            DenseMatrix result(this->rows_, this->cols_);
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
            if (this->rows_ != other.rows_ || this->cols_ != other.cols_)
                throw std::invalid_argument("Matrix dimensions must match for element-wise multiplication");

            DenseMatrix result(this->rows_, this->cols_);
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
            DenseMatrix result(this->rows_, this->cols_);
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

            DenseMatrix result(this->rows_, this->cols_);
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
            for (std::size_t i = 0; i < this->rows_; ++i) {
                for (std::size_t j = 0; j < this->cols_; ++j) {
                    new_data[j * this->rows_ + i] = data[i * this->cols_ + j];
                }
            }
            data.swap(new_data);
            std::swap(this->rows_, this->cols_);
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
            if (this->rows_ != this->cols_)
                throw std::invalid_argument("Only square matrices can be inverted");

            const std::size_t n = this->rows_;
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
            if (row >= this->rows_ || col >= this->cols_)
                throw std::out_of_range("Index out of range");
            
            std::size_t idx = row * this->cols_ + col;
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
        void set(std::size_t row, std::size_t col, T value) {
            if (row >= this->rows_ || col >= this->cols_)
                throw std::out_of_range("Index out of range");

            std::size_t idx = row * this->cols_ + col;
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
            if (row >= this->rows_ || col >= this->cols_)
                throw std::out_of_range("Index out of range");
            std::size_t idx = row * this->cols_ + col;
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
            if (row >= this->rows_ || col >= this->cols_)
                throw std::out_of_range("Update failed: index out of bounds");

            std::size_t index = row * this->cols_ + col;

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
            if (row >= this->rows_ || col >= this->cols_)
                throw std::out_of_range("Index out of range");
            return init[row * this->cols_ + col] != 0;
        }
    };
// ================================================================================ 

    /**
     * @brief Stream output operator for DenseMatrix.
     *
     * Prints the contents of the DenseMatrix<T> in row-major order.
     * 
     * - Initialized values are printed numerically.
     * - Uninitialized entries are shown as "." to visually indicate unset positions.
     *
     * Example output:
     * ```
     * 1.0 . .
     * . . 2.5
     * ```
     *
     * Useful for debugging and inspection.
     *
     * @tparam T Element type (float or double).
     * @param os Output stream (e.g. std::cout)
     * @param mat DenseMatrix<T> to print
     * @return Reference to the output stream
     *
     * Example usage:
     * @code
     * DenseMatrix<float> mat(2, 2);
     * mat.set(0, 0, 1.0f);
     * mat.set(1, 1, 2.0f);
     *
     * std::cout << mat;
     * // Output:
     * // 1.0 .
     * // .   2.0
     * @endcode
     */
    template<typename T>
    std::ostream& operator<<(std::ostream& os, const DenseMatrix<T>& mat) {
        for (std::size_t r = 0; r < mat.rows(); ++r) {
            for (std::size_t c = 0; c < mat.cols(); ++c) {
                if (mat.is_initialized(r, c)) {
                    os << mat(r, c);
                } else {
                    os << ".";
                }

                if (c != mat.cols() - 1)
                    os << " ";  // spacing between columns
            }
            os << "\n";  // new row
        }
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
     * @brief Computes the Euclidean magnitude (L2 norm) of a c-style array
     *
     * This function calculates the square root of the sum of squares of all elements
     * in a vector. Internally, it uses SIMD acceleration (via `simd_ops`) when available.
     * Supports `float` and `double` types, enforced by `static_assert`.
     *
     * @tparam T The numeric type of the vector elements (`float` or `double` only).
     * @param data Pointer to the start of the array.
     * @param size Number of elements in the array.
     * @return The Euclidean magnitude of the vector.
     *
     * Example (float array):
     * @code
     * #include <iostream>
     * float data[] = {3.0f, 4.0f};
     * float result = magnitude(data, 2);
     * std::cout << "Magnitude: " << result << std::endl;  // Output: 5.0
     * @endcode
     */
    template<typename T>
    T magnitude(const T* data, std::size_t size) {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                      "magnitude is only supported for float and double types");

        T mag_sq = simd_ops<T>::magnitude_squared(data, size);
        return std::sqrt(mag_sq);
    }
// -------------------------------------------------------------------------------- 

    /**
     * @brief Computes the Euclidean magnitude (L2 norm) of a vector.
     *
     * This function calculates the square root of the sum of squares of all elements
     * in the vector. Internally, it uses SIMD acceleration (via `simd_ops`) when available.
     * Supports `float` and `double` types, enforced by `static_assert`.
     *
     * @tparam T The numeric type of the vector elements (`float` or `double` only).
     * @param vec The input `std::vector<T>` whose magnitude will be computed.
     * @return The Euclidean magnitude of the vector.
     *
     * @code
     * int main() {
     *     std::vector<double> vec = {1.0, 2.0, 2.0};
     *     double result = magnitude(vec);
     *     std::cout << "Magnitude: " << result << std::endl;  // Output: 3.0
     *     return 0;
     * }
     * @endcode
     */
    template<typename T>
    T magnitude(const std::vector<T>& vec) {
        return magnitude(vec.data(), vec.size());
    }
// -------------------------------------------------------------------------------- 

       /**
     * @brief Computes the squared Euclidean norm of an array.
     *
     * This internal helper function calculates the sum of squares of all elements
     * in the input array. SIMD acceleration is used via `simd_ops<T>` if supported.
     * Typically used inside the `magnitude()` function.
     *
     * @tparam T The numeric type of the elements (`float` or `double` only).
     * @tparam N The size of the array.
     * @param arr The input `std::array<T, N>` whose magnitude will be computed.
     * @return The sum of squared elements (i.e., squared magnitude).
     *
     * Example:
     * @code
     * std::array<float, 3> v = {1.0f, 2.0f, 2.0f};
     * float mag = magnitude(v);
     * std::cout << "Squared magnitude: " << mag << std::endl;  // Output: 9.0
     * @endcode
     */ 
    template<typename T, std::size_t N>
    T magnitude(const std::array<T, N>& arr) {
        return magnitude(arr.data(), N);
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
     * @brief Represents a single non-zero entry in a sparse COO matrix.
     *
     * Stores the row index, column index, and value for a sparse matrix element.
     * Supports sorting and equality comparison based on (row, col) order.
     *
     * @tparam T Must be either float or double.
     */
    template<typename T>
    struct Triplet {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                      "Triplet<T> only supports float or double");

        std::size_t row{};  ///< Row index of the entry.
        std::size_t col{};  ///< Column index of the entry.
        T value{};          ///< Value at the specified matrix position.
// -------------------------------------------------------------------------------- 

        /**
         * @brief Constructs an empty Triplet with default values (0, 0, 0).
         *
         * Example:
         * @code
         * slt::Triplet<float> t;
         * assert(t.row == 0 && t.col == 0 && t.value == 0.0f);
         * @endcode
         */
        Triplet() = default;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Constructs a Triplet with specified row, column, and value.
         * 
         * @param r Row index.
         * @param c Column index.
         * @param v Value to store.
         * 
         * Example:
         * @code
         * slt::Triplet<double> t(1, 2, 3.14);
         * assert(t.row == 1 && t.col == 2 && t.value == 3.14);
         * @endcode
         */
        Triplet(std::size_t r, std::size_t c, T v)
            : row(r), col(c), value(v) {}
// -------------------------------------------------------------------------------- 

        /**
         * @brief Constructs a copy of another Triplet.
         * 
         * @param other The Triplet to copy.
         *
         * Example:
         * @code
         * slt::Triplet<float> t1(0, 1, 1.0f);
         * slt::Triplet<float> t2(t1);
         * assert(t2.equals(t1));
         * @endcode
         */
        Triplet(const Triplet& other) = default;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Move constructor for efficient transfer of Triplet.
         * 
         * @param other The Triplet to move from.
         *
         * Example:
         * @code
         * slt::Triplet<float> t1(0, 1, 2.0f);
         * slt::Triplet<float> t2(std::move(t1));
         * @endcode
         */
        Triplet(Triplet&& other) noexcept = default;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Copy-assigns another Triplet.
         * 
         * @param other The Triplet to copy from.
         * @return Reference to the current object.
         *
         * Example:
         * @code
         * slt::Triplet<double> t1(1, 2, 3.0);
         * slt::Triplet<double> t2;
         * t2 = t1;
         * assert(t2.equals(t1));
         * @endcode
         */
        Triplet& operator=(const Triplet& other) = default;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Move-assigns another Triplet.
         * 
         * @param other The Triplet to move from.
         * @return Reference to the current object.
         *
         * Example:
         * @code
         * slt::Triplet<float> t1(1, 2, 3.0f);
         * slt::Triplet<float> t2;
         * t2 = std::move(t1);
         * @endcode
         */
        Triplet& operator=(Triplet&& other) noexcept = default;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Compares Triplets by row-major order (row, then column).
         * 
         * @param other The Triplet to compare with.
         * @return True if this Triplet precedes the other.
         *
         * Example:
         * @code
         * slt::Triplet<float> a(1, 0, 2.0f);
         * slt::Triplet<float> b(1, 2, 2.0f);
         * assert(a < b);
         * @endcode
         */
        bool operator<(const Triplet& other) const {
            return std::tie(row, col) < std::tie(other.row, other.col);
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Checks if two Triplets refer to the same (row, col) location.
         * 
         * @param other The Triplet to compare with.
         * @return True if row and column are equal.
         *
         * Example:
         * @code
         * slt::Triplet<double> a(0, 0, 1.0);
         * slt::Triplet<double> b(0, 0, 2.0);
         * assert(a == b); // Value is ignored
         * @endcode
         */
        bool operator==(const Triplet& other) const {
            return row == other.row && col == other.col;
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Compares full equality (row, column, and value).
         * 
         * @param other The Triplet to compare with.
         * @return True if all fields are equal.
         *
         * Example:
         * @code
         * slt::Triplet<float> a(0, 1, 1.0f);
         * slt::Triplet<float> b(0, 1, 1.0f);
         * assert(a.equals(b));
         * @endcode
         */
        bool equals(const Triplet& other) const {
            return row == other.row && col == other.col && value == other.value;
        }
    };
// ================================================================================ 
// ================================================================================ 

    /**
     * @brief Sparse Coordinate (COO) Matrix class.
     *
     * This class represents a sparse matrix using the Coordinate List (COO) format.
     * Each non-zero element of the matrix is stored as a `Triplet` object, containing (row, column, value).
     * 
     * The matrix supports two internal insertion modes:
     * - **Fast insert mode (`fast_set == true`)**: Triplets are appended to the internal vector in O(1) time.
     *   This allows very efficient construction of the matrix, but requires the user to call `finalize()`
     *   before performing retrievals such as `get()` or binary searches.
     * - **Sorted mode (`fast_set == false`)**: Triplets are kept sorted by (row, column), allowing efficient
     *   retrieval using binary search, at the cost of slower insertions.
     *
     * Supported operations:
     * - Insertion of new elements (`set()`)
     * - Updating existing elements (`update()`)
     * - Element-wise arithmetic with other SparseCOOMatrix or DenseMatrix
     * - Scalar operations (+, -, *) 
     * - Conversion to DenseMatrix
     * - Iteration over stored triplets (`begin()`, `end()`)
     * 
     * Template parameter T must be either float or double.
     * 
     * Example usage:
     * @code
     * slt::SparseCOOMatrix<float> mat(5, 5);
     * mat.set(0, 0, 1.0f);
     * mat.set(2, 3, 5.5f);
     * mat.finalize();
     * float val = mat.get(2, 3);  // returns 5.5f
     * @endcode
     * 
     * @tparam T The type of the matrix elements (float or double).
     */
    template<typename T>
    class SparseCOOMatrix : public MatrixBase<T> {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                      "DenseMatrix only supports float or double");
    private:
        std::vector<Triplet<T>> triplet; ///< A vector of triplet objects
        bool fast_set = true;  ///< true if vectors are optimized for insertation, false if optimized for retrieval
// ================================================================================ 

    public:
        /**
         * @brief The total size of the matrix 
         *
         * @return The number of rows multiplied by the number of columns, 0 if not initialized
         */
        std::size_t size() const override {return this->rows_ * this->cols_;} 
// -------------------------------------------------------------------------------- 

        /**
         * @brief Returns the number of non-zero elements stored in the matrix.
         *
         * This function returns the number of elements currently stored in the internal
         * `triplet` vector, which corresponds to the number of explicitly initialized
         * non-zero entries in the sparse matrix.
         *
         * @return The number of non-zero elements.
         *
         * Example:
         * @code
         * slt::SparseCOOMatrix<float> mat(3, 3);
         * mat.set(0, 0, 1.0f);
         * mat.set(1, 2, 2.5f);
         * std::cout << "Non-zero count: " << mat.initialized_count() << std::endl;
         * @endcode
         *
         * Output:
         * @code
         * Non-zero count: 2
         * @endcode
         */
        std::size_t initialized_count() const override {return triplet.size();} 
// -------------------------------------------------------------------------------- 

        /**
         * @brief Constructs an empty sparse COO matrix with specified dimensions.
         *
         * This constructor initializes a matrix of size `r` x `c` in fast insertion mode.
         * Entries can be added efficiently using `set()`. To enable optimized access through
         * methods like `get()` or `is_initialized()`, the user must call `finalize()` after
         * all insertions are complete.
         *
         * @param r Number of rows in the matrix.
         * @param c Number of columns in the matrix.
         * @param initial_capacity Optional initial reservation size for internal storage vectors
         *                         (default: 16), which can reduce reallocations during insertion.
         *
         * @code
         * #include "sparse_coo_matrix.hpp"
         * #include <iostream>
         *
         * int main() {
         *     // Create a 3x3 sparse matrix
         *     slt::SparseCOOMatrix<float> mat(3, 3);
         *
         *     // Insert values (in fast insertion mode)
         *     mat.set(0, 0, 1.0f);
         *     mat.set(1, 2, 2.5f);
         *     mat.set(2, 1, -3.2f);
         *
         *     // Finalize before using get() or is_initialized()
         *     mat.finalize();
         *
         *     // Access and print a value
         *     if (mat.is_initialized(1, 2)) {
         *         std::cout << "mat(1, 2) = " << mat.get(1, 2) << std::endl;
         *     }
         *
         *     return 0;
         * }
         * @endcode
         *
         * **Output:**  
         * ``mat(1, 2) = 2.5``
         */
        explicit SparseCOOMatrix(std::size_t r, std::size_t c, std::size_t initial_capacity = 16) {
            this->rows_ = r;
            this->cols_ = c;
            triplet.reserve(initial_capacity);
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Constructs a SparseCOOMatrix from a vector of Triplet<T>.
         *
         * The resulting matrix is initialized with the provided triplets.
         * The internal storage is automatically sorted in row-major order (row, then col),
         * and the matrix is ready for optimized access (fast_set = false).
         *
         * @param r Number of rows in the matrix.
         * @param c Number of columns in the matrix.
         * @param triplets Vector of triplet values to insert.
         *
         * @code
         * std::vector<slt::Triplet<float>> triplets = {
         *     {0, 0, 1.0f},
         *     {1, 2, 2.5f},
         *     {4, 4, 3.1f}
         * };
         * slt::SparseCOOMatrix<float> mat(5, 5, triplets);
         * @endcode
         */
        SparseCOOMatrix(std::size_t r, std::size_t c, const std::vector<Triplet<T>>& triplets)
            : triplet(triplets), fast_set(false)
        {
            this->rows_ = r;
            this->cols_ = c;
            std::sort(triplet.begin(), triplet.end());
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Constructs a SparseCOOMatrix from an rvalue vector of Triplet<T> (move).
         *
         * Moves the contents of the given vector into the matrix to avoid unnecessary copying.
         * This is the most efficient way to initialize a large sparse matrix from a temporary
         * or intermediate vector of triplets. After the move, the input vector will be empty.
         *
         * The triplets are automatically sorted in row-major order (row first, then col),
         * and the matrix is ready for optimized access (`fast_set = false`).
         *
         * @param r Number of rows in the matrix.
         * @param c Number of columns in the matrix.
         * @param triplets Rvalue reference to a vector of triplet values to move into the matrix.
         *
         * @note The original vector passed in will be left empty after construction.
         *
         * @example
         * @code
         * std::vector<slt::Triplet<float>> triplets = {
         *     {0, 0, 1.0f},
         *     {1, 2, 2.5f},
         *     {4, 4, 3.1f}
         * };
         *
         * // Efficient move construction
         * slt::SparseCOOMatrix<float> mat(5, 5, std::move(triplets));
         *
         * // After this, triplets.size() == 0
         * @endcode
         */
        SparseCOOMatrix(std::size_t r, std::size_t c, std::vector<Triplet<T>>&& triplets)
            : triplet(std::move(triplets)), fast_set(false)
        {
            this->rows_ = r;
            this->cols_ = c;
            std::sort(triplet.begin(), triplet.end());
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Constructs a SparseCOOMatrix from a fixed-size std::array of Triplet<T>.
         *
         * The matrix is initialized with the given triplets and automatically sorted.
         *
         * @tparam N Number of triplets.
         * @param r Number of rows in the matrix.
         * @param c Number of columns in the matrix.
         * @param triplets Array of triplet values to insert.
         *
         * @code
         * std::array<slt::Triplet<double>, 2> triplets = {
         *     slt::Triplet<double>(0, 1, 3.14),
         *     slt::Triplet<double>(2, 2, 2.71)
         * };
         * slt::SparseCOOMatrix<double> mat(3, 3, triplets);
         * @endcode
         */
        template<std::size_t N>
        SparseCOOMatrix(std::size_t r, std::size_t c, const std::array<Triplet<T>, N>& triplets)
            : triplet(triplets.begin(), triplets.end()), fast_set(false)
        {
            this->rows_ = r;
            this->cols_ = c;
            std::sort(triplet.begin(), triplet.end());
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Constructs a SparseCOOMatrix from a C-style array of Triplet<T>.
         *
         * Useful for integrating with legacy code or simple fixed data.
         * The matrix is sorted automatically after construction.
         *
         * @param r Number of rows in the matrix.
         * @param c Number of columns in the matrix.
         * @param triplets Pointer to an array of Triplet<T>.
         * @param count Number of triplets in the array.
         *
         * @code
         * slt::Triplet<float> triplets[] = {
         *     {0, 0, 1.0f},
         *     {2, 3, 4.5f}
         * };
         * slt::SparseCOOMatrix<float> mat(4, 4, triplets, 2);
         * @endcode
         */
        SparseCOOMatrix(std::size_t r, std::size_t c, const Triplet<T>* triplets, std::size_t count)
            : triplet(triplets, triplets + count), fast_set(false)
        {
            this->rows_ = r;
            this->cols_ = c;
            std::sort(triplet.begin(), triplet.end());
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Constructs a SparseCOOMatrix from an initializer list of Triplet<T>.
         *
         * Allows convenient inline initialization using brace-enclosed lists.
         * The matrix is sorted automatically after construction.
         *
         * @param r Number of rows in the matrix.
         * @param c Number of columns in the matrix.
         * @param init_list List of triplets to insert.
         *
         * @code
         * slt::SparseCOOMatrix<float> mat(4, 4, {
         *     {0, 1, 1.5f},
         *     {2, 0, 3.0f},
         *     {3, 3, 2.0f}
         * });
         * @endcode
         */
        SparseCOOMatrix(std::size_t r, std::size_t c, std::initializer_list<Triplet<T>> init_list)
            : triplet(init_list), fast_set(false)
        {
            this->rows_ = r;
            this->cols_ = c;
            std::sort(triplet.begin(), triplet.end());
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Constructs an identity matrix as a SparseCOOMatrix<T>.
         *
         * Creates a square sparse matrix of size n x n, with ones on the main diagonal
         * and zeros elsewhere. The resulting matrix is sorted and ready for optimized access.
         *
         * @param n The number of rows and columns (matrix is n x n).
         *
         * @example
         * @code
         * slt::SparseCOOMatrix<float> I(4);  // 4x4 identity matrix
         * EXPECT_FLOAT_EQ(I.get(0, 0), 1.0f);
         * EXPECT_FLOAT_EQ(I.get(1, 1), 1.0f);
         * EXPECT_FLOAT_EQ(I.get(2, 2), 1.0f);
         * EXPECT_FLOAT_EQ(I.get(3, 3), 1.0f);
         * @endcode
         */
        explicit SparseCOOMatrix(std::size_t n)
            : fast_set(false)
        {
            this->rows_ = n;
            this->cols_ = n;
            triplet.reserve(n);
            for (std::size_t i = 0; i < n; ++i) {
                triplet.emplace_back(i, i, static_cast<T>(1));
            }
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Constructs a SparseCOOMatrix from a DenseMatrix<T>.
         *
         * This constructor creates a SparseCOOMatrix by copying all initialized and non-zero
         * elements from the given DenseMatrix<T>. Elements in the dense matrix that are either
         * uninitialized or exactly zero are omitted to preserve sparsity.
         *
         * The resulting matrix will have the same number of rows and columns as the input dense matrix.
         * After construction, the internal triplet list is sorted in row-major order and
         * `fast_set` is set to `false` for efficient retrieval.
         *
         * @param dense The input DenseMatrix<T> to convert.
         * @param accept_zeros Accepts 0 values if true, rejects them if false.  Defaulted to true
         *
         * @example
         * @code
         * slt::DenseMatrix<float> dense({
         *     {1.0f, 0.0f},
         *     {0.0f, 2.5f}
         * });
         *
         * slt::SparseCOOMatrix<float> sparse(dense);
         *
         * EXPECT_EQ(sparse.initialized_count(), 2);
         * EXPECT_FLOAT_EQ(sparse.get(0, 0), 1.0f);
         * EXPECT_FLOAT_EQ(sparse.get(1, 1), 2.5f);
         * @endcode
         */ 
        explicit SparseCOOMatrix(const DenseMatrix<T>& dense, bool accept_zeros = true)
            : fast_set(false)
        {
            this->cols_ = dense.cols();
            this->rows_ = dense.rows();
            triplet.reserve(dense.size());  // Conservative guess, not all will be used

            for (std::size_t r = 0; r < dense.rows(); ++r) {
                for (std::size_t c = 0; c < dense.cols(); ++c) {
                    if (dense.is_initialized(r, c)) {
                        T value = dense(r, c);
                        if (accept_zeros || value != T{}) {
                            triplet.emplace_back(r, c, value);
                        }
                    }
                }
            }

            std::sort(triplet.begin(), triplet.end());
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
            triplet(other.triplet),
            fast_set(other.fast_set) {
            this->rows_ = other.rows_;
            this->cols_ = other.cols_; 
        }
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
              triplet(std::move(other.triplet)),
              fast_set(std::exchange(other.fast_set, true)) {
            this->rows_ = std::exchange(other.rows_, 0);
            this->cols_ = std::exchange(other.cols_, 0);
        }
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
            if (this->rows_ != other.rows_ || this->cols_ != other.cols_)
                return false;

            if (triplet.size() != other.triplet.size())
                return false;

            for (std::size_t i = 0; i < triplet.size(); ++i) {
                if (triplet[i].row != other.triplet[i].row ||
                    triplet[i].col != other.triplet[i].col) {
                    return false;
                }

                if constexpr (std::is_floating_point_v<T>) {
                    if (std::fabs(triplet[i].value - other.triplet[i].value) > 1e-6)
                        return false;
                } else {
                    if (triplet[i].value != other.triplet[i].value)
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
                this->rows_ = other.rows_;
                this->cols_ = other.cols_;
                fast_set = other.fast_set;
                triplet = other.triplet;
            }
            return *this;
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Assignment from DenseMatrix<T> to SparseCOOMatrix<T>.
         *
         * This assignment operator replaces the contents of the SparseCOOMatrix with
         * all initialized and non-zero elements from the given DenseMatrix<T>.
         *
         * Previous sparse data is cleared. Elements in the dense matrix that are either
         * uninitialized or exactly zero are skipped to maintain sparsity.
         *
         * The resulting sparse matrix will match the dimensions of the input dense matrix.
         * The triplets are sorted after assignment and `fast_set` is set to `false`.
         *
         * @param dense The DenseMatrix<T> to assign from.
         * @return Reference to this SparseCOOMatrix<T>.
         *
         * @example
         * @code
         * slt::DenseMatrix<float> dense({
         *     {1.0f, 0.0f},
         *     {0.0f, 3.0f}
         * });
         *
         * slt::SparseCOOMatrix<float> sparse(2, 2);
         * sparse = dense;
         *
         * EXPECT_EQ(sparse.initialized_count(), 2);
         * EXPECT_FLOAT_EQ(sparse.get(0, 0), 1.0f);
         * EXPECT_FLOAT_EQ(sparse.get(1, 1), 3.0f);
         * @endcode
         */
        SparseCOOMatrix<T>& operator=(const DenseMatrix<T>& dense) {
            this->rows_ = dense.rows();
            this->cols_ = dense.cols();
            fast_set = false;

            triplet.clear();
            triplet.reserve(dense.size());  // Conservative

            for (std::size_t r = 0; r < dense.rows(); ++r) {
                for (std::size_t c = 0; c < dense.cols(); ++c) {
                    if (dense.is_initialized(r, c)) {
                        T value = dense(r, c);
                        triplet.emplace_back(r, c, value);
                    }
                }
            }

            std::sort(triplet.begin(), triplet.end());
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
                this->rows_ = std::exchange(other.rows_, 0);
                this->cols_ = std::exchange(other.cols_, 0);
                fast_set = std::exchange(other.fast_set, true);

                triplet = std::move(other.triplet);
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
            if (this->rows_ != other.rows_ || this->cols_ != other.cols_)
                throw std::invalid_argument("Matrix dimensions must match for addition");

            DenseMatrix<T> result(this->rows_, this->cols_);

            // Add all elements from this sparse matrix
            for (const auto& t : triplet)
                result.set(t.row, t.col, t.value);

            // Add all elements from the other sparse matrix
            for (const auto& t : other.triplet) {
                if (result.is_initialized(t.row, t.col))
                    result.update(t.row, t.col, result(t.row, t.col) + t.value);
                else
                    result.set(t.row, t.col, t.value);
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
            for (auto& t : result.triplet) {
                t.value += scalar;
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
            if (this->rows_ != other.rows_ || this->cols_ != other.cols_)
                throw std::invalid_argument("Matrix dimensions must match for subtraction");

            DenseMatrix<T> result(this->rows_, this->cols_);

            // Add all elements from this sparse matrix
            for (const auto& t : triplet) {
                result.set(t.row, t.col, t.value);
            }

            // Subtract all elements from the other sparse matrix
            for (const auto& t : other.triplet) {
                if (result.is_initialized(t.row, t.col))
                    result.update(t.row, t.col, result(t.row, t.col) - t.value);
                else
                    result.set(t.row, t.col, -t.value);  // store the negative value
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
            for (auto& t : result.triplet) {
                t.value -= scalar;
            }
            return result;
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Performs element-wise multiplication of two sparse matrices.
         *
         * Computes the Hadamard (element-wise) product of two sparse matrices in COO format.
         * Only non-zero entries that are present in both matrices at the same (row, col) index
         * will appear in the result. The output is also stored in sparse COO format.
         *
         * Both matrices must have identical dimensions.
         * The operation preserves sparsity — it does not densify the result.
         *
         * @param other The second sparse matrix to multiply with.
         * @return A new SparseCOOMatrix<T> representing the element-wise product.
         * @throws std::invalid_argument if matrix dimensions do not match.
         *
         * @example
         * @code
         * slt::SparseCOOMatrix<float> A(2, 2, {
         *     {0, 0, 1.0f},
         *     {0, 1, 2.0f}
         * });
         *
         * slt::SparseCOOMatrix<float> B(2, 2, {
         *     {0, 0, 3.0f},
         *     {1, 1, 4.0f}
         * });
         *
         * auto result = A * B;
         * // result contains: (0,0) = 1.0 * 3.0 = 3.0
         * // other entries are not in both matrices and are omitted.
         * @endcode
         */
        SparseCOOMatrix operator*(const SparseCOOMatrix& other) const {
            if (this->rows_ != other.rows_ || this->cols_ != other.cols_)
                throw std::invalid_argument("Matrix dimensions must match for element-wise multiplication");

            SparseCOOMatrix<T> result(this->rows_, this->cols_);

            for (const auto& t : triplet) {
                std::size_t r = t.row;
                std::size_t c = t.col;

                if (other.is_initialized(r, c)) {
                    T product = t.value * other.get(r, c);
                    result.set(r, c, product);
                }
            }

            return result;
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Multiplies every non-zero element of the sparse matrix by a scalar.
         *
         * Each stored value in the COO matrix is multiplied by the provided scalar.
         * The sparsity pattern remains unchanged — only the values are scaled.
         *
         * This operation is performed in a new SparseCOOMatrix<T>, leaving the original unchanged.
         *
         * @param scalar The scalar multiplier.
         * @return A new SparseCOOMatrix<T> with scaled values.
         *
         * @example
         * @code
         * slt::SparseCOOMatrix<float> A(2, 2, {
         *     {0, 0, 2.0f},
         *     {1, 1, 4.0f}
         * });
         *
         * auto result = A * 2.0f;
         * // result contains (0,0) = 4.0f, (1,1) = 8.0f
         * @endcode
         */
        SparseCOOMatrix operator*(T scalar) const {
            SparseCOOMatrix<T> result(this->rows_, this->cols_);

            for (const auto& t : triplet) {
                result.set(t.row, t.col, t.value * scalar);
            }

            return result;
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Divides each non-zero element of the sparse matrix by a scalar.
         *
         * Each stored value in the COO matrix is divided by the provided scalar.
         * The sparsity pattern is preserved — zero elements remain unrepresented.
         *
         * Division by zero is not allowed and will throw an exception.
         *
         * @param scalar The divisor.
         * @return A new SparseCOOMatrix<T> with divided values.
         * @throws std::invalid_argument if scalar == 0.
         *
         * @example
         * @code
         * slt::SparseCOOMatrix<float> A(2, 2, {
         *     {0, 0, 6.0f},
         *     {1, 1, 3.0f}
         * });
         *
         * auto result = A / 3.0f;
         * // result contains (0,0) = 2.0f, (1,1) = 1.0f
         * @endcode
         */
        SparseCOOMatrix operator/(T scalar) const {
            if (scalar == T{}) {
                throw std::invalid_argument("Division by zero");
            }

            SparseCOOMatrix<T> result(this->rows_, this->cols_);

            for (const auto& t : triplet) {
                result.set(t.row, t.col, t.value / scalar);
            }

            return result;
        }
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
            if (r >= this->rows_ || c >= this->cols_)
                throw std::out_of_range("Index out of bounds");

            Triplet<T> target(r, c, T{});

            if (fast_set) {
                // Linear search for unsorted triplet vector
                for (const auto& t : triplet) {
                    if (t.row == r && t.col == c)
                        return t.value;
                }
                throw std::runtime_error("Accessing uninitialized matrix element");
            } else {
                // Binary search for sorted triplet vector
                auto it = std::lower_bound(triplet.begin(), triplet.end(), target);

                if (it != triplet.end() && it->row == r && it->col == c) {
                    return it->value;
                } else {
                    throw std::runtime_error("Accessing uninitialized matrix element");
                }
            }
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
         * @brief Sets a value in the sparse matrix at the specified (row, column) position.
         *
         * This function inserts a new non-zero value into the matrix at coordinates (r, c).
         * 
         * - If `fast_set == true`:  
         *   The value is appended to the internal triplet vector in **O(1)** time.  
         *   This mode is intended for bulk construction of the matrix (fast insert mode).  
         *   No duplicate checking is performed, and triplets may be out of order.  
         *   You must call `finalize()` to sort the matrix before using `get()`, `operator()`, or binary search operations.
         * 
         * - If `fast_set == false`:  
         *   The function performs a **binary search** to maintain the sorted invariant (row-major order).  
         *   If a value already exists at the position (r, c), an exception is thrown.  
         *   This mode guarantees correct ordering and safe query operations.
         *
         * @param r Row index of the element (0-based).
         * @param c Column index of the element (0-based).
         * @param value The value to insert at (r, c).
         *
         * @throws std::out_of_range if the (r, c) indices are out of matrix bounds.
         * @throws std::runtime_error if fast_set is false and the position already contains a value.
         *
         * @example
         * @code
         * slt::SparseCOOMatrix<float> mat(3, 3);
         * mat.set(0, 1, 5.0f);
         * mat.set(2, 2, 7.0f);
         * mat.finalize();  // Now ready for queries
         *
         * float val = mat.get(0, 1);  // Returns 5.0f
         * @endcode
         */
        void set(std::size_t r, std::size_t c, T value) {
            if (r >= this->rows_ || c >= this->cols_)
                throw std::out_of_range("Index out of bounds");

            Triplet<T> target(r, c, value);

            if (fast_set) {
                // O(1) append, no duplicate checks
                triplet.push_back(target);
            } else {
                // Find insertion point using binary search
                auto it = std::lower_bound(triplet.begin(), triplet.end(), target);

                // If element already exists, throw
                if (it != triplet.end() && it->row == r && it->col == c) {
                    throw std::runtime_error("Value already set. Use update() instead.");
                }

                // Insert new triplet at correct position
                triplet.insert(it, target);
            }
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Updates an existing value in the matrix at the specified (row, column) position.
         *
         * This function modifies the value of an **already-inserted** element in the sparse matrix.
         *
         * - If `fast_set == true`:  
         *   A **linear search** is performed to locate the element.  
         *   The matrix must have previously stored this value via `set()`.  
         *   If the element is not found, an exception is thrown.
         *
         * - If `fast_set == false`:  
         *   A **binary search** is performed over the sorted triplet vector (requires prior `finalize()`).  
         *   If the element exists, its value is updated in-place.  
         *   If the element does not exist, an exception is thrown — you must call `set()` first.
         *
         * @param r Row index of the element (0-based).
         * @param c Column index of the element (0-based).
         * @param value New value to assign to the existing element.
         *
         * @throws std::out_of_range if the (r, c) indices are out of matrix bounds.
         * @throws std::runtime_error if the element does not exist and was never set.
         *
         * @example
         * @code
         * slt::SparseCOOMatrix<float> mat(4, 4);
         * mat.set(1, 1, 3.0f);
         * mat.finalize();
         * 
         * mat.update(1, 1, 7.5f);  // Successfully updates the existing element
         * 
         * float val = mat.get(1, 1);  // Returns 7.5f
         * 
         * // mat.update(2, 2, 9.0f);  // Would throw runtime_error -- element not set
         * @endcode
         */
        void update(std::size_t r, std::size_t c, T value) {
            if (r >= this->rows_ || c >= this->cols_)
                throw std::out_of_range("Index out of bounds");

            Triplet<T> target(r, c, T{});

            if (fast_set) {
                // Linear search for unsorted triplet vector
                for (auto& t : triplet) {
                    if (t.row == r && t.col == c) {
                        t.value = value;
                        return;
                    }
                }
                throw std::runtime_error("Element not set yet. Use set() first.");
            } else {
                // Binary search for sorted triplet vector
                auto it = std::lower_bound(
                    triplet.begin(), triplet.end(), target,
                    [](const Triplet<T>& a, const Triplet<T>& b) {
                        return std::tie(a.row, a.col) < std::tie(b.row, b.col);
                    });

                if (it != triplet.end() && it->row == r && it->col == c) {
                    it->value = value;
                } else {
                    throw std::runtime_error("Element not set yet. Use set() first.");
                }
            }
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
            if (r >= this->rows_ || c >= this->cols_)
                throw std::out_of_range("Index out of range");

            Triplet<T> target(r, c, T{});

            if (fast_set) {
                // Linear search for unsorted triplet vector
                for (const auto& t : triplet) {
                    if (t.row == r && t.col == c)
                        return true;
                }
                return false;
            } else {
                // Binary search for sorted triplet vector
                auto it = std::lower_bound(
                    triplet.begin(), triplet.end(), target,
                    [](const Triplet<T>& a, const Triplet<T>& b) {
                        return std::tie(a.row, a.col) < std::tie(b.row, b.col);
                    });

                return (it != triplet.end() && it->row == r && it->col == c);
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
            if (!fast_set)
                return;  // Already finalized

            std::sort(triplet.begin(), triplet.end());
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
// -------------------------------------------------------------------------------- 

        /**
         * @brief Returns an iterator to the beginning of the triplet vector.
         *
         * This allows iteration over all non-zero entries of the sparse matrix using range-based for loops or STL algorithms.
         *
         * @return Iterator to the first Triplet in the matrix.
         *
         * Example:
         * @code
         * slt::SparseCOOMatrix<float> mat(3, 3);
         * mat.set(0, 1, 2.5f);
         * mat.set(2, 0, 4.0f);
         *
         * for (const auto& t : mat) {
         *     std::cout << "(" << t.row << ", " << t.col << ") = " << t.value << std::endl;
         * }
         * @endcode
         */
        auto begin() const { return triplet.begin(); }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Returns a mutable iterator to the beginning of the triplet vector.
         *
         * This overload allows modification of the stored Triplet<T> objects — for example,
         * to adjust values in-place, or to apply bulk transformations.
         *
         * Example:
         * @code
         * slt::SparseCOOMatrix<float> mat(5, 5);
         * mat.set(0, 0, 2.0f);
         * mat.finalize();
         *
         * for (auto it = mat.begin(); it != mat.end(); ++it) {
         *     it->value *= 2.0f;  // scale all non-zero values
         * }
         * @endcode
         *
         * @return Iterator to the first Triplet<T> in the matrix.
         */
        auto begin() { return triplet.begin(); }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Returns an iterator to one-past-the-end of the triplet vector.
         *
         * Used for iteration over the matrix using range-based for loops or STL algorithms.
         *
         * @return Iterator to one past the last Triplet.
         *
         * Example:
         * @code
         * slt::SparseCOOMatrix<float> mat(3, 3);
         * mat.set(0, 1, 2.5f);
         * mat.set(2, 0, 4.0f);
         *
         * auto it = mat.begin();
         * while (it != mat.end()) {
         *     std::cout << "(" << it->row << ", " << it->col << ") = " << it->value << std::endl;
         *     ++it;
         * }
         * @endcode
         */
        auto end() const { return triplet.end(); }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Returns a mutable iterator to the end of the triplet vector.
         *
         * This overload allows modification of the stored Triplet<T> objects in algorithms
         * operating on ranges (e.g., std::for_each, std::transform).
         *
         * Example:
         * @code
         * slt::SparseCOOMatrix<float> mat(3, 3);
         * mat.set(0, 0, 1.0f);
         * mat.set(1, 2, 2.0f);
         * mat.finalize();
         *
         * std::for_each(mat.begin(), mat.end(), [](auto& t) {
         *     t.value += 1.0f;  // increment all non-zero values
         * });
         * @endcode
         *
         * @return Iterator one-past-the-last Triplet<T> in the matrix.
         */
        auto end() { return triplet.end(); }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Transposes the sparse matrix in-place.
         *
         * Swaps the row and column indices of each stored triplet.
         * After the transpose, the matrix shape becomes (cols, rows).
         *
         * If the matrix is finalized (`fast_set == false`), the triplet vector
         * is re-sorted to maintain row-major order for efficient lookup.
         *
         * @example
         * @code
         * slt::SparseCOOMatrix<float> mat(2, 3, {
         *     {0, 1, 1.0f},
         *     {1, 2, 2.0f}
         * });
         * mat.transpose();
         * // Now mat.rows() == 3, mat.cols() == 2
         * @endcode
         */
        void transpose() {
            std::swap(this->rows_, this->cols_);

            for (auto& t : triplet) {
                std::swap(t.row, t.col);
            }

            if (!fast_set) {
                std::sort(triplet.begin(), triplet.end());
            }
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Computes the matrix inverse of this SparseCOOMatrix as a dense matrix.
         *
         * This method returns a `DenseMatrix<T>` containing the inverse of the sparse matrix.
         * Internally, the sparse matrix is first converted to a dense format, and then
         * a standard dense matrix inversion algorithm (such as Gauss-Jordan elimination or LU decomposition)
         * is applied. The result is always a full dense matrix because in general the inverse
         * of a sparse matrix is not sparse.
         *
         * The current SparseCOOMatrix must represent a square (N x N) matrix and must be invertible
         * (i.e., full rank, non-singular). If the matrix is not square or is singular, this method
         * will throw an exception.
         *
         * SIMD acceleration (where available) is performed during the dense inversion stage
         * by the DenseMatrix<T>::inverse() method. No SIMD is used inside SparseCOOMatrix itself.
         *
         * @return A `DenseMatrix<T>` representing the inverse of this matrix.
         * @throws std::invalid_argument if the matrix is not square.
         * @throws std::runtime_error if the matrix is singular (non-invertible).
         *
         * @note The inverse of a sparse matrix is generally dense — expect memory usage to increase.
         *       If you want to preserve sparsity, use a dedicated sparse solver instead.
         *
         * @example
         * @code
         * slt::SparseCOOMatrix<float> A(2, 2, {
         *     {0, 0, 4.0f},
         *     {0, 1, 7.0f},
         *     {1, 0, 2.0f},
         *     {1, 1, 6.0f}
         * });
         *
         * slt::DenseMatrix<float> A_inv = A.inverse();
         * // A_inv now contains the full dense inverse of A
         * @endcode
         */ 
        DenseMatrix<T> inverse() const {
            if (this->rows_ != this->cols_)
                throw std::invalid_argument("Inverse is only defined for square matrices");

            // Step 1: Convert to dense
            DenseMatrix<T> dense(this->rows_, this->cols_, 0);
            for (const auto& t : triplet) {
                dense.update(t.row, t.col, t.value);
            }

            // Step 2: Invert
            DenseMatrix<T> inv = dense.inverse();  // You already have a DenseMatrix::inverse()

            return inv;
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Removes an element at the specified (row, column) position from the sparse matrix.
         *
         * If an entry with matching (row, col) exists, it is erased from the internal triplet vector.
         * If no such entry exists, the method does nothing — it is safe to call even if the entry is missing.
         *
         * In `fast_set` mode (unsorted triplet vector), this performs a linear search (O(n)).
         * In finalized mode (`fast_set == false`), this performs a binary search (O(log n)) 
         * because the triplets are sorted in row-major order.
         *
         * This operation preserves the matrix dimensions. It only affects the stored non-zero elements.
         *
         * @param r Row index of the element to remove.
         * @param c Column index of the element to remove.
         *
         * @throws std::out_of_range if the provided row or column index is invalid (out of matrix bounds).
         *
         * @example
         * @code
         * slt::SparseCOOMatrix<float> mat(3, 3);
         * mat.set(1, 2, 5.0f);
         * mat.finalize();
         * 
         * mat.remove(1, 2);  // Now (1,2) no longer exists
         * 
         * EXPECT_THROW(mat.get(1, 2), std::runtime_error);  // Confirm removal
         * @endcode
         */ 
        void remove(std::size_t r, std::size_t c) {
            if (r >= this->rows_ || c >= this->cols_)
                throw std::out_of_range("Index out of bounds");

            Triplet<T> target(r, c, T{});

            if (fast_set) {
                // Linear search and erase
                auto it = std::find_if(triplet.begin(), triplet.end(),
                    [=](const Triplet<T>& t) {
                        return t.row == r && t.col == c;
                    });

                if (it != triplet.end()) {
                    triplet.erase(it);
                }
            } else {
                // Binary search
                auto it = std::lower_bound(triplet.begin(), triplet.end(), target);

                if (it != triplet.end() && it->row == r && it->col == c) {
                    triplet.erase(it);
                }
            }
        }
    };
// // ================================================================================ 
// // ================================================================================ 
// SparseCOOMatrix friend functions 

    /**
     * @brief Output stream operator for printing a SparseCOOMatrix.
     *
     * Prints the matrix in triplet form:
     * (row, col) = value
     *
     * Example output:
     * @code
     * SparseCOOMatrix<float> (3 x 3), nonzeros = 2
     * (0, 0) = 1.0
     * (2, 1) = 5.0
     * @endcode
     *
     * @tparam T The matrix element type (float or double)
     * @param os The output stream (std::ostream)
     * @param mat The SparseCOOMatrix to print
     * @return std::ostream& for chaining
     */
    template<typename T>
    std::ostream& operator<<(std::ostream& os, const SparseCOOMatrix<T>& mat) {
        os << "SparseCOOMatrix<" << (std::is_same_v<T, float> ? "float" : "double")
           << "> (" << mat.rows() << " x " << mat.cols() << "), nonzeros = "
           << mat.initialized_count() << "\n";

        for (const auto& t : mat) {
            os << "(" << t.row << ", " << t.col << ") = " << t.value << "\n";
        }

        return os;
    }
// -------------------------------------------------------------------------------- 

    /**
     * @brief Adds a scalar to each non-zero element of a SparseCOOMatrix.
     *
     * This free function allows symmetric scalar addition: `scalar + matrix`.
     * 
     * The operation is equivalent to `matrix + scalar` (implemented via member function),
     * and preserves the sparsity pattern: only stored elements are modified.
     * 
     * Unstored zero elements remain unaffected.
     *
     * @param scalar Scalar value to add.
     * @param matrix Sparse matrix to operate on.
     * @return A new `SparseCOOMatrix<T>` with updated values.
     *
     * @example
     * @code
     * slt::SparseCOOMatrix<float> A(2, 2, {
     *     {0, 0, 2.0f},
     *     {1, 1, 5.0f}
     * });
     * 
     * auto result = 3.0f + A;
     * 
     * // result.get(0, 0) == 5.0f
     * // result.get(1, 1) == 8.0f
     * @endcode
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
        if (dense.rows() != sparse.rows() || dense.cols() != sparse.cols()) {
            throw std::invalid_argument("Matrix dimensions must match for addition");
        }

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
        for (const auto& t : sparse) {
            std::size_t r = t.row;
            std::size_t c = t.col;
            result.update(r, c, result(r, c) + t.value);
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
        // Simply call the existing operator+ where DenseMatrix is the first argument
        return dense + sparse;
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
        SparseCOOMatrix<T> result(matrix.rows(), matrix.cols(), matrix.initialized_count());

        for (const auto& t : matrix) {
            T val = scalar - t.value;
            result.set(t.row, t.col, val);
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
        for (const auto& t : sparse) {
            result.update(t.row, t.col, result(t.row, t.col) + t.value);
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

        // Copy dense values to result
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
        for (const auto& t : sparse) {
            result.update(t.row, t.col, result(t.row, t.col) - t.value);
        }

        return result;
    }
// -------------------------------------------------------------------------------- 

    /**
     * @brief Multiplies every non-zero element of a SparseCOOMatrix by a scalar value.
     *
     * Creates a new SparseCOOMatrix where each stored value is multiplied by the given scalar.
     * This operation preserves the sparsity pattern of the input matrix — zero elements remain zero
     * and are not explicitly added to the result.
     *
     * Internally, this operator simply delegates to the member `SparseCOOMatrix::operator*(T scalar)` function.
     *
     * @param scalar The scalar value to multiply each matrix element by.
     * @param matrix The input SparseCOOMatrix.
     * @return A new SparseCOOMatrix with updated values.
     *
     * @example
     * @code
     * slt::SparseCOOMatrix<float> A(2, 2);
     * A.set(0, 0, 2.0f);
     * A.set(1, 1, 4.0f);
     *
     * auto B = 3.0f * A;
     * // B.get(0, 0) == 6.0f
     * // B.get(1, 1) == 12.0f
     * @endcode
     */
    template<typename T>
    SparseCOOMatrix<T> operator*(T scalar, const SparseCOOMatrix<T>& matrix) {
        return matrix * scalar;  // Leverage existing member function
    }
// -------------------------------------------------------------------------------- 

    /**
     * @brief Performs element-wise multiplication of a DenseMatrix and a SparseCOOMatrix.
     *
     * Returns a new DenseMatrix<T> where each element is the product of corresponding entries
     * in the dense and sparse matrices. Multiplication is performed only at non-zero positions
     * of the sparse matrix — implicit zero entries are skipped.
     *
     * The result is fully initialized as a DenseMatrix. Positions in the dense matrix that
     * do not correspond to a non-zero entry in the sparse matrix are left as zero.
     *
     * This operation does not affect the sparsity of the sparse operand. It is intended for
     * element-wise scaling of selected elements in a dense matrix.
     *
     * @tparam T Floating-point type (float or double)
     * @param dense The dense matrix operand.
     * @param sparse The sparse COO matrix operand.
     * @return A new DenseMatrix<T> containing the element-wise product.
     * @throws std::invalid_argument if matrix dimensions do not match.
     *
     * @example
     * @code
     * slt::DenseMatrix<float> A(2, 2);
     * A.set(0, 0, 1.0f);
     * A.set(0, 1, 2.0f);
     * A.set(1, 0, 3.0f);
     * A.set(1, 1, 4.0f);
     *
     * slt::SparseCOOMatrix<float> B(2, 2);
     * B.set(0, 1, 5.0f);
     * B.set(1, 0, 6.0f);
     *
     * slt::DenseMatrix<float> C = A * B;
     *
     * // C(0,0) == 0.0f
     * // C(0,1) == 10.0f  (2.0 * 5.0)
     * // C(1,0) == 18.0f  (3.0 * 6.0)
     * // C(1,1) == 0.0f
     * @endcode
     */
    template<typename T>
    DenseMatrix<T> operator*(const DenseMatrix<T>& dense, const SparseCOOMatrix<T>& sparse) {
        if (dense.rows() != sparse.rows() || dense.cols() != sparse.cols())
            throw std::invalid_argument("Matrix dimensions must match for element-wise multiplication");

        DenseMatrix<T> result(dense.rows(), dense.cols());

        // Initialize result with zeros
        std::fill(result.begin(), result.begin() + result.size(), T{});
        std::fill(result.begin(), result.begin() + result.size(), 0);

        // Only multiply at sparse matrix locations
        for (std::size_t i = 0; i < sparse.initialized_count(); ++i) {
            std::size_t r = sparse.row_index(i);
            std::size_t c = sparse.col_index(i);
            T value = dense(r, c) * sparse.value(i);
            result.set(r, c, value);
        }

        return result;
    }
// -------------------------------------------------------------------------------- 

    /**
     * @brief Performs element-wise multiplication of a SparseCOOMatrix and a DenseMatrix.
     *
     * Returns a new DenseMatrix<T> containing the element-wise product of the operands.
     * 
     * This function is equivalent to: `dense * sparse`, and simply reuses that implementation.
     * The multiplication is commutative in this case — only positions with non-zero entries
     * in the sparse matrix are affected.
     *
     * @tparam T Floating-point type (float or double)
     * @param sparse The sparse COO matrix operand.
     * @param dense The dense matrix operand.
     * @return A new DenseMatrix<T> with the element-wise product.
     * @throws std::invalid_argument if matrix dimensions do not match.
     *
     * @example
     * @code
     * slt::SparseCOOMatrix<float> A(2, 2);
     * A.set(0, 1, 2.0f);
     * A.set(1, 0, 4.0f);
     *
     * slt::DenseMatrix<float> B(2, 2);
     * B.set(0, 0, 10.0f);
     * B.set(0, 1, 20.0f);
     * B.set(1, 0, 30.0f);
     * B.set(1, 1, 40.0f);
     *
     * slt::DenseMatrix<float> C = A * B;
     *
     * // C(0,1) == 40.0f (2.0 * 20.0)
     * // C(1,0) == 120.0f (4.0 * 30.0)
     * // Other entries == 0.0f
     * @endcode
     */
    template<typename T>
    DenseMatrix<T> operator*(const SparseCOOMatrix<T>& sparse, const DenseMatrix<T>& dense) {
        // Reuse the function above — multiplication is commutative
        return dense * sparse;
    }
// -------------------------------------------------------------------------------- 

    /**
     * @brief Performs sparse matrix multiplication: result = A * B.
     *
     * Multiplies two sparse COO matrices `A` and `B`, returning the result
     * as a dense matrix. Internally, this uses a hash-based lookup to avoid
     * unnecessary dense conversion, while preserving the sparsity advantage
     * during computation.
     *
     * The algorithm performs:
     * - For each row `i` in A:
     *    - For each column `j` in B:
     *       - Computes dot product of row `i` of A with column `j` of B
     *       - Stores result in position (i, j)
     *
     * Sparse to dense multiplication is chosen for simplicity — the result
     * is always returned as a full DenseMatrix.
     *
     * Memory-friendly: does not allocate temporary dense buffers for A or B.
     *
     * @note This implementation is not SIMD accelerated. For future SIMD,
     * conversion to CSR/CSC would be needed.
     *
     * @tparam T Element type (float or double).
     * @param A Left-hand operand (SparseCOOMatrix).
     * @param B Right-hand operand (SparseCOOMatrix).
     * @return DenseMatrix<T> result of A * B.
     * @throws std::invalid_argument if dimensions are incompatible for multiplication.
     *
     * @example
     * @code
     * slt::SparseCOOMatrix<float> A(2, 3);
     * slt::SparseCOOMatrix<float> B(3, 2);
     * 
     * A.set(0, 1, 4.0f);
     * A.set(1, 2, 5.0f);
     * 
     * B.set(1, 0, 2.0f);
     * B.set(2, 1, 3.0f);
     * 
     * auto C = mat_mul(A, B);
     * 
     * EXPECT_FLOAT_EQ(C(0, 0), 8.0f);  // 4 * 2
     * EXPECT_FLOAT_EQ(C(1, 1), 15.0f); // 5 * 3
     * @endcode
     */
    template<typename T>
    DenseMatrix<T> mat_mul(const SparseCOOMatrix<T>& A, const SparseCOOMatrix<T>& B) {
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

        // Build temporary map: for each (k,j) in B, map from k → (j,value)
        std::unordered_map<std::size_t, std::vector<std::pair<std::size_t, T>>> B_map;
        for (const auto& tB : B) {
            B_map[tB.row].emplace_back(tB.col, tB.value);
        }

        // For each (i,k) in A, multiply against corresponding B[k,:]
        for (const auto& tA : A) {
            const std::size_t i = tA.row;
            const std::size_t k = tA.col;
            const T value_A = tA.value;

            if (B_map.count(k)) {
                for (const auto& [j, value_B] : B_map[k]) {
                    T old_val = result.is_initialized(i, j) ? result(i, j) : T{};
                    result.set(i, j, old_val + value_A * value_B);
                }
            }
        }

        return result;
    }
// -------------------------------------------------------------------------------- 

    /**
     * @brief Multiplies a SparseCOOMatrix with a DenseMatrix (A × B).
     *
     * This function performs matrix multiplication between a sparse COO matrix (A)
     * and a dense matrix (B), producing a fully dense result matrix.
     *
     * The result is computed as:
     *    result(i,j) = sum_k A(i,k) * B(k,j)
     *
     * The result is a DenseMatrix<T> of dimensions (A.rows() × B.cols()).
     * Zero entries in A do not contribute to the result.
     *
     * - This implementation is sparse-friendly: avoids expanding A to a dense matrix.
     * - The result is always fully initialized.
     * - The multiplication is not SIMD accelerated.
     *
     * @tparam T The element type (float or double).
     * @param A The sparse matrix operand (SparseCOOMatrix).
     * @param B The dense matrix operand (DenseMatrix).
     * @return A DenseMatrix<T> representing A × B.
     *
     * @throws std::invalid_argument if A.cols() != B.rows()
     *
     * @example
     * @code
     * slt::SparseCOOMatrix<float> A(2, 3);
     * A.set(0, 1, 4.0f);
     * A.set(1, 2, 5.0f);
     *
     * slt::DenseMatrix<float> B({
     *     {1.0f, 2.0f},
     *     {3.0f, 4.0f},
     *     {5.0f, 6.0f}
     * });
     *
     * auto C = mat_mul(A, B);
     * // C(0,0) = 4.0f * 3.0f = 12.0f
     * // C(1,1) = 5.0f * 6.0f = 30.0f
     * @endcode
     */
    template<typename T>
    DenseMatrix<T> mat_mul(const SparseCOOMatrix<T>& A, const DenseMatrix<T>& B) {
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

        // Initialize result with zeros
        std::fill(result.begin(), result.end(), T{});

        for (const auto& tA : A) {
            std::size_t i = tA.row;
            std::size_t k = tA.col;
            T value_A = tA.value;

            for (std::size_t j = 0; j < B_cols; ++j) {
                T value_B = B(k, j);
                T old_val = result.is_initialized(i, j) ? result(i, j) : T{};
                result.set(i, j, old_val + value_A * value_B);
            }
        }

        return result;
    }
// -------------------------------------------------------------------------------- 

    /**
     * @brief Multiplies a DenseMatrix with a SparseCOOMatrix (A × B).
     *
     * Performs matrix multiplication between a dense matrix (A) and a sparse COO matrix (B),
     * producing a dense matrix result.
     *
     * The result is computed as:
     *    result(i,j) = sum_k A(i,k) * B(k,j)
     *
     * The result is a DenseMatrix<T> of dimensions (A.rows() × B.cols()).
     * Zero entries in B do not contribute to the result.
     *
     * - This implementation avoids dense expansion of B.
     * - The result is fully initialized.
     * - No SIMD acceleration is used.
     *
     * @tparam T The element type (float or double).
     * @param A The dense matrix operand (DenseMatrix).
     * @param B The sparse matrix operand (SparseCOOMatrix).
     * @return A DenseMatrix<T> representing A × B.
     *
     * @throws std::invalid_argument if A.cols() != B.rows()
     *
     * @example
     * @code
     * slt::DenseMatrix<float> A({
     *     {1.0f, 2.0f},
     *     {3.0f, 4.0f}
     * });
     *
     * slt::SparseCOOMatrix<float> B(2, 3);
     * B.set(0, 0, 5.0f);
     * B.set(1, 2, 6.0f);
     *
     * auto C = mat_mul(A, B);
     *
     * // C(0, 0) = 1.0 * 5.0 = 5.0
     * // C(0, 2) = 2.0 * 6.0 = 12.0
     * @endcode
     */
    template<typename T>
    DenseMatrix<T> mat_mul(const DenseMatrix<T>& A, const SparseCOOMatrix<T>& B) {
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

        // Initialize result with zeros
        std::fill(result.begin(), result.end(), T{});

        for (const auto& tB : B) {
            std::size_t k = tB.row;
            std::size_t j = tB.col;
            T value_B = tB.value;

            for (std::size_t i = 0; i < A_rows; ++i) {
                T value_A = A(i, k);
                T old_val = result.is_initialized(i, j) ? result(i, j) : T{};
                result.set(i, j, old_val + value_A * value_B);
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
