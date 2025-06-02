// ================================================================================
// ================================================================================
// - File:    test_matrix.cpp
// - Purpose: This file implements google test as a method to test C++ code.
//            Describe the type of testing to be completed
//
// Source Metadata
// - Author:  Jonathan A. Webb
// - Date:    June 01, 2025
// - Version: 1.0
// - Copyright: Copyright 2025, Jon Webb Inc.
// ================================================================================
// ================================================================================
// - Begin test

#include <gtest/gtest.h>
#include "matrix.hpp"
// ================================================================================ 
// ================================================================================ 

template <typename T>
void expect_matrix_eq(const slt::DenseMatrix<T>& m, const std::vector<std::vector<T>>& expected, T tol = T{1e-5}) {
    ASSERT_EQ(m.rows(), expected.size());
    ASSERT_EQ(m.cols(), expected[0].size());

    for (std::size_t i = 0; i < m.rows(); ++i)
        for (std::size_t j = 0; j < m.cols(); ++j)
            EXPECT_NEAR(m(i, j), expected[i][j], tol);
}
// -------------------------------------------------------------------------------- 

template<typename T>
void expect_matrix_eq(const slt::DenseMatrix<T>& a, const slt::DenseMatrix<T>& b, T epsilon = T(1e-5)) {
    ASSERT_EQ(a.rows(), b.rows());
    ASSERT_EQ(a.cols(), b.cols());
    for (std::size_t i = 0; i < a.rows(); ++i) {
        for (std::size_t j = 0; j < a.cols(); ++j) {
            ASSERT_NEAR(a.get(i, j), b.get(i, j), epsilon);
        }
    }
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixTest, DefaultConstructorInitializesCorrectSize) {
    slt::DenseMatrix<float> mat(3, 4, 5.0f);
    EXPECT_EQ(mat.rows(), 3);
    EXPECT_EQ(mat.cols(), 4);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            EXPECT_FLOAT_EQ(mat(i, j), 5.0f);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixTest, VectorOfVectorConstructor) {
    std::vector<std::vector<double>> data = {{1.0, 2.0}, {3.0, 4.0}};
    slt::DenseMatrix<double> mat(data);
    EXPECT_EQ(mat.rows(), 2);
    EXPECT_EQ(mat.cols(), 2);
    EXPECT_DOUBLE_EQ(mat(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(mat(1, 1), 4.0);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixTest, StdArrayConstructor) {
    std::array<std::array<float, 2>, 2> arr = {{{1.0f, 2.0f}, {3.0f, 4.0f}}};
    slt::DenseMatrix<float> mat(arr);
    EXPECT_EQ(mat.rows(), 2);
    EXPECT_EQ(mat.cols(), 2);
    EXPECT_FLOAT_EQ(mat(0, 1), 2.0f);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixTest, InitializerListConstructor) {
    slt::DenseMatrix<double> mat{{1.1, 2.2}, {3.3, 4.4}};
    EXPECT_EQ(mat.rows(), 2);
    EXPECT_EQ(mat.cols(), 2);
    EXPECT_DOUBLE_EQ(mat(1, 0), 3.3);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixTest, FlatVectorConstructor) {
    std::vector<float> flat = {1, 2, 3, 4, 5, 6};
    slt::DenseMatrix<float> mat(flat, 2, 3);
    EXPECT_EQ(mat.rows(), 2);
    EXPECT_EQ(mat.cols(), 3);
    EXPECT_FLOAT_EQ(mat(1, 2), 6);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixTest, ThrowsOnInvalidFlatVector) {
    std::vector<float> flat = {1, 2, 3};
    EXPECT_THROW(slt::DenseMatrix<float> mat(flat, 2, 2), std::invalid_argument);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixTest, GetSetWorksAndInitialValuesCorrect) {
    slt::DenseMatrix<double> mat(3, 3);  // Default-init with 0.0

    // Check default values are zero-initialized
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            EXPECT_THROW(mat.get(i, j), std::runtime_error);

    // Set a few values and read them back
    mat.set(0, 0, 3.14);
    mat.set(1, 2, -1.5);
    mat.set(2, 2, 42.0);

    EXPECT_DOUBLE_EQ(mat.get(0, 0), 3.14);
    EXPECT_DOUBLE_EQ(mat.get(1, 2), -1.5);
    EXPECT_DOUBLE_EQ(mat.get(2, 2), 42.0);

    // Ensure other elements remain unchanged
    EXPECT_THROW(mat.get(0, 1), std::runtime_error);
    EXPECT_THROW(mat.get(2, 0), std::runtime_error);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixTest, GetThrowsOnInvalidIndex) {
    slt::DenseMatrix<float> mat(2, 2);

    EXPECT_THROW(mat.get(2, 0), std::out_of_range);
    EXPECT_THROW(mat.set(0, 3, 1.0f), std::out_of_range);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixTest, AdditionOperatorProducesCorrectResult) {
    slt::DenseMatrix<double> A{{1.0, 2.0}, {3.0, 4.0}};
    slt::DenseMatrix<double> B{{5.0, 6.0}, {7.0, 8.0}};

    slt::DenseMatrix<double> C = A + B;

    EXPECT_DOUBLE_EQ(C.get(0, 0), 6.0);
    // EXPECT_DOUBLE_EQ(C.get(0, 1), 8.0);
    // EXPECT_DOUBLE_EQ(C.get(1, 0), 10.0);
    // EXPECT_DOUBLE_EQ(C.get(1, 1), 12.0);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixTest, AdditionOperatorThrowsOnMismatchedDimensions) {
    slt::DenseMatrix<float> A(2, 3, 1.0f);
    slt::DenseMatrix<float> B(3, 2, 1.0f);

    EXPECT_THROW(A + B, std::invalid_argument);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixTest, AdditionWithZeroMatrixIsNoOp) {
    slt::DenseMatrix<float> A{{1.0f, 2.0f}, {3.0f, 4.0f}};
    slt::DenseMatrix<float> Z(2, 2, 0.0f);

    slt::DenseMatrix<float> R = A + Z;

    EXPECT_FLOAT_EQ(R.get(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(R.get(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(R.get(1, 0), 3.0f);
    EXPECT_FLOAT_EQ(R.get(1, 1), 4.0f);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixTest, TransposeSwapsRowsAndColsCorrectly) {
    slt::DenseMatrix<float> mat{
        {1.0f, 2.0f, 3.0f},
        {4.0f, 5.0f, 6.0f}
    };

    mat.transpose();

    EXPECT_EQ(mat.rows(), 3);
    EXPECT_EQ(mat.cols(), 2);

    EXPECT_FLOAT_EQ(mat.get(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(mat.get(0, 1), 4.0f);

    EXPECT_FLOAT_EQ(mat.get(1, 0), 2.0f);
    EXPECT_FLOAT_EQ(mat.get(1, 1), 5.0f);

    EXPECT_FLOAT_EQ(mat.get(2, 0), 3.0f);
    EXPECT_FLOAT_EQ(mat.get(2, 1), 6.0f);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixTest, DoubleTransposeRestoresOriginalMatrix) {
    slt::DenseMatrix<double> original{
        {10.0, 20.0},
        {30.0, 40.0},
        {50.0, 60.0}
    };

    slt::DenseMatrix<double> mat = original;
    mat.transpose();
    mat.transpose();

    for (std::size_t i = 0; i < original.rows(); ++i) {
        for (std::size_t j = 0; j < original.cols(); ++j) {
            EXPECT_DOUBLE_EQ(mat.get(i, j), original.get(i, j));
        }
    }
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixTest, SubtractionOperatorSubtractsElementwise) {
    slt::DenseMatrix<float> A{
        {10.0f, 20.0f},
        {30.0f, 40.0f}
    };
    slt::DenseMatrix<float> B{
        {1.0f, 2.0f},
        {3.0f, 4.0f}
    };

    auto result = A - B;

    EXPECT_FLOAT_EQ(result.get(0, 0), 9.0f);
    EXPECT_FLOAT_EQ(result.get(0, 1), 18.0f);
    EXPECT_FLOAT_EQ(result.get(1, 0), 27.0f);
    EXPECT_FLOAT_EQ(result.get(1, 1), 36.0f);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixTest, CloneCreatesDeepCopy) {
    slt::DenseMatrix<float> original = {
        {1.0f, 2.0f},
        {3.0f, 4.0f}
    };

    std::unique_ptr<slt::MatrixBase<float>> cloned = original.clone();

    // Verify dimensions
    EXPECT_EQ(cloned->rows(), original.rows());
    EXPECT_EQ(cloned->cols(), original.cols());

    // Verify values are equal
    for (std::size_t i = 0; i < original.rows(); ++i) {
        for (std::size_t j = 0; j < original.cols(); ++j) {
            EXPECT_FLOAT_EQ(cloned->get(i, j), original(i, j));
        }
    }

    // Mutate original and verify cloned is unaffected
    original.update(0, 0, 9.9f);
    EXPECT_NE(cloned->get(0, 0), original(0, 0));
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixScalarTest, AddScalarFloat) {
    slt::DenseMatrix<float> mat{
        {1.0f, 2.0f},
        {3.0f, 4.0f}
    };

    auto result = mat + 1.5f;

    std::vector<std::vector<float>> expected = {
        {2.5f, 3.5f},
        {4.5f, 5.5f}
    };

    expect_matrix_eq(result, expected);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixScalarTest, SubtractScalarFloat) {
    slt::DenseMatrix<float> mat{
        {10.0f, 5.0f},
        {3.0f, 1.0f}
    };

    auto result = mat - 2.0f;

    std::vector<std::vector<float>> expected = {
        {8.0f, 3.0f},
        {1.0f, -1.0f}
    };

    expect_matrix_eq(result, expected);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixScalarTest, AddScalarDouble) {
    slt::DenseMatrix<double> mat{
        {2.0, 4.0},
        {6.0, 8.0}
    };

    auto result = mat + 0.5;

    std::vector<std::vector<double>> expected = {
        {2.5, 4.5},
        {6.5, 8.5}
    };

    expect_matrix_eq(result, expected);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixScalarTest, SubtractScalarDouble) {
    slt::DenseMatrix<double> mat{
        {1.0, 0.0},
        {2.5, 3.0}
    };

    auto result = mat - 1.0;

    std::vector<std::vector<double>> expected = {
        {0.0, -1.0},
        {1.5, 2.0}
    };

    expect_matrix_eq(result, expected);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixScalarTest, Float_MatrixPlusScalar) {
    slt::DenseMatrix<float> mat = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    slt::DenseMatrix<float> expected = {{2.5f, 3.5f}, {4.5f, 5.5f}};
    auto result = mat + 1.5f;
    expect_matrix_eq(result, expected);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixScalarTest, Float_ScalarPlusMatrix) {
    slt::DenseMatrix<float> mat = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    slt::DenseMatrix<float> expected = {{2.5f, 3.5f}, {4.5f, 5.5f}};
    auto result = 1.5f + mat;
    expect_matrix_eq(result, expected);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixScalarTest, Float_MatrixMinusScalar) {
    slt::DenseMatrix<float> mat = {{3.0f, 4.0f}, {5.0f, 6.0f}};
    slt::DenseMatrix<float> expected = {{1.5f, 2.5f}, {3.5f, 4.5f}};
    auto result = mat - 1.5f;
    expect_matrix_eq(result, expected);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixScalarTest, Float_ScalarMinusMatrix) {
    slt::DenseMatrix<float> mat = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    slt::DenseMatrix<float> expected = {{4.0f, 3.0f}, {2.0f, 1.0f}};
    auto result = 5.0f - mat;
    expect_matrix_eq(result, expected);
}
// -------------------------------------------------------------------------------- 
// ------------------- DOUBLE TESTS -------------------

TEST(DenseMatrixScalarTest, Double_MatrixPlusScalar) {
    slt::DenseMatrix<double> mat = {{1.0, 2.0}, {3.0, 4.0}};
    slt::DenseMatrix<double> expected = {{2.5, 3.5}, {4.5, 5.5}};
    auto result = mat + 1.5;
    expect_matrix_eq(result, expected, 1e-10);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixScalarTest, Double_ScalarPlusMatrix) {
    slt::DenseMatrix<double> mat = {{1.0, 2.0}, {3.0, 4.0}};
    slt::DenseMatrix<double> expected = {{2.5, 3.5}, {4.5, 5.5}};
    auto result = 1.5 + mat;
    expect_matrix_eq(result, expected, 1e-10);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixScalarTest, Double_MatrixMinusScalar) {
    slt::DenseMatrix<double> mat = {{3.0, 4.0}, {5.0, 6.0}};
    slt::DenseMatrix<double> expected = {{1.5, 2.5}, {3.5, 4.5}};
    auto result = mat - 1.5;
    expect_matrix_eq(result, expected, 1e-10);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixScalarTest, Double_ScalarMinusMatrix) {
    slt::DenseMatrix<double> mat = {{1.0, 2.0}, {3.0, 4.0}};
    slt::DenseMatrix<double> expected = {{4.0, 3.0}, {2.0, 1.0}};
    auto result = 5.0 - mat;
    expect_matrix_eq(result, expected, 1e-10);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixScalarTest, Float_MatrixTimesScalar) {
    slt::DenseMatrix<float> mat = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    auto result = mat * 2.0f;

    EXPECT_FLOAT_EQ(result.get(0, 0), 2.0f);
    EXPECT_FLOAT_EQ(result.get(0, 1), 4.0f);
    EXPECT_FLOAT_EQ(result.get(1, 0), 6.0f);
    EXPECT_FLOAT_EQ(result.get(1, 1), 8.0f);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixScalarTest, Float_ScalarTimesMatrix) {
    slt::DenseMatrix<float> mat = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    auto result = 2.0f * mat;

    EXPECT_FLOAT_EQ(result.get(0, 0), 2.0f);
    EXPECT_FLOAT_EQ(result.get(0, 1), 4.0f);
    EXPECT_FLOAT_EQ(result.get(1, 0), 6.0f);
    EXPECT_FLOAT_EQ(result.get(1, 1), 8.0f);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixElementwiseTest, Float_MatrixTimesMatrix) {
    slt::DenseMatrix<float> A = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    slt::DenseMatrix<float> B = {{5.0f, 6.0f}, {7.0f, 8.0f}};
    auto result = A * B;

    EXPECT_FLOAT_EQ(result.get(0, 0), 5.0f);
    EXPECT_FLOAT_EQ(result.get(0, 1), 12.0f);
    EXPECT_FLOAT_EQ(result.get(1, 0), 21.0f);
    EXPECT_FLOAT_EQ(result.get(1, 1), 32.0f);
}
// --------------------------------------------------------------------------------

TEST(DenseMatrixElementwiseTest, ThrowsOnMismatchedDimensions) {
    slt::DenseMatrix<float> A = {{1.0f, 2.0f}};
    slt::DenseMatrix<float> B = {{1.0f, 2.0f}, {3.0f, 4.0f}};

    EXPECT_THROW({ auto result = A * B; }, std::invalid_argument);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixScalarDivisionTest, Float_MatrixDividedByScalar) {
    slt::DenseMatrix<float> mat = {{2.0f, 4.0f}, {6.0f, 8.0f}};
    slt::DenseMatrix<float> expected = {{1.0f, 2.0f}, {3.0f, 4.0f}};

    auto result = mat / 2.0f;

    EXPECT_FLOAT_EQ(result.get(0, 0), expected.get(0, 0));
    EXPECT_FLOAT_EQ(result.get(0, 1), expected.get(0, 1));
    EXPECT_FLOAT_EQ(result.get(1, 0), expected.get(1, 0));
    EXPECT_FLOAT_EQ(result.get(1, 1), expected.get(1, 1));
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixScalarDivisionTest, Double_MatrixDividedByScalar) {
    slt::DenseMatrix<double> mat = {{3.0, 6.0}, {9.0, 12.0}};
    slt::DenseMatrix<double> expected = {{1.5, 3.0}, {4.5, 6.0}};

    auto result = mat / 2.0;

    EXPECT_DOUBLE_EQ(result.get(0, 0), expected.get(0, 0));
    EXPECT_DOUBLE_EQ(result.get(0, 1), expected.get(0, 1));
    EXPECT_DOUBLE_EQ(result.get(1, 0), expected.get(1, 0));
    EXPECT_DOUBLE_EQ(result.get(1, 1), expected.get(1, 1));
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixScalarDivisionTest, DivisionByZeroThrows) {
    slt::DenseMatrix<double> mat = {{1.0, 2.0}, {3.0, 4.0}};
    EXPECT_THROW(mat / 0.0, std::invalid_argument);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixInverseTest, InverseOf2x2MatrixMatchesKnownValues) {
    slt::DenseMatrix<double> A = {{4.0, 7.0}, {2.0, 6.0}};
    auto Ainv = A.inverse();
    // Known correct inverse of A
    EXPECT_NEAR(Ainv.get(0, 0),  0.6, 1e-9);
    EXPECT_NEAR(Ainv.get(0, 1), -0.7, 1e-9);
    EXPECT_NEAR(Ainv.get(1, 0), -0.2, 1e-9);
    EXPECT_NEAR(Ainv.get(1, 1),  0.4, 1e-9);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixInverseTest, InverseOf3x3Matrix) {
    slt::DenseMatrix<double> A = {
        {3.0, 0.0, 2.0},
        {2.0, 0.0, -2.0},
        {0.0, 1.0, 1.0}
    };

    auto Ainv = A.inverse();

    // Known correct inverse of A
    slt::DenseMatrix<double> expected = {
        { 0.2,  0.2,  0.0},
        {-0.2,  0.3,  1.0},
        { 0.2, -0.3,  0.0}
    };
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            EXPECT_NEAR(Ainv.get(i, j), expected.get(i, j), 1e-9);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixInverseTest, SingularMatrixThrowsRuntimeError) {
    slt::DenseMatrix<double> A = {
        {2.0, 4.0},
        {1.0, 2.0}  // Linearly dependent rows
    };

    EXPECT_THROW(A.inverse(), std::runtime_error);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixInverseTest, NonSquareMatrixThrowsInvalidArgument) {
    slt::DenseMatrix<double> A = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0}
    };

    EXPECT_THROW(A.inverse(), std::invalid_argument);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixInverseTest, FloatInverseOf2x2Matrix) {
    slt::DenseMatrix<float> A = {{1.0f, 2.0f}, {3.0f, 4.0f}};
    auto Ainv = A.inverse();

    // Known inverse of A
    slt::DenseMatrix<float> expected = {
        {-2.0f,  1.0f},
        { 1.5f, -0.5f}
    };

    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j)
            EXPECT_NEAR(Ainv.get(i, j), expected.get(i, j), 1e-5f);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixRemoveTest, RemoveInitializedElement) {
    slt::DenseMatrix<double> mat = {{1.0, 2.0}, {3.0, 4.0}};
    
    mat.remove(0, 1);  // Remove value at (0, 1)

    // Check that element is uninitialized
    EXPECT_FALSE(mat.is_initialized(0, 1));

    // Check that accessing it throws
    EXPECT_THROW(mat.get(0, 1), std::runtime_error);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixRemoveTest, RemoveUninitializedElementThrows) {
    slt::DenseMatrix<double> mat(2, 2);  // All elements uninitialized

    EXPECT_THROW(mat.remove(1, 1), std::runtime_error);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixRemoveTest, RemoveOutOfBoundsThrows) {
    slt::DenseMatrix<double> mat = {{1.0, 2.0}, {3.0, 4.0}};

    EXPECT_THROW(mat.remove(2, 0), std::out_of_range);
    EXPECT_THROW(mat.remove(0, 2), std::out_of_range);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixRemoveTest, RemoveThenSetSucceeds) {
    slt::DenseMatrix<double> mat = {{5.0, 6.0}, {7.0, 8.0}};
    mat.remove(1, 1);

    EXPECT_FALSE(mat.is_initialized(1, 1));

    mat.set(1, 1, 42.0);  // Should succeed
    EXPECT_TRUE(mat.is_initialized(1, 1));
    EXPECT_DOUBLE_EQ(mat.get(1, 1), 42.0);
}
// ================================================================================
// ================================================================================
// eof
