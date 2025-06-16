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
// -------------------------------------------------------------------------------- 

TEST(DotProductTest, FloatBasic) {
    float a[] = {1.0f, 2.0f, 3.0f};
    float b[] = {4.0f, 5.0f, 6.0f};
    float result = slt::dot(a, b, 3);
    EXPECT_FLOAT_EQ(result, 32.0f);  // 1*4 + 2*5 + 3*6
}
// -------------------------------------------------------------------------------- 

TEST(DotProductTest, DoubleBasic) {
    double a[] = {1.0, 2.0, 3.0};
    double b[] = {4.0, 5.0, 6.0};
    double result = slt::dot(a, b, 3);
    EXPECT_DOUBLE_EQ(result, 32.0);  // 1*4 + 2*5 + 3*6
}
// -------------------------------------------------------------------------------- 

TEST(DotProductTest, FloatNegativeValues) {
    float a[] = {-1.0f, -2.0f, -3.0f};
    float b[] = {1.0f, 2.0f, 3.0f};
    float result = slt::dot(a, b, 3);
    EXPECT_FLOAT_EQ(result, -14.0f);
}
// -------------------------------------------------------------------------------- 

TEST(DotProductTest, DoubleMixedSigns) {
    double a[] = {1.5, -2.5, 3.0};
    double b[] = {-1.0, 2.0, -3.0};
    double result = slt::dot(a, b, 3);
    EXPECT_DOUBLE_EQ(result, -15.5);
}
// -------------------------------------------------------------------------------- 

TEST(DotProductTest, VectorFloat) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {4.0f, 5.0f, 6.0f};
    float result = slt::dot(a, b);
    EXPECT_FLOAT_EQ(result, 32.0f);
}
// -------------------------------------------------------------------------------- 

TEST(DotProductTest, VectorDouble) {
    std::vector<double> a = {1.0, 2.0, 3.0};
    std::vector<double> b = {4.0, 5.0, 6.0};
    double result = slt::dot(a, b);
    EXPECT_DOUBLE_EQ(result, 32.0);
}
// -------------------------------------------------------------------------------- 

TEST(DotProductTest, ArrayFloat) {
    std::array<float, 3> a = {1.0f, 2.0f, 3.0f};
    std::array<float, 3> b = {4.0f, 5.0f, 6.0f};
    float result = slt::dot(a, b);
    EXPECT_FLOAT_EQ(result, 32.0f);
}
// -------------------------------------------------------------------------------- 

TEST(DotProductTest, ArrayDouble) {
    std::array<double, 3> a = {1.0, 2.0, 3.0};
    std::array<double, 3> b = {4.0, 5.0, 6.0};
    double result = slt::dot(a, b);
    EXPECT_DOUBLE_EQ(result, 32.0);
}
// -------------------------------------------------------------------------------- 

TEST(CrossProductTest, CStyleArrayFloat) {
    float a[3] = {1.0f, 0.0f, 0.0f};
    float b[3] = {0.0f, 1.0f, 0.0f};
    float result[3];

    slt::cross(a, b, result);

    EXPECT_FLOAT_EQ(result[0], 0.0f);
    EXPECT_FLOAT_EQ(result[1], 0.0f);
    EXPECT_FLOAT_EQ(result[2], 1.0f);
}
// -------------------------------------------------------------------------------- 

TEST(CrossProductTest, CStyleArrayDouble) {
    double a[3] = {0.0, 1.0, 0.0};
    double b[3] = {0.0, 0.0, 1.0};
    double result[3];

    slt::cross(a, b, result);

    EXPECT_DOUBLE_EQ(result[0], 1.0);
    EXPECT_DOUBLE_EQ(result[1], 0.0);
    EXPECT_DOUBLE_EQ(result[2], 0.0);
}
// ----------------------------------------------------------------------------

TEST(CrossProductTest, StdArrayFloat) {
    std::array<float, 3> a = {0.0f, 0.0f, 1.0f};
    std::array<float, 3> b = {1.0f, 0.0f, 0.0f};

    auto result = slt::cross(a, b);

    EXPECT_FLOAT_EQ(result[0], 0.0f);
    EXPECT_FLOAT_EQ(result[1], 1.0f);
    EXPECT_FLOAT_EQ(result[2], 0.0f);
}
// -------------------------------------------------------------------------------- 

TEST(CrossProductTest, StdArrayDouble) {
    std::array<double, 3> a = {1.0, 2.0, 3.0};
    std::array<double, 3> b = {4.0, 5.0, 6.0};

    auto result = slt::cross(a, b);

    EXPECT_DOUBLE_EQ(result[0], -3.0);
    EXPECT_DOUBLE_EQ(result[1], 6.0);
    EXPECT_DOUBLE_EQ(result[2], -3.0);
}
// ----------------------------------------------------------------------------

TEST(CrossProductTest, StdVectorFloat) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {4.0f, 5.0f, 6.0f};

    auto result = slt::cross(a, b);

    EXPECT_FLOAT_EQ(result[0], -3.0f);
    EXPECT_FLOAT_EQ(result[1], 6.0f);
    EXPECT_FLOAT_EQ(result[2], -3.0f);
}
// -------------------------------------------------------------------------------- 

TEST(CrossProductTest, StdVectorDouble) {
    std::vector<double> a = {0.0, 0.0, 1.0};
    std::vector<double> b = {1.0, 0.0, 0.0};

    auto result = slt::cross(a, b);

    EXPECT_DOUBLE_EQ(result[0], 0.0);
    EXPECT_DOUBLE_EQ(result[1], 1.0);
    EXPECT_DOUBLE_EQ(result[2], 0.0);
}
// -------------------------------------------------------------------------------- 
TEST(DenseMatrixMatMulTest, BasicMultiplicationFloat) {
    slt::DenseMatrix<float> A = {{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    slt::DenseMatrix<float> B = {{7.0f, 8.0f}, {9.0f, 10.0f}, {11.0f, 12.0f}};

    slt::DenseMatrix<float> C = slt::mat_mul(A, B);

    ASSERT_FLOAT_EQ(C.get(0, 0), 58.0f);
    ASSERT_FLOAT_EQ(C.get(0, 1), 64.0f);
    ASSERT_FLOAT_EQ(C.get(1, 0), 139.0f);
    ASSERT_FLOAT_EQ(C.get(1, 1), 154.0f);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixMatMulTest, IdentityMatrixMultiplication) {
    slt::DenseMatrix<double> I = {{1.0, 0.0, 0.0},
                                  {0.0, 1.0, 0.0},
                                  {0.0, 0.0, 1.0}};

    // for (size_t i = 0; i < 3; ++i)
    //     I.set(i, i, 1.0);

    slt::DenseMatrix<double> A(3, 3);
    A.set(0, 0, 5.0); A.set(0, 1, 6.0); A.set(0, 2, 7.0);
    A.set(1, 0, 1.0); A.set(1, 1, 2.0); A.set(1, 2, 3.0);
    A.set(2, 0, 4.0); A.set(2, 1, 0.0); A.set(2, 2, 8.0);
    slt::DenseMatrix<double> C = slt::mat_mul(A, I);
    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            ASSERT_DOUBLE_EQ(C.get(i, j), A.get(i, j));
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixMatMulTest, ZeroMatrixMultiplication) {
    slt::DenseMatrix<float> A(2, 3, 5.0f);  // All values set to 0.0f and marked initialized
    slt::DenseMatrix<float> B(3, 4, 0.0f);  // Same here
    float a = A.get(0,0);
    std::cout << a << "\n";
    slt::DenseMatrix<float> C = slt::mat_mul(A, B);

    for (size_t i = 0; i < C.rows(); ++i)
        for (size_t j = 0; j < C.cols(); ++j)
            ASSERT_FLOAT_EQ(C.get(i, j), 0.0f);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixMatMulTest, DimensionMismatchThrows) {
    slt::DenseMatrix<float> A(2, 3, 0.0f);
    slt::DenseMatrix<float> B(4, 2, 0.0f);  // Mismatched inner dimensions

    EXPECT_THROW({
        slt::DenseMatrix<float> C = slt::mat_mul(A, B);
    }, std::invalid_argument);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixSparseAdditionTest, BasicAddition) {
    slt::DenseMatrix<float> dense({
        {1.0f, 2.0f},
        {3.0f, 4.0f}
    });

    slt::SparseCOOMatrix<float> sparse({
        {0.0f, 5.0f},
        {0.0f, 1.0f}
    });

    auto result = dense + sparse;

    EXPECT_FLOAT_EQ(result(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(result(0, 1), 7.0f);
    EXPECT_FLOAT_EQ(result(1, 0), 3.0f);
    EXPECT_FLOAT_EQ(result(1, 1), 5.0f);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixSparseAdditionTest, AllZeroSparseMatrix) {
    slt::DenseMatrix<float> dense({
        {1.0f, 2.0f},
        {3.0f, 4.0f}
    });

    slt::SparseCOOMatrix<float> sparse({
        {0.0f, 0.0f},
        {0.0f, 0.0f}
    });

    auto result = dense + sparse;

    EXPECT_FLOAT_EQ(result(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(result(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(result(1, 0), 3.0f);
    EXPECT_FLOAT_EQ(result(1, 1), 4.0f);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixSparseAdditionTest, ThrowsOnSizeMismatch) {
    slt::DenseMatrix<float> dense({
        {1.0f, 2.0f}
    });

    slt::SparseCOOMatrix<float> sparse({
        {0.0f, 1.0f},
        {2.0f, 3.0f}
    });

    EXPECT_THROW({
        auto result = dense + sparse;
    }, std::invalid_argument);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixSparseAdditionTest, SparseOnlyAffectsSpecifiedEntries) {
    slt::DenseMatrix<float> dense({
        {10.0f, 20.0f},
        {30.0f, 40.0f}
    });

    slt::SparseCOOMatrix<float> sparse({
        {0.0f, 0.0f},
        {0.0f, -40.0f}
    });

    auto result = dense + sparse;

    EXPECT_FLOAT_EQ(result(0, 0), 10.0f);
    EXPECT_FLOAT_EQ(result(0, 1), 20.0f);
    EXPECT_FLOAT_EQ(result(1, 0), 30.0f);
    EXPECT_FLOAT_EQ(result(1, 1), 0.0f);  // 40 - 40
}
// ================================================================================ 
// ================================================================================


TEST(DenseMatrixTest, CopyConstructor) {
    slt::DenseMatrix<float> A({1.0f, 2.0f, 3.0f, 4.0f}, 2, 2);
    slt::DenseMatrix<float> B(A);  // Copy constructor

    EXPECT_EQ(B.rows(), 2);
    EXPECT_EQ(B.cols(), 2);
    EXPECT_EQ(B(0, 0), 1.0f);
    EXPECT_EQ(B(0, 1), 2.0f);
    EXPECT_EQ(B(1, 0), 3.0f);
    EXPECT_EQ(B(1, 1), 4.0f);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixTest, MoveConstructor) {
    slt::DenseMatrix<float> A({5.0f, 6.0f, 7.0f, 8.0f}, 2, 2);
    slt::DenseMatrix<float> B(std::move(A));  // Move constructor

    EXPECT_EQ(B.rows(), 2);
    EXPECT_EQ(B.cols(), 2);
    EXPECT_EQ(B(0, 0), 5.0f);
    EXPECT_EQ(B(0, 1), 6.0f);
    EXPECT_EQ(B(1, 0), 7.0f);
    EXPECT_EQ(B(1, 1), 8.0f);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixTest, CopyAssignment) {
    slt::DenseMatrix<double> A({1.1, 2.2, 3.3, 4.4}, 2, 2);
    slt::DenseMatrix<double> B(2, 2);
    B = A;  // Copy assignment

    EXPECT_EQ(B.rows(), 2);
    EXPECT_EQ(B.cols(), 2);
    EXPECT_DOUBLE_EQ(B(0, 0), 1.1);
    EXPECT_DOUBLE_EQ(B(0, 1), 2.2);
    EXPECT_DOUBLE_EQ(B(1, 0), 3.3);
    EXPECT_DOUBLE_EQ(B(1, 1), 4.4);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixTest, MoveAssignment) {
    slt::DenseMatrix<float> A({9.0f, 10.0f, 11.0f, 12.0f}, 2, 2);
    slt::DenseMatrix<float> B(2, 2);
    B = std::move(A);  // Move assignment

    EXPECT_EQ(B.rows(), 2);
    EXPECT_EQ(B.cols(), 2);
    EXPECT_EQ(B(0, 0), 9.0f);
    EXPECT_EQ(B(0, 1), 10.0f);
    EXPECT_EQ(B(1, 0), 11.0f);
    EXPECT_EQ(B(1, 1), 12.0f);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixSizeAndNonzeroCountTest, SizeAndNonzeroCountBehavior) {
    // Construct 3x3 matrix
    slt::DenseMatrix<float> mat(3, 3);

    // Initially all values should be uninitialized
    EXPECT_EQ(mat.size(), 9);
    EXPECT_EQ(mat.nonzero_count(), 0);

    // Set a single element
    mat.set(0, 0, 1.0f);
    EXPECT_EQ(mat.nonzero_count(), 1);

    // Update another element
    mat.set(1, 1, 2.0f);
    EXPECT_EQ(mat.nonzero_count(), 2);

    // Remove one element
    mat.remove(1, 1);
    EXPECT_EQ(mat.nonzero_count(), 1);

    // Fill all elements
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < 3; ++j) {
            if (!mat.is_initialized(i, j)) {
                mat.set(i, j, static_cast<float>(i * 3 + j));
            }
        }
    }

    EXPECT_EQ(mat.nonzero_count(), 9);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixArrayConstructorTest, ValidArrayInitialization) {
    std::array<float, 6> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

    slt::DenseMatrix<float> mat(values, 2, 3);

    EXPECT_EQ(mat.rows(), 2);
    EXPECT_EQ(mat.cols(), 3);
    EXPECT_EQ(mat.size(), 6);
    EXPECT_EQ(mat.nonzero_count(), 6);

    EXPECT_FLOAT_EQ(mat(0, 0), 1.0f);
    EXPECT_FLOAT_EQ(mat(0, 1), 2.0f);
    EXPECT_FLOAT_EQ(mat(0, 2), 3.0f);
    EXPECT_FLOAT_EQ(mat(1, 0), 4.0f);
    EXPECT_FLOAT_EQ(mat(1, 1), 5.0f);
    EXPECT_FLOAT_EQ(mat(1, 2), 6.0f);
}
// -------------------------------------------------------------------------------- 

TEST(DenseMatrixArrayConstructorTest, MismatchedDimensionsThrows) {
    std::array<float, 5> bad_values = {1, 2, 3, 4, 5};  // 5 elements

    EXPECT_THROW({
        slt::DenseMatrix<float> bad_mat(bad_values, 2, 3);  // 2x3 = 6 expected
    }, std::invalid_argument);
}
// ================================================================================ 
// ================================================================================ 

// TEST(SparseCOOMatrixTest, ConstructorInitializesDimensionsCorrectly) {
//     slt::SparseCOOMatrix<float> mat(5, 7);
//     EXPECT_EQ(mat.rows(), 5);
//     EXPECT_EQ(mat.cols(), 7);
// }
// // -------------------------------------------------------------------------------- 
//
// TEST(SparseCOOMatrixTest, ConstructorDefaultsToFastInsertTrue) {
//     slt::SparseCOOMatrix<double> mat(3, 3);
//     // Try inserting two values and then call get (should throw because not finalized)
//     mat.set(0, 0, 1.23);
//     mat.set(2, 1, 4.56);
//     EXPECT_THROW(mat.get(2, 0), std::runtime_error);  // Not finalized
//     mat.finalize();
//     EXPECT_NO_THROW(mat.get(0, 0));
//     EXPECT_DOUBLE_EQ(mat.get(0, 0), 1.23);
//     EXPECT_DOUBLE_EQ(mat.get(2, 1), 4.56);
// }
// // -------------------------------------------------------------------------------- 
//
// TEST(SparseCOOMatrixTest, FastInsertFalseAllowsImmediateAccess) {
//     slt::SparseCOOMatrix<float> mat(4, 4, false);  // Sorted mode
//     mat.set(1, 2, 3.14f);
//     EXPECT_NO_THROW(mat.get(1, 2));
//     EXPECT_FLOAT_EQ(mat.get(1, 2), 3.14f);
// }
// // -------------------------------------------------------------------------------- 
//
// TEST(SparseCOOMatrixTest, IsInitializedReturnsFalseInitially) {
//     slt::SparseCOOMatrix<float> mat(2, 2);
//     EXPECT_FALSE(mat.is_initialized(0, 0));
//     EXPECT_FALSE(mat.is_initialized(1, 1));
// }
// -------------------------------------------------------------------------------- 

// TEST(SparseCOOMatrixTest, OutOfBoundsThrowsInGetAndSet) {
//     slt::SparseCOOMatrix<float> mat(2, 2);
//     EXPECT_THROW(mat.set(5, 0, 1.0f), std::out_of_range);
//     EXPECT_THROW(mat.set(0, 5, 1.0f), std::out_of_range);
//     EXPECT_THROW(mat.get(3, 1), std::out_of_range);
// }
// // -------------------------------------------------------------------------------- 
//
// // Test: Constructor from std::vector<std::vector<T>>
// TEST(SparseCOOMatrixTest, ConstructFrom2DVector) {
//     std::vector<std::vector<double>> input = {
//         {1.0, 0.0, 2.0},
//         {0.0, 0.0, 0.0},
//         {0.0, 3.0, 0.0}
//     };
//
//     slt::SparseCOOMatrix mat(input);
//     EXPECT_EQ(mat.rows(), 3);
//     EXPECT_EQ(mat.cols(), 3);
//
//     mat.finalize();
//
//     EXPECT_TRUE(mat.is_initialized(0, 0));
//     EXPECT_TRUE(mat.is_initialized(0, 2));
//     EXPECT_TRUE(mat.is_initialized(2, 1));
//
//     EXPECT_DOUBLE_EQ(mat.get(0, 0), 1.0);
//     EXPECT_DOUBLE_EQ(mat.get(0, 2), 2.0);
//     EXPECT_DOUBLE_EQ(mat.get(2, 1), 3.0);
//
//     EXPECT_FALSE(mat.is_initialized(1, 1));
//     EXPECT_FALSE(mat.is_initialized(2, 2));
//
//     EXPECT_THROW(mat.get(1, 1), std::runtime_error);
// }
//
// // ---------------------------------------------------------------------------
//
// TEST(SparseCOOMatrixTest, ThrowsOnInvalidShape) {
//     std::vector<std::vector<double>> bad_input = {
//         {1.0, 0.0},
//         {2.0}  // shorter row
//     };
//
//     EXPECT_THROW(slt::SparseCOOMatrix mat(bad_input), std::invalid_argument);
// }
// // -------------------------------------------------------------------------------- 
//
// TEST(SparseCOOMatrixTest, ConstructFromFixedSizeArray) {
//     std::array<std::array<double, 3>, 3> arr = {{
//         {1.0, 0.0, 2.0},
//         {0.0, 0.0, 0.0},
//         {0.0, 3.0, 0.0}
//     }};
//
//     slt::SparseCOOMatrix mat(arr);
//     EXPECT_EQ(mat.rows(), 3);
//     EXPECT_EQ(mat.cols(), 3);
//
//     mat.finalize();
//
//     EXPECT_TRUE(mat.is_initialized(0, 0));
//     EXPECT_TRUE(mat.is_initialized(0, 2));
//     EXPECT_TRUE(mat.is_initialized(2, 1));
//
//     EXPECT_DOUBLE_EQ(mat.get(0, 0), 1.0);
//     EXPECT_DOUBLE_EQ(mat.get(0, 2), 2.0);
//     EXPECT_DOUBLE_EQ(mat.get(2, 1), 3.0);
//
//     EXPECT_FALSE(mat.is_initialized(1, 1));
//     EXPECT_FALSE(mat.is_initialized(2, 2));
//
//     EXPECT_THROW(mat.get(1, 1), std::runtime_error);
// }
// // -------------------------------------------------------------------------------- 
//
// TEST(SparseCOOMatrixTest, ConstructFromInitializerList) {
//     slt::SparseCOOMatrix mat{
//         {1.0, 0.0, 2.0},
//         {0.0, 0.0, 0.0},
//         {0.0, 3.0, 0.0}
//     };
//
//     EXPECT_EQ(mat.rows(), 3);
//     EXPECT_EQ(mat.cols(), 3);
//
//     mat.finalize();  // Required for accurate lookup after fast insertion
//
//     // Check expected initialized elements
//     EXPECT_TRUE(mat.is_initialized(0, 0));
//     EXPECT_TRUE(mat.is_initialized(0, 2));
//     EXPECT_TRUE(mat.is_initialized(2, 1));
//
//     EXPECT_DOUBLE_EQ(mat.get(0, 0), 1.0);
//     EXPECT_DOUBLE_EQ(mat.get(0, 2), 2.0);
//     EXPECT_DOUBLE_EQ(mat.get(2, 1), 3.0);
//
//     // Check uninitialized entries
//     EXPECT_FALSE(mat.is_initialized(1, 1));
//     EXPECT_FALSE(mat.set_fast());
//     EXPECT_THROW(mat.get(1, 1), std::runtime_error);
// }
// // -------------------------------------------------------------------------------- 
//
// TEST(SparseCOOMatrixUpdateTest, UpdatesExistingValueAfterFinalize) {
//     slt::SparseCOOMatrix<float> mat(3, 3);
//     mat.set(0, 1, 2.0f);
//     mat.set(2, 2, 5.0f);
//     mat.finalize();
//
//     EXPECT_FLOAT_EQ(mat.get(0, 1), 2.0f);
// }
// // -------------------------------------------------------------------------------- 
//
// TEST(SparseCOOMatrixUpdateTest, ThrowsIfElementNotSet) {
//     slt::SparseCOOMatrix<double> mat(2, 2);
//     mat.set(0, 0, 3.14);
//     mat.finalize();
//
//     EXPECT_THROW(mat.update(1, 1, 2.71), std::runtime_error);
// }
// // -------------------------------------------------------------------------------- 
//
// TEST(SparseCOOMatrixUpdateTest, ThrowsIfNotFinalized) {
//     slt::SparseCOOMatrix<float> mat(2, 2);
//     mat.set(0, 0, 1.0f);
//     // Forgot to call finalize()
//
//     EXPECT_THROW(mat.update(0, 1, 3.0f), std::runtime_error);
// }
// // -------------------------------------------------------------------------------- 
//
// TEST(SparseCOOMatrixUpdateTest, ThrowsOnInvalidIndex) {
//     slt::SparseCOOMatrix<float> mat(2, 2);
//     mat.set(0, 0, 1.0f);
//     mat.finalize();
//
//     EXPECT_THROW(mat.update(2, 2, 5.0f), std::out_of_range);
// }
// // -------------------------------------------------------------------------------- 
//
// TEST(SparseCOOMatrixAdditionTest, NonOverlappingEntries) {
//     // Matrix A:
//     // [1 0]
//     // [0 0]
//     slt::SparseCOOMatrix<float> A({
//         {1.0f, 0.0f},
//         {0.0f, 0.0f}
//     });
//
//     // Matrix B:
//     // [0 0]
//     // [0 2]
//     slt::SparseCOOMatrix<float> B({
//         {0.0f, 0.0f},
//         {0.0f, 2.0f}
//     });
//
//     slt::DenseMatrix<float> result = A + B;
//
//     ASSERT_EQ(result.rows(), 2);
//     ASSERT_EQ(result.cols(), 2);
//
//     // Check expected values
//     EXPECT_FLOAT_EQ(result(0, 0), 1.0f);  // from A
//     EXPECT_FLOAT_EQ(result(0, 1), 0.0f);  // empty
//     EXPECT_FLOAT_EQ(result(1, 0), 0.0f);  // empty
//     EXPECT_FLOAT_EQ(result(1, 1), 2.0f);  // from B
// }
// // -------------------------------------------------------------------------------- 
//
// TEST(SparseCOOMatrixAdditionTest, OverlappingEntries) {
//     // Matrix A:
//     // [1 0]
//     // [0 2]
//     slt::SparseCOOMatrix<float> A({
//         {1.0f, 0.0f},
//         {0.0f, 2.0f}
//     });
//
//     // Matrix B:
//     // [3 0]
//     // [0 4]
//     slt::SparseCOOMatrix<float> B({
//         {3.0f, 0.0f},
//         {0.0f, 4.0f}
//     });
//
//     slt::DenseMatrix<float> result = A + B;
//
//     ASSERT_EQ(result.rows(), 2);
//     ASSERT_EQ(result.cols(), 2);
//
//     // Check element-wise sum
//     EXPECT_FLOAT_EQ(result(0, 0), 4.0f);  // 1 + 3
//     EXPECT_FLOAT_EQ(result(0, 1), 0.0f);
//     EXPECT_FLOAT_EQ(result(1, 0), 0.0f);
//     EXPECT_FLOAT_EQ(result(1, 1), 6.0f);  // 2 + 4
// }
// // -------------------------------------------------------------------------------- 
//
// TEST(DenseSparseAdditionTest, DensePlusSparseCOO) {
//     // Dense matrix:
//     // [1 2]
//     // [3 4]
//     slt::DenseMatrix<float> dense({
//         {1.0f, 2.0f},
//         {3.0f, 4.0f}
//     });
//
//     // Sparse matrix:
//     // [0 5]
//     // [6 0]
//     slt::SparseCOOMatrix<float> sparse({
//         {0.0f, 5.0f},
//         {6.0f, 0.0f}
//     });
//
//     slt::DenseMatrix<float> result = dense + sparse;
//
//     ASSERT_EQ(result.rows(), 2);
//     ASSERT_EQ(result.cols(), 2);
//
//     // Check values
//     EXPECT_FLOAT_EQ(result(0, 0), 1.0f);  // 1 + 0
//     EXPECT_FLOAT_EQ(result(0, 1), 7.0f);  // 2 + 5
//     EXPECT_FLOAT_EQ(result(1, 0), 9.0f);  // 3 + 6
//     EXPECT_FLOAT_EQ(result(1, 1), 4.0f);  // 4 + 0
// }
// // -------------------------------------------------------------------------------- 
//
// TEST(SparseScalarAdditionTest, SparsePlusScalar) {
//     // Sparse matrix:
//     // [0 5]
//     // [6 0]
//     slt::SparseCOOMatrix<float> sparse({
//         {0.0f, 5.0f},
//         {6.0f, 0.0f}
//     });
//
//     float scalar = 3.0f;
//     auto result = sparse + scalar;
//
//     ASSERT_EQ(result.rows(), 2);
//     ASSERT_EQ(result.cols(), 2);
//     ASSERT_EQ(result.nonzero_count(), 2);
//
//     EXPECT_EQ(result.row_index(0), 0);
//     EXPECT_EQ(result.col_index(0), 1);
//     EXPECT_FLOAT_EQ(result.value(0), 8.0f);  // 5 + 3
//
//     EXPECT_EQ(result.row_index(1), 1);
//     EXPECT_EQ(result.col_index(1), 0);
//     EXPECT_FLOAT_EQ(result.value(1), 9.0f);  // 6 + 3
// }
// // -------------------------------------------------------------------------------- 
//
// TEST(SparseScalarAdditionTest, ScalarPlusSparse) {
//     // Sparse matrix:
//     // [0 2]
//     // [3 0]
//     slt::SparseCOOMatrix<float> sparse({
//         {0.0f, 2.0f},
//         {3.0f, 0.0f}
//     });
//
//     float scalar = 1.5f;
//     auto result = scalar + sparse;
//
//     ASSERT_EQ(result.rows(), 2);
//     ASSERT_EQ(result.cols(), 2);
//     ASSERT_EQ(result.nonzero_count(), 2);
//
//     EXPECT_EQ(result.row_index(0), 0);
//     EXPECT_EQ(result.col_index(0), 1);
//     EXPECT_FLOAT_EQ(result.value(0), 3.5f);  // 2 + 1.5
//
//     EXPECT_EQ(result.row_index(1), 1);
//     EXPECT_EQ(result.col_index(1), 0);
//     EXPECT_FLOAT_EQ(result.value(1), 4.5f);  // 3 + 1.5
// }
// // -------------------------------------------------------------------------------- 
//
// TEST(SparseCOOMatrixTest, FlatConstructor_PopulatesNonZeroValues) {
//     std::vector<float> flat = {
//         0.0f, 2.0f,
//         3.0f, 0.0f
//     };
//
//     slt::SparseCOOMatrix<float> mat(flat, 2, 2);
//     mat.finalize();
//
//     EXPECT_EQ(mat.nonzero_count(), 2);
//     EXPECT_TRUE(mat.is_initialized(0, 1));
//     EXPECT_TRUE(mat.is_initialized(1, 0));
//     EXPECT_FLOAT_EQ(mat(0, 1), 2.0f);
//     EXPECT_FLOAT_EQ(mat(1, 0), 3.0f);
// }
// // -------------------------------------------------------------------------------- 
//
// TEST(SparseCOOMatrixTest, FlatConstructor_AllZero) {
//     std::vector<double> flat(6, 0.0);
//     slt::SparseCOOMatrix<double> mat(flat, 2, 3);
//     mat.finalize();
//
//     EXPECT_EQ(mat.nonzero_count(), 0);
// }
// // -------------------------------------------------------------------------------- 
//
// TEST(SparseCOOMatrixTest, FlatConstructor_InvalidSizeThrows) {
//     std::vector<float> flat = {1.0f, 2.0f, 3.0f};  // Should be 2x2 but only 3 elements
//
//     EXPECT_THROW({
//             slt::SparseCOOMatrix<float> mat(flat, 2, 2);
//     }, std::invalid_argument);
// }
// // -------------------------------------------------------------------------------- 
//
// TEST(SparseCOOMatrixTest, FlatConstructor_CorrectIndexing) {
//     std::vector<float> flat = {
//         1.0f, 0.0f,
//         0.0f, 4.0f
//     };
//
//     slt::SparseCOOMatrix<float> mat(flat, 2, 2);
//     mat.finalize();
//
//     EXPECT_EQ(mat.row_index(0), 0);
//     EXPECT_EQ(mat.col_index(0), 0);
//     EXPECT_EQ(mat.row_index(1), 1);
//     EXPECT_EQ(mat.col_index(1), 1);
// }
// // -------------------------------------------------------------------------------- 
//
// TEST(SparseCOOMatrixEquality, IdenticalMatrices) {
//     slt::SparseCOOMatrix<float> A(2, 2, false);
//     A.set(0, 0, 1.0f);
//     A.set(1, 1, 2.0f);
//     A.finalize();
//
//     slt::SparseCOOMatrix<float> B(2, 2, false);
//     B.set(0, 0, 1.0f);
//     B.set(1, 1, 2.0f);
//     B.finalize();
//
//     EXPECT_TRUE(A == B);
// }
// // -------------------------------------------------------------------------------- 
//
// TEST(SparseCOOMatrixEquality, DifferentDimensions) {
//     slt::SparseCOOMatrix<float> A(2, 2, false);
//     A.set(0, 0, 1.0f);
//
//     slt::SparseCOOMatrix<float> B(3, 2, false);
//     B.set(0, 0, 1.0f);
//
//     EXPECT_FALSE(A == B);
// }
// // -------------------------------------------------------------------------------- 
//
// TEST(SparseCOOMatrixEquality, DifferentValues) {
//     slt::SparseCOOMatrix<float> A(2, 2, false);
//     A.set(0, 0, 1.0f);
//     A.set(1, 1, 2.0f);
//     A.finalize();
//
//     slt::SparseCOOMatrix<float> B(2, 2, false);
//     B.set(0, 0, 1.0f);
//     B.set(1, 1, 3.0f);  // different value
//     B.finalize();
//
//     EXPECT_FALSE(A == B);
// }
// // -------------------------------------------------------------------------------- 
//
// TEST(SparseCOOMatrixEquality, SameValuesDifferentOrder) {
//     slt::SparseCOOMatrix<float> A(2, 2, false);
//     A.set(0, 0, 1.0f);
//     A.set(1, 1, 2.0f);
//     A.finalize();
//
//     slt::SparseCOOMatrix<float> B(2, 2, false);
//     B.set(1, 1, 2.0f);
//     B.set(0, 0, 1.0f);
//     B.finalize();
//
//     EXPECT_TRUE(A == B);
// }
// // -------------------------------------------------------------------------------- 
//
// TEST(SparseCOOMatrixEquality, NotFinalizedVsFinalized) {
//     slt::SparseCOOMatrix<float> A(2, 2, true);
//     A.set(0, 0, 1.0f);
//     A.set(1, 1, 2.0f);
//
//     slt::SparseCOOMatrix<float> B(2, 2, false);
//     B.set(0, 0, 1.0f);
//     B.set(1, 1, 2.0f);
//     B.finalize();
//
//     EXPECT_TRUE(A == B);
// }
// // -------------------------------------------------------------------------------- 
//
// TEST(SparseCOOMatrixAssignment, CopyAssignmentIndependence) {
//     slt::SparseCOOMatrix<float> A(2, 2, false);
//     A.set(0, 0, 1.0f);
//     A.set(1, 1, 2.0f);
//     A.finalize();
//
//     slt::SparseCOOMatrix<float> B(1, 1);
//     B = A;  // Deep copy
//
//     // Matrices should be equal
//     EXPECT_TRUE(A == B);
//
//     // Modify B independently
//     B.update(1, 1, 3.0f);
//
//     // Original A should remain unchanged
//     EXPECT_FLOAT_EQ(A.get(1, 1), 2.0f);
//     EXPECT_FLOAT_EQ(B.get(1, 1), 3.0f);
// }
// // -------------------------------------------------------------------------------- 
//
// TEST(SparseCOOMatrixAssignment, MoveAssignmentTransfersResources) {
//     slt::SparseCOOMatrix<float> A(2, 2, false);
//     A.set(0, 0, 4.0f);
//     A.set(1, 1, 5.0f);
//     A.finalize();
//
//     slt::SparseCOOMatrix<float> B(1, 1);
//     B = std::move(A);  // Move assignment
//
//     // Values should exist in B
//     EXPECT_FLOAT_EQ(B.get(0, 0), 4.0f);
//     EXPECT_FLOAT_EQ(B.get(1, 1), 5.0f);
//
//     // Dimensions should match
//     EXPECT_EQ(B.rows(), 2);
//     EXPECT_EQ(B.cols(), 2);
//
//     // A is in a valid but empty state; we check dimensions and size
//     EXPECT_EQ(A.rows(), 0);
//     EXPECT_EQ(A.cols(), 0);
//     EXPECT_EQ(A.nonzero_count(), 0);
// }
// // -------------------------------------------------------------------------------- 
//
// TEST(SparseCOOMatrixTest, SubtractScalarFromSparseMatrix) {
//     slt::SparseCOOMatrix<float> A(2, 2);
//     A.set(0, 0, 1.0f);
//     A.set(1, 1, 2.0f);
//
//     auto result = A - 1.0f;
//
//     EXPECT_EQ(result.rows(), 2);
//     EXPECT_EQ(result.cols(), 2);
//     EXPECT_EQ(result.nonzero_count(), 2);
//     EXPECT_FLOAT_EQ(result(0,0), 0.0f);
//     EXPECT_FLOAT_EQ(result(1,1), 1.0f);
// }
// // -------------------------------------------------------------------------------- 
//
// TEST(SparseCOOMatrixTest, SubtractSparseFromSparse) {
//     slt::SparseCOOMatrix<float> A{
//         {1.0, 0.0},
//         {0.0, 2.0}
//     };
//     slt::SparseCOOMatrix<float> B{
//         {0.0, 3.0},
//         {4.0, 0.0}
//     };
//
//     slt::DenseMatrix<float> result = A - B;
//
//     EXPECT_EQ(result.rows(), 2);
//     EXPECT_EQ(result.cols(), 2);
//     EXPECT_FLOAT_EQ(result(0, 0), 1.0f);
//     EXPECT_FLOAT_EQ(result(0, 1), -3.0f);
//     EXPECT_FLOAT_EQ(result(1, 0), -4.0f);
//     EXPECT_FLOAT_EQ(result(1, 1), 2.0f);
// }
// // -------------------------------------------------------------------------------- 
//
// TEST(SparseCOOMatrixOperators, ScalarMinusMatrix) {
//     slt::SparseCOOMatrix<float> A(2, 2);
//     A.set(0, 0, 3.0f);
//     A.set(1, 1, 1.0f);
//
//     // Perform scalar - matrix
//     slt::SparseCOOMatrix<float> B = 5.0f - A;
//
//     // Check values: B(0,0) = 2.0, B(1,1) = 4.0
//     EXPECT_FLOAT_EQ(B.get(0, 0), 2.0f);
//     EXPECT_FLOAT_EQ(B.get(1, 1), 4.0f);
//
//     // Check that dimensions are preserved
//     EXPECT_EQ(B.rows(), 2);
//     EXPECT_EQ(B.cols(), 2);
// }
// // -------------------------------------------------------------------------------- 
//
// TEST(SparseCOOMatrixOperators, SparseMinusDense) {
//     slt::SparseCOOMatrix<float> A(2, 2);
//     A.set(0, 0, 3.0f);
//     A.set(1, 1, 1.0f);
//
//     slt::DenseMatrix<float> B(2, 2, 0.0f);
//     B.update(0, 0, 1.0f);
//     B.update(0, 1, 2.0f);
//     B.update(1, 0, 3.0f);
//     B.update(1, 1, 4.0f);
//
//     slt::DenseMatrix<float> result = A - B;
//
//     EXPECT_FLOAT_EQ(result(0, 0), 2.0f);   // 3 - 1
//     EXPECT_FLOAT_EQ(result(0, 1), -2.0f);  // 0 - 2
//     EXPECT_FLOAT_EQ(result(1, 0), -3.0f);  // 0 - 3
//     EXPECT_FLOAT_EQ(result(1, 1), -3.0f);  // 1 - 4
//
//     EXPECT_EQ(result.rows(), 2);
//     EXPECT_EQ(result.cols(), 2);
// }
// // -------------------------------------------------------------------------------- 
//
// TEST(SparseCOOMatrixOperators, DenseMinusSparse) {
//     slt::DenseMatrix<float> A(2, 2, 5.0f);
//     A.update(0, 0, 7.0f);
//     A.update(1, 1, 9.0f);
//
//     slt::SparseCOOMatrix<float> B(2, 2);
//     B.set(0, 0, 2.0f);
//     B.set(1, 1, 4.0f);
//
//     slt::DenseMatrix<float> result = A - B;
//
//     EXPECT_FLOAT_EQ(result(0, 0), 5.0f);   // 7 - 2
//     EXPECT_FLOAT_EQ(result(0, 1), 5.0f);   // 5 - 0
//     EXPECT_FLOAT_EQ(result(1, 0), 5.0f);   // 5 - 0
//     EXPECT_FLOAT_EQ(result(1, 1), 5.0f);   // 9 - 4
//
//     EXPECT_EQ(result.rows(), 2);
//     EXPECT_EQ(result.cols(), 2);
// }
// // -------------------------------------------------------------------------------- 
//
// TEST(SparseCOOMatrixOperators, ElementWiseMultiplyMatchingEntries) {
//     slt::SparseCOOMatrix<float> A{
//         {1.0f, 0.0f},
//         {0.0f, 2.0f}
//     };
//     slt::SparseCOOMatrix<float> B{
//         {0.0f, 3.0f},
//         {0.0f, 4.0f}
//     };
//
//     slt::SparseCOOMatrix<float> result = A * B;
//
//     EXPECT_EQ(result.rows(), 2);
//     EXPECT_EQ(result.cols(), 2);
//
//     // Only one overlapping non-zero at (1,1): 2.0 * 4.0 = 8.0
//     EXPECT_FLOAT_EQ(result.get(1, 1), 8.0f);
//
//     // Non-overlapping positions should not be present
//     EXPECT_THROW(result.get(0, 0), std::runtime_error);
//     EXPECT_THROW(result.get(0, 1), std::runtime_error);
//     EXPECT_THROW(result.get(1, 0), std::runtime_error);
// }
// // -------------------------------------------------------------------------------- 
//
// TEST(SparseCOOMatrixOperators, MultiplyWithNoOverlap) {
//     slt::SparseCOOMatrix<float> A(2, 2);
//     A.set(0, 0, 5.0f);
//
//     slt::SparseCOOMatrix<float> B(2, 2);
//     B.set(1, 1, 10.0f);
//
//     slt::SparseCOOMatrix<float> result = A * B;
//
//     EXPECT_EQ(result.rows(), 2);
//     EXPECT_EQ(result.cols(), 2);
//
//     // No overlapping positions -> result should be empty
//     EXPECT_THROW(result.get(0, 0), std::runtime_error);
//     EXPECT_THROW(result.get(1, 1), std::runtime_error);
// }
// // -------------------------------------------------------------------------------- 
//
// TEST(SparseCOOMatrixOperators, MismatchedDimensionsThrows) {
//     slt::SparseCOOMatrix<float> A(2, 3);
//     slt::SparseCOOMatrix<float> B(3, 2);
//
//     EXPECT_THROW({
//         auto result = A * B;
//     }, std::invalid_argument);
// }
// ================================================================================
// ================================================================================
// eof
