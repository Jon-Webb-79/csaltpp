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
// ================================================================================
// ================================================================================
// eof
