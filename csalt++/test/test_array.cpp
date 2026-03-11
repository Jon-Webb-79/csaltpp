// ================================================================================
// ================================================================================
// - File:    test_array.cpp
// - Purpose: This file implements google test as a method to test C++ code.
//            Describe the type of testing to be completed
//
// Source Metadata
// - Author:  Jonathan A. Webb
// - Date:    March 10, 2026
// - Version: 1.0
// - Copyright: Copyright 2026, Jon Webb Inc.
// ================================================================================
// ================================================================================
// - Begin test

#include <gtest/gtest.h>
#include "array.hpp"
#include "allocator.hpp"
// ================================================================================ 
// ================================================================================ 

// A minimal plain struct to exercise init() with a non-scalar trivially
// copyable type.  It has no user-defined constructor, destructor, or copy
// members so std::is_trivially_copyable_v<Point> is true.
struct Point {
    int   x;
    int   y;

    bool operator==(const Point& other) const noexcept {
        return x == other.x && y == other.y;
    }
};
// ================================================================================
// ================================================================================

/**
 * @class ArrayInitTest
 * @brief Test fixture for cslt::Array::init() tests
 *
 * @details Provides a shared HeapAllocator instance for all init() tests.
 *          Each test receives a freshly constructed fixture so the allocator
 *          state is clean at the start of every case.
 */
class ArrayInitTest : public ::testing::Test {
protected:
    cslt::HeapAllocator alloc;
};
// ================================================================================
// ================================================================================

// ============================================================================
// Successful initialisation tests
// ============================================================================

/**
 * @test Verify that init() succeeds with a valid capacity for int
 */
TEST_F(ArrayInitTest, InitIntSucceeds) {
    auto result = cslt::Array<int>::init(8, alloc);
    ASSERT_TRUE(result.hasValue());

    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
    EXPECT_EQ(arr->size(),     0u);
    EXPECT_EQ(arr->capacity(), 8u);
}
// --------------------------------------------------------------------------------

/**
 * @test Verify that init() succeeds with a valid capacity for double
 */
TEST_F(ArrayInitTest, InitDoubleSucceeds) {
    auto result = cslt::Array<double>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());

    cslt::UniquePtr<cslt::Array<double>, cslt::ArrayDeleter<double>> arr(result.value());
    EXPECT_EQ(arr->size(),     0u);
    EXPECT_EQ(arr->capacity(), 4u);
}
// --------------------------------------------------------------------------------

/**
 * @test Verify that init() succeeds with a valid capacity for a plain struct
 */
TEST_F(ArrayInitTest, InitPointSucceeds) {
    auto result = cslt::Array<Point>::init(16, alloc);
    ASSERT_TRUE(result.hasValue());

    cslt::UniquePtr<cslt::Array<Point>, cslt::ArrayDeleter<Point>> arr(result.value());
    EXPECT_EQ(arr->size(),     0u);
    EXPECT_EQ(arr->capacity(), 16u);
}
// --------------------------------------------------------------------------------

/**
 * @test Verify that init() with capacity of 1 succeeds (minimum valid capacity)
 */
TEST_F(ArrayInitTest, InitCapacityOneSucceeds) {
    auto result = cslt::Array<int>::init(1, alloc);
    ASSERT_TRUE(result.hasValue());

    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
    EXPECT_EQ(arr->size(),     0u);
    EXPECT_EQ(arr->capacity(), 1u);
}
// --------------------------------------------------------------------------------

/**
 * @test Verify that a freshly initialised array reports is_empty() == true
 */
TEST_F(ArrayInitTest, InitArrayIsEmpty) {
    auto result = cslt::Array<int>::init(8, alloc);
    ASSERT_TRUE(result.hasValue());

    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
    EXPECT_TRUE(arr->is_empty());
}
// --------------------------------------------------------------------------------

/**
 * @test Verify that a freshly initialised array with capacity 1 reports
 *       is_full() == false (no elements have been pushed yet)
 */
TEST_F(ArrayInitTest, InitArrayIsNotFull) {
    auto result = cslt::Array<int>::init(1, alloc);
    ASSERT_TRUE(result.hasValue());

    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
    EXPECT_FALSE(arr->is_full());
}
// --------------------------------------------------------------------------------

/**
 * @test Verify that the data() pointer is non-null after a successful init()
 */
TEST_F(ArrayInitTest, InitDataPointerIsNonNull) {
    auto result = cslt::Array<int>::init(8, alloc);
    ASSERT_TRUE(result.hasValue());

    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
    EXPECT_NE(arr->data(), nullptr);
}
// --------------------------------------------------------------------------------

/**
 * @test Verify that two independent init() calls produce distinct data buffers
 */
TEST_F(ArrayInitTest, TwoInitCallsProduceDistinctBuffers) {
    auto r1 = cslt::Array<int>::init(8, alloc);
    auto r2 = cslt::Array<int>::init(8, alloc);
    ASSERT_TRUE(r1.hasValue());
    ASSERT_TRUE(r2.hasValue());

    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr1(r1.value());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr2(r2.value());
    EXPECT_NE(arr1->data(), arr2->data());
}
// --------------------------------------------------------------------------------

/**
 * @test Verify that a large capacity initialisation succeeds and reports the
 *       correct capacity (exercises the allocator with a sizeable request)
 */
TEST_F(ArrayInitTest, InitLargeCapacitySucceeds) {
    auto result = cslt::Array<double>::init(1024, alloc);
    ASSERT_TRUE(result.hasValue());

    cslt::UniquePtr<cslt::Array<double>, cslt::ArrayDeleter<double>> arr(result.value());
    EXPECT_EQ(arr->capacity(), 1024u);
    EXPECT_EQ(arr->size(),     0u);
}

// ============================================================================
// Error condition tests
// ============================================================================

/**
 * @test Verify that init() with capacity 0 returns an error and not a value
 */
TEST_F(ArrayInitTest, InitZeroCapacityReturnsError) {
    auto result = cslt::Array<int>::init(0, alloc);
    EXPECT_FALSE(result.hasValue());
}
// --------------------------------------------------------------------------------

/**
 * @test Verify that init() with capacity 0 for double returns an error
 */
TEST_F(ArrayInitTest, InitZeroCapacityDoubleReturnsError) {
    auto result = cslt::Array<double>::init(0, alloc);
    EXPECT_FALSE(result.hasValue());
}
// --------------------------------------------------------------------------------

/**
 * @test Verify that init() with capacity 0 for Point returns an error
 */
TEST_F(ArrayInitTest, InitZeroCapacityPointReturnsError) {
    auto result = cslt::Array<Point>::init(0, alloc);
    EXPECT_FALSE(result.hasValue());
}
// ================================================================================
// ================================================================================

// ============================================================================
// push_back tests
// ============================================================================

/**
 * @test Verify that push_back() on an int array increases size by one and
 *       stores the correct value, read back via operator[]
 */
TEST_F(ArrayInitTest, PushBackIntSingleElement) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());

    EXPECT_TRUE(arr->push_back(42));
    EXPECT_EQ(arr->size(), 1u);

    auto r = (*arr)[0];
    ASSERT_TRUE(r.hasValue());
    EXPECT_EQ(r.value(), 42);
}
// --------------------------------------------------------------------------------

/**
 * @test Verify that push_back() appends multiple int values in order
 */
TEST_F(ArrayInitTest, PushBackIntMultipleElements) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());

    EXPECT_TRUE(arr->push_back(10));
    EXPECT_TRUE(arr->push_back(20));
    EXPECT_TRUE(arr->push_back(30));
    EXPECT_EQ(arr->size(), 3u);

    EXPECT_EQ((*arr)[0].value(), 10);
    EXPECT_EQ((*arr)[1].value(), 20);
    EXPECT_EQ((*arr)[2].value(), 30);
}
// --------------------------------------------------------------------------------

/**
 * @test Verify that push_back() on a double array stores the correct value
 */
TEST_F(ArrayInitTest, PushBackDoubleElement) {
    auto result = cslt::Array<double>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<double>, cslt::ArrayDeleter<double>> arr(result.value());

    EXPECT_TRUE(arr->push_back(3.14));
    EXPECT_EQ(arr->size(), 1u);

    auto r = (*arr)[0];
    ASSERT_TRUE(r.hasValue());
    EXPECT_DOUBLE_EQ(r.value(), 3.14);
}
// --------------------------------------------------------------------------------

/**
 * @test Verify that push_back() on a Point array stores the correct value
 */
TEST_F(ArrayInitTest, PushBackPointElement) {
    auto result = cslt::Array<Point>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<Point>, cslt::ArrayDeleter<Point>> arr(result.value());

    Point p{3, 7};
    EXPECT_TRUE(arr->push_back(p));
    EXPECT_EQ(arr->size(), 1u);

    auto r = (*arr)[0];
    ASSERT_TRUE(r.hasValue());
    EXPECT_EQ(r.value(), p);
}
// --------------------------------------------------------------------------------

/**
 * @test Verify that push_back() triggers growth when capacity is exceeded
 *       and all previously pushed values remain correct
 */
TEST_F(ArrayInitTest, PushBackTriggersgrowth) {
    auto result = cslt::Array<int>::init(2, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());

    // Push beyond initial capacity of 2
    EXPECT_TRUE(arr->push_back(1));
    EXPECT_TRUE(arr->push_back(2));
    EXPECT_TRUE(arr->push_back(3));  // triggers growth

    EXPECT_EQ(arr->size(), 3u);
    EXPECT_GE(arr->capacity(), 3u);

    EXPECT_EQ((*arr)[0].value(), 1);
    EXPECT_EQ((*arr)[1].value(), 2);
    EXPECT_EQ((*arr)[2].value(), 3);
}
// --------------------------------------------------------------------------------

/**
 * @test Verify that operator[] returns an error for an out-of-bounds read
 *       on an int array
 */
TEST_F(ArrayInitTest, PushBackOutOfBoundsReadReturnsError) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());

    arr->push_back(99);

    auto r = (*arr)[1];  // index 1 is beyond the populated region
    EXPECT_FALSE(r.hasValue());
}
// --------------------------------------------------------------------------------

/**
 * @test Verify that push_back() leaves the array non-empty after the first push
 */
TEST_F(ArrayInitTest, PushBackArrayIsNotEmptyAfterPush) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());

    arr->push_back(1);
    EXPECT_FALSE(arr->is_empty());
}

// ============================================================================
// push_front tests
// ============================================================================

/**
 * @test Verify that push_front() inserts a single int at index 0
 */
TEST_F(ArrayInitTest, PushFrontIntSingleElement) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());

    EXPECT_TRUE(arr->push_front(42));
    EXPECT_EQ(arr->size(), 1u);

    auto r = (*arr)[0];
    ASSERT_TRUE(r.hasValue());
    EXPECT_EQ(r.value(), 42);
}
// --------------------------------------------------------------------------------

/**
 * @test Verify that push_front() correctly shifts existing elements right
 *       and places the new element at index 0
 */
TEST_F(ArrayInitTest, PushFrontIntShiftsElements) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());

    arr->push_back(2);
    arr->push_back(3);
    EXPECT_TRUE(arr->push_front(1));

    EXPECT_EQ(arr->size(), 3u);
    EXPECT_EQ((*arr)[0].value(), 1);
    EXPECT_EQ((*arr)[1].value(), 2);
    EXPECT_EQ((*arr)[2].value(), 3);
}
// --------------------------------------------------------------------------------

/**
 * @test Verify that push_front() on a double array inserts and shifts correctly
 */
TEST_F(ArrayInitTest, PushFrontDoubleShiftsElements) {
    auto result = cslt::Array<double>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<double>, cslt::ArrayDeleter<double>> arr(result.value());

    arr->push_back(2.0);
    arr->push_back(3.0);
    EXPECT_TRUE(arr->push_front(1.0));

    EXPECT_EQ(arr->size(), 3u);
    EXPECT_DOUBLE_EQ((*arr)[0].value(), 1.0);
    EXPECT_DOUBLE_EQ((*arr)[1].value(), 2.0);
    EXPECT_DOUBLE_EQ((*arr)[2].value(), 3.0);
}
// --------------------------------------------------------------------------------

/**
 * @test Verify that push_front() on a Point array inserts and shifts correctly
 */
TEST_F(ArrayInitTest, PushFrontPointShiftsElements) {
    auto result = cslt::Array<Point>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<Point>, cslt::ArrayDeleter<Point>> arr(result.value());

    Point p2{2, 0};
    Point p3{3, 0};
    Point p1{1, 0};
    arr->push_back(p2);
    arr->push_back(p3);
    EXPECT_TRUE(arr->push_front(p1));

    EXPECT_EQ(arr->size(), 3u);
    EXPECT_EQ((*arr)[0].value(), p1);
    EXPECT_EQ((*arr)[1].value(), p2);
    EXPECT_EQ((*arr)[2].value(), p3);
}
// --------------------------------------------------------------------------------

/**
 * @test Verify that push_front() triggers growth when capacity is exceeded
 *       and all values remain in the correct order
 */
TEST_F(ArrayInitTest, PushFrontTriggersGrowth) {
    auto result = cslt::Array<int>::init(2, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());

    arr->push_back(3);
    arr->push_back(2);
    EXPECT_TRUE(arr->push_front(1));  // triggers growth

    EXPECT_EQ(arr->size(), 3u);
    EXPECT_GE(arr->capacity(), 3u);
    EXPECT_EQ((*arr)[0].value(), 1);
    EXPECT_EQ((*arr)[1].value(), 3);
    EXPECT_EQ((*arr)[2].value(), 2);
}
// --------------------------------------------------------------------------------

/**
 * @test Verify that repeated push_front() calls reverse the insertion order
 */
TEST_F(ArrayInitTest, PushFrontRepeatedReversesOrder) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());

    arr->push_front(3);
    arr->push_front(2);
    arr->push_front(1);

    EXPECT_EQ((*arr)[0].value(), 1);
    EXPECT_EQ((*arr)[1].value(), 2);
    EXPECT_EQ((*arr)[2].value(), 3);
}

// ============================================================================
// push_any tests
// ============================================================================

/**
 * @test Verify that push_any() at index 0 behaves identically to push_front()
 */
TEST_F(ArrayInitTest, PushAnyAtIndexZeroMatchesPushFront) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());

    arr->push_back(2);
    arr->push_back(3);
    EXPECT_TRUE(arr->push_any(0, 1));

    EXPECT_EQ(arr->size(), 3u);
    EXPECT_EQ((*arr)[0].value(), 1);
    EXPECT_EQ((*arr)[1].value(), 2);
    EXPECT_EQ((*arr)[2].value(), 3);
}
// --------------------------------------------------------------------------------

/**
 * @test Verify that push_any() at index == size() behaves identically to push_back()
 */
TEST_F(ArrayInitTest, PushAnyAtSizeMatchesPushBack) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());

    arr->push_back(1);
    arr->push_back(2);
    EXPECT_TRUE(arr->push_any(arr->size(), 3));

    EXPECT_EQ(arr->size(), 3u);
    EXPECT_EQ((*arr)[0].value(), 1);
    EXPECT_EQ((*arr)[1].value(), 2);
    EXPECT_EQ((*arr)[2].value(), 3);
}
// --------------------------------------------------------------------------------

/**
 * @test Verify that push_any() inserts correctly in the middle of an int array
 */
TEST_F(ArrayInitTest, PushAnyIntMiddleInsertion) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());

    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(4);
    EXPECT_TRUE(arr->push_any(2, 3));  // insert 3 between 2 and 4

    EXPECT_EQ(arr->size(), 4u);
    EXPECT_EQ((*arr)[0].value(), 1);
    EXPECT_EQ((*arr)[1].value(), 2);
    EXPECT_EQ((*arr)[2].value(), 3);
    EXPECT_EQ((*arr)[3].value(), 4);
}
// --------------------------------------------------------------------------------

/**
 * @test Verify that push_any() inserts correctly in the middle of a double array
 */
TEST_F(ArrayInitTest, PushAnyDoubleMiddleInsertion) {
    auto result = cslt::Array<double>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<double>, cslt::ArrayDeleter<double>> arr(result.value());

    arr->push_back(1.0);
    arr->push_back(2.0);
    arr->push_back(4.0);
    EXPECT_TRUE(arr->push_any(2, 3.0));

    EXPECT_EQ(arr->size(), 4u);
    EXPECT_DOUBLE_EQ((*arr)[0].value(), 1.0);
    EXPECT_DOUBLE_EQ((*arr)[1].value(), 2.0);
    EXPECT_DOUBLE_EQ((*arr)[2].value(), 3.0);
    EXPECT_DOUBLE_EQ((*arr)[3].value(), 4.0);
}
// --------------------------------------------------------------------------------

/**
 * @test Verify that push_any() inserts correctly in the middle of a Point array
 */
TEST_F(ArrayInitTest, PushAnyPointMiddleInsertion) {
    auto result = cslt::Array<Point>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<Point>, cslt::ArrayDeleter<Point>> arr(result.value());

    Point p1{1, 0}, p2{2, 0}, p3{3, 0}, p4{4, 0};
    arr->push_back(p1);
    arr->push_back(p2);
    arr->push_back(p4);
    EXPECT_TRUE(arr->push_any(2, p3));

    EXPECT_EQ(arr->size(), 4u);
    EXPECT_EQ((*arr)[0].value(), p1);
    EXPECT_EQ((*arr)[1].value(), p2);
    EXPECT_EQ((*arr)[2].value(), p3);
    EXPECT_EQ((*arr)[3].value(), p4);
}
// --------------------------------------------------------------------------------

/**
 * @test Verify that push_any() with an out-of-range index returns false
 *       and leaves the array unchanged
 */
TEST_F(ArrayInitTest, PushAnyOutOfRangeReturnsFalse) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());

    arr->push_back(1);
    arr->push_back(2);

    EXPECT_FALSE(arr->push_any(5, 99));  // index 5 > size 2
    EXPECT_EQ(arr->size(), 2u);          // array unchanged
}
// --------------------------------------------------------------------------------

/**
 * @test Verify that push_any() triggers growth when capacity is exceeded
 *       and values remain in the correct order after insertion
 */
TEST_F(ArrayInitTest, PushAnyTriggersGrowth) {
    auto result = cslt::Array<int>::init(2, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());

    arr->push_back(1);
    arr->push_back(3);
    EXPECT_TRUE(arr->push_any(1, 2));  // triggers growth, inserts 2 between 1 and 3

    EXPECT_EQ(arr->size(), 3u);
    EXPECT_GE(arr->capacity(), 3u);
    EXPECT_EQ((*arr)[0].value(), 1);
    EXPECT_EQ((*arr)[1].value(), 2);
    EXPECT_EQ((*arr)[2].value(), 3);
}
// ================================================================================
// ================================================================================
// eof
