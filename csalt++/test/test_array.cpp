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
// -------------------------------------------------------------------------------- 

// ================================================================================
// ================================================================================
 
// ============================================================================
// pop_back tests
// ============================================================================
 
/**
 * @test Verify that pop_back() removes the last int element and decrements size
 */
TEST_F(ArrayInitTest, PopBackIntRemovesLastElement) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
 
    EXPECT_TRUE(arr->pop_back());
    EXPECT_EQ(arr->size(), 2u);
    EXPECT_EQ((*arr)[0].value(), 1);
    EXPECT_EQ((*arr)[1].value(), 2);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that pop_back() removes the last double element correctly
 */
TEST_F(ArrayInitTest, PopBackDoubleRemovesLastElement) {
    auto result = cslt::Array<double>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<double>, cslt::ArrayDeleter<double>> arr(result.value());
 
    arr->push_back(1.1);
    arr->push_back(2.2);
    arr->push_back(3.3);
 
    EXPECT_TRUE(arr->pop_back());
    EXPECT_EQ(arr->size(), 2u);
    EXPECT_DOUBLE_EQ((*arr)[0].value(), 1.1);
    EXPECT_DOUBLE_EQ((*arr)[1].value(), 2.2);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that pop_back() removes the last Point element correctly
 */
TEST_F(ArrayInitTest, PopBackPointRemovesLastElement) {
    auto result = cslt::Array<Point>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<Point>, cslt::ArrayDeleter<Point>> arr(result.value());
 
    Point p1{1, 0}, p2{2, 0}, p3{3, 0};
    arr->push_back(p1);
    arr->push_back(p2);
    arr->push_back(p3);
 
    EXPECT_TRUE(arr->pop_back());
    EXPECT_EQ(arr->size(), 2u);
    EXPECT_EQ((*arr)[0].value(), p1);
    EXPECT_EQ((*arr)[1].value(), p2);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that repeated pop_back() calls drain the array to empty
 */
TEST_F(ArrayInitTest, PopBackDrainsToEmpty) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
 
    EXPECT_TRUE(arr->pop_back());
    EXPECT_TRUE(arr->pop_back());
    EXPECT_TRUE(arr->pop_back());
    EXPECT_EQ(arr->size(), 0u);
    EXPECT_TRUE(arr->is_empty());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that pop_back() on an empty array returns false
 */
TEST_F(ArrayInitTest, PopBackOnEmptyArrayReturnsFalse) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    EXPECT_FALSE(arr->pop_back());
    EXPECT_EQ(arr->size(), 0u);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that pop_back() does not reduce the buffer capacity
 */
TEST_F(ArrayInitTest, PopBackDoesNotReduceCapacity) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    size_t const cap_before = arr->capacity();
 
    arr->pop_back();
    EXPECT_EQ(arr->capacity(), cap_before);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that an element pushed after pop_back() lands at the correct index
 */
TEST_F(ArrayInitTest, PopBackThenPushBackRestoresElement) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->pop_back();
    arr->push_back(99);
 
    EXPECT_EQ(arr->size(), 2u);
    EXPECT_EQ((*arr)[0].value(), 1);
    EXPECT_EQ((*arr)[1].value(), 99);
}
 
// ============================================================================
// pop_front tests
// ============================================================================
 
/**
 * @test Verify that pop_front() removes the first int element and shifts
 *       remaining elements left
 */
TEST_F(ArrayInitTest, PopFrontIntRemovesFirstElement) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
 
    EXPECT_TRUE(arr->pop_front());
    EXPECT_EQ(arr->size(), 2u);
    EXPECT_EQ((*arr)[0].value(), 2);
    EXPECT_EQ((*arr)[1].value(), 3);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that pop_front() removes the first double element correctly
 */
TEST_F(ArrayInitTest, PopFrontDoubleRemovesFirstElement) {
    auto result = cslt::Array<double>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<double>, cslt::ArrayDeleter<double>> arr(result.value());
 
    arr->push_back(1.1);
    arr->push_back(2.2);
    arr->push_back(3.3);
 
    EXPECT_TRUE(arr->pop_front());
    EXPECT_EQ(arr->size(), 2u);
    EXPECT_DOUBLE_EQ((*arr)[0].value(), 2.2);
    EXPECT_DOUBLE_EQ((*arr)[1].value(), 3.3);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that pop_front() removes the first Point element correctly
 */
TEST_F(ArrayInitTest, PopFrontPointRemovesFirstElement) {
    auto result = cslt::Array<Point>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<Point>, cslt::ArrayDeleter<Point>> arr(result.value());
 
    Point p1{1, 0}, p2{2, 0}, p3{3, 0};
    arr->push_back(p1);
    arr->push_back(p2);
    arr->push_back(p3);
 
    EXPECT_TRUE(arr->pop_front());
    EXPECT_EQ(arr->size(), 2u);
    EXPECT_EQ((*arr)[0].value(), p2);
    EXPECT_EQ((*arr)[1].value(), p3);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that repeated pop_front() calls drain the array to empty
 */
TEST_F(ArrayInitTest, PopFrontDrainsToEmpty) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
 
    EXPECT_TRUE(arr->pop_front());
    EXPECT_TRUE(arr->pop_front());
    EXPECT_TRUE(arr->pop_front());
    EXPECT_EQ(arr->size(), 0u);
    EXPECT_TRUE(arr->is_empty());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that pop_front() on an empty array returns false
 */
TEST_F(ArrayInitTest, PopFrontOnEmptyArrayReturnsFalse) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    EXPECT_FALSE(arr->pop_front());
    EXPECT_EQ(arr->size(), 0u);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that pop_front() does not reduce the buffer capacity
 */
TEST_F(ArrayInitTest, PopFrontDoesNotReduceCapacity) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    size_t const cap_before = arr->capacity();
 
    arr->pop_front();
    EXPECT_EQ(arr->capacity(), cap_before);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that a single-element array is empty after pop_front()
 */
TEST_F(ArrayInitTest, PopFrontSingleElementLeavesArrayEmpty) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(42);
    EXPECT_TRUE(arr->pop_front());
    EXPECT_TRUE(arr->is_empty());
}
 
// ============================================================================
// pop_any tests
// ============================================================================
 
/**
 * @test Verify that pop_any() at index 0 behaves identically to pop_front()
 */
TEST_F(ArrayInitTest, PopAnyAtIndexZeroMatchesPopFront) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
 
    EXPECT_TRUE(arr->pop_any(0));
    EXPECT_EQ(arr->size(), 2u);
    EXPECT_EQ((*arr)[0].value(), 2);
    EXPECT_EQ((*arr)[1].value(), 3);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that pop_any() at index == size()-1 behaves identically to pop_back()
 */
TEST_F(ArrayInitTest, PopAnyAtLastIndexMatchesPopBack) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
 
    EXPECT_TRUE(arr->pop_any(arr->size() - 1u));
    EXPECT_EQ(arr->size(), 2u);
    EXPECT_EQ((*arr)[0].value(), 1);
    EXPECT_EQ((*arr)[1].value(), 2);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that pop_any() removes the correct middle int element and
 *       shifts remaining elements left
 */
TEST_F(ArrayInitTest, PopAnyIntMiddleRemoval) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
    arr->push_back(4);
 
    EXPECT_TRUE(arr->pop_any(1));  // remove 2
    EXPECT_EQ(arr->size(), 3u);
    EXPECT_EQ((*arr)[0].value(), 1);
    EXPECT_EQ((*arr)[1].value(), 3);
    EXPECT_EQ((*arr)[2].value(), 4);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that pop_any() removes the correct middle double element
 */
TEST_F(ArrayInitTest, PopAnyDoubleMiddleRemoval) {
    auto result = cslt::Array<double>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<double>, cslt::ArrayDeleter<double>> arr(result.value());
 
    arr->push_back(1.0);
    arr->push_back(2.0);
    arr->push_back(3.0);
    arr->push_back(4.0);
 
    EXPECT_TRUE(arr->pop_any(2));  // remove 3.0
    EXPECT_EQ(arr->size(), 3u);
    EXPECT_DOUBLE_EQ((*arr)[0].value(), 1.0);
    EXPECT_DOUBLE_EQ((*arr)[1].value(), 2.0);
    EXPECT_DOUBLE_EQ((*arr)[2].value(), 4.0);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that pop_any() removes the correct middle Point element
 */
TEST_F(ArrayInitTest, PopAnyPointMiddleRemoval) {
    auto result = cslt::Array<Point>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<Point>, cslt::ArrayDeleter<Point>> arr(result.value());
 
    Point p1{1, 0}, p2{2, 0}, p3{3, 0}, p4{4, 0};
    arr->push_back(p1);
    arr->push_back(p2);
    arr->push_back(p3);
    arr->push_back(p4);
 
    EXPECT_TRUE(arr->pop_any(2));  // remove p3
    EXPECT_EQ(arr->size(), 3u);
    EXPECT_EQ((*arr)[0].value(), p1);
    EXPECT_EQ((*arr)[1].value(), p2);
    EXPECT_EQ((*arr)[2].value(), p4);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that pop_any() on an empty array returns false
 */
TEST_F(ArrayInitTest, PopAnyOnEmptyArrayReturnsFalse) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    EXPECT_FALSE(arr->pop_any(0));
    EXPECT_EQ(arr->size(), 0u);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that pop_any() with an out-of-range index returns false and
 *       leaves the array unchanged
 */
TEST_F(ArrayInitTest, PopAnyOutOfRangeReturnsFalse) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
 
    EXPECT_FALSE(arr->pop_any(5));  // index 5 >= size 2
    EXPECT_EQ(arr->size(), 2u);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that pop_any() does not reduce the buffer capacity
 */
TEST_F(ArrayInitTest, PopAnyDoesNotReduceCapacity) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
    size_t const cap_before = arr->capacity();
 
    arr->pop_any(1);
    EXPECT_EQ(arr->capacity(), cap_before);
}
// -------------------------------------------------------------------------------- 

// ============================================================================
// set() tests
// ============================================================================
 
/**
 * @test Verify that set() overwrites an existing int element at a valid index
 */
TEST_F(ArrayInitTest, SetIntOverwritesExistingElement) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
 
    auto r = arr->set(1, 99);
    ASSERT_TRUE(r.hasValue());
    EXPECT_TRUE(r.value());
    EXPECT_EQ(arr->size(), 3u);
    EXPECT_EQ((*arr)[1].value(), 99);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that set() overwrites an existing double element correctly
 */
TEST_F(ArrayInitTest, SetDoubleOverwritesExistingElement) {
    auto result = cslt::Array<double>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<double>, cslt::ArrayDeleter<double>> arr(result.value());
 
    arr->push_back(1.0);
    arr->push_back(2.0);
    arr->push_back(3.0);
 
    auto r = arr->set(0, 9.9);
    ASSERT_TRUE(r.hasValue());
    EXPECT_DOUBLE_EQ((*arr)[0].value(), 9.9);
    EXPECT_DOUBLE_EQ((*arr)[1].value(), 2.0);
    EXPECT_DOUBLE_EQ((*arr)[2].value(), 3.0);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that set() overwrites an existing Point element correctly
 */
TEST_F(ArrayInitTest, SetPointOverwritesExistingElement) {
    auto result = cslt::Array<Point>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<Point>, cslt::ArrayDeleter<Point>> arr(result.value());
 
    Point p1{1, 0}, p2{2, 0}, p3{3, 0}, p99{99, 99};
    arr->push_back(p1);
    arr->push_back(p2);
    arr->push_back(p3);
 
    auto r = arr->set(2, p99);
    ASSERT_TRUE(r.hasValue());
    EXPECT_EQ((*arr)[0].value(), p1);
    EXPECT_EQ((*arr)[1].value(), p2);
    EXPECT_EQ((*arr)[2].value(), p99);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that set() at index == size() appends a new int element
 *       and increments size
 */
TEST_F(ArrayInitTest, SetIntAtSizeAppendsElement) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
 
    auto r = arr->set(arr->size(), 3);
    ASSERT_TRUE(r.hasValue());
    EXPECT_EQ(arr->size(), 3u);
    EXPECT_EQ((*arr)[2].value(), 3);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that set() at index == size() triggers growth when the buffer
 *       is full and appends the element correctly
 */
TEST_F(ArrayInitTest, SetIntAtSizeTriggersGrowth) {
    auto result = cslt::Array<int>::init(2, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);  // now full
 
    auto r = arr->set(arr->size(), 3);
    ASSERT_TRUE(r.hasValue());
    EXPECT_EQ(arr->size(), 3u);
    EXPECT_GE(arr->capacity(), 3u);
    EXPECT_EQ((*arr)[0].value(), 1);
    EXPECT_EQ((*arr)[1].value(), 2);
    EXPECT_EQ((*arr)[2].value(), 3);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that set() does not alter neighbouring elements when
 *       overwriting a middle element
 */
TEST_F(ArrayInitTest, SetIntDoesNotAlterNeighbours) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(10);
    arr->push_back(20);
    arr->push_back(30);
 
    arr->set(1, 99);
 
    EXPECT_EQ((*arr)[0].value(), 10);
    EXPECT_EQ((*arr)[1].value(), 99);
    EXPECT_EQ((*arr)[2].value(), 30);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that set() returns an OutOfBoundsError when index > size()
 */
TEST_F(ArrayInitTest, SetOutOfRangeReturnsError) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
 
    auto r = arr->set(5, 99);  // index 5 > size 2
    EXPECT_FALSE(r.hasValue());
    EXPECT_EQ(arr->size(), 2u);  // array unchanged
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that set() does not change size when overwriting an existing element
 */
TEST_F(ArrayInitTest, SetOverwriteDoesNotChangeSize) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
 
    arr->set(0, 99);
    EXPECT_EQ(arr->size(), 3u);
}
 
// ============================================================================
// operator[] (const read overload) tests
// ============================================================================
 
/**
 * @test Verify that operator[] returns the correct int value at each valid index
 */
TEST_F(ArrayInitTest, OperatorBracketIntReturnsCorrectValues) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(10);
    arr->push_back(20);
    arr->push_back(30);
 
    EXPECT_EQ((*arr)[0].value(), 10);
    EXPECT_EQ((*arr)[1].value(), 20);
    EXPECT_EQ((*arr)[2].value(), 30);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that operator[] returns the correct double value at each valid index
 */
TEST_F(ArrayInitTest, OperatorBracketDoubleReturnsCorrectValues) {
    auto result = cslt::Array<double>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<double>, cslt::ArrayDeleter<double>> arr(result.value());
 
    arr->push_back(1.1);
    arr->push_back(2.2);
    arr->push_back(3.3);
 
    EXPECT_DOUBLE_EQ((*arr)[0].value(), 1.1);
    EXPECT_DOUBLE_EQ((*arr)[1].value(), 2.2);
    EXPECT_DOUBLE_EQ((*arr)[2].value(), 3.3);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that operator[] returns the correct Point value at each valid index
 */
TEST_F(ArrayInitTest, OperatorBracketPointReturnsCorrectValues) {
    auto result = cslt::Array<Point>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<Point>, cslt::ArrayDeleter<Point>> arr(result.value());
 
    Point p1{1, 2}, p2{3, 4}, p3{5, 6};
    arr->push_back(p1);
    arr->push_back(p2);
    arr->push_back(p3);
 
    EXPECT_EQ((*arr)[0].value(), p1);
    EXPECT_EQ((*arr)[1].value(), p2);
    EXPECT_EQ((*arr)[2].value(), p3);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that operator[] returns an error when index equals size()
 */
TEST_F(ArrayInitTest, OperatorBracketAtSizeReturnsError) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
 
    auto r = (*arr)[arr->size()];
    EXPECT_FALSE(r.hasValue());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that operator[] returns an error on a well out-of-range index
 */
TEST_F(ArrayInitTest, OperatorBracketFarOutOfRangeReturnsError) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
 
    auto r = (*arr)[100];
    EXPECT_FALSE(r.hasValue());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that operator[] returns an error on any index when the array
 *       is empty
 */
TEST_F(ArrayInitTest, OperatorBracketOnEmptyArrayReturnsError) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    auto r = (*arr)[0];
    EXPECT_FALSE(r.hasValue());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that operator[] reflects an updated value after a set() call
 */
TEST_F(ArrayInitTest, OperatorBracketReflectsSetUpdate) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
 
    arr->set(1, 99);
    EXPECT_EQ((*arr)[1].value(), 99);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that operator[] at each index returns correct values after a
 *       series of push_front() calls
 */
TEST_F(ArrayInitTest, OperatorBracketAfterPushFrontReturnsCorrectOrder) {
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
// -------------------------------------------------------------------------------- 

// ============================================================================
// is_empty() tests
// ============================================================================
 
/**
 * @test Verify that a freshly initialised int array reports is_empty() == true
 */
TEST_F(ArrayInitTest, IsEmptyTrueOnFreshIntArray) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    EXPECT_TRUE(arr->is_empty());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that is_empty() returns false after a push_back()
 */
TEST_F(ArrayInitTest, IsEmptyFalseAfterPushBack) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    EXPECT_FALSE(arr->is_empty());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that is_empty() returns true after all elements are popped
 */
TEST_F(ArrayInitTest, IsEmptyTrueAfterAllElementsPopped) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->pop_back();
    arr->pop_back();
 
    EXPECT_TRUE(arr->is_empty());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that is_empty() returns true after clear()
 */
TEST_F(ArrayInitTest, IsEmptyTrueAfterClear) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->clear();
 
    EXPECT_TRUE(arr->is_empty());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that is_empty() works correctly for double arrays
 */
TEST_F(ArrayInitTest, IsEmptyDoubleArray) {
    auto result = cslt::Array<double>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<double>, cslt::ArrayDeleter<double>> arr(result.value());
 
    EXPECT_TRUE(arr->is_empty());
    arr->push_back(1.0);
    EXPECT_FALSE(arr->is_empty());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that is_empty() works correctly for Point arrays
 */
TEST_F(ArrayInitTest, IsEmptyPointArray) {
    auto result = cslt::Array<Point>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<Point>, cslt::ArrayDeleter<Point>> arr(result.value());
 
    EXPECT_TRUE(arr->is_empty());
    arr->push_back({1, 2});
    EXPECT_FALSE(arr->is_empty());
}
 
// ============================================================================
// is_full() tests
// ============================================================================
 
/**
 * @test Verify that is_full() returns false on a freshly initialised array
 */
TEST_F(ArrayInitTest, IsFullFalseOnFreshArray) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    EXPECT_FALSE(arr->is_full());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that is_full() returns true when size equals capacity for int
 */
TEST_F(ArrayInitTest, IsFullTrueWhenIntArrayAtCapacity) {
    auto result = cslt::Array<int>::init(3, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
 
    EXPECT_TRUE(arr->is_full());
    EXPECT_EQ(arr->size(), arr->capacity());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that is_full() returns true when size equals capacity for double
 */
TEST_F(ArrayInitTest, IsFullTrueWhenDoubleArrayAtCapacity) {
    auto result = cslt::Array<double>::init(2, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<double>, cslt::ArrayDeleter<double>> arr(result.value());
 
    arr->push_back(1.0);
    arr->push_back(2.0);
 
    EXPECT_TRUE(arr->is_full());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that is_full() returns true when size equals capacity for Point
 */
TEST_F(ArrayInitTest, IsFullTrueWhenPointArrayAtCapacity) {
    auto result = cslt::Array<Point>::init(2, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<Point>, cslt::ArrayDeleter<Point>> arr(result.value());
 
    arr->push_back({1, 2});
    arr->push_back({3, 4});
 
    EXPECT_TRUE(arr->is_full());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that is_full() returns false after a pop reduces size below capacity
 */
TEST_F(ArrayInitTest, IsFullFalseAfterPopBack) {
    auto result = cslt::Array<int>::init(2, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    EXPECT_TRUE(arr->is_full());
 
    arr->pop_back();
    EXPECT_FALSE(arr->is_full());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that is_full() returns false after growth — push_back()
 *       beyond capacity grows the buffer so size < new capacity
 */
TEST_F(ArrayInitTest, IsFullFalseAfterGrowth) {
    auto result = cslt::Array<int>::init(2, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);  // triggers growth: capacity doubles to 4, size is 3
 
    EXPECT_FALSE(arr->is_full());
    EXPECT_GT(arr->capacity(), arr->size());
}
 
// ============================================================================
// is_ptr() tests
// ============================================================================
 
/**
 * @test Verify that is_ptr() returns true for a pointer to the first element
 */
TEST_F(ArrayInitTest, IsPtrTrueForFirstElement) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
 
    const int* p = arr->data();
    EXPECT_TRUE(arr->is_ptr(p));
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that is_ptr() returns true for a pointer to a middle element
 */
TEST_F(ArrayInitTest, IsPtrTrueForMiddleElement) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(10);
    arr->push_back(20);
    arr->push_back(30);
 
    const int* p = arr->data() + 1;
    EXPECT_TRUE(arr->is_ptr(p));
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that is_ptr() returns true for a pointer to the last element
 */
TEST_F(ArrayInitTest, IsPtrTrueForLastElement) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
 
    const int* p = arr->data() + 2;  // last populated element
    EXPECT_TRUE(arr->is_ptr(p));
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that is_ptr() returns false for a pointer one past the last
 *       populated element (i.e. data() + size())
 */
TEST_F(ArrayInitTest, IsPtrFalseForOnePastEnd) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
 
    const int* p = arr->data() + arr->size();  // one past populated region
    EXPECT_FALSE(arr->is_ptr(p));
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that is_ptr() returns false for a pointer before the buffer
 */
TEST_F(ArrayInitTest, IsPtrFalseForPointerBeforeBuffer) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
 
    const int* p = arr->data() - 1;
    EXPECT_FALSE(arr->is_ptr(p));
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that is_ptr() returns false for a nullptr
 */
TEST_F(ArrayInitTest, IsPtrFalseForNullptr) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
 
    EXPECT_FALSE(arr->is_ptr(nullptr));
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that is_ptr() returns false when the array is empty
 */
TEST_F(ArrayInitTest, IsPtrFalseWhenArrayIsEmpty) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    const int* p = arr->data();
    EXPECT_FALSE(arr->is_ptr(p));
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that is_ptr() returns true for all valid element pointers
 *       in a Point array
 */
TEST_F(ArrayInitTest, IsPtrTrueForAllPointElements) {
    auto result = cslt::Array<Point>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<Point>, cslt::ArrayDeleter<Point>> arr(result.value());
 
    arr->push_back({1, 1});
    arr->push_back({2, 2});
    arr->push_back({3, 3});
 
    for (size_t i = 0u; i < arr->size(); ++i) {
        EXPECT_TRUE(arr->is_ptr(arr->data() + i));
    }
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that is_ptr() returns false for an unaligned byte offset
 *       within the buffer (pointer is inside the buffer but not on an
 *       element boundary)
 */
TEST_F(ArrayInitTest, IsPtrFalseForUnalignedPointer) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
 
    // Advance by one byte into the buffer — not aligned to sizeof(int)
    const char* byte_ptr = reinterpret_cast<const char*>(arr->data()) + 1;
    const int*  unaligned = reinterpret_cast<const int*>(byte_ptr);
    EXPECT_FALSE(arr->is_ptr(unaligned));
}
// -------------------------------------------------------------------------------- 

// ============================================================================
// cumulative() tests
// ============================================================================
 
/**
 * @test Verify that cumulative() produces the correct prefix sum for an int
 *       array using a caller-supplied allocator
 */
TEST_F(ArrayInitTest, CumulativeIntSumWithAllocator) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
    arr->push_back(4);
 
    auto cr = cslt::Array<int>::cumulative(
        *arr,
        [](int& accum, const int& elem) { accum += elem; },
        alloc);
 
    ASSERT_TRUE(cr.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> cum(cr.value());
 
    EXPECT_EQ(cum->size(), 4u);
    EXPECT_EQ((*cum)[0].value(), 1);   // 1
    EXPECT_EQ((*cum)[1].value(), 3);   // 1+2
    EXPECT_EQ((*cum)[2].value(), 6);   // 1+2+3
    EXPECT_EQ((*cum)[3].value(), 10);  // 1+2+3+4
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that cumulative() produces the correct prefix sum using the
 *       source array's own allocator (single-argument overload)
 */
TEST_F(ArrayInitTest, CumulativeIntSumWithSourceAllocator) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
    arr->push_back(4);
 
    auto cr = cslt::Array<int>::cumulative(
        *arr,
        [](int& accum, const int& elem) { accum += elem; });
 
    ASSERT_TRUE(cr.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> cum(cr.value());
 
    EXPECT_EQ(cum->size(), 4u);
    EXPECT_EQ((*cum)[0].value(), 1);
    EXPECT_EQ((*cum)[1].value(), 3);
    EXPECT_EQ((*cum)[2].value(), 6);
    EXPECT_EQ((*cum)[3].value(), 10);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that cumulative() produces the correct prefix product for an
 *       int array (exercises a non-additive callable)
 */
TEST_F(ArrayInitTest, CumulativeIntProduct) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
    arr->push_back(4);
 
    auto cr = cslt::Array<int>::cumulative(
        *arr,
        [](int& accum, const int& elem) { accum *= elem; },
        alloc);
 
    ASSERT_TRUE(cr.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> cum(cr.value());
 
    EXPECT_EQ(cum->size(), 4u);
    EXPECT_EQ((*cum)[0].value(), 1);   // 1
    EXPECT_EQ((*cum)[1].value(), 2);   // 1*2
    EXPECT_EQ((*cum)[2].value(), 6);   // 1*2*3
    EXPECT_EQ((*cum)[3].value(), 24);  // 1*2*3*4
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that cumulative() produces the correct prefix sum for a double
 *       array
 */
TEST_F(ArrayInitTest, CumulativeDoubleSum) {
    auto result = cslt::Array<double>::init(3, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<double>, cslt::ArrayDeleter<double>> arr(result.value());
 
    arr->push_back(1.5);
    arr->push_back(2.5);
    arr->push_back(1.0);
 
    auto cr = cslt::Array<double>::cumulative(
        *arr,
        [](double& accum, const double& elem) { accum += elem; },
        alloc);
 
    ASSERT_TRUE(cr.hasValue());
    cslt::UniquePtr<cslt::Array<double>, cslt::ArrayDeleter<double>> cum(cr.value());
 
    EXPECT_EQ(cum->size(), 3u);
    EXPECT_DOUBLE_EQ((*cum)[0].value(), 1.5);
    EXPECT_DOUBLE_EQ((*cum)[1].value(), 4.0);
    EXPECT_DOUBLE_EQ((*cum)[2].value(), 5.0);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that cumulative() on a single-element array returns a
 *       single-element result equal to the seed
 */
TEST_F(ArrayInitTest, CumulativeSingleElementReturnsSeed) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(42);
 
    auto cr = cslt::Array<int>::cumulative(
        *arr,
        [](int& accum, const int& elem) { accum += elem; },
        alloc);
 
    ASSERT_TRUE(cr.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> cum(cr.value());
 
    EXPECT_EQ(cum->size(), 1u);
    EXPECT_EQ((*cum)[0].value(), 42);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that the result of cumulative() has capacity equal to the
 *       source size (fixed-length snapshot with no extra room)
 */
TEST_F(ArrayInitTest, CumulativeResultCapacityEqualsSourceSize) {
    auto result = cslt::Array<int>::init(8, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
 
    auto cr = cslt::Array<int>::cumulative(
        *arr,
        [](int& accum, const int& elem) { accum += elem; },
        alloc);
 
    ASSERT_TRUE(cr.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> cum(cr.value());
 
    // Capacity must equal exactly the number of source elements
    EXPECT_EQ(cum->capacity(), arr->size());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that cumulative() does not modify the source array
 */
TEST_F(ArrayInitTest, CumulativeDoesNotModifySource) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
 
    auto cr = cslt::Array<int>::cumulative(
        *arr,
        [](int& accum, const int& elem) { accum += elem; },
        alloc);
    ASSERT_TRUE(cr.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> cum(cr.value());
 
    // Source must be unchanged
    EXPECT_EQ(arr->size(), 3u);
    EXPECT_EQ((*arr)[0].value(), 1);
    EXPECT_EQ((*arr)[1].value(), 2);
    EXPECT_EQ((*arr)[2].value(), 3);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that cumulative() on an empty array returns an error
 */
TEST_F(ArrayInitTest, CumulativeEmptyArrayReturnsError) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    auto cr = cslt::Array<int>::cumulative(
        *arr,
        [](int& accum, const int& elem) { accum += elem; },
        alloc);
 
    EXPECT_FALSE(cr.hasValue());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that the two cumulative() overloads produce identical results
 *       for the same source and callable
 */
TEST_F(ArrayInitTest, CumulativeBothOverloadsProduceSameResult) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
    arr->push_back(4);
 
    auto add = [](int& accum, const int& elem) { accum += elem; };
 
    auto cr1 = cslt::Array<int>::cumulative(*arr, add, alloc);
    auto cr2 = cslt::Array<int>::cumulative(*arr, add);
 
    ASSERT_TRUE(cr1.hasValue());
    ASSERT_TRUE(cr2.hasValue());
 
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> cum1(cr1.value());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> cum2(cr2.value());
 
    ASSERT_EQ(cum1->size(), cum2->size());
    for (size_t i = 0u; i < cum1->size(); ++i) {
        EXPECT_EQ((*cum1)[i].value(), (*cum2)[i].value());
    }
}
// -------------------------------------------------------------------------------- 

// ============================================================================
// slice() tests
// ============================================================================
 
/**
 * @test Verify that slice() returns the correct elements for a middle range
 *       of an int array using a caller-supplied allocator
 */
TEST_F(ArrayInitTest, SliceIntMiddleRangeWithAllocator) {
    auto result = cslt::Array<int>::init(6, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(10);
    arr->push_back(20);
    arr->push_back(30);
    arr->push_back(40);
    arr->push_back(50);
 
    // slice [1, 4) -> {20, 30, 40}
    auto sr = cslt::Array<int>::slice(*arr, 1, 4, alloc);
    ASSERT_TRUE(sr.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> sl(sr.value());
 
    EXPECT_EQ(sl->size(), 3u);
    EXPECT_EQ((*sl)[0].value(), 20);
    EXPECT_EQ((*sl)[1].value(), 30);
    EXPECT_EQ((*sl)[2].value(), 40);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that slice() returns the correct elements using the source
 *       array's own allocator (single-argument overload)
 */
TEST_F(ArrayInitTest, SliceIntMiddleRangeWithSourceAllocator) {
    auto result = cslt::Array<int>::init(6, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(10);
    arr->push_back(20);
    arr->push_back(30);
    arr->push_back(40);
    arr->push_back(50);
 
    auto sr = cslt::Array<int>::slice(*arr, 1, 4);
    ASSERT_TRUE(sr.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> sl(sr.value());
 
    EXPECT_EQ(sl->size(), 3u);
    EXPECT_EQ((*sl)[0].value(), 20);
    EXPECT_EQ((*sl)[1].value(), 30);
    EXPECT_EQ((*sl)[2].value(), 40);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that slice() from index 0 to size() returns a full copy of
 *       the source array
 */
TEST_F(ArrayInitTest, SliceIntFullRange) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
    arr->push_back(4);
 
    auto sr = cslt::Array<int>::slice(*arr, 0, 4, alloc);
    ASSERT_TRUE(sr.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> sl(sr.value());
 
    EXPECT_EQ(sl->size(), 4u);
    EXPECT_EQ((*sl)[0].value(), 1);
    EXPECT_EQ((*sl)[1].value(), 2);
    EXPECT_EQ((*sl)[2].value(), 3);
    EXPECT_EQ((*sl)[3].value(), 4);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that slice() of a single element returns a one-element array
 */
TEST_F(ArrayInitTest, SliceIntSingleElement) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(10);
    arr->push_back(20);
    arr->push_back(30);
 
    // slice [1, 2) -> {20}
    auto sr = cslt::Array<int>::slice(*arr, 1, 2, alloc);
    ASSERT_TRUE(sr.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> sl(sr.value());
 
    EXPECT_EQ(sl->size(), 1u);
    EXPECT_EQ((*sl)[0].value(), 20);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that slice() from the start of the array produces the correct
 *       prefix
 */
TEST_F(ArrayInitTest, SliceIntFromStart) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
    arr->push_back(4);
 
    // slice [0, 2) -> {1, 2}
    auto sr = cslt::Array<int>::slice(*arr, 0, 2, alloc);
    ASSERT_TRUE(sr.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> sl(sr.value());
 
    EXPECT_EQ(sl->size(), 2u);
    EXPECT_EQ((*sl)[0].value(), 1);
    EXPECT_EQ((*sl)[1].value(), 2);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that slice() to the end of the array produces the correct suffix
 */
TEST_F(ArrayInitTest, SliceIntToEnd) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
    arr->push_back(4);
 
    // slice [2, 4) -> {3, 4}
    auto sr = cslt::Array<int>::slice(*arr, 2, 4, alloc);
    ASSERT_TRUE(sr.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> sl(sr.value());
 
    EXPECT_EQ(sl->size(), 2u);
    EXPECT_EQ((*sl)[0].value(), 3);
    EXPECT_EQ((*sl)[1].value(), 4);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that slice() produces correct results for a double array
 */
TEST_F(ArrayInitTest, SliceDoubleMiddleRange) {
    auto result = cslt::Array<double>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<double>, cslt::ArrayDeleter<double>> arr(result.value());
 
    arr->push_back(1.1);
    arr->push_back(2.2);
    arr->push_back(3.3);
    arr->push_back(4.4);
 
    // slice [1, 3) -> {2.2, 3.3}
    auto sr = cslt::Array<double>::slice(*arr, 1, 3, alloc);
    ASSERT_TRUE(sr.hasValue());
    cslt::UniquePtr<cslt::Array<double>, cslt::ArrayDeleter<double>> sl(sr.value());
 
    EXPECT_EQ(sl->size(), 2u);
    EXPECT_DOUBLE_EQ((*sl)[0].value(), 2.2);
    EXPECT_DOUBLE_EQ((*sl)[1].value(), 3.3);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that slice() produces correct results for a Point array
 */
TEST_F(ArrayInitTest, SlicePointMiddleRange) {
    auto result = cslt::Array<Point>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<Point>, cslt::ArrayDeleter<Point>> arr(result.value());
 
    Point p1{1, 0}, p2{2, 0}, p3{3, 0}, p4{4, 0};
    arr->push_back(p1);
    arr->push_back(p2);
    arr->push_back(p3);
    arr->push_back(p4);
 
    // slice [1, 3) -> {p2, p3}
    auto sr = cslt::Array<Point>::slice(*arr, 1, 3, alloc);
    ASSERT_TRUE(sr.hasValue());
    cslt::UniquePtr<cslt::Array<Point>, cslt::ArrayDeleter<Point>> sl(sr.value());
 
    EXPECT_EQ(sl->size(), 2u);
    EXPECT_EQ((*sl)[0].value(), p2);
    EXPECT_EQ((*sl)[1].value(), p3);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that the slice result has capacity equal to the slice length
 *       (fixed-length snapshot)
 */
TEST_F(ArrayInitTest, SliceResultCapacityEqualsSliceLength) {
    auto result = cslt::Array<int>::init(8, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
    arr->push_back(4);
    arr->push_back(5);
 
    // slice [1, 4) -> 3 elements
    auto sr = cslt::Array<int>::slice(*arr, 1, 4, alloc);
    ASSERT_TRUE(sr.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> sl(sr.value());
 
    EXPECT_EQ(sl->capacity(), 3u);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that slice() does not modify the source array
 */
TEST_F(ArrayInitTest, SliceDoesNotModifySource) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
    arr->push_back(4);
 
    auto sr = cslt::Array<int>::slice(*arr, 1, 3, alloc);
    ASSERT_TRUE(sr.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> sl(sr.value());
 
    EXPECT_EQ(arr->size(), 4u);
    EXPECT_EQ((*arr)[0].value(), 1);
    EXPECT_EQ((*arr)[1].value(), 2);
    EXPECT_EQ((*arr)[2].value(), 3);
    EXPECT_EQ((*arr)[3].value(), 4);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that the two slice() overloads produce identical results
 */
TEST_F(ArrayInitTest, SliceBothOverloadsProduceSameResult) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
    arr->push_back(4);
 
    auto sr1 = cslt::Array<int>::slice(*arr, 1, 3, alloc);
    auto sr2 = cslt::Array<int>::slice(*arr, 1, 3);
 
    ASSERT_TRUE(sr1.hasValue());
    ASSERT_TRUE(sr2.hasValue());
 
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> sl1(sr1.value());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> sl2(sr2.value());
 
    ASSERT_EQ(sl1->size(), sl2->size());
    for (size_t i = 0u; i < sl1->size(); ++i) {
        EXPECT_EQ((*sl1)[i].value(), (*sl2)[i].value());
    }
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that slice() returns an error when start >= end
 */
TEST_F(ArrayInitTest, SliceStartEqualToEndReturnsError) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
 
    auto sr = cslt::Array<int>::slice(*arr, 2, 2, alloc);  // start == end
    EXPECT_FALSE(sr.hasValue());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that slice() returns an error when start > end
 */
TEST_F(ArrayInitTest, SliceStartGreaterThanEndReturnsError) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
 
    auto sr = cslt::Array<int>::slice(*arr, 3, 1, alloc);  // start > end
    EXPECT_FALSE(sr.hasValue());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that slice() returns an error when end exceeds size()
 */
TEST_F(ArrayInitTest, SliceEndExceedsSizeReturnsError) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
 
    auto sr = cslt::Array<int>::slice(*arr, 0, 10, alloc);  // end > size
    EXPECT_FALSE(sr.hasValue());
}
// -------------------------------------------------------------------------------- 

// ============================================================================
// concat() tests
// ============================================================================
 
/**
 * @test Verify that concat() appends all int elements of another array
 *       in the correct order
 */
TEST_F(ArrayInitTest, ConcatIntAppendsCorrectly) {
    auto ra = cslt::Array<int>::init(4, alloc);
    auto rb = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(ra.hasValue());
    ASSERT_TRUE(rb.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> a(ra.value());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> b(rb.value());
 
    a->push_back(1);
    a->push_back(2);
    a->push_back(3);
    b->push_back(4);
    b->push_back(5);
    b->push_back(6);
 
    EXPECT_TRUE(a->concat(*b));
 
    EXPECT_EQ(a->size(), 6u);
    EXPECT_EQ((*a)[0].value(), 1);
    EXPECT_EQ((*a)[1].value(), 2);
    EXPECT_EQ((*a)[2].value(), 3);
    EXPECT_EQ((*a)[3].value(), 4);
    EXPECT_EQ((*a)[4].value(), 5);
    EXPECT_EQ((*a)[5].value(), 6);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that concat() appends all double elements correctly
 */
TEST_F(ArrayInitTest, ConcatDoubleAppendsCorrectly) {
    auto ra = cslt::Array<double>::init(4, alloc);
    auto rb = cslt::Array<double>::init(4, alloc);
    ASSERT_TRUE(ra.hasValue());
    ASSERT_TRUE(rb.hasValue());
    cslt::UniquePtr<cslt::Array<double>, cslt::ArrayDeleter<double>> a(ra.value());
    cslt::UniquePtr<cslt::Array<double>, cslt::ArrayDeleter<double>> b(rb.value());
 
    a->push_back(1.1);
    a->push_back(2.2);
    b->push_back(3.3);
    b->push_back(4.4);
 
    EXPECT_TRUE(a->concat(*b));
 
    EXPECT_EQ(a->size(), 4u);
    EXPECT_DOUBLE_EQ((*a)[0].value(), 1.1);
    EXPECT_DOUBLE_EQ((*a)[1].value(), 2.2);
    EXPECT_DOUBLE_EQ((*a)[2].value(), 3.3);
    EXPECT_DOUBLE_EQ((*a)[3].value(), 4.4);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that concat() appends all Point elements correctly
 */
TEST_F(ArrayInitTest, ConcatPointAppendsCorrectly) {
    auto ra = cslt::Array<Point>::init(4, alloc);
    auto rb = cslt::Array<Point>::init(4, alloc);
    ASSERT_TRUE(ra.hasValue());
    ASSERT_TRUE(rb.hasValue());
    cslt::UniquePtr<cslt::Array<Point>, cslt::ArrayDeleter<Point>> a(ra.value());
    cslt::UniquePtr<cslt::Array<Point>, cslt::ArrayDeleter<Point>> b(rb.value());
 
    Point p1{1, 0}, p2{2, 0}, p3{3, 0}, p4{4, 0};
    a->push_back(p1);
    a->push_back(p2);
    b->push_back(p3);
    b->push_back(p4);
 
    EXPECT_TRUE(a->concat(*b));
 
    EXPECT_EQ(a->size(), 4u);
    EXPECT_EQ((*a)[0].value(), p1);
    EXPECT_EQ((*a)[1].value(), p2);
    EXPECT_EQ((*a)[2].value(), p3);
    EXPECT_EQ((*a)[3].value(), p4);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that concat() onto an empty destination produces a copy of
 *       the source
 */
TEST_F(ArrayInitTest, ConcatOntoEmptyArray) {
    auto ra = cslt::Array<int>::init(4, alloc);
    auto rb = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(ra.hasValue());
    ASSERT_TRUE(rb.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> a(ra.value());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> b(rb.value());
 
    b->push_back(1);
    b->push_back(2);
    b->push_back(3);
 
    EXPECT_TRUE(a->concat(*b));
 
    EXPECT_EQ(a->size(), 3u);
    EXPECT_EQ((*a)[0].value(), 1);
    EXPECT_EQ((*a)[1].value(), 2);
    EXPECT_EQ((*a)[2].value(), 3);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that concat() with an empty source leaves the destination
 *       unchanged and returns true
 */
TEST_F(ArrayInitTest, ConcatEmptySourceReturnsTrue) {
    auto ra = cslt::Array<int>::init(4, alloc);
    auto rb = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(ra.hasValue());
    ASSERT_TRUE(rb.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> a(ra.value());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> b(rb.value());
 
    a->push_back(1);
    a->push_back(2);
 
    EXPECT_TRUE(a->concat(*b));  // b is empty
 
    EXPECT_EQ(a->size(), 2u);
    EXPECT_EQ((*a)[0].value(), 1);
    EXPECT_EQ((*a)[1].value(), 2);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that concat() triggers growth when the combined size exceeds
 *       the current capacity and all values are preserved
 */
TEST_F(ArrayInitTest, ConcatTriggersGrowth) {
    auto ra = cslt::Array<int>::init(2, alloc);
    auto rb = cslt::Array<int>::init(2, alloc);
    ASSERT_TRUE(ra.hasValue());
    ASSERT_TRUE(rb.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> a(ra.value());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> b(rb.value());
 
    a->push_back(1);
    a->push_back(2);  // a is now full (cap == 2)
    b->push_back(3);
    b->push_back(4);
 
    EXPECT_TRUE(a->concat(*b));
 
    EXPECT_EQ(a->size(), 4u);
    EXPECT_GE(a->capacity(), 4u);
    EXPECT_EQ((*a)[0].value(), 1);
    EXPECT_EQ((*a)[1].value(), 2);
    EXPECT_EQ((*a)[2].value(), 3);
    EXPECT_EQ((*a)[3].value(), 4);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that concat() does not modify the source array
 */
TEST_F(ArrayInitTest, ConcatDoesNotModifySource) {
    auto ra = cslt::Array<int>::init(4, alloc);
    auto rb = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(ra.hasValue());
    ASSERT_TRUE(rb.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> a(ra.value());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> b(rb.value());
 
    a->push_back(1);
    a->push_back(2);
    b->push_back(3);
    b->push_back(4);
 
    a->concat(*b);
 
    EXPECT_EQ(b->size(), 2u);
    EXPECT_EQ((*b)[0].value(), 3);
    EXPECT_EQ((*b)[1].value(), 4);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that self-concatenation doubles the array correctly
 */
TEST_F(ArrayInitTest, ConcatSelfDoubles) {
    auto ra = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(ra.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> a(ra.value());
 
    a->push_back(1);
    a->push_back(2);
    a->push_back(3);
 
    EXPECT_TRUE(a->concat(*a));
 
    EXPECT_EQ(a->size(), 6u);
    EXPECT_EQ((*a)[0].value(), 1);
    EXPECT_EQ((*a)[1].value(), 2);
    EXPECT_EQ((*a)[2].value(), 3);
    EXPECT_EQ((*a)[3].value(), 1);
    EXPECT_EQ((*a)[4].value(), 2);
    EXPECT_EQ((*a)[5].value(), 3);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that multiple sequential concat() calls accumulate correctly
 */
TEST_F(ArrayInitTest, ConcatMultipleSequentialCalls) {
    auto ra = cslt::Array<int>::init(4, alloc);
    auto rb = cslt::Array<int>::init(4, alloc);
    auto rc = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(ra.hasValue());
    ASSERT_TRUE(rb.hasValue());
    ASSERT_TRUE(rc.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> a(ra.value());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> b(rb.value());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> c(rc.value());
 
    a->push_back(1);
    b->push_back(2);
    b->push_back(3);
    c->push_back(4);
    c->push_back(5);
    c->push_back(6);
 
    EXPECT_TRUE(a->concat(*b));
    EXPECT_TRUE(a->concat(*c));
 
    EXPECT_EQ(a->size(), 6u);
    EXPECT_EQ((*a)[0].value(), 1);
    EXPECT_EQ((*a)[1].value(), 2);
    EXPECT_EQ((*a)[2].value(), 3);
    EXPECT_EQ((*a)[3].value(), 4);
    EXPECT_EQ((*a)[4].value(), 5);
    EXPECT_EQ((*a)[5].value(), 6);
}
// ================================================================================ 
// ================================================================================ 

// ============================================================================
// reverse() tests
// ============================================================================
 
/**
 * @test Verify that reverse() correctly reverses an odd-length int array
 */
TEST_F(ArrayInitTest, ReverseIntOddLength) {
    auto result = cslt::Array<int>::init(8, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
    arr->push_back(4);
    arr->push_back(5);
 
    arr->reverse();
 
    EXPECT_EQ(arr->size(), 5u);
    EXPECT_EQ(arr->data()[0], 5);
    EXPECT_EQ(arr->data()[1], 4);
    EXPECT_EQ(arr->data()[2], 3);
    EXPECT_EQ(arr->data()[3], 2);
    EXPECT_EQ(arr->data()[4], 1);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that reverse() correctly reverses an even-length int array
 */
TEST_F(ArrayInitTest, ReverseIntEvenLength) {
    auto result = cslt::Array<int>::init(8, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
    arr->push_back(4);
 
    arr->reverse();
 
    EXPECT_EQ(arr->size(), 4u);
    EXPECT_EQ(arr->data()[0], 4);
    EXPECT_EQ(arr->data()[1], 3);
    EXPECT_EQ(arr->data()[2], 2);
    EXPECT_EQ(arr->data()[3], 1);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that reverse() correctly reverses a double array
 */
TEST_F(ArrayInitTest, ReverseDoubleArray) {
    auto result = cslt::Array<double>::init(8, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<double>, cslt::ArrayDeleter<double>> arr(result.value());
 
    arr->push_back(1.1);
    arr->push_back(2.2);
    arr->push_back(3.3);
    arr->push_back(4.4);
 
    arr->reverse();
 
    EXPECT_EQ(arr->size(), 4u);
    EXPECT_DOUBLE_EQ(arr->data()[0], 4.4);
    EXPECT_DOUBLE_EQ(arr->data()[1], 3.3);
    EXPECT_DOUBLE_EQ(arr->data()[2], 2.2);
    EXPECT_DOUBLE_EQ(arr->data()[3], 1.1);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that reverse() correctly reverses a Point array
 *       (non-trivially-copyable path)
 */
TEST_F(ArrayInitTest, ReversePointArray) {
    auto result = cslt::Array<Point>::init(8, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<Point>, cslt::ArrayDeleter<Point>> arr(result.value());
 
    Point p1{1, 0}, p2{2, 0}, p3{3, 0}, p4{4, 0};
    arr->push_back(p1);
    arr->push_back(p2);
    arr->push_back(p3);
    arr->push_back(p4);
 
    arr->reverse();
 
    EXPECT_EQ(arr->size(), 4u);
    EXPECT_EQ(arr->data()[0], p4);
    // EXPECT_EQ(arr->data()[1], p3);
    // EXPECT_EQ(arr->data()[2], p2);
    // EXPECT_EQ(arr->data()[3], p1);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that reverse() on a single-element array leaves it unchanged
 */
TEST_F(ArrayInitTest, ReverseSingleElementIsNoop) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(42);
    arr->reverse();
 
    EXPECT_EQ(arr->size(), 1u);
    EXPECT_EQ(arr->data()[0], 42);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that reverse() on an empty array is a no-op and does not crash
 */
TEST_F(ArrayInitTest, ReverseEmptyArrayIsNoop) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->reverse();
 
    EXPECT_EQ(arr->size(), 0u);
    EXPECT_TRUE(arr->is_empty());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that reverse() does not alter the size or capacity
 */
TEST_F(ArrayInitTest, ReversePreservesSizeAndCapacity) {
    auto result = cslt::Array<int>::init(8, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
    size_t const size_before = arr->size();
    size_t const cap_before  = arr->capacity();
 
    arr->reverse();
 
    EXPECT_EQ(arr->size(),     size_before);
    EXPECT_EQ(arr->capacity(), cap_before);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that two successive reverse() calls restore the original order
 */
TEST_F(ArrayInitTest, ReverseDoubleInversionRestoresOrder) {
    auto result = cslt::Array<int>::init(8, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
    arr->push_back(4);
    arr->push_back(5);
 
    arr->reverse();
    arr->reverse();
 
    EXPECT_EQ(arr->data()[0], 1);
    EXPECT_EQ(arr->data()[1], 2);
    EXPECT_EQ(arr->data()[2], 3);
    EXPECT_EQ(arr->data()[3], 4);
    EXPECT_EQ(arr->data()[4], 5);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that reverse() on a two-element int array swaps the elements
 */
TEST_F(ArrayInitTest, ReverseTwoElementArray) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(10);
    arr->push_back(20);
 
    arr->reverse();
 
    EXPECT_EQ(arr->data()[0], 20);
    EXPECT_EQ(arr->data()[1], 10);
}
// -------------------------------------------------------------------------------- 

// ============================================================================
// Comparators used across sort() tests
// ============================================================================
 
static int cmp_int_asc(const int& a, const int& b) {
    return (a > b) - (a < b);
}
 
static int cmp_double_asc(const double& a, const double& b) {
    return (a > b) - (a < b);
}
 
static int cmp_point_x_asc(const Point& a, const Point& b) {
    return (a.x > b.x) - (a.x < b.x);
}
 
// ============================================================================
// sort() tests
// ============================================================================
 
/**
 * @test Verify that sort() ascending correctly sorts an unsorted int array
 */
TEST_F(ArrayInitTest, SortIntAscending) {
    auto result = cslt::Array<int>::init(8, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(5);
    arr->push_back(2);
    arr->push_back(8);
    arr->push_back(1);
    arr->push_back(9);
    arr->push_back(3);
 
    EXPECT_TRUE(arr->sort(cmp_int_asc, cslt::Direction::FORWARD));
 
    EXPECT_EQ(arr->size(), 6u);
    EXPECT_EQ(arr->data()[0], 1);
    EXPECT_EQ(arr->data()[1], 2);
    EXPECT_EQ(arr->data()[2], 3);
    EXPECT_EQ(arr->data()[3], 5);
    EXPECT_EQ(arr->data()[4], 8);
    EXPECT_EQ(arr->data()[5], 9);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that sort() descending correctly sorts an unsorted int array
 */
TEST_F(ArrayInitTest, SortIntDescending) {
    auto result = cslt::Array<int>::init(8, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(5);
    arr->push_back(2);
    arr->push_back(8);
    arr->push_back(1);
    arr->push_back(9);
    arr->push_back(3);
 
    EXPECT_TRUE(arr->sort(cmp_int_asc, cslt::Direction::REVERSE));
 
    EXPECT_EQ(arr->size(), 6u);
    EXPECT_EQ(arr->data()[0], 9);
    EXPECT_EQ(arr->data()[1], 8);
    EXPECT_EQ(arr->data()[2], 5);
    EXPECT_EQ(arr->data()[3], 3);
    EXPECT_EQ(arr->data()[4], 2);
    EXPECT_EQ(arr->data()[5], 1);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that sort() correctly sorts a double array ascending
 */
TEST_F(ArrayInitTest, SortDoubleAscending) {
    auto result = cslt::Array<double>::init(8, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<double>, cslt::ArrayDeleter<double>> arr(result.value());
 
    arr->push_back(3.3);
    arr->push_back(1.1);
    arr->push_back(4.4);
    arr->push_back(2.2);
 
    EXPECT_TRUE(arr->sort(cmp_double_asc, cslt::Direction::FORWARD));
 
    EXPECT_DOUBLE_EQ(arr->data()[0], 1.1);
    EXPECT_DOUBLE_EQ(arr->data()[1], 2.2);
    EXPECT_DOUBLE_EQ(arr->data()[2], 3.3);
    EXPECT_DOUBLE_EQ(arr->data()[3], 4.4);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that sort() correctly sorts a double array descending
 */
TEST_F(ArrayInitTest, SortDoubleDescending) {
    auto result = cslt::Array<double>::init(8, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<double>, cslt::ArrayDeleter<double>> arr(result.value());
 
    arr->push_back(3.3);
    arr->push_back(1.1);
    arr->push_back(4.4);
    arr->push_back(2.2);
 
    EXPECT_TRUE(arr->sort(cmp_double_asc, cslt::Direction::REVERSE));
 
    EXPECT_DOUBLE_EQ(arr->data()[0], 4.4);
    EXPECT_DOUBLE_EQ(arr->data()[1], 3.3);
    EXPECT_DOUBLE_EQ(arr->data()[2], 2.2);
    EXPECT_DOUBLE_EQ(arr->data()[3], 1.1);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that sort() correctly sorts a Point array ascending by x
 */
TEST_F(ArrayInitTest, SortPointAscendingByX) {
    auto result = cslt::Array<Point>::init(8, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<Point>, cslt::ArrayDeleter<Point>> arr(result.value());
 
    arr->push_back({4, 0});
    arr->push_back({1, 0});
    arr->push_back({3, 0});
    arr->push_back({2, 0});
 
    EXPECT_TRUE(arr->sort(cmp_point_x_asc, cslt::Direction::FORWARD));
 
    EXPECT_EQ(arr->data()[0].x, 1);
    EXPECT_EQ(arr->data()[1].x, 2);
    EXPECT_EQ(arr->data()[2].x, 3);
    EXPECT_EQ(arr->data()[3].x, 4);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that sort() correctly sorts a Point array descending by x
 */
TEST_F(ArrayInitTest, SortPointDescendingByX) {
    auto result = cslt::Array<Point>::init(8, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<Point>, cslt::ArrayDeleter<Point>> arr(result.value());
 
    arr->push_back({4, 0});
    arr->push_back({1, 0});
    arr->push_back({3, 0});
    arr->push_back({2, 0});
 
    EXPECT_TRUE(arr->sort(cmp_point_x_asc, cslt::Direction::REVERSE));
 
    EXPECT_EQ(arr->data()[0].x, 4);
    EXPECT_EQ(arr->data()[1].x, 3);
    EXPECT_EQ(arr->data()[2].x, 2);
    EXPECT_EQ(arr->data()[3].x, 1);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that sort() with a lambda comparator works correctly
 */
TEST_F(ArrayInitTest, SortIntAscendingWithLambda) {
    auto result = cslt::Array<int>::init(8, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(4);
    arr->push_back(2);
    arr->push_back(7);
    arr->push_back(1);
    arr->push_back(5);
 
    EXPECT_TRUE(arr->sort(
        [](const int& a, const int& b) { return (a > b) - (a < b); },
        cslt::Direction::FORWARD));
 
    EXPECT_EQ(arr->data()[0], 1);
    EXPECT_EQ(arr->data()[1], 2);
    EXPECT_EQ(arr->data()[2], 4);
    EXPECT_EQ(arr->data()[3], 5);
    EXPECT_EQ(arr->data()[4], 7);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that sort() on an already-sorted array leaves it unchanged
 */
TEST_F(ArrayInitTest, SortAlreadySortedArray) {
    auto result = cslt::Array<int>::init(8, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(1);
    arr->push_back(2);
    arr->push_back(3);
    arr->push_back(4);
    arr->push_back(5);
 
    EXPECT_TRUE(arr->sort(cmp_int_asc, cslt::Direction::FORWARD));
 
    EXPECT_EQ(arr->data()[0], 1);
    EXPECT_EQ(arr->data()[1], 2);
    EXPECT_EQ(arr->data()[2], 3);
    EXPECT_EQ(arr->data()[3], 4);
    EXPECT_EQ(arr->data()[4], 5);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that sort() on a reverse-sorted array sorts correctly
 */
TEST_F(ArrayInitTest, SortReverseSortedArray) {
    auto result = cslt::Array<int>::init(8, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(5);
    arr->push_back(4);
    arr->push_back(3);
    arr->push_back(2);
    arr->push_back(1);
 
    EXPECT_TRUE(arr->sort(cmp_int_asc, cslt::Direction::FORWARD));
 
    EXPECT_EQ(arr->data()[0], 1);
    EXPECT_EQ(arr->data()[1], 2);
    EXPECT_EQ(arr->data()[2], 3);
    EXPECT_EQ(arr->data()[3], 4);
    EXPECT_EQ(arr->data()[4], 5);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that sort() handles an array with all duplicate values
 */
TEST_F(ArrayInitTest, SortAllDuplicates) {
    auto result = cslt::Array<int>::init(8, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(7);
    arr->push_back(7);
    arr->push_back(7);
    arr->push_back(7);
 
    EXPECT_TRUE(arr->sort(cmp_int_asc, cslt::Direction::FORWARD));
 
    EXPECT_EQ(arr->data()[0], 7);
    EXPECT_EQ(arr->data()[1], 7);
    EXPECT_EQ(arr->data()[2], 7);
    EXPECT_EQ(arr->data()[3], 7);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that sort() on a two-element unsorted array swaps correctly
 */
TEST_F(ArrayInitTest, SortTwoElementArray) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(9);
    arr->push_back(1);
 
    EXPECT_TRUE(arr->sort(cmp_int_asc, cslt::Direction::FORWARD));
 
    EXPECT_EQ(arr->data()[0], 1);
    EXPECT_EQ(arr->data()[1], 9);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that sort() returns false and is a no-op on a single-element array
 */
TEST_F(ArrayInitTest, SortSingleElementReturnsFalse) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(42);
 
    EXPECT_FALSE(arr->sort(cmp_int_asc, cslt::Direction::FORWARD));
    EXPECT_EQ(arr->data()[0], 42);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that sort() returns false on an empty array
 */
TEST_F(ArrayInitTest, SortEmptyArrayReturnsFalse) {
    auto result = cslt::Array<int>::init(4, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    EXPECT_FALSE(arr->sort(cmp_int_asc, cslt::Direction::FORWARD));
    EXPECT_TRUE(arr->is_empty());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that sort() does not alter the size or capacity
 */
TEST_F(ArrayInitTest, SortPreservesSizeAndCapacity) {
    auto result = cslt::Array<int>::init(8, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(3);
    arr->push_back(1);
    arr->push_back(2);
    size_t const size_before = arr->size();
    size_t const cap_before  = arr->capacity();
 
    arr->sort(cmp_int_asc, cslt::Direction::FORWARD);
 
    EXPECT_EQ(arr->size(),     size_before);
    EXPECT_EQ(arr->capacity(), cap_before);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that sort() ascending followed by sort() descending produces
 *       the reverse of the ascending result
 */
TEST_F(ArrayInitTest, SortAscendingThenDescendingProducesReverse) {
    auto result = cslt::Array<int>::init(8, alloc);
    ASSERT_TRUE(result.hasValue());
    cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
 
    arr->push_back(3);
    arr->push_back(1);
    arr->push_back(4);
    arr->push_back(2);
    arr->push_back(5);
 
    arr->sort(cmp_int_asc, cslt::Direction::FORWARD);
    arr->sort(cmp_int_asc, cslt::Direction::REVERSE);
 
    EXPECT_EQ(arr->data()[0], 5);
    EXPECT_EQ(arr->data()[1], 4);
    EXPECT_EQ(arr->data()[2], 3);
    EXPECT_EQ(arr->data()[3], 2);
    EXPECT_EQ(arr->data()[4], 1);
}
// ================================================================================
// ================================================================================
// eof
