// ================================================================================
// ================================================================================
// - File:    test_heap.cpp
// - Purpose: This file implements google test as a method to test C++ code.
//            Describe the type of testing to be completed
//
// Source Metadata
// - Author:  Jonathan A. Webb
// - Date:    April 02, 2026
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

/**
 * @struct Task
 * @brief A struct used to exercise Heap with a custom lambda comparator.
 *        Default-constructible as required by Expected<T>.
 */
struct Task {
    int    priority;
    int    id;
 
    Task() : priority(0), id(0) {}
    Task(int p, int i) : priority(p), id(i) {}
    Task(const Task&) = default;
    ~Task() {}
 
    bool operator==(const Task& o) const noexcept {
        return priority == o.priority && id == o.id;
    }
};
// ================================================================================
// ================================================================================
 
/**
 * @class HeapMinTest
 * @brief Test fixture for min-heap (std::greater<int>) tests.
 */
class HeapMinTest : public ::testing::Test {
protected:
    cslt::HeapAllocator alloc;
    using MinHeap    = cslt::Heap<int, std::less<int>>;
    using MinDeleter = cslt::HeapDeleter<int, std::less<int>>;
    using MinPtr     = cslt::UniquePtr<MinHeap, MinDeleter>;
};
 
/**
 * @class HeapMaxTest
 * @brief Test fixture for max-heap (std::less<int>) tests.
 */
class HeapMaxTest : public ::testing::Test {
protected:
    cslt::HeapAllocator alloc;
    using MaxHeap    = cslt::Heap<int, std::greater<int>>;
    using MaxDeleter = cslt::HeapDeleter<int, std::greater<int>>;
    using MaxPtr     = cslt::UniquePtr<MaxHeap, MaxDeleter>;
};
 
/**
 * @class HeapFloatTest
 * @brief Test fixture for float min-heap tests.
 */
class HeapFloatTest : public ::testing::Test {
protected:
    cslt::HeapAllocator alloc;
    using FHeap    = cslt::Heap<float, std::greater<float>>;
    using FDeleter = cslt::HeapDeleter<float, std::greater<float>>;
    using FPtr     = cslt::UniquePtr<FHeap, FDeleter>;
};
// ================================================================================
// ================================================================================
 
// ============================================================================
// init() tests
// ============================================================================
 
/**
 * @test Verify that init() succeeds and the heap starts empty
 */
TEST_F(HeapMinTest, InitSucceedsEmptyState) {
    auto r = MinHeap::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    MinPtr h(r.value());
 
    EXPECT_TRUE(h->is_empty());
    EXPECT_EQ(h->size(), 0u);
    EXPECT_GE(h->capacity(), 8u);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that init() returns an error when capacity is zero
 */
TEST_F(HeapMinTest, InitZeroCapacityReturnsError) {
    auto r = MinHeap::init(0, true, alloc);
    EXPECT_FALSE(r.hasValue());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that init() works for a max-heap
 */
TEST_F(HeapMaxTest, InitMaxHeapSucceeds) {
    auto r = MaxHeap::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    MaxPtr h(r.value());
 
    EXPECT_TRUE(h->is_empty());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that init() works with a lambda comparator
 */
TEST_F(HeapMinTest, InitWithLambdaComparatorSucceeds) {
    auto cmp = [](const int& a, const int& b) { return a > b; };
    using LHeap    = cslt::Heap<int, decltype(cmp)>;
    using LDeleter = cslt::HeapDeleter<int, decltype(cmp)>;
 
    auto r = LHeap::init(8, true, alloc, cmp);
    ASSERT_TRUE(r.hasValue());
    cslt::UniquePtr<LHeap, LDeleter> h(r.value());
 
    EXPECT_TRUE(h->is_empty());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that init() with growth disabled sets is_full correctly
 */
TEST_F(HeapMinTest, InitNoGrowthIsFullWhenAtCapacity) {
    auto r = MinHeap::init(2, false, alloc);
    ASSERT_TRUE(r.hasValue());
    MinPtr h(r.value());
 
    EXPECT_FALSE(h->is_full());
    h->push(1);
    h->push(2);
    EXPECT_TRUE(h->is_full());
}
 
// ============================================================================
// push() tests
// ============================================================================
 
/**
 * @test Verify that pushing a single element makes it the root of a min-heap
 */
TEST_F(HeapMinTest, PushSingleElementBecomesRoot) {
    auto r = MinHeap::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    MinPtr h(r.value());
 
    EXPECT_TRUE(h->push(42));
    EXPECT_EQ(h->size(), 1u);
 
    auto pr = h->peek();
    ASSERT_TRUE(pr.hasValue());
    EXPECT_EQ(pr.value(), 42);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that the min-heap root is always the smallest element
 */
TEST_F(HeapMinTest, MinHeapRootIsAlwaysSmallest) {
    auto r = MinHeap::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    MinPtr h(r.value());
 
    h->push(5);
    h->push(3);
    h->push(7);
    h->push(1);
    h->push(4);
 
    auto pr = h->peek();
    ASSERT_TRUE(pr.hasValue());
    EXPECT_EQ(pr.value(), 1);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that the max-heap root is always the largest element
 */
TEST_F(HeapMaxTest, MaxHeapRootIsAlwaysLargest) {
    auto r = MaxHeap::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    MaxPtr h(r.value());
 
    h->push(5);
    h->push(3);
    h->push(7);
    h->push(1);
    h->push(4);
 
    auto pr = h->peek();
    ASSERT_TRUE(pr.hasValue());
    EXPECT_EQ(pr.value(), 7);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that push() increments size correctly
 */
TEST_F(HeapMinTest, PushIncrementsSize) {
    auto r = MinHeap::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    MinPtr h(r.value());
 
    EXPECT_EQ(h->size(), 0u);
    h->push(1);
    EXPECT_EQ(h->size(), 1u);
    h->push(2);
    EXPECT_EQ(h->size(), 2u);
    h->push(3);
    EXPECT_EQ(h->size(), 3u);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that push() returns false when growth is disabled and heap is full
 */
TEST_F(HeapMinTest, PushReturnsFalseWhenFullNoGrowth) {
    auto r = MinHeap::init(2, false, alloc);
    ASSERT_TRUE(r.hasValue());
    MinPtr h(r.value());
 
    EXPECT_TRUE(h->push(1));
    EXPECT_TRUE(h->push(2));
    EXPECT_FALSE(h->push(3));
    EXPECT_EQ(h->size(), 2u);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that push() triggers growth and succeeds when growth is enabled
 */
TEST_F(HeapMinTest, PushTriggersGrowthWhenEnabled) {
    auto r = MinHeap::init(2, true, alloc);
    ASSERT_TRUE(r.hasValue());
    MinPtr h(r.value());
 
    for (int i = 0; i < 10; ++i)
        EXPECT_TRUE(h->push(i));
 
    EXPECT_EQ(h->size(), 10u);
    EXPECT_GT(h->capacity(), 2u);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that push() maintains the min-heap property after many insertions
 */
TEST_F(HeapMinTest, PushManyElementsMaintainsMinHeapProperty) {
    auto r = MinHeap::init(16, true, alloc);
    ASSERT_TRUE(r.hasValue());
    MinPtr h(r.value());
 
    int values[] = {9, 4, 7, 1, 8, 3, 6, 2, 5};
    for (int v : values)
        h->push(v);
 
    // Pop all elements — must come out in ascending order
    int prev = -1;
    while (!h->is_empty()) {
        auto pr = h->pop();
        ASSERT_TRUE(pr.hasValue());
        EXPECT_GE(pr.value(), prev);
        prev = pr.value();
    }
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that push() maintains the max-heap property after many insertions
 */
TEST_F(HeapMaxTest, PushManyElementsMaintainsMaxHeapProperty) {
    auto r = MaxHeap::init(16, true, alloc);
    ASSERT_TRUE(r.hasValue());
    MaxPtr h(r.value());
 
    int values[] = {9, 4, 7, 1, 8, 3, 6, 2, 5};
    for (int v : values)
        h->push(v);
 
    // Pop all elements — must come out in descending order
    int prev = 100;
    while (!h->is_empty()) {
        auto pr = h->pop();
        ASSERT_TRUE(pr.hasValue());
        EXPECT_LE(pr.value(), prev);
        prev = pr.value();
    }
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that push() works for float values
 */
TEST_F(HeapFloatTest, PushFloatMinHeapWorks) {
    auto r = FHeap::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    FPtr h(r.value());
 
    h->push(3.14f);
    h->push(1.41f);
    h->push(2.72f);
 
    auto pr = h->peek();
    ASSERT_TRUE(pr.hasValue());
    EXPECT_FLOAT_EQ(pr.value(), 3.14f);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify push() with duplicate values maintains heap property
 */
TEST_F(HeapMinTest, PushDuplicateValuesHandledCorrectly) {
    auto r = MinHeap::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    MinPtr h(r.value());
 
    h->push(5);
    h->push(5);
    h->push(5);
 
    EXPECT_EQ(h->size(), 3u);
    auto pr = h->peek();
    ASSERT_TRUE(pr.hasValue());
    EXPECT_EQ(pr.value(), 5);
}
 
// ============================================================================
// pop() tests
// ============================================================================
 
/**
 * @test Verify that pop() on an empty heap returns an error
 */
TEST_F(HeapMinTest, PopEmptyHeapReturnsError) {
    auto r = MinHeap::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    MinPtr h(r.value());
 
    auto pr = h->pop();
    EXPECT_FALSE(pr.hasValue());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that pop() on a single-element heap leaves it empty
 */
TEST_F(HeapMinTest, PopSingleElementLeavesEmpty) {
    auto r = MinHeap::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    MinPtr h(r.value());
 
    h->push(99);
    auto pr = h->pop();
    ASSERT_TRUE(pr.hasValue());
    EXPECT_EQ(pr.value(), 99);
    EXPECT_TRUE(h->is_empty());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that pop() returns the min-heap root and decrements size
 */
TEST_F(HeapMinTest, PopReturnsRootAndDecrementsSize) {
    auto r = MinHeap::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    MinPtr h(r.value());
 
    h->push(3);
    h->push(1);
    h->push(2);
 
    auto pr = h->pop();
    ASSERT_TRUE(pr.hasValue());
    EXPECT_EQ(pr.value(), 1);
    EXPECT_EQ(h->size(), 2u);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that pop() returns the max-heap root correctly
 */
TEST_F(HeapMaxTest, PopReturnsMaxRoot) {
    auto r = MaxHeap::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    MaxPtr h(r.value());
 
    h->push(3);
    h->push(1);
    h->push(2);
 
    auto pr = h->pop();
    ASSERT_TRUE(pr.hasValue());
    EXPECT_EQ(pr.value(), 3);
    EXPECT_EQ(h->size(), 2u);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that successive pop() calls return elements in sorted order
 *       for a min-heap
 */
TEST_F(HeapMinTest, SuccessivePopsReturnAscendingOrder) {
    auto r = MinHeap::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    MinPtr h(r.value());
 
    h->push(4);
    h->push(2);
    h->push(6);
    h->push(1);
    h->push(3);
    h->push(5);
 
    int expected[] = {1, 2, 3, 4, 5, 6};
    for (int e : expected) {
        auto pr = h->pop();
        ASSERT_TRUE(pr.hasValue());
        EXPECT_EQ(pr.value(), e);
    }
    EXPECT_TRUE(h->is_empty());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that successive pop() calls return elements in sorted order
 *       for a max-heap
 */
TEST_F(HeapMaxTest, SuccessivePopsReturnDescendingOrder) {
    auto r = MaxHeap::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    MaxPtr h(r.value());
 
    h->push(4);
    h->push(2);
    h->push(6);
    h->push(1);
    h->push(3);
    h->push(5);
 
    int expected[] = {6, 5, 4, 3, 2, 1};
    for (int e : expected) {
        auto pr = h->pop();
        ASSERT_TRUE(pr.hasValue());
        EXPECT_EQ(pr.value(), e);
    }
    EXPECT_TRUE(h->is_empty());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that pop() restores the heap property after removal
 */
TEST_F(HeapMinTest, PopRestoresHeapProperty) {
    auto r = MinHeap::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    MinPtr h(r.value());
 
    h->push(10);
    h->push(5);
    h->push(15);
    h->push(3);
    h->push(7);
 
    h->pop();   // removes 3
 
    // Root must still be the smallest remaining
    auto pr = h->peek();
    ASSERT_TRUE(pr.hasValue());
    EXPECT_EQ(pr.value(), 5);
}
 
// ============================================================================
// peek() tests
// ============================================================================
 
/**
 * @test Verify that peek() returns the root without removing it
 */
TEST_F(HeapMinTest, PeekReturnsRootWithoutRemoving) {
    auto r = MinHeap::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    MinPtr h(r.value());
 
    h->push(3);
    h->push(1);
    h->push(2);
 
    auto pr = h->peek();
    ASSERT_TRUE(pr.hasValue());
    EXPECT_EQ(pr.value(), 1);
    EXPECT_EQ(h->size(), 3u);  // unchanged
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that peek() on an empty heap returns an error
 */
TEST_F(HeapMinTest, PeekEmptyHeapReturnsError) {
    auto r = MinHeap::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    MinPtr h(r.value());
 
    auto pr = h->peek();
    EXPECT_FALSE(pr.hasValue());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that multiple peek() calls return the same root
 */
TEST_F(HeapMinTest, MultipleConsecutivePeeksReturnSameRoot) {
    auto r = MinHeap::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    MinPtr h(r.value());
 
    h->push(5);
    h->push(2);
    h->push(8);
 
    for (int i = 0; i < 3; ++i) {
        auto pr = h->peek();
        ASSERT_TRUE(pr.hasValue());
        EXPECT_EQ(pr.value(), 2);
    }
    EXPECT_EQ(h->size(), 3u);
}
 
// ============================================================================
// copy() tests
// ============================================================================
 
/**
 * @test Verify that copy() produces an independent deep copy
 */
TEST_F(HeapMinTest, CopyProducesIndependentHeap) {
    auto r = MinHeap::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    MinPtr src(r.value());
 
    src->push(3);
    src->push(1);
    src->push(2);
 
    auto cr = MinHeap::copy(*src, alloc);
    ASSERT_TRUE(cr.hasValue());
    MinPtr dst(cr.value());
 
    EXPECT_EQ(dst->size(), 3u);
 
    // Root of copy must equal root of original
    EXPECT_EQ(dst->peek().value(), src->peek().value());
 
    // Mutating dst must not affect src
    dst->push(0);
    EXPECT_EQ(src->size(), 3u);
    EXPECT_EQ(dst->peek().value(), 0);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that copy() preserves the heap property in the copy
 */
TEST_F(HeapMinTest, CopyPreservesHeapProperty) {
    auto r = MinHeap::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    MinPtr src(r.value());
 
    src->push(7);
    src->push(3);
    src->push(9);
    src->push(1);
    src->push(5);
 
    auto cr = MinHeap::copy(*src, alloc);
    ASSERT_TRUE(cr.hasValue());
    MinPtr dst(cr.value());
 
    // Pop all from copy — must come out in ascending order
    int prev = -1;
    while (!dst->is_empty()) {
        auto pr = dst->pop();
        ASSERT_TRUE(pr.hasValue());
        EXPECT_GE(pr.value(), prev);
        prev = pr.value();
    }
 
    // Source must be unchanged
    EXPECT_EQ(src->size(), 5u);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that the single-argument copy() uses the source's own allocator
 */
TEST_F(HeapMinTest, CopyOneArgOverloadSucceeds) {
    auto r = MinHeap::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    MinPtr src(r.value());
 
    src->push(4);
    src->push(2);
 
    auto cr = MinHeap::copy(*src);
    ASSERT_TRUE(cr.hasValue());
    MinPtr dst(cr.value());
 
    EXPECT_EQ(dst->size(), 2u);
    EXPECT_EQ(dst->peek().value(), 2);
}
 
// ============================================================================
// foreach() tests
// ============================================================================
 
/**
 * @test Verify that foreach() visits every element exactly once
 */
TEST_F(HeapMinTest, ForeachVisitsAllElements) {
    auto r = MinHeap::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    MinPtr h(r.value());
 
    h->push(1);
    h->push(2);
    h->push(3);
 
    int count = 0;
    int sum   = 0;
    h->foreach([&](const int& v) {
        ++count;
        sum += v;
    });
 
    EXPECT_EQ(count, 3);
    EXPECT_EQ(sum,   6);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that foreach() returns false on an empty heap
 */
TEST_F(HeapMinTest, ForeachOnEmptyHeapReturnsFalse) {
    auto r = MinHeap::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    MinPtr h(r.value());
 
    EXPECT_FALSE(h->foreach([](const int&) {}));
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that foreach() works with a function pointer
 */
static int g_heap_foreach_sum = 0;
static void sum_element(const int& v) { g_heap_foreach_sum += v; }
 
TEST_F(HeapMinTest, ForeachWithFunctionPointerWorks) {
    auto r = MinHeap::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    MinPtr h(r.value());
 
    h->push(10);
    h->push(20);
    h->push(30);
 
    g_heap_foreach_sum = 0;
    h->foreach(sum_element);
    EXPECT_EQ(g_heap_foreach_sum, 60);
}
 
// ============================================================================
// Introspection tests
// ============================================================================
 
/**
 * @test Verify that size() tracks pushes and pops correctly
 */
TEST_F(HeapMinTest, SizeTracksOperations) {
    auto r = MinHeap::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    MinPtr h(r.value());
 
    EXPECT_EQ(h->size(), 0u);
    h->push(1);
    EXPECT_EQ(h->size(), 1u);
    h->push(2);
    EXPECT_EQ(h->size(), 2u);
    h->pop();
    EXPECT_EQ(h->size(), 1u);
    h->pop();
    EXPECT_EQ(h->size(), 0u);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that is_empty() returns true on a fresh heap and false after push
 */
TEST_F(HeapMinTest, IsEmptyFreshAndAfterPush) {
    auto r = MinHeap::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    MinPtr h(r.value());
 
    EXPECT_TRUE(h->is_empty());
    h->push(1);
    EXPECT_FALSE(h->is_empty());
    h->pop();
    EXPECT_TRUE(h->is_empty());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that capacity() returns a value >= the requested initial capacity
 */
TEST_F(HeapMinTest, CapacityAtLeastRequestedSize) {
    auto r = MinHeap::init(16, true, alloc);
    ASSERT_TRUE(r.hasValue());
    MinPtr h(r.value());
 
    EXPECT_GE(h->capacity(), 16u);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that is_full() is false when growth is enabled
 */
TEST_F(HeapMinTest, IsFullFalseWhenGrowthEnabled) {
    auto r = MinHeap::init(2, true, alloc);
    ASSERT_TRUE(r.hasValue());
    MinPtr h(r.value());
 
    h->push(1);
    h->push(2);
    // Slab is at capacity but growth is enabled — is_full must be false
    EXPECT_FALSE(h->is_full());
}
 
// ============================================================================
// Custom comparator / struct tests
// ============================================================================
 
/**
 * @test Verify that a lambda comparator correctly implements a min-heap
 *       of Task structs by priority
 */
TEST(HeapTaskTest, LambdaComparatorMinPriorityHeap) {
    cslt::HeapAllocator alloc;
 
    // Lower priority number = higher priority (like a real priority queue)
    auto cmp = [](const Task& a, const Task& b) {
        return a.priority < b.priority;
    };
    using TaskHeap    = cslt::Heap<Task, decltype(cmp)>;
    using TaskDeleter = cslt::HeapDeleter<Task, decltype(cmp)>;
 
    auto r = TaskHeap::init(8, true, alloc, cmp);
    ASSERT_TRUE(r.hasValue());
    cslt::UniquePtr<TaskHeap, TaskDeleter> h(r.value());
 
    h->push(Task{5, 1});
    h->push(Task{1, 2});
    h->push(Task{3, 3});
 
    // Highest-priority task (lowest priority number) should be at root
    auto pr = h->peek();
    ASSERT_TRUE(pr.hasValue());
    EXPECT_EQ(pr.value().priority, 1);
    EXPECT_EQ(pr.value().id,       2);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that popping all Task elements returns them in priority order
 */
TEST(HeapTaskTest, PopTasksInPriorityOrder) {
    cslt::HeapAllocator alloc;
 
    auto cmp = [](const Task& a, const Task& b) {
        return a.priority < b.priority;
    };
    using TaskHeap    = cslt::Heap<Task, decltype(cmp)>;
    using TaskDeleter = cslt::HeapDeleter<Task, decltype(cmp)>;
 
    auto r = TaskHeap::init(8, true, alloc, cmp);
    ASSERT_TRUE(r.hasValue());
    cslt::UniquePtr<TaskHeap, TaskDeleter> h(r.value());
 
    h->push(Task{4, 1});
    h->push(Task{1, 2});
    h->push(Task{3, 3});
    h->push(Task{2, 4});
 
    int expected_priorities[] = {1, 2, 3, 4};
    for (int ep : expected_priorities) {
        auto pr = h->pop();
        ASSERT_TRUE(pr.hasValue());
        EXPECT_EQ(pr.value().priority, ep);
    }
    EXPECT_TRUE(h->is_empty());
}
// ================================================================================
// ================================================================================
// eof
