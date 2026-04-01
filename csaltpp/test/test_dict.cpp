// ================================================================================
// ================================================================================
// - File:    test_dict.cpp
// - Purpose: This file implements google test as a method to test C++ code.
//            Describe the type of testing to be completed
//
// Source Metadata
// - Author:  Jonathan A. Webb
// - Date:    March 25, 2026
// - Version: 1.0
// - Copyright: Copyright 2026, Jon Webb Inc.
// ================================================================================
// ================================================================================
// - Begin test

#include <gtest/gtest.h>
#include "dict.hpp"
#include "allocator.hpp"
// ================================================================================
// ================================================================================
 
/**
 * @struct Point3i
 * @brief A trivially copyable struct used to exercise Dict with a non-scalar
 *        key type.  No user-defined constructor, destructor, or copy members,
 *        so std::is_trivially_copyable_v<Point3i> is true.
 */
struct Point3i {
    int x, y, z;
    bool operator==(const Point3i& o) const noexcept {
        return x == o.x && y == o.y && z == o.z;
    }
};
 
/**
 * @struct Payload
 * @brief A non-trivially-copyable value type used to verify that Dict<K,V>
 *        correctly calls constructors and destructors for non-trivial V.
 */
struct Payload {
    int   id;
    float score;
 
    Payload()              : id(0), score(0.0f) {}  // required by Expected<T>
    Payload(int i, float s) : id(i), score(s) {}
    Payload(const Payload&) = default;
    ~Payload() {}  // user-defined destructor makes it non-trivially copyable
 
    bool operator==(const Payload& o) const noexcept {
        return id == o.id && score == o.score;
    }
};
// ================================================================================
// ================================================================================
 
/**
 * @class DictTest
 * @brief Test fixture providing a shared HeapAllocator for all Dict tests.
 */
class DictTest : public ::testing::Test {
protected:
    cslt::HeapAllocator alloc;
};
 
// Helper alias to reduce line length in tests
template <typename K, typename V>
using DictPtr = cslt::UniquePtr<cslt::Dict<K,V>, cslt::DictDeleter<K,V>>;
// ================================================================================
// ================================================================================
 
// ============================================================================
// init() tests
// ============================================================================
 
/**
 * @test Verify that init() succeeds with a valid capacity and returns a
 *       non-null pointer with correct initial state
 */
TEST_F(DictTest, InitIntFloatSucceeds) {
    auto r = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> d(r.value());
 
    EXPECT_TRUE(d->is_empty());
    EXPECT_EQ(d->hash_size(), 0u);
    EXPECT_EQ(d->size(), 0u);
    EXPECT_GE(d->bucket_count(), 8u);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that init() with capacity 1 rounds up to the minimum of 8
 */
TEST_F(DictTest, InitCapacityRoundsUpToMinimum) {
    auto r = cslt::Dict<int, float>::init(1, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> d(r.value());
 
    EXPECT_EQ(d->bucket_count(), 8u);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that init() rounds capacity up to the next power of two
 */
TEST_F(DictTest, InitCapacityRoundsUpToPowerOfTwo) {
    auto r = cslt::Dict<int, float>::init(10, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> d(r.value());
 
    EXPECT_EQ(d->bucket_count(), 16u);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that init() returns an error when capacity is zero
 */
TEST_F(DictTest, InitZeroCapacityReturnsError) {
    auto r = cslt::Dict<int, float>::init(0, true, alloc);
    EXPECT_FALSE(r.hasValue());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that init() works correctly for a non-trivial value type
 */
TEST_F(DictTest, InitNonTrivialValueSucceeds) {
    auto r = cslt::Dict<int, Payload>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,Payload> d(r.value());
 
    EXPECT_TRUE(d->is_empty());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that init() works correctly for a struct key type
 */
TEST_F(DictTest, InitStructKeySucceeds) {
    auto r = cslt::Dict<Point3i, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<Point3i,float> d(r.value());
 
    EXPECT_TRUE(d->is_empty());
}
 
// ============================================================================
// insert() tests
// ============================================================================
 
/**
 * @test Verify that insert() stores an int->float pair and increments hash_size
 */
TEST_F(DictTest, InsertIntFloatSucceeds) {
    auto r = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> d(r.value());
 
    EXPECT_TRUE(d->insert(1, 1.1f));
    EXPECT_EQ(d->hash_size(), 1u);
    EXPECT_FALSE(d->is_empty());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that insert() correctly stores multiple distinct int->float pairs
 */
TEST_F(DictTest, InsertMultipleIntFloatPairs) {
    auto r = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> d(r.value());
 
    EXPECT_TRUE(d->insert(1, 1.1f));
    EXPECT_TRUE(d->insert(2, 2.2f));
    EXPECT_TRUE(d->insert(3, 3.3f));
    EXPECT_EQ(d->hash_size(), 3u);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that insert() rejects a duplicate key and returns false
 */
TEST_F(DictTest, InsertDuplicateKeyReturnsFalse) {
    auto r = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> d(r.value());
 
    EXPECT_TRUE(d->insert(42, 1.0f));
    EXPECT_FALSE(d->insert(42, 9.9f));
    EXPECT_EQ(d->hash_size(), 1u);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that insert() works correctly for a struct key type
 */
TEST_F(DictTest, InsertStructKeySucceeds) {
    auto r = cslt::Dict<Point3i, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<Point3i,float> d(r.value());
 
    EXPECT_TRUE(d->insert({1, 2, 3}, 1.0f));
    EXPECT_EQ(d->hash_size(), 1u);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that insert() works correctly for a non-trivial value type
 */
TEST_F(DictTest, InsertNonTrivialValueSucceeds) {
    auto r = cslt::Dict<int, Payload>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,Payload> d(r.value());
 
    EXPECT_TRUE(d->insert(1, Payload{10, 9.9f}));
    EXPECT_EQ(d->hash_size(), 1u);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that insert() triggers growth when the load factor is exceeded
 */
TEST_F(DictTest, InsertTriggersGrowth) {
    auto r = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> d(r.value());
 
    // 8 * 0.75 = 6 — inserting 7 should trigger a resize
    for (int i = 0; i < 7; ++i)
        EXPECT_TRUE(d->insert(i, static_cast<float>(i)));
 
    EXPECT_EQ(d->hash_size(), 7u);
    EXPECT_GT(d->bucket_count(), 8u);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that insert() returns false when growth is disabled and the
 *       dict is full
 */
TEST_F(DictTest, InsertFailsWhenFullAndGrowthDisabled) {
    auto r = cslt::Dict<int, float>::init(8, false, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> d(r.value());
 
    // Fill all 8 buckets
    for (int i = 0; i < 8; ++i)
        d->insert(i, static_cast<float>(i));
 
    EXPECT_FALSE(d->insert(99, 0.0f));
}
 
// ============================================================================
// update() tests
// ============================================================================
 
/**
 * @test Verify that update() overwrites the value for an existing int key
 */
TEST_F(DictTest, UpdateIntFloatOverwritesValue) {
    auto r = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> d(r.value());
 
    d->insert(42, 1.0f);
    EXPECT_TRUE(d->update(42, 9.9f));
 
    auto gr = d->get(42);
    ASSERT_TRUE(gr.hasValue());
    EXPECT_FLOAT_EQ(gr.value(), 9.9f);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that update() does not change hash_size
 */
TEST_F(DictTest, UpdateDoesNotChangeHashSize) {
    auto r = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> d(r.value());
 
    d->insert(1, 1.0f);
    d->insert(2, 2.0f);
    d->update(1, 9.9f);
 
    EXPECT_EQ(d->hash_size(), 2u);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that update() returns false for a key that does not exist
 */
TEST_F(DictTest, UpdateMissingKeyReturnsFalse) {
    auto r = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> d(r.value());
 
    EXPECT_FALSE(d->update(99, 1.0f));
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that update() works correctly for a non-trivial value type
 */
TEST_F(DictTest, UpdateNonTrivialValueSucceeds) {
    auto r = cslt::Dict<int, Payload>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,Payload> d(r.value());
 
    d->insert(1, Payload{10, 1.0f});
    EXPECT_TRUE(d->update(1, Payload{20, 2.0f}));
 
    auto gr = d->get(1);
    ASSERT_TRUE(gr.hasValue());
    EXPECT_EQ(gr.value().id, 20);
    EXPECT_FLOAT_EQ(gr.value().score, 2.0f);
}
 
// ============================================================================
// pop() tests
// ============================================================================
 
/**
 * @test Verify that pop() removes an int key and returns the correct value
 */
TEST_F(DictTest, PopIntFloatReturnsValue) {
    auto r = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> d(r.value());
 
    d->insert(42, 3.14f);
 
    auto pr = d->pop(42);
    ASSERT_TRUE(pr.hasValue());
    EXPECT_FLOAT_EQ(pr.value(), 3.14f);
    EXPECT_EQ(d->hash_size(), 0u);
    EXPECT_FALSE(d->has_key(42));
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that pop() decrements hash_size correctly
 */
TEST_F(DictTest, PopDecrementsHashSize) {
    auto r = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> d(r.value());
 
    d->insert(1, 1.0f);
    d->insert(2, 2.0f);
    d->insert(3, 3.0f);
 
    d->pop(2);
    EXPECT_EQ(d->hash_size(), 2u);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that pop() returns an error for a key that does not exist
 */
TEST_F(DictTest, PopMissingKeyReturnsError) {
    auto r = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> d(r.value());
 
    auto pr = d->pop(99);
    EXPECT_FALSE(pr.hasValue());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that pop() works correctly for a non-trivial value type
 */
TEST_F(DictTest, PopNonTrivialValueReturnsValue) {
    auto r = cslt::Dict<int, Payload>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,Payload> d(r.value());
 
    d->insert(7, Payload{7, 7.7f});
 
    auto pr = d->pop(7);
    ASSERT_TRUE(pr.hasValue());
    EXPECT_EQ(pr.value().id, 7);
    EXPECT_FLOAT_EQ(pr.value().score, 7.7f);
    EXPECT_EQ(d->hash_size(), 0u);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that the same key can be re-inserted after being popped
 */
TEST_F(DictTest, PopThenReinsertSucceeds) {
    auto r = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> d(r.value());
 
    d->insert(5, 5.5f);
    d->pop(5);
    EXPECT_TRUE(d->insert(5, 6.6f));
 
    auto gr = d->get(5);
    ASSERT_TRUE(gr.hasValue());
    EXPECT_FLOAT_EQ(gr.value(), 6.6f);
}
 
// ============================================================================
// get() tests
// ============================================================================
 
/**
 * @test Verify that get() returns the correct float value for an existing key
 */
TEST_F(DictTest, GetIntFloatReturnsCorrectValue) {
    auto r = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> d(r.value());
 
    d->insert(10, 1.1f);
    d->insert(20, 2.2f);
    d->insert(30, 3.3f);
 
    auto gr = d->get(20);
    ASSERT_TRUE(gr.hasValue());
    EXPECT_FLOAT_EQ(gr.value(), 2.2f);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that get() returns an error for a missing key
 */
TEST_F(DictTest, GetMissingKeyReturnsError) {
    auto r = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> d(r.value());
 
    d->insert(1, 1.0f);
 
    auto gr = d->get(99);
    EXPECT_FALSE(gr.hasValue());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that get() does not modify the dict
 */
TEST_F(DictTest, GetDoesNotModifyDict) {
    auto r = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> d(r.value());
 
    d->insert(1, 1.0f);
    d->get(1);
 
    EXPECT_EQ(d->hash_size(), 1u);
    EXPECT_TRUE(d->has_key(1));
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that get() works correctly for a struct key
 */
TEST_F(DictTest, GetStructKeyReturnsCorrectValue) {
    auto r = cslt::Dict<Point3i, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<Point3i,float> d(r.value());
 
    d->insert({1, 2, 3}, 9.9f);
 
    auto gr = d->get({1, 2, 3});
    ASSERT_TRUE(gr.hasValue());
    EXPECT_FLOAT_EQ(gr.value(), 9.9f);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that get() works correctly for a non-trivial value type
 */
TEST_F(DictTest, GetNonTrivialValueReturnsCorrectValue) {
    auto r = cslt::Dict<int, Payload>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,Payload> d(r.value());
 
    d->insert(3, Payload{33, 3.3f});
 
    auto gr = d->get(3);
    ASSERT_TRUE(gr.hasValue());
    EXPECT_EQ(gr.value().id, 33);
    EXPECT_FLOAT_EQ(gr.value().score, 3.3f);
}
 
// ============================================================================
// get_ptr() tests
// ============================================================================
 
/**
 * @test Verify that get_ptr() returns a non-null pointer to the correct value
 */
TEST_F(DictTest, GetPtrReturnsCorrectPointer) {
    auto r = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> d(r.value());
 
    d->insert(7, 7.7f);
 
    const float* ptr = d->get_ptr(7);
    ASSERT_NE(ptr, nullptr);
    EXPECT_FLOAT_EQ(*ptr, 7.7f);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that get_ptr() returns nullptr for a missing key
 */
TEST_F(DictTest, GetPtrMissingKeyReturnsNullptr) {
    auto r = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> d(r.value());
 
    d->insert(1, 1.0f);
 
    EXPECT_EQ(d->get_ptr(99), nullptr);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that get_ptr() reflects a value updated via update()
 */
TEST_F(DictTest, GetPtrReflectsUpdateChange) {
    auto r = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> d(r.value());
 
    d->insert(5, 1.0f);
    d->update(5, 9.9f);
 
    const float* ptr = d->get_ptr(5);
    ASSERT_NE(ptr, nullptr);
    EXPECT_FLOAT_EQ(*ptr, 9.9f);
}
 
// ============================================================================
// has_key() tests
// ============================================================================
 
/**
 * @test Verify that has_key() returns true for an existing int key
 */
TEST_F(DictTest, HasKeyReturnsTrueForExistingKey) {
    auto r = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> d(r.value());
 
    d->insert(42, 1.0f);
    EXPECT_TRUE(d->has_key(42));
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that has_key() returns false for a missing key
 */
TEST_F(DictTest, HasKeyReturnsFalseForMissingKey) {
    auto r = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> d(r.value());
 
    d->insert(1, 1.0f);
    EXPECT_FALSE(d->has_key(99));
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that has_key() returns false after a key is popped
 */
TEST_F(DictTest, HasKeyReturnsFalseAfterPop) {
    auto r = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> d(r.value());
 
    d->insert(5, 5.5f);
    d->pop(5);
    EXPECT_FALSE(d->has_key(5));
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that has_key() works correctly for a struct key type
 */
TEST_F(DictTest, HasKeyStructKeyWorks) {
    auto r = cslt::Dict<Point3i, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<Point3i,float> d(r.value());
 
    d->insert({1, 2, 3}, 1.0f);
    EXPECT_TRUE(d->has_key({1, 2, 3}));
    EXPECT_FALSE(d->has_key({9, 9, 9}));
}
 
// ============================================================================
// clear() tests
// ============================================================================
 
/**
 * @test Verify that clear() removes all entries and resets hash_size to zero
 */
TEST_F(DictTest, ClearRemovesAllEntries) {
    auto r = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> d(r.value());
 
    d->insert(1, 1.0f);
    d->insert(2, 2.0f);
    d->insert(3, 3.0f);
 
    d->clear();
 
    EXPECT_EQ(d->hash_size(), 0u);
    EXPECT_TRUE(d->is_empty());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that clear() does not change the bucket count
 */
TEST_F(DictTest, ClearPreservesBucketCount) {
    auto r = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> d(r.value());
 
    d->insert(1, 1.0f);
    d->insert(2, 2.0f);
    size_t const bc = d->bucket_count();
 
    d->clear();
    EXPECT_EQ(d->bucket_count(), bc);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that insert() works correctly after a clear()
 */
TEST_F(DictTest, InsertAfterClearSucceeds) {
    auto r = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> d(r.value());
 
    d->insert(1, 1.0f);
    d->clear();
    EXPECT_TRUE(d->insert(1, 9.9f));
    EXPECT_EQ(d->hash_size(), 1u);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that clear() correctly destructs non-trivial value types
 */
TEST_F(DictTest, ClearNonTrivialValueDestructsCorrectly) {
    auto r = cslt::Dict<int, Payload>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,Payload> d(r.value());
 
    d->insert(1, Payload{1, 1.0f});
    d->insert(2, Payload{2, 2.0f});
    d->clear();
 
    EXPECT_TRUE(d->is_empty());
}
 
// ============================================================================
// foreach() tests
// ============================================================================
 
/**
 * @test Verify that foreach() visits every key-value pair exactly once
 */
TEST_F(DictTest, ForeachVisitsAllPairs) {
    auto r = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> d(r.value());
 
    d->insert(1, 1.0f);
    d->insert(2, 2.0f);
    d->insert(3, 3.0f);
 
    int   count     = 0;
    float sum_vals  = 0.0f;
    int   sum_keys  = 0;
 
    d->foreach([&](const int& k, const float& v) {
        ++count;
        sum_keys += k;
        sum_vals += v;
    });
 
    EXPECT_EQ(count, 3);
    EXPECT_EQ(sum_keys, 6);
    EXPECT_FLOAT_EQ(sum_vals, 6.0f);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that foreach() on an empty dict calls the callback zero times
 */
TEST_F(DictTest, ForeachOnEmptyDictCallsCallbackZeroTimes) {
    auto r = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> d(r.value());
 
    int count = 0;
    d->foreach([&](const int&, const float&) { ++count; });
    EXPECT_EQ(count, 0);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that foreach() works with a file-scope function pointer
 */
static int g_foreach_count = 0;
static void foreach_counter(const int&, const float&) { ++g_foreach_count; }
 
TEST_F(DictTest, ForeachWithFunctionPointerWorks) {
    auto r = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> d(r.value());
 
    d->insert(10, 1.0f);
    d->insert(20, 2.0f);
 
    g_foreach_count = 0;
    d->foreach(foreach_counter);
    EXPECT_EQ(g_foreach_count, 2);
}
 
// ============================================================================
// copy() tests
// ============================================================================
 
/**
 * @test Verify that copy() produces an independent deep copy
 */
TEST_F(DictTest, CopyProducesIndependentDict) {
    auto r = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> src(r.value());
 
    src->insert(1, 1.0f);
    src->insert(2, 2.0f);
 
    auto cr = cslt::Dict<int, float>::copy(*src, alloc);
    ASSERT_TRUE(cr.hasValue());
    DictPtr<int,float> dst(cr.value());
 
    EXPECT_EQ(dst->hash_size(), 2u);
    EXPECT_TRUE(dst->has_key(1));
    EXPECT_TRUE(dst->has_key(2));
 
    // Mutating dst must not affect src
    dst->insert(3, 3.0f);
    EXPECT_FALSE(src->has_key(3));
    EXPECT_EQ(src->hash_size(), 2u);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that the single-argument copy() overload uses the source allocator
 */
TEST_F(DictTest, CopyOneArgOverloadSucceeds) {
    auto r = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> src(r.value());
 
    src->insert(7, 7.7f);
 
    auto cr = cslt::Dict<int, float>::copy(*src);
    ASSERT_TRUE(cr.hasValue());
    DictPtr<int,float> dst(cr.value());
 
    EXPECT_EQ(dst->hash_size(), 1u);
    auto gr = dst->get(7);
    ASSERT_TRUE(gr.hasValue());
    EXPECT_FLOAT_EQ(gr.value(), 7.7f);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that copy() correctly copies non-trivial value types
 */
TEST_F(DictTest, CopyNonTrivialValueSucceeds) {
    auto r = cslt::Dict<int, Payload>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,Payload> src(r.value());
 
    src->insert(5, Payload{50, 5.5f});
 
    auto cr = cslt::Dict<int, Payload>::copy(*src, alloc);
    ASSERT_TRUE(cr.hasValue());
    DictPtr<int,Payload> dst(cr.value());
 
    auto gr = dst->get(5);
    ASSERT_TRUE(gr.hasValue());
    EXPECT_EQ(gr.value().id, 50);
    EXPECT_FLOAT_EQ(gr.value().score, 5.5f);
}
 
// ============================================================================
// merge() tests
// ============================================================================
 
/**
 * @test Verify that merge() combines two non-overlapping dicts correctly
 */
TEST_F(DictTest, MergeNonOverlappingDicts) {
    auto ra = cslt::Dict<int, float>::init(8, true, alloc);
    auto rb = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(ra.hasValue());
    ASSERT_TRUE(rb.hasValue());
    DictPtr<int,float> a(ra.value());
    DictPtr<int,float> b(rb.value());
 
    a->insert(1, 1.0f);
    a->insert(2, 2.0f);
    b->insert(3, 3.0f);
    b->insert(4, 4.0f);
 
    auto mr = cslt::Dict<int, float>::merge(*a, *b, false, alloc);
    ASSERT_TRUE(mr.hasValue());
    DictPtr<int,float> m(mr.value());
 
    EXPECT_EQ(m->hash_size(), 4u);
    EXPECT_TRUE(m->has_key(1));
    EXPECT_TRUE(m->has_key(2));
    EXPECT_TRUE(m->has_key(3));
    EXPECT_TRUE(m->has_key(4));
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that merge() with overwrite == true uses b's value on conflict
 */
TEST_F(DictTest, MergeOverwriteTrueBWins) {
    auto ra = cslt::Dict<int, float>::init(8, true, alloc);
    auto rb = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(ra.hasValue());
    ASSERT_TRUE(rb.hasValue());
    DictPtr<int,float> a(ra.value());
    DictPtr<int,float> b(rb.value());
 
    a->insert(1, 1.0f);
    b->insert(1, 9.9f);  // conflict
 
    auto mr = cslt::Dict<int, float>::merge(*a, *b, true, alloc);
    ASSERT_TRUE(mr.hasValue());
    DictPtr<int,float> m(mr.value());
 
    auto gr = m->get(1);
    ASSERT_TRUE(gr.hasValue());
    EXPECT_FLOAT_EQ(gr.value(), 9.9f);  // b wins
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that merge() with overwrite == false keeps a's value on conflict
 */
TEST_F(DictTest, MergeOverwriteFalseAWins) {
    auto ra = cslt::Dict<int, float>::init(8, true, alloc);
    auto rb = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(ra.hasValue());
    ASSERT_TRUE(rb.hasValue());
    DictPtr<int,float> a(ra.value());
    DictPtr<int,float> b(rb.value());
 
    a->insert(1, 1.0f);
    b->insert(1, 9.9f);  // conflict
 
    auto mr = cslt::Dict<int, float>::merge(*a, *b, false, alloc);
    ASSERT_TRUE(mr.hasValue());
    DictPtr<int,float> m(mr.value());
 
    auto gr = m->get(1);
    ASSERT_TRUE(gr.hasValue());
    EXPECT_FLOAT_EQ(gr.value(), 1.0f);  // a wins
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that merge() does not modify the source dicts
 */
TEST_F(DictTest, MergeDoesNotModifySources) {
    auto ra = cslt::Dict<int, float>::init(8, true, alloc);
    auto rb = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(ra.hasValue());
    ASSERT_TRUE(rb.hasValue());

    DictPtr<int, float> a(ra.value());
    DictPtr<int, float> b(rb.value());

    a->insert(1, 1.0f);
    b->insert(2, 2.0f);

    auto rm = cslt::Dict<int, float>::merge(*a, *b, true, alloc);
    ASSERT_TRUE(rm.hasValue());
    DictPtr<int, float> merged(rm.value());

    EXPECT_EQ(a->hash_size(), 1u);
    EXPECT_EQ(b->hash_size(), 1u);

    EXPECT_EQ(merged->hash_size(), 2u);
    EXPECT_TRUE(merged->has_key(1));
    EXPECT_TRUE(merged->has_key(2));
}
 
// ============================================================================
// size(), hash_size(), bucket_count(), is_empty() tests
// ============================================================================
 
/**
 * @test Verify that hash_size() tracks insertions and pops correctly
 */
TEST_F(DictTest, HashSizeTracksInsertionsAndPops) {
    auto r = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> d(r.value());
 
    EXPECT_EQ(d->hash_size(), 0u);
    d->insert(1, 1.0f);
    EXPECT_EQ(d->hash_size(), 1u);
    d->insert(2, 2.0f);
    EXPECT_EQ(d->hash_size(), 2u);
    d->pop(1);
    EXPECT_EQ(d->hash_size(), 1u);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that is_empty() returns true on a fresh dict and false after insert
 */
TEST_F(DictTest, IsEmptyFreshAndAfterInsert) {
    auto r = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> d(r.value());
 
    EXPECT_TRUE(d->is_empty());
    d->insert(1, 1.0f);
    EXPECT_FALSE(d->is_empty());
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that bucket_count() returns a power of two >= the requested capacity
 */
TEST_F(DictTest, BucketCountIsPowerOfTwoGECapacity) {
    auto r = cslt::Dict<int, float>::init(12, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> d(r.value());
 
    size_t const bc = d->bucket_count();
    EXPECT_GE(bc, 12u);
    // Check power-of-two: bc & (bc-1) == 0
    EXPECT_EQ(bc & (bc - 1u), 0u);
}
// --------------------------------------------------------------------------------
 
/**
 * @test Verify that size() returns the number of occupied buckets, which is
 *       <= hash_size()
 */
TEST_F(DictTest, SizeIsLeOrEqualToHashSize) {
    auto r = cslt::Dict<int, float>::init(8, true, alloc);
    ASSERT_TRUE(r.hasValue());
    DictPtr<int,float> d(r.value());
 
    for (int i = 0; i < 5; ++i)
        d->insert(i, static_cast<float>(i));
 
    EXPECT_LE(d->size(), d->hash_size());
    EXPECT_EQ(d->hash_size(), 5u);
}
// ================================================================================
// ================================================================================
// eof
