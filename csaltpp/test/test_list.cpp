// ================================================================================
// ================================================================================
// - File:    test_list.cpp
// - Purpose: This file implements google test as a method to test C++ code.
//            Describe the type of testing to be completed
//
// Source Metadata
// - Author:  Jonathan A. Webb
// - Date:    April 01, 2026
// - Version: 1.0
// - Copyright: Copyright 2026, Jon Webb Inc.
// ================================================================================
// ================================================================================
// - Begin test

#include <gtest/gtest.h>

#include "list.hpp"
#include "allocator.hpp"

#include <vector>
#include <string>
#include <cstddef>
// ================================================================================ 
// ================================================================================ 

namespace cslt {
    namespace {

        template <typename T>
        using ListPtr = UniquePtr<SList<T>, SListDeleter<T>>;
        // -----------------------------------------------------------------------------

        template <typename T>
        ListPtr<T> make_list(std::size_t slab_nodes, bool allow_overflow, HeapAllocator& alloc) {
            auto r = SList<T>::init(slab_nodes, allow_overflow, alloc);
            EXPECT_TRUE(r.hasValue());
            if (!r.hasValue()) {
                return ListPtr<T>(nullptr);
            }
            return ListPtr<T>(r.value());
        }
        // -----------------------------------------------------------------------------

        TEST(SListTest, InitRejectsZeroCapacity) {
            HeapAllocator alloc;
            auto r = SList<int>::init(0, false, alloc);

            EXPECT_FALSE(r.hasValue());
        }
        // -----------------------------------------------------------------------------

        TEST(SListTest, InitCreatesEmptyList) {
            HeapAllocator alloc;
            auto list = make_list<int>(4, false, alloc);

            ASSERT_NE(list.get(), nullptr);
            EXPECT_TRUE(list->is_empty());
            EXPECT_EQ(list->size(), 0u);

            EXPECT_EQ(list->slab_capacity(), 4u);
            EXPECT_EQ(list->slab_used(), 0u);
            EXPECT_EQ(list->slab_free_count(), 0u);
            EXPECT_EQ(list->slab_live_count(), 0u);
            EXPECT_EQ(list->slab_remaining(), 4u);

            EXPECT_FALSE(list->is_slab_full());
            EXPECT_FALSE(list->in_overflow());
            EXPECT_EQ(list->overflow_count(), 0u);
        }
        // -----------------------------------------------------------------------------

        TEST(SListTest, PushBackPreservesOrder) {
            HeapAllocator alloc;
            auto list = make_list<int>(4, false, alloc);

            ASSERT_TRUE(list->push_back(10));
            ASSERT_TRUE(list->push_back(20));
            ASSERT_TRUE(list->push_back(30));

            EXPECT_EQ(list->size(), 3u);
            EXPECT_EQ(list->get(0).value(), 10);
            EXPECT_EQ(list->get(1).value(), 20);
            EXPECT_EQ(list->get(2).value(), 30);

            EXPECT_EQ(list->peek_front().value(), 10);
            EXPECT_EQ(list->peek_back().value(), 30);

            EXPECT_EQ(list->slab_used(), 3u);
            EXPECT_EQ(list->slab_free_count(), 0u);
            EXPECT_EQ(list->slab_live_count(), 3u);
            EXPECT_EQ(list->slab_remaining(), 1u);
        }
        // -----------------------------------------------------------------------------

        TEST(SListTest, PushFrontPreservesOrder) {
            HeapAllocator alloc;
            auto list = make_list<int>(4, false, alloc);

            ASSERT_TRUE(list->push_front(3));
            ASSERT_TRUE(list->push_front(2));
            ASSERT_TRUE(list->push_front(1));

            EXPECT_EQ(list->size(), 3u);
            EXPECT_EQ(list->get(0).value(), 1);
            EXPECT_EQ(list->get(1).value(), 2);
            EXPECT_EQ(list->get(2).value(), 3);
        }
        // -----------------------------------------------------------------------------

        TEST(SListTest, PushAtInsertsInMiddle) {
            HeapAllocator alloc;
            auto list = make_list<int>(5, false, alloc);

            ASSERT_TRUE(list->push_back(1));
            ASSERT_TRUE(list->push_back(3));
            ASSERT_TRUE(list->push_at(1, 2));

            EXPECT_EQ(list->size(), 3u);
            EXPECT_EQ(list->get(0).value(), 1);
            EXPECT_EQ(list->get(1).value(), 2);
            EXPECT_EQ(list->get(2).value(), 3);
        }
        // -----------------------------------------------------------------------------

        TEST(SListTest, PushAtRejectsOutOfRangeIndex) {
            HeapAllocator alloc;
            auto list = make_list<int>(4, false, alloc);

            ASSERT_TRUE(list->push_back(1));
            EXPECT_FALSE(list->push_at(2, 99));
            EXPECT_EQ(list->size(), 1u);
        }
        // -----------------------------------------------------------------------------

        TEST(SListTest, PopFrontRemovesHeadAndRecyclesSlabSlot) {
            HeapAllocator alloc;
            auto list = make_list<int>(4, false, alloc);

            ASSERT_TRUE(list->push_back(1));
            ASSERT_TRUE(list->push_back(2));
            ASSERT_TRUE(list->push_back(3));

            auto r = list->pop_front();
            ASSERT_TRUE(r.hasValue());
            EXPECT_EQ(r.value(), 1);

            EXPECT_EQ(list->size(), 2u);
            EXPECT_EQ(list->get(0).value(), 2);
            EXPECT_EQ(list->get(1).value(), 3);

            EXPECT_EQ(list->slab_used(), 2u);
            EXPECT_EQ(list->slab_free_count(), 1u);
            EXPECT_EQ(list->slab_live_count(), 2u);
            EXPECT_EQ(list->slab_remaining(), 2u);
        }
        // -----------------------------------------------------------------------------

        TEST(SListTest, PopBackRemovesTailAndRecyclesSlabSlot) {
            HeapAllocator alloc;
            auto list = make_list<int>(4, false, alloc);

            ASSERT_TRUE(list->push_back(1));
            ASSERT_TRUE(list->push_back(2));
            ASSERT_TRUE(list->push_back(3));

            auto r = list->pop_back();
            ASSERT_TRUE(r.hasValue());
            EXPECT_EQ(r.value(), 3);

            EXPECT_EQ(list->size(), 2u);
            EXPECT_EQ(list->peek_back().value(), 2);

            EXPECT_EQ(list->slab_used(), 2u);
            EXPECT_EQ(list->slab_free_count(), 1u);
            EXPECT_EQ(list->slab_live_count(), 2u);
            EXPECT_EQ(list->slab_remaining(), 2u);
        }
        // -----------------------------------------------------------------------------

        TEST(SListTest, PopAtRemovesMiddleAndRecyclesSlabSlot) {
            HeapAllocator alloc;
            auto list = make_list<int>(5, false, alloc);

            ASSERT_TRUE(list->push_back(10));
            ASSERT_TRUE(list->push_back(20));
            ASSERT_TRUE(list->push_back(30));

            auto r = list->pop_at(1);
            ASSERT_TRUE(r.hasValue());
            EXPECT_EQ(r.value(), 20);

            EXPECT_EQ(list->size(), 2u);
            EXPECT_EQ(list->get(0).value(), 10);
            EXPECT_EQ(list->get(1).value(), 30);

            EXPECT_EQ(list->slab_used(), 2u);
            EXPECT_EQ(list->slab_free_count(), 1u);
            EXPECT_EQ(list->slab_live_count(), 2u);
            EXPECT_EQ(list->slab_remaining(), 3u);
        }
        // -----------------------------------------------------------------------------

        TEST(SListTest, PopOperationsFailOnEmptyList) {
            HeapAllocator alloc;
            auto list = make_list<int>(4, false, alloc);

            EXPECT_FALSE(list->pop_front().hasValue());
            EXPECT_FALSE(list->pop_back().hasValue());
            EXPECT_FALSE(list->pop_at(0).hasValue());
        }
        // -----------------------------------------------------------------------------

        TEST(SListTest, GetPeekAndContainsWork) {
            HeapAllocator alloc;
            auto list = make_list<int>(4, false, alloc);

            ASSERT_TRUE(list->push_back(5));
            ASSERT_TRUE(list->push_back(10));
            ASSERT_TRUE(list->push_back(15));

            EXPECT_EQ(list->get(2).value(), 15);
            EXPECT_EQ(list->peek_front().value(), 5);
            EXPECT_EQ(list->peek_back().value(), 15);

            auto c1 = list->contains(10);
            ASSERT_TRUE(c1.hasValue());
            EXPECT_EQ(c1.value(), 1u);

            auto c2 = list->contains(999);
            EXPECT_FALSE(c2.hasValue());
        }
        // -----------------------------------------------------------------------------

        TEST(SListTest, ForeachVisitsAllElementsInOrder) {
            HeapAllocator alloc;
            auto list = make_list<int>(4, false, alloc);

            ASSERT_TRUE(list->push_back(7));
            ASSERT_TRUE(list->push_back(8));
            ASSERT_TRUE(list->push_back(9));

            std::vector<int> values;
            std::vector<std::size_t> indices;

            ASSERT_TRUE(list->foreach([&](const int& v, std::size_t i) {
                values.push_back(v);
                indices.push_back(i);
            }));

            ASSERT_EQ(values.size(), 3u);
            ASSERT_EQ(indices.size(), 3u);

            EXPECT_EQ(values[0], 7);
            EXPECT_EQ(values[1], 8);
            EXPECT_EQ(values[2], 9);

            EXPECT_EQ(indices[0], 0u);
            EXPECT_EQ(indices[1], 1u);
            EXPECT_EQ(indices[2], 2u);
        }
        // -----------------------------------------------------------------------------

        TEST(SListTest, ReverseWorks) {
            HeapAllocator alloc;
            auto list = make_list<int>(5, false, alloc);

            ASSERT_TRUE(list->push_back(1));
            ASSERT_TRUE(list->push_back(2));
            ASSERT_TRUE(list->push_back(3));
            ASSERT_TRUE(list->push_back(4));

            ASSERT_TRUE(list->reverse());

            EXPECT_EQ(list->size(), 4u);
            EXPECT_EQ(list->get(0).value(), 4);
            EXPECT_EQ(list->get(1).value(), 3);
            EXPECT_EQ(list->get(2).value(), 2);
            EXPECT_EQ(list->get(3).value(), 1);

            EXPECT_EQ(list->peek_front().value(), 4);
            EXPECT_EQ(list->peek_back().value(), 1);
        }
        // -----------------------------------------------------------------------------

        TEST(SListTest, ClearResetsListToFreshState) {
            HeapAllocator alloc;
            auto list = make_list<int>(3, true, alloc);

            ASSERT_TRUE(list->push_back(1));
            ASSERT_TRUE(list->push_back(2));
            ASSERT_TRUE(list->push_back(3));
            ASSERT_TRUE(list->push_back(4));  // overflow

            ASSERT_TRUE(list->clear());

            EXPECT_TRUE(list->is_empty());
            EXPECT_EQ(list->size(), 0u);

            EXPECT_EQ(list->slab_capacity(), 3u);
            EXPECT_EQ(list->slab_used(), 0u);
            EXPECT_EQ(list->slab_free_count(), 0u);
            EXPECT_EQ(list->slab_live_count(), 0u);
            EXPECT_EQ(list->slab_remaining(), 3u);

            EXPECT_FALSE(list->is_slab_full());
            EXPECT_FALSE(list->in_overflow());
            EXPECT_EQ(list->overflow_count(), 0u);

            ASSERT_TRUE(list->push_back(42));
            EXPECT_EQ(list->size(), 1u);
            EXPECT_EQ(list->peek_front().value(), 42);
            EXPECT_EQ(list->slab_used(), 1u);
            EXPECT_EQ(list->slab_free_count(), 0u);
            EXPECT_EQ(list->slab_remaining(), 2u);
            EXPECT_EQ(list->overflow_count(), 0u);
        }
        // -----------------------------------------------------------------------------

        TEST(SListTest, CopyCreatesIndependentList) {
            HeapAllocator alloc;
            auto src = make_list<int>(4, false, alloc);

            ASSERT_TRUE(src->push_back(11));
            ASSERT_TRUE(src->push_back(22));
            ASSERT_TRUE(src->push_back(33));

            auto r = SList<int>::copy(*src, alloc);
            ASSERT_TRUE(r.hasValue());

            ListPtr<int> dst(r.value());

            ASSERT_NE(dst.get(), nullptr);
            EXPECT_EQ(dst->size(), 3u);
            EXPECT_EQ(dst->get(0).value(), 11);
            EXPECT_EQ(dst->get(1).value(), 22);
            EXPECT_EQ(dst->get(2).value(), 33);

            auto popped = src->pop_front();
            ASSERT_TRUE(popped.hasValue());
            EXPECT_EQ(popped.value(), 11);

            EXPECT_EQ(src->size(), 2u);
            EXPECT_EQ(dst->size(), 3u);
            EXPECT_EQ(dst->get(0).value(), 11);
        }
        // -----------------------------------------------------------------------------

        TEST(SListTest, ConcatAppendsSourceWithoutModifyingIt) {
            HeapAllocator alloc;
            auto a = make_list<int>(6, false, alloc);
            auto b = make_list<int>(4, false, alloc);

            ASSERT_TRUE(a->push_back(1));
            ASSERT_TRUE(a->push_back(2));

            ASSERT_TRUE(b->push_back(3));
            ASSERT_TRUE(b->push_back(4));

            ASSERT_TRUE(a->concat(*b));

            EXPECT_EQ(a->size(), 4u);
            EXPECT_EQ(a->get(0).value(), 1);
            EXPECT_EQ(a->get(1).value(), 2);
            EXPECT_EQ(a->get(2).value(), 3);
            EXPECT_EQ(a->get(3).value(), 4);

            EXPECT_EQ(b->size(), 2u);
            EXPECT_EQ(b->get(0).value(), 3);
            EXPECT_EQ(b->get(1).value(), 4);
        }
        // -----------------------------------------------------------------------------

        TEST(SListTest, SlabFullWithoutOverflowRejectsFurtherPushes) {
            HeapAllocator alloc;
            auto list = make_list<int>(2, false, alloc);

            ASSERT_TRUE(list->push_back(1));
            ASSERT_TRUE(list->push_back(2));

            EXPECT_TRUE(list->is_slab_full());
            EXPECT_EQ(list->slab_used(), 2u);
            EXPECT_EQ(list->slab_free_count(), 0u);
            EXPECT_EQ(list->slab_remaining(), 0u);

            EXPECT_FALSE(list->push_back(3));
            EXPECT_EQ(list->size(), 2u);
            EXPECT_FALSE(list->in_overflow());
            EXPECT_EQ(list->overflow_count(), 0u);
        }
        // -----------------------------------------------------------------------------

        TEST(SListTest, PoppingFromFullSlabMakesSpaceAvailableAgain) {
            HeapAllocator alloc;
            auto list = make_list<int>(2, false, alloc);

            ASSERT_TRUE(list->push_back(1));
            ASSERT_TRUE(list->push_back(2));

            EXPECT_TRUE(list->is_slab_full());
            EXPECT_EQ(list->slab_remaining(), 0u);

            auto popped = list->pop_front();
            ASSERT_TRUE(popped.hasValue());
            EXPECT_EQ(popped.value(), 1);

            EXPECT_FALSE(list->is_slab_full());
            EXPECT_EQ(list->slab_used(), 1u);
            EXPECT_EQ(list->slab_free_count(), 1u);
            EXPECT_EQ(list->slab_live_count(), 1u);
            EXPECT_EQ(list->slab_remaining(), 1u);

            ASSERT_TRUE(list->push_back(3));
            EXPECT_EQ(list->size(), 2u);
            EXPECT_EQ(list->get(0).value(), 2);
            EXPECT_EQ(list->get(1).value(), 3);

            EXPECT_TRUE(list->is_slab_full());
            EXPECT_EQ(list->slab_used(), 2u);
            EXPECT_EQ(list->slab_free_count(), 0u);
            EXPECT_EQ(list->overflow_count(), 0u);
        }
        // -----------------------------------------------------------------------------

        TEST(SListTest, ReusesSlabNodeInsteadOfOverflowAfterPop) {
            HeapAllocator alloc;
            auto list = make_list<int>(2, true, alloc);

            ASSERT_TRUE(list->push_back(1));
            ASSERT_TRUE(list->push_back(2));
            EXPECT_TRUE(list->is_slab_full());

            auto popped = list->pop_back();
            ASSERT_TRUE(popped.hasValue());
            EXPECT_EQ(popped.value(), 2);

            EXPECT_EQ(list->slab_used(), 1u);
            EXPECT_EQ(list->slab_free_count(), 1u);
            EXPECT_EQ(list->overflow_count(), 0u);
            EXPECT_FALSE(list->is_slab_full());

            ASSERT_TRUE(list->push_back(3));

            EXPECT_EQ(list->size(), 2u);
            EXPECT_EQ(list->get(0).value(), 1);
            EXPECT_EQ(list->get(1).value(), 3);

            EXPECT_EQ(list->slab_used(), 2u);
            EXPECT_EQ(list->overflow_count(), 0u);
            EXPECT_FALSE(list->in_overflow());
            EXPECT_EQ(list->slab_free_count(), 0u);
            EXPECT_TRUE(list->is_slab_full());
        }
        // -----------------------------------------------------------------------------

        TEST(SListTest, OverflowEnabledAllowsGrowthPastSlab) {
            HeapAllocator alloc;
            auto list = make_list<int>(2, true, alloc);

            ASSERT_TRUE(list->push_back(1));
            ASSERT_TRUE(list->push_back(2));
            ASSERT_TRUE(list->push_back(3));
            ASSERT_TRUE(list->push_back(4));

            EXPECT_EQ(list->size(), 4u);
            EXPECT_EQ(list->slab_capacity(), 2u);
            EXPECT_EQ(list->slab_used(), 2u);
            EXPECT_EQ(list->slab_free_count(), 0u);
            EXPECT_EQ(list->slab_live_count(), 2u);

            EXPECT_TRUE(list->is_slab_full());
            EXPECT_TRUE(list->in_overflow());
            EXPECT_EQ(list->overflow_count(), 2u);

            EXPECT_EQ(list->get(0).value(), 1);
            EXPECT_EQ(list->get(1).value(), 2);
            EXPECT_EQ(list->get(2).value(), 3);
            EXPECT_EQ(list->get(3).value(), 4);
        }
        // -----------------------------------------------------------------------------

        TEST(SListTest, RemovingOverflowNodeReducesOverflowCount) {
            HeapAllocator alloc;
            auto list = make_list<int>(2, true, alloc);

            ASSERT_TRUE(list->push_back(1));
            ASSERT_TRUE(list->push_back(2));
            ASSERT_TRUE(list->push_back(3));  // overflow
            ASSERT_TRUE(list->push_back(4));  // overflow

            EXPECT_EQ(list->overflow_count(), 2u);
            EXPECT_TRUE(list->in_overflow());

            auto r = list->pop_back();
            ASSERT_TRUE(r.hasValue());
            EXPECT_EQ(r.value(), 4);

            EXPECT_EQ(list->overflow_count(), 1u);
            EXPECT_TRUE(list->in_overflow());

            r = list->pop_back();
            ASSERT_TRUE(r.hasValue());
            EXPECT_EQ(r.value(), 3);

            EXPECT_EQ(list->overflow_count(), 0u);
            EXPECT_FALSE(list->in_overflow());
        }
        // -----------------------------------------------------------------------------

        TEST(SListTest, SlabRemainingTracksCurrentOccupancy) {
            HeapAllocator alloc;
            auto list = make_list<int>(5, false, alloc);

            ASSERT_TRUE(list->push_back(1));
            ASSERT_TRUE(list->push_back(2));
            ASSERT_TRUE(list->push_back(3));

            EXPECT_EQ(list->slab_used(), 3u);
            EXPECT_EQ(list->slab_free_count(), 0u);
            EXPECT_EQ(list->slab_remaining(), 2u);

            auto r = list->pop_at(1);
            ASSERT_TRUE(r.hasValue());
            EXPECT_EQ(r.value(), 2);

            EXPECT_EQ(list->slab_used(), 2u);
            EXPECT_EQ(list->slab_free_count(), 1u);
            EXPECT_EQ(list->slab_remaining(), 3u);
        }
        // -----------------------------------------------------------------------------

        TEST(SListTest, StringPayloadWorks) {
            HeapAllocator alloc;
            auto list = make_list<std::string>(3, true, alloc);

            ASSERT_TRUE(list->push_back("alpha"));
            ASSERT_TRUE(list->push_back("beta"));
            ASSERT_TRUE(list->push_front("zero"));

            EXPECT_EQ(list->size(), 3u);
            EXPECT_EQ(list->get(0).value(), "zero");
            EXPECT_EQ(list->get(1).value(), "alpha");
            EXPECT_EQ(list->get(2).value(), "beta");

            auto r = list->pop_at(1);
            ASSERT_TRUE(r.hasValue());
            EXPECT_EQ(r.value(), "alpha");
            EXPECT_EQ(list->size(), 2u);

            EXPECT_EQ(list->slab_used(), 2u);
            EXPECT_EQ(list->slab_free_count(), 1u);
            EXPECT_EQ(list->slab_remaining(), 1u);
        }
        // -----------------------------------------------------------------------------

        TEST(SListTest, GetRejectsOutOfRangeIndex) {
            HeapAllocator alloc;
            auto list = make_list<int>(2, false, alloc);

            ASSERT_TRUE(list->push_back(1));

            EXPECT_FALSE(list->get(1).hasValue());
            EXPECT_FALSE(list->get(99).hasValue());
        }

    }  // namespace
}
// ================================================================================
// ================================================================================
// eof
