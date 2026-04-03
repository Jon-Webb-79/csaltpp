// ================================================================================
// ================================================================================
// - File:    test_avl.cpp
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

#include <cstring>
#include <vector>

#include "tree.hpp"
#include "allocator.hpp"
#include "pointers.hpp"
// ================================================================================ 
// ================================================================================ 

namespace {

inline bool is_no_error(const cslt::Error& err) {
    return std::strcmp(err.what(), "No Error") == 0;
}

inline bool is_not_no_error(const cslt::Error& err) {
    return !is_no_error(err);
}

struct IntCompare {
    int operator()(const int& lhs, const int& rhs) const noexcept {
        return (lhs > rhs) - (lhs < rhs);
    }
};

struct ReverseIntCompare {
    int operator()(const int& lhs, const int& rhs) const noexcept {
        return (rhs > lhs) - (rhs < lhs);
    }
};

} // anonymous namespace

// ================================================================================
// ================================================================================

class AVLTreeIntTest : public ::testing::Test {
protected:
    using Tree        = cslt::AVLTree<int, IntCompare>;
    using TreeDeleter = cslt::AVLTreeDeleter<int, IntCompare>;

    cslt::HeapAllocator allocator;

    static cslt::UniquePtr<Tree, TreeDeleter>
    make_tree(cslt::HeapAllocator& alloc,
              size_t capacity = 16,
              bool overflow = true,
              bool allow_duplicates = false) {
        auto r = Tree::init(capacity, alloc, overflow, allow_duplicates, IntCompare{});
        EXPECT_TRUE(r.hasValue()) << r.error().what();
        return cslt::UniquePtr<Tree, TreeDeleter>(r.value());
    }

    static std::vector<int> inorder_values(const Tree& tree) {
        std::vector<int> vals;
        auto err = tree.foreach([&vals](const int& v) noexcept {
            vals.push_back(v);
        });
        EXPECT_TRUE(is_no_error(err)) << err.what();
        return vals;
    }
};

// ================================================================================
// Initialization
// ================================================================================

TEST_F(AVLTreeIntTest, InitSuccess) {
    auto r = Tree::init(8, allocator, true, false, IntCompare{});
    ASSERT_TRUE(r.hasValue()) << r.error().what();

    cslt::UniquePtr<Tree, TreeDeleter> tree(r.value());

    EXPECT_EQ(tree->size(), 0u);
    EXPECT_TRUE(tree->empty());
    EXPECT_EQ(tree->height(), 0);
    EXPECT_EQ(tree->slab_capacity(), 8u);
    EXPECT_EQ(tree->slab_used(), 0u);
    EXPECT_TRUE(tree->overflow_enabled());
    EXPECT_FALSE(tree->duplicates_allowed());
}

// --------------------------------------------------------------------------------

TEST_F(AVLTreeIntTest, InitFailsForZeroCapacity) {
    auto r = Tree::init(0, allocator, true, false, IntCompare{});
    EXPECT_FALSE(r.hasValue());
}

// ================================================================================
// Insert / contains / find
// ================================================================================

TEST_F(AVLTreeIntTest, InsertOneElement) {
    auto tree = make_tree(allocator);

    auto err = tree->insert(42);
    EXPECT_TRUE(is_no_error(err)) << err.what();

    EXPECT_EQ(tree->size(), 1u);
    EXPECT_FALSE(tree->empty());
    EXPECT_EQ(tree->height(), 1);
    EXPECT_TRUE(tree->contains(42));
}

// --------------------------------------------------------------------------------

TEST_F(AVLTreeIntTest, FindInsertedValues) {
    auto tree = make_tree(allocator);

    EXPECT_TRUE(is_no_error(tree->insert(10)));
    EXPECT_TRUE(is_no_error(tree->insert(5)));
    EXPECT_TRUE(is_no_error(tree->insert(20)));

    auto r1 = tree->find(10);
    ASSERT_TRUE(r1.hasValue()) << r1.error().what();
    EXPECT_EQ(r1.value(), 10);

    auto r2 = tree->find(5);
    ASSERT_TRUE(r2.hasValue()) << r2.error().what();
    EXPECT_EQ(r2.value(), 5);

    auto r3 = tree->find(20);
    ASSERT_TRUE(r3.hasValue()) << r3.error().what();
    EXPECT_EQ(r3.value(), 20);
}

// --------------------------------------------------------------------------------

TEST_F(AVLTreeIntTest, FindMissingValueFails) {
    auto tree = make_tree(allocator);

    EXPECT_TRUE(is_no_error(tree->insert(10)));
    EXPECT_TRUE(is_no_error(tree->insert(5)));
    EXPECT_TRUE(is_no_error(tree->insert(20)));

    auto r = tree->find(99);
    EXPECT_FALSE(r.hasValue());
}

// --------------------------------------------------------------------------------

TEST_F(AVLTreeIntTest, ContainsReturnsFalseForMissingValue) {
    auto tree = make_tree(allocator);

    EXPECT_TRUE(is_no_error(tree->insert(1)));
    EXPECT_TRUE(is_no_error(tree->insert(2)));
    EXPECT_TRUE(is_no_error(tree->insert(3)));

    EXPECT_TRUE(tree->contains(1));
    EXPECT_TRUE(tree->contains(2));
    EXPECT_TRUE(tree->contains(3));
    EXPECT_FALSE(tree->contains(99));
}

// ================================================================================
// Min / Max
// ================================================================================

TEST_F(AVLTreeIntTest, MinAndMaxReturnExpectedValues) {
    auto tree = make_tree(allocator);

    EXPECT_TRUE(is_no_error(tree->insert(10)));
    EXPECT_TRUE(is_no_error(tree->insert(3)));
    EXPECT_TRUE(is_no_error(tree->insert(17)));
    EXPECT_TRUE(is_no_error(tree->insert(1)));
    EXPECT_TRUE(is_no_error(tree->insert(9)));

    auto min_r = tree->min();
    auto max_r = tree->max();

    ASSERT_TRUE(min_r.hasValue()) << min_r.error().what();
    ASSERT_TRUE(max_r.hasValue()) << max_r.error().what();

    EXPECT_EQ(min_r.value(), 1);
    EXPECT_EQ(max_r.value(), 17);
}

// --------------------------------------------------------------------------------

TEST_F(AVLTreeIntTest, MinFailsOnEmptyTree) {
    auto tree = make_tree(allocator);
    auto r = tree->min();
    EXPECT_FALSE(r.hasValue());
}

// --------------------------------------------------------------------------------

TEST_F(AVLTreeIntTest, MaxFailsOnEmptyTree) {
    auto tree = make_tree(allocator);
    auto r = tree->max();
    EXPECT_FALSE(r.hasValue());
}

// ================================================================================
// Duplicate handling
// ================================================================================

TEST_F(AVLTreeIntTest, DuplicateInsertRejectedWhenDisabled) {
    auto tree = make_tree(allocator, 16, true, false);

    EXPECT_TRUE(is_no_error(tree->insert(10)));

    auto err = tree->insert(10);
    EXPECT_TRUE(is_not_no_error(err));
    EXPECT_EQ(tree->size(), 1u);
}

// --------------------------------------------------------------------------------

TEST_F(AVLTreeIntTest, DuplicateInsertAcceptedWhenEnabled) {
    auto tree = make_tree(allocator, 16, true, true);

    EXPECT_TRUE(is_no_error(tree->insert(10)));
    EXPECT_TRUE(is_no_error(tree->insert(10)));
    EXPECT_TRUE(is_no_error(tree->insert(10)));

    EXPECT_EQ(tree->size(), 3u);

    auto vals = inorder_values(*tree);
    ASSERT_EQ(vals.size(), 3u);
    EXPECT_EQ(vals[0], 10);
    EXPECT_EQ(vals[1], 10);
    EXPECT_EQ(vals[2], 10);
}

// ================================================================================
// Removal
// ================================================================================

TEST_F(AVLTreeIntTest, RemoveLeafNode) {
    auto tree = make_tree(allocator);

    EXPECT_TRUE(is_no_error(tree->insert(10)));
    EXPECT_TRUE(is_no_error(tree->insert(5)));
    EXPECT_TRUE(is_no_error(tree->insert(20)));
    
    auto r = tree->remove(5);
    ASSERT_TRUE(r.hasValue()) << r.error().what();
    EXPECT_EQ(r.value(), 5);
    
    EXPECT_EQ(tree->size(), 2u);
    EXPECT_FALSE(tree->contains(5));
    EXPECT_TRUE(tree->contains(10));
    EXPECT_TRUE(tree->contains(20));
}

// --------------------------------------------------------------------------------

TEST_F(AVLTreeIntTest, RemoveNodeWithOneChild) {
    auto tree = make_tree(allocator);

    EXPECT_TRUE(is_no_error(tree->insert(10)));
    EXPECT_TRUE(is_no_error(tree->insert(5)));
    EXPECT_TRUE(is_no_error(tree->insert(2)));

    auto r = tree->remove(5);
    ASSERT_TRUE(r.hasValue()) << r.error().what();
    EXPECT_EQ(r.value(), 5);

    EXPECT_EQ(tree->size(), 2u);
    EXPECT_FALSE(tree->contains(5));
    EXPECT_TRUE(tree->contains(10));
    EXPECT_TRUE(tree->contains(2));
}

// --------------------------------------------------------------------------------

TEST_F(AVLTreeIntTest, RemoveNodeWithTwoChildren) {
    auto tree = make_tree(allocator);

    for (int v : {10, 5, 20, 3, 7, 15, 25}) {
        EXPECT_TRUE(is_no_error(tree->insert(v))) << "insert failed for " << v;
    }

    auto r = tree->remove(10);
    ASSERT_TRUE(r.hasValue()) << r.error().what();
    EXPECT_EQ(r.value(), 10);

    EXPECT_EQ(tree->size(), 6u);
    EXPECT_FALSE(tree->contains(10));

    auto vals = inorder_values(*tree);
    std::vector<int> expected{3, 5, 7, 15, 20, 25};
    EXPECT_EQ(vals, expected);
}

// --------------------------------------------------------------------------------

TEST_F(AVLTreeIntTest, RemoveMissingValueFails) {
    auto tree = make_tree(allocator);

    EXPECT_TRUE(is_no_error(tree->insert(1)));
    EXPECT_TRUE(is_no_error(tree->insert(2)));
    EXPECT_TRUE(is_no_error(tree->insert(3)));

    auto r = tree->remove(99);
    EXPECT_FALSE(r.hasValue());
    EXPECT_EQ(tree->size(), 3u);
}

// --------------------------------------------------------------------------------

TEST_F(AVLTreeIntTest, RemoveFailsOnEmptyTree) {
    auto tree = make_tree(allocator);

    auto r = tree->remove(1);
    EXPECT_FALSE(r.hasValue());
}

// ================================================================================
// Traversal
// ================================================================================

TEST_F(AVLTreeIntTest, ForeachTraversesInSortedOrder) {
    auto tree = make_tree(allocator);

    for (int v : {10, 4, 15, 2, 8, 12, 20}) {
        EXPECT_TRUE(is_no_error(tree->insert(v)));
    }

    auto vals = inorder_values(*tree);
    std::vector<int> expected{2, 4, 8, 10, 12, 15, 20};
    EXPECT_EQ(vals, expected);
}

// --------------------------------------------------------------------------------

TEST_F(AVLTreeIntTest, ForeachRangeTraversesInclusiveRange) {
    auto tree = make_tree(allocator);

    for (int v : {10, 4, 15, 2, 8, 12, 20, 6, 9}) {
        EXPECT_TRUE(is_no_error(tree->insert(v)));
    }

    std::vector<int> vals;
    auto err = tree->foreach_range(6, 12, [&vals](const int& v) noexcept {
        vals.push_back(v);
    });

    EXPECT_TRUE(is_no_error(err)) << err.what();

    std::vector<int> expected{6, 8, 9, 10, 12};
    EXPECT_EQ(vals, expected);
}

// --------------------------------------------------------------------------------

TEST_F(AVLTreeIntTest, ForeachRangeFailsForInvalidBounds) {
    auto tree = make_tree(allocator);

    for (int v : {1, 2, 3}) {
        EXPECT_TRUE(is_no_error(tree->insert(v)));
    }

    auto err = tree->foreach_range(10, 5, [](const int&) noexcept {});
    EXPECT_TRUE(is_not_no_error(err));
}

// --------------------------------------------------------------------------------

TEST_F(AVLTreeIntTest, ForeachFailsOnEmptyTree) {
    auto tree = make_tree(allocator);

    auto err = tree->foreach([](const int&) noexcept {});
    EXPECT_TRUE(is_not_no_error(err));
}

// ================================================================================
// Balance / height sanity
// ================================================================================

TEST_F(AVLTreeIntTest, HeightRemainsLogarithmicForSequentialInsertions) {
    auto tree = make_tree(allocator, 64, true, false);

    for (int i = 1; i <= 15; ++i) {
        EXPECT_TRUE(is_no_error(tree->insert(i)));
    }

    EXPECT_EQ(tree->size(), 15u);

    // A perfectly balanced tree with 15 nodes has height 4.
    // AVL should remain close to that.
    EXPECT_LE(tree->height(), 5);
}

// --------------------------------------------------------------------------------

TEST_F(AVLTreeIntTest, ThreeSequentialInsertsBalanceTree) {
    auto tree = make_tree(allocator);

    EXPECT_TRUE(is_no_error(tree->insert(1)));
    EXPECT_TRUE(is_no_error(tree->insert(2)));
    EXPECT_TRUE(is_no_error(tree->insert(3)));

    EXPECT_EQ(tree->size(), 3u);
    EXPECT_EQ(tree->height(), 2);

    auto vals = inorder_values(*tree);
    std::vector<int> expected{1, 2, 3};
    EXPECT_EQ(vals, expected);
}

// ================================================================================
// Copy
// ================================================================================

TEST_F(AVLTreeIntTest, CopyProducesIndependentTree) {
    auto tree = make_tree(allocator);

    for (int v : {10, 5, 20, 3, 7, 15, 25}) {
        EXPECT_TRUE(is_no_error(tree->insert(v)));
    }

    auto copy_r = Tree::copy(*tree, allocator);
    ASSERT_TRUE(copy_r.hasValue()) << copy_r.error().what();

    cslt::UniquePtr<Tree, TreeDeleter> copy(copy_r.value());

    auto orig_vals = inorder_values(*tree);
    auto copy_vals = inorder_values(*copy);

    EXPECT_EQ(orig_vals, copy_vals);

    EXPECT_TRUE(is_no_error(copy->insert(100)));
    EXPECT_TRUE(copy->contains(100));
    EXPECT_FALSE(tree->contains(100));
}

// ================================================================================
// Clear
// ================================================================================

TEST_F(AVLTreeIntTest, ClearResetsTreeState) {
    auto tree = make_tree(allocator);

    for (int v : {10, 5, 20, 3, 7}) {
        EXPECT_TRUE(is_no_error(tree->insert(v)));
    }

    EXPECT_FALSE(tree->empty());
    EXPECT_GT(tree->size(), 0u);

    tree->clear();

    EXPECT_TRUE(tree->empty());
    EXPECT_EQ(tree->size(), 0u);
    EXPECT_EQ(tree->height(), 0);
    EXPECT_FALSE(tree->contains(10));

    auto min_r = tree->min();
    EXPECT_FALSE(min_r.hasValue());
}

// ================================================================================
// Custom comparator
// ================================================================================

TEST(AVLTreeCustomComparatorTest, ReverseComparatorChangesTraversalOrder) {
    using Tree        = cslt::AVLTree<int, ReverseIntCompare>;
    using TreeDeleter = cslt::AVLTreeDeleter<int, ReverseIntCompare>;

    cslt::HeapAllocator allocator;

    auto r = Tree::init(16, allocator, true, false, ReverseIntCompare{});
    ASSERT_TRUE(r.hasValue()) << r.error().what();

    cslt::UniquePtr<Tree, TreeDeleter> tree(r.value());

    for (int v : {10, 5, 20, 3, 7, 15, 25}) {
        EXPECT_TRUE(is_no_error(tree->insert(v)));
    }

    std::vector<int> vals;
    auto err = tree->foreach([&vals](const int& v) noexcept {
        vals.push_back(v);
    });

    EXPECT_TRUE(is_no_error(err)) << err.what();

    std::vector<int> expected{25, 20, 15, 10, 7, 5, 3};
    EXPECT_EQ(vals, expected);
}
// ================================================================================
// ================================================================================
// eof
