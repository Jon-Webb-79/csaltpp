// ================================================================================
// ================================================================================
// - File:    test_pointers.cpp
// - Purpose: This file implements google test as a method to test C++ code.
//            Describe the type of testing to be completed
//
// Source Metadata
// - Author:  Jonathan A. Webb
// - Date:    January 04, 2026
// - Version: 1.0
// - Copyright: Copyright 2026, Jon Webb Inc.
// ================================================================================
// ================================================================================
// - Begin test

#include <gtest/gtest.h>
#include "utilities.hpp"
#include "pointers.hpp"
// ================================================================================ 
// ================================================================================ 

namespace {

// -----------------------------------------------------------------------------
// Helpers / fixtures
// -----------------------------------------------------------------------------

struct Counter {
    static inline int alive = 0;
    static inline int ctor  = 0;
    static inline int dtor  = 0;

    static void reset() { alive = ctor = dtor = 0; }

    Counter()  { ++alive; ++ctor; }
    ~Counter() { --alive; ++dtor; }
};

struct Base {
    virtual ~Base() = default;
};
struct Derived : Base {
    static inline int dtor = 0;
    ~Derived() override { ++dtor; }
    static void reset() { dtor = 0; }
};

// Value deleter that counts calls and stores last pointer
template <class T>
struct CountingDeleter {
    static inline int calls = 0;
    static inline T*  last  = nullptr;

    static void reset() { calls = 0; last = nullptr; }

    void operator()(T* p) const noexcept {
        ++calls;
        last = p;
        delete p;
    }
};

// Array deleter that counts calls
template <class T>
struct CountingArrayDeleter {
    static inline int calls = 0;
    static inline T*  last  = nullptr;

    static void reset() { calls = 0; last = nullptr; }

    void operator()(T* p) const noexcept {
        ++calls;
        last = p;
        delete[] p;
    }
};

// Reference deleter type (non-copyable, non-movable) to prove reference storage works
struct RefDeleter {
    int calls = 0;
    void* last = nullptr;

    RefDeleter() = default;
    RefDeleter(const RefDeleter&) = delete;
    RefDeleter& operator=(const RefDeleter&) = delete;

    void operator()(Counter* p) noexcept {
        ++calls;
        last = p;
        delete p;
    }
};

} // namespace

// =============================================================================
// Compile-time checks (basic API shape)
// =============================================================================

TEST(UniquePtrCompileTime, IsMoveOnly) {
    static_assert(!cslt::IsCopyConstructible<cslt::UniquePtr<int>>::value,
                  "UniquePtr should not be copy constructible");
    static_assert(!cslt::IsCopyAssignable<cslt::UniquePtr<int>>::value,
                  "UniquePtr should not be copy assignable");

    static_assert(cslt::IsMoveConstructible<cslt::UniquePtr<int>>::value,
                  "UniquePtr should be move constructible");
    static_assert(cslt::IsMoveAssignable<cslt::UniquePtr<int>>::value,
                  "UniquePtr should be move assignable");

    SUCCEED();
}

// =============================================================================
// DefaultDelete behavior
// =============================================================================

TEST(UniquePtrDefaultDelete, DeletesOnScopeExit) {
    Counter::reset();

    {
        cslt::UniquePtr<Counter> p(new Counter());
        EXPECT_TRUE(static_cast<bool>(p));
        EXPECT_EQ(Counter::alive, 1);
    }

    EXPECT_EQ(Counter::alive, 0);
    EXPECT_EQ(Counter::ctor, 1);
    EXPECT_EQ(Counter::dtor, 1);
}

TEST(UniquePtrDefaultDelete, ResetDeletesOld) {
    Counter::reset();

    cslt::UniquePtr<Counter> p(new Counter());
    EXPECT_EQ(Counter::alive, 1);

    p.reset(new Counter()); // deletes previous
    EXPECT_EQ(Counter::alive, 1);
    EXPECT_EQ(Counter::ctor, 2);
    EXPECT_EQ(Counter::dtor, 1);

    p.reset(); // deletes current
    EXPECT_FALSE(p);
    EXPECT_EQ(Counter::alive, 0);
    EXPECT_EQ(Counter::dtor, 2);
}

TEST(UniquePtrDefaultDelete, ReleaseDoesNotDelete) {
    Counter::reset();

    Counter* raw = nullptr;
    {
        cslt::UniquePtr<Counter> p(new Counter());
        raw = p.release();
        EXPECT_FALSE(p);
        EXPECT_NE(raw, nullptr);
        EXPECT_EQ(Counter::alive, 1);
    }

    // still alive because we released ownership
    EXPECT_EQ(Counter::alive, 1);
    delete raw;
    EXPECT_EQ(Counter::alive, 0);
    EXPECT_EQ(Counter::dtor, 1);
}

TEST(UniquePtrDefaultDelete, NullptrAssignmentResets) {
    Counter::reset();

    cslt::UniquePtr<Counter> p(new Counter());
    EXPECT_TRUE(p);
    EXPECT_EQ(Counter::alive, 1);

    p = nullptr;
    EXPECT_FALSE(p);
    EXPECT_EQ(Counter::alive, 0);
    EXPECT_EQ(Counter::dtor, 1);
}

// =============================================================================
// Move semantics
// =============================================================================

TEST(UniquePtrMove, MoveConstructorTransfersOwnership) {
    Counter::reset();

    cslt::UniquePtr<Counter> a(new Counter());
    Counter* raw = a.get();

    cslt::UniquePtr<Counter> b(cslt::move(a));

    EXPECT_FALSE(a);
    EXPECT_TRUE(b);
    EXPECT_EQ(b.get(), raw);
    EXPECT_EQ(Counter::alive, 1);
}

TEST(UniquePtrMove, MoveAssignmentTransfersOwnershipAndDeletesOld) {
    Counter::reset();

    cslt::UniquePtr<Counter> a(new Counter());
    cslt::UniquePtr<Counter> b(new Counter());

    EXPECT_EQ(Counter::alive, 2);

    Counter* raw_a = a.get();
    b = cslt::move(a);

    EXPECT_FALSE(a);
    EXPECT_TRUE(b);
    EXPECT_EQ(b.get(), raw_a);

    // b's old object should have been deleted
    EXPECT_EQ(Counter::alive, 1);
    EXPECT_EQ(Counter::dtor, 1);
}

// =============================================================================
// Custom deleter (value)
// =============================================================================

TEST(UniquePtrDeleterValue, UsesCustomDeleterOnResetAndDestruction) {
    CountingDeleter<Counter>::reset();
    Counter::reset();

    {
        cslt::UniquePtr<Counter, CountingDeleter<Counter>> p(new Counter(),
                                                            CountingDeleter<Counter>{});
        EXPECT_EQ(Counter::alive, 1);

        p.reset(new Counter()); // should call deleter on old
        EXPECT_EQ(CountingDeleter<Counter>::calls, 1);
        EXPECT_EQ(Counter::dtor, 1);
        EXPECT_EQ(Counter::alive, 1);
    }

    // destructor should delete last one
    EXPECT_EQ(CountingDeleter<Counter>::calls, 2);
    EXPECT_EQ(Counter::dtor, 2);
    EXPECT_EQ(Counter::alive, 0);
}

// =============================================================================
// Reference deleter
// =============================================================================

TEST(UniquePtrDeleterRef, BindsToExternalDeleterAndCallsIt) {
    Counter::reset();

    RefDeleter d;
    {
        cslt::UniquePtr<Counter, RefDeleter&> p(new Counter(), d);
        EXPECT_EQ(d.calls, 0);
        EXPECT_EQ(Counter::alive, 1);

        p.reset(); // should call external deleter
        EXPECT_EQ(d.calls, 1);
        EXPECT_EQ(Counter::alive, 0);
        EXPECT_EQ(Counter::dtor, 1);
        EXPECT_FALSE(p);
    }
}

TEST(UniquePtrDeleterRef, MoveDoesNotRebindDeleter) {
    Counter::reset();

    RefDeleter d;
    cslt::UniquePtr<Counter, RefDeleter&> a(new Counter(), d);

    cslt::UniquePtr<Counter, RefDeleter&> b(cslt::move(a));
    EXPECT_FALSE(a);
    EXPECT_TRUE(b);

    b.reset();
    EXPECT_EQ(d.calls, 1);
    EXPECT_EQ(Counter::alive, 0);
}

// =============================================================================
// swap
// =============================================================================

TEST(UniquePtrOps, SwapExchangesPointers) {
    Counter::reset();

    cslt::UniquePtr<Counter> a(new Counter());
    cslt::UniquePtr<Counter> b(new Counter());

    Counter* pa = a.get();
    Counter* pb = b.get();

    a.swap(b);
    EXPECT_EQ(a.get(), pb);
    EXPECT_EQ(b.get(), pa);
    EXPECT_EQ(Counter::alive, 2);
}

// =============================================================================
// Comparisons
// =============================================================================

TEST(UniquePtrCompare, EqualityAndNullptr) {
    Counter::reset();

    cslt::UniquePtr<Counter> a;
    cslt::UniquePtr<Counter> b;

    EXPECT_TRUE(a == nullptr);
    EXPECT_TRUE(nullptr == a);
    EXPECT_FALSE(a != nullptr);

    a.reset(new Counter());
    EXPECT_TRUE(a != nullptr);
    EXPECT_FALSE(a == nullptr);

    b.reset(a.get()); // NOTE: this is intentionally dangerous; just for pointer compare test
    // To avoid double delete, release b immediately
    EXPECT_TRUE(a == b);
    (void)b.release();
}

// =============================================================================
// Array specialization
// =============================================================================

TEST(UniquePtrArray, DeletesArrayOnScopeExit) {
    CountingArrayDeleter<int>::reset();

    {
        cslt::UniquePtr<int[], CountingArrayDeleter<int>> p(new int[4]{1,2,3,4},
                                                           CountingArrayDeleter<int>{});
        EXPECT_TRUE(p);
        EXPECT_EQ(p[0], 1);
        EXPECT_EQ(p[3], 4);
    }

    EXPECT_EQ(CountingArrayDeleter<int>::calls, 1);
    EXPECT_NE(CountingArrayDeleter<int>::last, nullptr);
}

TEST(UniquePtrArray, ResetDeletesOldArray) {
    CountingArrayDeleter<int>::reset();

    cslt::UniquePtr<int[], CountingArrayDeleter<int>> p(new int[2]{7,8},
                                                       CountingArrayDeleter<int>{});
    EXPECT_EQ(p[0], 7);

    p.reset(new int[3]{1,2,3});
    EXPECT_EQ(CountingArrayDeleter<int>::calls, 1);
    EXPECT_EQ(p[2], 3);

    p.reset();
    EXPECT_EQ(CountingArrayDeleter<int>::calls, 2);
    EXPECT_FALSE(p);
}

// =============================================================================
// Converting move (Derived -> Base) with default deleter
// =============================================================================

TEST(UniquePtrConvert, DerivedToBaseMoveConstruct) {
    Derived::reset();

    cslt::UniquePtr<Derived> d(new Derived());
    cslt::UniquePtr<Base> b(cslt::move(d));

    EXPECT_FALSE(d);
    EXPECT_TRUE(b);

    b.reset();
    EXPECT_EQ(Derived::dtor, 1);
}

// ================================================================================
// ================================================================================
// eof
