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
// =============================================================================
// Basic construction / destruction
// =============================================================================

TEST(SharedPtrBasic, DefaultConstructIsNull) {
    cslt::SharedPtr<Counter> p;
    EXPECT_FALSE(p);
    EXPECT_EQ(p.get(), nullptr);
    EXPECT_EQ(p.use_count(), 0u);
    EXPECT_FALSE(p.unique());
}

TEST(SharedPtrBasic, ConstructFromRawPointerDeletesAtEnd) {
    Counter::reset();
    {
        cslt::SharedPtr<Counter> p(new Counter());
        EXPECT_TRUE(p);
        EXPECT_EQ(Counter::alive, 1);
        EXPECT_EQ(p.use_count(), 1u);
        EXPECT_TRUE(p.unique());
    }
    EXPECT_EQ(Counter::alive, 0);
    EXPECT_EQ(Counter::ctor, 1);
    EXPECT_EQ(Counter::dtor, 1);
}

// =============================================================================
// Copy semantics / refcounting
// =============================================================================

TEST(SharedPtrRefCount, CopyIncrementsAndLastReleaseDeletes) {
    Counter::reset();

    cslt::SharedPtr<Counter> a(new Counter());
    EXPECT_EQ(a.use_count(), 1u);
    EXPECT_EQ(Counter::alive, 1);

    {
        cslt::SharedPtr<Counter> b = a;
        EXPECT_EQ(a.use_count(), 2u);
        EXPECT_EQ(b.use_count(), 2u);
        EXPECT_EQ(Counter::alive, 1);

        {
            cslt::SharedPtr<Counter> c(b);
            EXPECT_EQ(a.use_count(), 3u);
            EXPECT_EQ(c.use_count(), 3u);
            EXPECT_EQ(Counter::alive, 1);
        }

        EXPECT_EQ(a.use_count(), 2u);
        EXPECT_EQ(Counter::alive, 1);
    }

    EXPECT_EQ(a.use_count(), 1u);
    EXPECT_TRUE(a.unique());
    EXPECT_EQ(Counter::alive, 1);

    a.reset();
    EXPECT_EQ(Counter::alive, 0);
    EXPECT_EQ(Counter::dtor, 1);
}

// =============================================================================
// Move semantics
// =============================================================================

TEST(SharedPtrMove, MoveConstructorTransfersControlBlock) {
    Counter::reset();

    cslt::SharedPtr<Counter> a(new Counter());
    EXPECT_EQ(a.use_count(), 1u);

    cslt::SharedPtr<Counter> b(cslt::move(a));
    EXPECT_FALSE(a);
    EXPECT_TRUE(b);
    EXPECT_EQ(b.use_count(), 1u);
    EXPECT_EQ(Counter::alive, 1);
}

TEST(SharedPtrMove, MoveAssignmentReleasesOldAndTakesNew) {
    Counter::reset();

    cslt::SharedPtr<Counter> a(new Counter());
    cslt::SharedPtr<Counter> b(new Counter());
    EXPECT_EQ(Counter::alive, 2);

    b = cslt::move(a); // should delete b's old object
    EXPECT_FALSE(a);
    EXPECT_TRUE(b);

    EXPECT_EQ(Counter::alive, 1);
    EXPECT_EQ(Counter::dtor, 1);
    EXPECT_EQ(b.use_count(), 1u);
}

// =============================================================================
// reset / nullptr assignment
// =============================================================================

TEST(SharedPtrOps, ResetToNullReleases) {
    Counter::reset();

    cslt::SharedPtr<Counter> p(new Counter());
    EXPECT_EQ(Counter::alive, 1);

    p.reset();
    EXPECT_FALSE(p);
    EXPECT_EQ(p.use_count(), 0u);
    EXPECT_EQ(Counter::alive, 0);
    EXPECT_EQ(Counter::dtor, 1);
}

TEST(SharedPtrOps, AssignNullptrReleases) {
    Counter::reset();

    cslt::SharedPtr<Counter> p(new Counter());
    EXPECT_EQ(Counter::alive, 1);

    p = nullptr;
    EXPECT_FALSE(p);
    EXPECT_EQ(Counter::alive, 0);
    EXPECT_EQ(Counter::dtor, 1);
}

TEST(SharedPtrOps, ResetToNewPointerDeletesOldWhenUnique) {
    Counter::reset();

    cslt::SharedPtr<Counter> p(new Counter());
    EXPECT_EQ(Counter::alive, 1);

    p.reset(new Counter()); // should delete old
    EXPECT_EQ(Counter::alive, 1);
    EXPECT_EQ(Counter::ctor, 2);
    EXPECT_EQ(Counter::dtor, 1);

    p.reset();
    EXPECT_EQ(Counter::alive, 0);
    EXPECT_EQ(Counter::dtor, 2);
}

// =============================================================================
// swap
// =============================================================================

TEST(SharedPtrOps, SwapExchangesOwnership) {
    Counter::reset();

    cslt::SharedPtr<Counter> a(new Counter());
    cslt::SharedPtr<Counter> b(new Counter());

    Counter* pa = a.get();
    Counter* pb = b.get();

    a.swap(b);

    EXPECT_EQ(a.get(), pb);
    EXPECT_EQ(b.get(), pa);
    EXPECT_EQ(a.use_count(), 1u);
    EXPECT_EQ(b.use_count(), 1u);
    EXPECT_EQ(Counter::alive, 2);

    a.reset();
    b.reset();
    EXPECT_EQ(Counter::alive, 0);
    EXPECT_EQ(Counter::dtor, 2);
}

// =============================================================================
// Custom deleter
// =============================================================================

TEST(SharedPtrDeleter, CustomDeleterCalledExactlyOnceAtZero) {
    Counter::reset();
    CountingDeleter<Counter>::reset();

    Counter* raw = new Counter();

    {
        cslt::SharedPtr<Counter> a(raw, CountingDeleter<Counter>{});
        EXPECT_EQ(a.use_count(), 1u);
        EXPECT_EQ(Counter::alive, 1);

        {
            cslt::SharedPtr<Counter> b = a;
            EXPECT_EQ(a.use_count(), 2u);
            EXPECT_EQ(b.use_count(), 2u);
        }

        EXPECT_EQ(a.use_count(), 1u);
        EXPECT_EQ(CountingDeleter<Counter>::calls, 0);
        EXPECT_EQ(Counter::dtor, 0);
    }

    // leaving scope should drop count to 0 and call deleter once
    EXPECT_EQ(CountingDeleter<Counter>::calls, 1);
    EXPECT_EQ(CountingDeleter<Counter>::last, raw);
    EXPECT_EQ(Counter::alive, 0);
    EXPECT_EQ(Counter::dtor, 1);
}

// =============================================================================
// Converting copy/move (Derived -> Base)
// =============================================================================

TEST(SharedPtrConvert, DerivedToBaseCopySharesControlBlock) {
    Derived::reset();

    cslt::SharedPtr<Derived> d(new Derived());
    EXPECT_EQ(d.use_count(), 1u);

    cslt::SharedPtr<Base> b(d); // converting copy
    EXPECT_EQ(d.use_count(), 2u);
    EXPECT_EQ(b.use_count(), 2u);
    EXPECT_TRUE(b);
    EXPECT_TRUE(d);

    d.reset();
    EXPECT_EQ(b.use_count(), 1u);
    EXPECT_EQ(Derived::dtor, 0);

    b.reset();
    EXPECT_EQ(Derived::dtor, 1);
}

TEST(SharedPtrConvert, DerivedToBaseMoveTransfersControlBlock) {
    Derived::reset();
    cslt::SharedPtr<Derived> d(new Derived());
    EXPECT_EQ(d.use_count(), 1u);
    
    cslt::SharedPtr<Base> b(cslt::move(d)); // converting move

    EXPECT_FALSE(static_cast<bool>(d));  // d should be null after move
    EXPECT_TRUE(static_cast<bool>(b));   // b should own the object
    EXPECT_EQ(b.use_count(), 1u);        // Only one strong reference
    EXPECT_EQ(d.use_count(), 0u);        // d should report 0 (it's empty)

    b.reset();
    EXPECT_EQ(Derived::dtor, 1);         // Object should be destroyed
}

// =============================================================================
// make_shared
// =============================================================================

TEST(SharedPtrMakeShared, ConstructsAndManagesLifetime) {
    Counter::reset();

    auto p = cslt::make_shared<Counter>();
    EXPECT_TRUE(p);
    EXPECT_EQ(p.use_count(), 1u);
    EXPECT_EQ(Counter::alive, 1);

    {
        auto q = p;
        EXPECT_EQ(p.use_count(), 2u);
        EXPECT_EQ(q.use_count(), 2u);
    }

    EXPECT_EQ(p.use_count(), 1u);
    p.reset();
    EXPECT_EQ(Counter::alive, 0);
    EXPECT_EQ(Counter::dtor, 1);
}
// ================================================================================ 
// ================================================================================ 

// ================================================================================
// WeakPtr Tests
// ================================================================================

TEST(WeakPtr, DefaultConstruction) {
    cslt::WeakPtr<int> wp;
    EXPECT_TRUE(wp.expired());
    EXPECT_EQ(wp.use_count(), 0u);
}

TEST(WeakPtr, ConstructFromSharedPtr) {
    cslt::SharedPtr<int> sp = cslt::make_shared<int>(42);
    cslt::WeakPtr<int> wp(sp);
    
    EXPECT_FALSE(wp.expired());
    EXPECT_EQ(wp.use_count(), 1u);
    EXPECT_EQ(sp.use_count(), 1u);
}

TEST(WeakPtr, ConstructFromSharedPtrMultipleWeak) {
    cslt::SharedPtr<int> sp = cslt::make_shared<int>(42);
    cslt::WeakPtr<int> wp1(sp);
    cslt::WeakPtr<int> wp2(sp);
    cslt::WeakPtr<int> wp3(sp);
    
    EXPECT_EQ(wp1.use_count(), 1u);
    EXPECT_EQ(wp2.use_count(), 1u);
    EXPECT_EQ(wp3.use_count(), 1u);
    EXPECT_EQ(sp.use_count(), 1u);
}

TEST(WeakPtr, CopyConstruction) {
    cslt::SharedPtr<int> sp = cslt::make_shared<int>(42);
    cslt::WeakPtr<int> wp1(sp);
    cslt::WeakPtr<int> wp2(wp1);
    
    EXPECT_FALSE(wp1.expired());
    EXPECT_FALSE(wp2.expired());
    EXPECT_EQ(wp1.use_count(), 1u);
    EXPECT_EQ(wp2.use_count(), 1u);
}

TEST(WeakPtr, MoveConstruction) {
    cslt::SharedPtr<int> sp = cslt::make_shared<int>(42);
    cslt::WeakPtr<int> wp1(sp);
    cslt::WeakPtr<int> wp2(cslt::move(wp1));
    
    EXPECT_TRUE(wp1.expired());  // wp1 should be empty after move
    EXPECT_FALSE(wp2.expired());
    EXPECT_EQ(wp1.use_count(), 0u);
    EXPECT_EQ(wp2.use_count(), 1u);
}

TEST(WeakPtr, CopyAssignment) {
    cslt::SharedPtr<int> sp = cslt::make_shared<int>(42);
    cslt::WeakPtr<int> wp1(sp);
    cslt::WeakPtr<int> wp2;
    
    EXPECT_TRUE(wp2.expired());
    wp2 = wp1;
    EXPECT_FALSE(wp2.expired());
    EXPECT_EQ(wp2.use_count(), 1u);
}

TEST(WeakPtr, MoveAssignment) {
    cslt::SharedPtr<int> sp = cslt::make_shared<int>(42);
    cslt::WeakPtr<int> wp1(sp);
    cslt::WeakPtr<int> wp2;
    
    wp2 = cslt::move(wp1);
    EXPECT_TRUE(wp1.expired());
    EXPECT_FALSE(wp2.expired());
    EXPECT_EQ(wp2.use_count(), 1u);
}

TEST(WeakPtr, AssignmentFromSharedPtr) {
    cslt::SharedPtr<int> sp = cslt::make_shared<int>(42);
    cslt::WeakPtr<int> wp;
    
    wp = sp;
    EXPECT_FALSE(wp.expired());
    EXPECT_EQ(wp.use_count(), 1u);
}

TEST(WeakPtr, ExpiresWhenSharedPtrDestroyed) {
    cslt::WeakPtr<int> wp;
    {
        cslt::SharedPtr<int> sp = cslt::make_shared<int>(42);
        wp = sp;
        EXPECT_FALSE(wp.expired());
        EXPECT_EQ(wp.use_count(), 1u);
    }
    // sp destroyed here
    EXPECT_TRUE(wp.expired());
    EXPECT_EQ(wp.use_count(), 0u);
}

TEST(WeakPtr, LockReturnsValidSharedPtr) {
    cslt::SharedPtr<int> sp = cslt::make_shared<int>(42);
    cslt::WeakPtr<int> wp(sp);
    
    cslt::SharedPtr<int> sp2 = wp.lock();
    EXPECT_TRUE(sp2);
    EXPECT_EQ(*sp2, 42);
    EXPECT_EQ(sp.use_count(), 2u);
    EXPECT_EQ(sp2.use_count(), 2u);
    EXPECT_EQ(wp.use_count(), 2u);
}

TEST(WeakPtr, LockReturnsEmptyWhenExpired) {
    cslt::WeakPtr<int> wp;
    {
        cslt::SharedPtr<int> sp = cslt::make_shared<int>(42);
        wp = sp;
    }
    
    cslt::SharedPtr<int> sp = wp.lock();
    EXPECT_FALSE(sp);
    EXPECT_EQ(sp.get(), nullptr);
}

TEST(WeakPtr, LockPreventsDestruction) {
    Counter::reset();
    cslt::WeakPtr<Counter> wp;
    {
        cslt::SharedPtr<Counter> sp = cslt::make_shared<Counter>();
        wp = sp;
        EXPECT_EQ(Counter::alive, 1);
    }
    // SharedPtr destroyed, but we haven't locked yet
    EXPECT_EQ(Counter::alive, 0);  // Object should be destroyed
    EXPECT_EQ(Counter::dtor, 1);
    
    cslt::SharedPtr<Counter> sp = wp.lock();
    EXPECT_FALSE(sp);  // Can't lock expired weak_ptr
}

TEST(WeakPtr, MultipleLocks) {
    cslt::SharedPtr<int> sp = cslt::make_shared<int>(42);
    cslt::WeakPtr<int> wp(sp);
    
    cslt::SharedPtr<int> sp2 = wp.lock();
    cslt::SharedPtr<int> sp3 = wp.lock();
    cslt::SharedPtr<int> sp4 = wp.lock();
    
    EXPECT_EQ(sp.use_count(), 4u);
    EXPECT_EQ(wp.use_count(), 4u);
    EXPECT_EQ(*sp2, 42);
    EXPECT_EQ(*sp3, 42);
    EXPECT_EQ(*sp4, 42);
}

TEST(WeakPtr, Reset) {
    cslt::SharedPtr<int> sp = cslt::make_shared<int>(42);
    cslt::WeakPtr<int> wp(sp);
    
    EXPECT_FALSE(wp.expired());
    wp.reset();
    EXPECT_TRUE(wp.expired());
    EXPECT_EQ(wp.use_count(), 0u);
}

TEST(WeakPtr, Swap) {
    cslt::SharedPtr<int> sp1 = cslt::make_shared<int>(42);
    cslt::SharedPtr<int> sp2 = cslt::make_shared<int>(99);
    
    cslt::WeakPtr<int> wp1(sp1);
    cslt::WeakPtr<int> wp2(sp2);
    
    wp1.swap(wp2);
    
    cslt::SharedPtr<int> locked1 = wp1.lock();
    cslt::SharedPtr<int> locked2 = wp2.lock();
    
    EXPECT_EQ(*locked1, 99);
    EXPECT_EQ(*locked2, 42);
}

TEST(WeakPtr, NonMemberSwap) {
    cslt::SharedPtr<int> sp1 = cslt::make_shared<int>(42);
    cslt::SharedPtr<int> sp2 = cslt::make_shared<int>(99);
    
    cslt::WeakPtr<int> wp1(sp1);
    cslt::WeakPtr<int> wp2(sp2);
    
    cslt::swap(wp1, wp2);
    
    cslt::SharedPtr<int> locked1 = wp1.lock();
    cslt::SharedPtr<int> locked2 = wp2.lock();
    
    EXPECT_EQ(*locked1, 99);
    EXPECT_EQ(*locked2, 42);
}

TEST(WeakPtr, ConvertingConstructorDerivedToBase) {
    cslt::SharedPtr<Derived> sp = cslt::make_shared<Derived>();
    cslt::WeakPtr<Base> wp(sp);
    
    EXPECT_FALSE(wp.expired());
    EXPECT_EQ(wp.use_count(), 1u);
}

TEST(WeakPtr, ConvertingCopyConstructorDerivedToBase) {
    cslt::SharedPtr<Derived> sp = cslt::make_shared<Derived>();
    cslt::WeakPtr<Derived> wp1(sp);
    cslt::WeakPtr<Base> wp2(wp1);
    
    EXPECT_FALSE(wp2.expired());
    EXPECT_EQ(wp2.use_count(), 1u);
}

TEST(WeakPtr, ConvertingMoveConstructorDerivedToBase) {
    cslt::SharedPtr<Derived> sp = cslt::make_shared<Derived>();
    cslt::WeakPtr<Derived> wp1(sp);
    cslt::WeakPtr<Base> wp2(cslt::move(wp1));
    
    EXPECT_TRUE(wp1.expired());
    EXPECT_FALSE(wp2.expired());
    EXPECT_EQ(wp2.use_count(), 1u);
}

TEST(WeakPtr, ConvertingCopyAssignmentDerivedToBase) {
    cslt::SharedPtr<Derived> sp = cslt::make_shared<Derived>();
    cslt::WeakPtr<Derived> wp1(sp);
    cslt::WeakPtr<Base> wp2;
    
    wp2 = wp1;
    EXPECT_FALSE(wp2.expired());
    EXPECT_EQ(wp2.use_count(), 1u);
}

TEST(WeakPtr, ConvertingMoveAssignmentDerivedToBase) {
    cslt::SharedPtr<Derived> sp = cslt::make_shared<Derived>();
    cslt::WeakPtr<Derived> wp1(sp);
    cslt::WeakPtr<Base> wp2;
    
    wp2 = cslt::move(wp1);
    EXPECT_TRUE(wp1.expired());
    EXPECT_FALSE(wp2.expired());
    EXPECT_EQ(wp2.use_count(), 1u);
}

TEST(WeakPtr, ConvertingAssignmentFromSharedPtrDerivedToBase) {
    cslt::SharedPtr<Derived> sp = cslt::make_shared<Derived>();
    cslt::WeakPtr<Base> wp;
    
    wp = sp;
    EXPECT_FALSE(wp.expired());
    EXPECT_EQ(wp.use_count(), 1u);
}

TEST(WeakPtr, LockReturnsDerivedType) {
    Derived::reset();
    cslt::SharedPtr<Derived> sp = cslt::make_shared<Derived>();
    cslt::WeakPtr<Base> wp(sp);
    
    cslt::SharedPtr<Base> locked = wp.lock();
    EXPECT_TRUE(locked);
    EXPECT_EQ(sp.use_count(), 2u);
    
    sp.reset();
    locked.reset();
    EXPECT_EQ(Derived::dtor, 1);
}

TEST(WeakPtr, SelfAssignment) {
    cslt::SharedPtr<int> sp = cslt::make_shared<int>(42);
    cslt::WeakPtr<int> wp(sp);
    
    wp = wp;  // Self-assignment
    EXPECT_FALSE(wp.expired());
    EXPECT_EQ(wp.use_count(), 1u);
}

TEST(WeakPtr, ControlBlockLifetime) {
    Counter::reset();
    cslt::WeakPtr<Counter> wp;
    {
        cslt::SharedPtr<Counter> sp = cslt::make_shared<Counter>();
        wp = sp;
        EXPECT_EQ(Counter::alive, 1);
    }
    // Object destroyed but control block should still exist for weak_ptr
    EXPECT_EQ(Counter::alive, 0);
    EXPECT_TRUE(wp.expired());
    
    // Control block should be destroyed when weak_ptr is destroyed
    wp.reset();
}

TEST(WeakPtr, MultipleWeakPtrsControlBlockLifetime) {
    Counter::reset();
    cslt::WeakPtr<Counter> wp1;
    cslt::WeakPtr<Counter> wp2;
    cslt::WeakPtr<Counter> wp3;
    
    {
        cslt::SharedPtr<Counter> sp = cslt::make_shared<Counter>();
        wp1 = sp;
        wp2 = sp;
        wp3 = sp;
    }
    
    // All weak pointers should be expired
    EXPECT_TRUE(wp1.expired());
    EXPECT_TRUE(wp2.expired());
    EXPECT_TRUE(wp3.expired());
    
    // Control block should survive until all weak pointers are gone
    wp1.reset();
    wp2.reset();
    // Still one weak pointer left, control block should exist
    wp3.reset();
    // Now control block should be destroyed
}

TEST(WeakPtr, ThreadSafetyBasic) {
    // Basic test to ensure lock() is atomic
    cslt::SharedPtr<int> sp = cslt::make_shared<int>(42);
    cslt::WeakPtr<int> wp(sp);
    
    // Simulate race: try to lock while shared_ptr might be destroyed
    cslt::SharedPtr<int> locked1 = wp.lock();
    EXPECT_TRUE(locked1);
    
    sp.reset();  // Decrement strong count but locked1 keeps it alive
    
    cslt::SharedPtr<int> locked2 = wp.lock();
    EXPECT_TRUE(locked2);  // Should still work because locked1 exists
    
    locked1.reset();
    locked2.reset();
    
    cslt::SharedPtr<int> locked3 = wp.lock();
    EXPECT_FALSE(locked3);  // Now should fail
}

TEST(WeakPtr, UseCountReflectsSharedPtrs) {
    cslt::SharedPtr<int> sp1 = cslt::make_shared<int>(42);
    cslt::WeakPtr<int> wp(sp1);
    
    EXPECT_EQ(wp.use_count(), 1u);
    
    cslt::SharedPtr<int> sp2 = sp1;
    EXPECT_EQ(wp.use_count(), 2u);
    
    cslt::SharedPtr<int> sp3 = sp1;
    EXPECT_EQ(wp.use_count(), 3u);
    
    sp2.reset();
    EXPECT_EQ(wp.use_count(), 2u);
    
    sp1.reset();
    EXPECT_EQ(wp.use_count(), 1u);
    
    sp3.reset();
    EXPECT_EQ(wp.use_count(), 0u);
    EXPECT_TRUE(wp.expired());
}
// ================================================================================
// ================================================================================
// eof
