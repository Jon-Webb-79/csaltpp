// ================================================================================
// ================================================================================
// - File:    test_utilities.cpp
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
// ================================================================================ 
// ================================================================================ 

namespace {

    struct Base {};
    struct Derived : Base {};
    struct Unrelated {};

    struct Movable {
        int*  p = nullptr;

        static inline int ctor     = 0;
        static inline int dtor     = 0;
        static inline int copyCtor = 0;
        static inline int moveCtor = 0;
        static inline int copyAsg  = 0;
        static inline int moveAsg  = 0;

        static void reset_counts() {
            ctor = dtor = copyCtor = moveCtor = copyAsg = moveAsg = 0;
        }

        Movable() : p(new int(123)) { ++ctor; }

        explicit Movable(int v) : p(new int(v)) { ++ctor; }

        ~Movable() {
            ++dtor;
            delete p;
            p = nullptr;
        }

        Movable(const Movable& other) : p(other.p ? new int(*other.p) : nullptr) {
            ++copyCtor;
        }

        Movable(Movable&& other) noexcept : p(other.p) {
            ++moveCtor;
            other.p = nullptr;
        }

        Movable& operator=(const Movable& other) {
            ++copyAsg;
            if (this != &other) {
                delete p;
                p = other.p ? new int(*other.p) : nullptr;
            }
            return *this;
        }

        Movable& operator=(Movable&& other) noexcept {
            ++moveAsg;
            if (this != &other) {
                delete p;
                p = other.p;
                other.p = nullptr;
            }
            return *this;
        }
    };

    // Helper overload set to test forward preserves value category
    int category(Base&) { return 1; }
    int category(Base&&) { return 2; }

    template <class T>
    int forward_category(T&& x) {
        // Perfect-forward into overload set
        return category(cslt::forward<T>(x));
    }

} // namespace

// =============================================================================
// Compile-time trait tests (static_assert)
// =============================================================================

TEST(UtilitiesTraits, BoolConstantTrueFalse) {
    static_assert(cslt::TrueType::value, "TrueType should be true");
    static_assert(!cslt::FalseType::value, "FalseType should be false");

    EXPECT_TRUE(cslt::TrueType::value);
    EXPECT_FALSE(cslt::FalseType::value);
}

TEST(UtilitiesTraits, RemoveRef) {
    static_assert(cslt::IsSame<cslt::RemoveRefT<int>, int>::value, "RemoveRefT<int> should be int");
    static_assert(cslt::IsSame<cslt::RemoveRefT<int&>, int>::value, "RemoveRefT<int&> should be int");
    static_assert(cslt::IsSame<cslt::RemoveRefT<int&&>, int>::value, "RemoveRefT<int&&> should be int");

    EXPECT_TRUE((cslt::IsSame<cslt::RemoveRefT<int&>, int>::value));
    EXPECT_TRUE((cslt::IsSame<cslt::RemoveRefT<int&&>, int>::value));
}

TEST(UtilitiesTraits, IsLValueRef) {
    static_assert(cslt::IsLValueRef<int&>::value, "int& should be lvalue ref");
    static_assert(!cslt::IsLValueRef<int&&>::value, "int&& should not be lvalue ref");
    static_assert(!cslt::IsLValueRef<int>::value, "int should not be lvalue ref");

    EXPECT_TRUE((cslt::IsLValueRef<int&>::value));
    EXPECT_FALSE((cslt::IsLValueRef<int&&>::value));
    EXPECT_FALSE((cslt::IsLValueRef<int>::value));
}

TEST(UtilitiesTraits, EnableIf) {
    // If EnableIf is wrong, these typedefs will fail to compile.
    using A = cslt::EnableIfT<true, int>;
    (void)sizeof(A);

    // This should not compile if uncommented:
    // using B = cslt::EnableIfT<false, int>;

    SUCCEED();
}

TEST(UtilitiesTraits, IsSame) {
    static_assert(cslt::IsSame<int, int>::value, "int == int");
    static_assert(!cslt::IsSame<int, float>::value, "int != float");

    EXPECT_TRUE((cslt::IsSame<int, int>::value));
    EXPECT_FALSE((cslt::IsSame<int, float>::value));
}

TEST(UtilitiesTraits, IsArray) {
    static_assert(cslt::IsArray<int[]>::value, "int[] should be array");
    static_assert(cslt::IsArray<int[3]>::value, "int[3] should be array");
    static_assert(!cslt::IsArray<int>::value, "int should not be array");
    static_assert(!cslt::IsArray<int*>::value, "int* should not be array");

    EXPECT_TRUE((cslt::IsArray<int[]>::value));
    EXPECT_TRUE((cslt::IsArray<int[3]>::value));
    EXPECT_FALSE((cslt::IsArray<int>::value));
    EXPECT_FALSE((cslt::IsArray<int*>::value));
}

TEST(UtilitiesTraits, IsConvertibleCommonCases) {
    static_assert(cslt::IsConvertible<int, double>::value, "int -> double convertible");
    static_assert(cslt::IsConvertible<Derived*, Base*>::value, "Derived* -> Base* convertible");
    static_assert(!cslt::IsConvertible<Base*, Derived*>::value, "Base* -> Derived* NOT convertible");
    static_assert(!cslt::IsConvertible<Unrelated*, Base*>::value, "Unrelated* -> Base* NOT convertible");

    // qualifiers
    static_assert(cslt::IsConvertible<int*, const int*>::value, "int* -> const int* convertible");
    static_assert(!cslt::IsConvertible<const int*, int*>::value, "const int* -> int* NOT convertible");

    EXPECT_TRUE((cslt::IsConvertible<int, double>::value));
    EXPECT_TRUE((cslt::IsConvertible<Derived*, Base*>::value));
    EXPECT_FALSE((cslt::IsConvertible<Base*, Derived*>::value));
    EXPECT_FALSE((cslt::IsConvertible<Unrelated*, Base*>::value));
    EXPECT_TRUE((cslt::IsConvertible<int*, const int*>::value));
    EXPECT_FALSE((cslt::IsConvertible<const int*, int*>::value));
}

// =============================================================================
// Runtime behavior tests for move / forward / swap
// =============================================================================

TEST(UtilitiesFuncs, MoveCastsToRvalueAndEnablesMoveCtor) {
    Movable::reset_counts();

    Movable a(7);
    EXPECT_NE(a.p, nullptr);

    Movable b(cslt::move(a)); // should use move ctor
    EXPECT_EQ(Movable::moveCtor, 1);
    EXPECT_EQ(Movable::copyCtor, 0);

    // moved-from should relinquish ownership in our type
    EXPECT_EQ(a.p, nullptr);
    EXPECT_NE(b.p, nullptr);
    EXPECT_EQ(*b.p, 7);
}

TEST(UtilitiesFuncs, SwapUsesMovesNotCopies) {
    Movable::reset_counts();

    Movable a(1);
    Movable b(2);

    cslt::swap(a, b);

    // Our swap implementation uses 1 move-ctor (tmp) and 2 move-assignments.
    EXPECT_EQ(Movable::copyCtor, 0);
    EXPECT_EQ(Movable::copyAsg, 0);
    EXPECT_EQ(Movable::moveCtor, 1);
    EXPECT_EQ(Movable::moveAsg, 2);

    ASSERT_NE(a.p, nullptr);
    ASSERT_NE(b.p, nullptr);
    EXPECT_EQ(*a.p, 2);
    EXPECT_EQ(*b.p, 1);
}

TEST(UtilitiesFuncs, ForwardPreservesLvalueCategory) {
    Base b;
    // lvalue should pick Base& overload => 1
    EXPECT_EQ(forward_category(b), 1);
}

TEST(UtilitiesFuncs, ForwardPreservesRvalueCategory) {
    // rvalue should pick Base&& overload => 2
    EXPECT_EQ(forward_category(Base{}), 2);
}
// ================================================================================
// ================================================================================
// eof
