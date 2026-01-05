.. _utilities_file:

*************************
Utilities and Type Traits
*************************

The cslt utilities module provides a minimal set of
standalone type traits and utility functions designed
for embedded and freestanding environments.

This module intentionally avoids std::type_traits,
compiler builtins, and dynamic allocation.


Fundamental Types
=================

nullptr_t
---------
Alias for the type of ``nullptr``.

This is provided to support APIs that accept or compare against ``nullptr``
without relying on the C++ standard library type aliases.

Example::

    cslt::nullptr_t p = nullptr;


Type Traits
===========

BoolConstant
------------
Represents a compile-time boolean value.

Used as a base for other traits.

- ``TrueType`` → true
- ``FalseType`` → false

Example::

    static_assert(cslt::TrueType::value, "Expected true");
    static_assert(!cslt::FalseType::value, "Expected false");

RemoveRef
---------
Removes reference qualifiers from a type.

- ``RemoveRef<T>::type`` yields the underlying type.
- ``RemoveRefT<T>`` is a convenience alias.

Example::

    static_assert(cslt::IsSame<cslt::RemoveRefT<int&>, int>::value);
    static_assert(cslt::IsSame<cslt::RemoveRefT<int&&>, int>::value);

IsLValueRef
-----------
Detects whether a type is an lvalue reference.

- ``IsLValueRef<T>::value`` is true only when ``T`` is of the form ``U&``.

Example::

    static_assert(cslt::IsLValueRef<int&>::value);
    static_assert(!cslt::IsLValueRef<int&&>::value);
    static_assert(!cslt::IsLValueRef<int>::value);

EnableIf
--------
Enables or disables template instantiations based on compile-time conditions
(SFINAE).

- ``EnableIf<cond, T>::type`` exists only when ``cond`` is true.
- ``EnableIfT<cond, T>`` is a convenience alias.

Example::

    template <class T, class = cslt::EnableIfT<cslt::IsSame<T,int>::value>>
    int only_for_ints(T) { return 0; }

IsSame
------
Checks whether two types are exactly the same.

- ``IsSame<A,B>::value`` is true only if ``A`` and ``B`` are identical types.

Example::

    static_assert(cslt::IsSame<int, int>::value);
    static_assert(!cslt::IsSame<int, float>::value);

IsArray
-------
Detects whether a type is an array.

Supports:
- ``T[]`` (unknown bound arrays)
- ``T[N]`` (known bound arrays)

Example::

    static_assert(cslt::IsArray<int[]>::value);
    static_assert(cslt::IsArray<int[4]>::value);
    static_assert(!cslt::IsArray<int>::value);

IsConvertible
-------------
Checks whether a value of one type can be implicitly converted to another.

- ``IsConvertible<From,To>::value`` is true if ``From`` is implicitly convertible
  to ``To``.

This is used internally to support safe pointer conversions (e.g., ``Derived*`` → ``Base*``)
and to constrain template overloads.

Example::

    struct Base {};
    struct Derived : Base {};

    static_assert(cslt::IsConvertible<Derived*, Base*>::value);
    static_assert(!cslt::IsConvertible<Base*, Derived*>::value);


Utility Functions
=================

move
----
Casts a value to an rvalue reference, enabling move semantics.

This function is equivalent in intent to ``std::move``.

Example::

    struct X { X(){} X(X&&){} };
    X a;
    X b = cslt::move(a);

forward
-------
Perfect-forwards a value with its original value category (lvalue/rvalue).

This function is equivalent in intent to ``std::forward`` and is typically used
in forwarding constructors and factory functions.

Notes
^^^^^
``cslt::forward`` is provided as two overloads (lvalue and rvalue). Passing an rvalue
through the lvalue form is prevented by a compile-time check.

Example::

    template <class T>
    void sink(T&& x) {
        // forwards x as lvalue if caller passed lvalue, otherwise as rvalue
        consume(cslt::forward<T>(x));
    }

swap
----
Swaps two objects using move operations.

This function is intended as a minimal standalone equivalent of ``std::swap``.

Example::

    int a = 1;
    int b = 2;
    cslt::swap(a, b);

declval
-------
Produces a reference to a type in an unevaluated context.

``declval`` is used for template metaprogramming and compile-time detection,
and must not be evaluated at runtime. It is declared but intentionally not defined.

This is provided primarily to implement traits such as ``IsConvertible``.

Example::

    // Example pattern (illustrative):
    // decltype(cslt::declval<T&>()) yields T& in unevaluated contexts.

