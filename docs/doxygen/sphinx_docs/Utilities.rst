.. _utilities_file:

**********************************
Utilities and Type Traits Overview
**********************************

**utilities.hpp** provides foundational type traits, metaprogramming utilities, 
and core C++ utilities that serve as building blocks for more complex library 
components. The header is designed to be self-contained with minimal 
dependencies on the standard library, emphasizing clarity, correctness, and 
compile-time type safety.

Design Philosophy
==================

The utilities library follows these core principles:

* **Self-Contained**: Minimal dependencies - only requires ``<cstddef>`` and ``<atomic>``
* **Type-Safe**: Heavy use of SFINAE and template metaprogramming to catch errors at compile-time
* **Zero Runtime Overhead**: All type traits and utilities are resolved at compile-time
* **Standard-Compliant**: Closely mirrors C++ standard library type traits and utilities
* **Educational**: Code is structured to be readable and demonstrate template metaprogramming techniques

Library Components
==================

Type Manipulation Utilities
____________________________

Reference Removal
~~~~~~~~~~~~~~~~~

``RemoveRef<T>`` removes reference qualifiers from a type.

.. code-block:: cpp

   RemoveRef<int&>::type    // int
   RemoveRef<int&&>::type   // int
   RemoveRef<int>::type     // int

**Alias**: ``RemoveRefT<T>`` provides convenient access to ``RemoveRef<T>::type``

Type Query Traits
_________________

IsLValueRef<T>
~~~~~~~~~~~~~~

Detects whether a type is an lvalue reference.

.. code-block:: cpp

   IsLValueRef<int&>::value   // true
   IsLValueRef<int&&>::value  // false
   IsLValueRef<int>::value    // false

IsSame<A, B>
~~~~~~~~~~~~

Checks if two types are identical.

.. code-block:: cpp

   IsSame<int, int>::value           // true
   IsSame<int, const int>::value     // false
   IsSame<int&, int>::value          // false

IsArray<T>
~~~~~~~~~~

Detects array types (both bounded and unbounded).

.. code-block:: cpp

   IsArray<int[]>::value      // true
   IsArray<int[10]>::value    // true
   IsArray<int>::value        // false
   IsArray<int*>::value       // false

Type Relationship Traits
________________________

IsConvertible<From, To>
~~~~~~~~~~~~~~~~~~~~~~~

Determines if type ``From`` can be implicitly converted to type ``To``.

.. code-block:: cpp

   IsConvertible<int, double>::value        // true
   IsConvertible<Derived*, Base*>::value    // true
   IsConvertible<Base*, Derived*>::value    // false

**Implementation Note**: Uses the detection idiom with a helper function ``_accept()`` to test convertibility via overload resolution.

IsConstructible<T, Args...>
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Checks if type ``T`` can be constructed from the given argument types.

.. code-block:: cpp

   IsConstructible<int, double>::value           // true
   IsConstructible<std::string, const char*>::value  // true
   IsConstructible<int>::value                   // true (default constructible)

**Derived Traits**:

* ``IsDefaultConstructible<T>`` - Can ``T`` be default-constructed?
* ``IsMoveConstructible<T>`` - Can ``T`` be constructed from ``T&&``?
* ``IsCopyConstructible<T>`` - Can ``T`` be constructed from ``const T&``?

IsAssignable<T, U>
~~~~~~~~~~~~~~~~~~

Determines if an object of type ``T`` can be assigned a value of type ``U``.

.. code-block:: cpp

   IsAssignable<int&, int>::value           // true
   IsAssignable<const int&, int>::value     // false
   IsAssignable<int, int>::value            // false (not a reference)

**Derived Traits**:

* ``IsMoveAssignable<T>`` - Can ``T&`` be assigned from ``T&&``?
* ``IsCopyAssignable<T>`` - Can ``T&`` be assigned from ``const T&``?

Metaprogramming Utilities
__________________________

BoolConstant<B>
~~~~~~~~~~~~~~~

Wraps a compile-time boolean value in a type.

.. code-block:: cpp

   BoolConstant<true>::value   // true
   BoolConstant<false>::value  // false

**Aliases**:

* ``TrueType`` - Alias for ``BoolConstant<true>``
* ``FalseType`` - Alias for ``BoolConstant<false>``

**Usage**: Commonly used for tag dispatch to select function overloads at compile-time.

EnableIf<Cond, T>
~~~~~~~~~~~~~~~~~

SFINAE helper that defines a member ``type`` only if the condition is true.

.. code-block:: cpp

   // Only defines ::type if Cond is true
   EnableIf<true, int>::type   // int
   EnableIf<false, int>::type  // Does not exist (SFINAE)

**Alias**: ``EnableIfT<Cond, T>`` provides convenient access to ``EnableIf<Cond, T>::type``

**Typical Usage**:

.. code-block:: cpp

   template <class T, class = EnableIfT<IsIntegral<T>::value>>
   void process(T value) {
       // Only enabled for integral types
   }

VoidT<Ts...>
~~~~~~~~~~~~

Maps any sequence of types to ``void``. Used in detection idioms.

.. code-block:: cpp

   VoidT<int, double, char>  // void

**Typical Usage**:

.. code-block:: cpp

   template <class T, class = void>
   struct HasTypedef : FalseType { };
   
   template <class T>
   struct HasTypedef<T, VoidT<typename T::value_type>> : TrueType { };

Core Utility Functions
______________________

move()
~~~~~~

Casts an lvalue to an rvalue reference, enabling move semantics.

.. code-block:: cpp

   template <class T>
   constexpr RemoveRefT<T>&& move(T&& x) noexcept;

**Usage**:

.. code-block:: cpp

   std::string s1 = "hello";
   std::string s2 = cslt::move(s1);  // s1 is now in moved-from state

forward()
~~~~~~~~~

Perfect forwarding utility that preserves value categories.

.. code-block:: cpp

   template <class T>
   constexpr T&& forward(RemoveRefT<T>& x) noexcept;
   
   template <class T>
   constexpr T&& forward(RemoveRefT<T>&& x) noexcept;

**Usage**:

.. code-block:: cpp

   template <class T, class... Args>
   void construct(T* ptr, Args&&... args) {
       new (ptr) T(cslt::forward<Args>(args)...);
   }

swap()
~~~~~~

Exchanges the values of two objects.

.. code-block:: cpp

   template <class T>
   constexpr void swap(T& a, T& b) noexcept;

**Usage**:

.. code-block:: cpp

   int x = 1, y = 2;
   cslt::swap(x, y);  // x == 2, y == 1

declval()
~~~~~~~~~

Obtains a reference to a type without constructing an object. Used in unevaluated contexts.

.. code-block:: cpp

   template <class T>
   T&& declval() noexcept;

**Usage**:

.. code-block:: cpp

   // Detect if T has a size() member function
   template <class T>
   using HasSize = decltype(declval<T>().size());

Internal Implementation Details
===============================

The library uses the ``detail`` namespace for implementation details that are not part of the public API but are documented here for completeness.

Detection Idiom Helpers
_______________________

IsConvertibleImpl<From, To>
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implementation of ``IsConvertible`` using SFINAE and overload resolution:

.. code-block:: cpp

   template <class From, class To>
   struct IsConvertibleImpl {
   private:
       template <class F, class = decltype(_accept<To>(declval<F>()))>
       static char test(int);
       
       template <class>
       static long test(...);
       
   public:
       static constexpr bool value = (sizeof(test<From>(0)) == sizeof(char));
   };

IsConstructibleImpl<T, Args...>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implementation of ``IsConstructible`` using direct construction syntax:

.. code-block:: cpp

   template <class T, class... Args>
   struct IsConstructibleImpl {
   private:
       template <class U, class = decltype(U(declval<Args>()...))>
       static char test(int);
       
       template <class, class...>
       static long test(...);
       
   public:
       static constexpr bool value = (sizeof(test<T>(0)) == sizeof(char));
   };

IsAssignableImpl<T, U>
~~~~~~~~~~~~~~~~~~~~~~

Implementation of ``IsAssignable`` testing assignment expression validity:

.. code-block:: cpp

   template <class T, class U>
   struct IsAssignableImpl {
   private:
       template <class X, class Y, class = decltype(declval<X>() = declval<Y>())>
       static char test(int);
       
       template <class, class>
       static long test(...);
       
   public:
       static constexpr bool value = (sizeof(test<T, U>(0)) == sizeof(char));
   };

Reference Counting Primitives
______________________________

Control Block Structure
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cpp

   struct ControlBlock {
       std::atomic<size_t> strong;  // Strong reference count
       std::atomic<size_t> weak;    // Weak reference count
       
       ControlBlock() : strong(1u), weak(1u) {}
       virtual ~ControlBlock() {}
       
       virtual void destroy_object() noexcept = 0;
       virtual void* get_ptr() const noexcept = 0;
   };

The control block is used by smart pointers for thread-safe reference counting. It maintains separate counters for strong (owning) and weak (observing) references.

ControlBlockPtr<T, Deleter>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Concrete control block that stores a pointer and custom deleter:

.. code-block:: cpp

   template <class T, class Deleter>
   struct ControlBlockPtr final : ControlBlock {
       T* ptr;
       Deleter del;
       
       void destroy_object() noexcept override {
           if (ptr) {
               del(ptr);
               ptr = nullptr;
           }
       }
   };

Reference Counting Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Thread-safe atomic reference counting operations:

**Strong Reference Operations**:

* ``incref_strong(ControlBlock*)`` - Atomically increment strong count
* ``decref_strong(ControlBlock*)`` - Atomically decrement strong count, returns true if reached zero
* ``get_strong_count(const ControlBlock*)`` - Read current strong count

**Weak Reference Operations**:

* ``incref_weak(ControlBlock*)`` - Atomically increment weak count
* ``decref_weak(ControlBlock*)`` - Atomically decrement weak count, returns true if reached zero
* ``get_weak_count(const ControlBlock*)`` - Read current weak count
* ``try_incref_strong(ControlBlock*)`` - Atomically try to promote weak to strong reference

**Memory Ordering**: Uses ``memory_order_relaxed`` for increments and ``memory_order_acq_rel`` for decrements to ensure proper synchronization without unnecessary overhead.

Requirements
============

Compiler Support
________________

* C++11 or later
* Support for variadic templates
* Support for template aliases (``using`` syntax)
* Support for ``constexpr`` functions
* Support for ``noexcept`` specifications
* Support for ``std::atomic`` (C++11)

Dependencies
____________

* ``<cstddef>`` - For ``size_t`` and ``nullptr_t``
* ``<atomic>`` - For thread-safe reference counting primitives

Usage Examples
==============

Type Trait Queries
__________________

.. code-block:: cpp

   #include "utilities.hpp"
   
   struct Base { };
   struct Derived : Base { };
   
   static_assert(cslt::IsConvertible<Derived*, Base*>::value, "");
   static_assert(!cslt::IsConvertible<Base*, Derived*>::value, "");
   static_assert(cslt::IsSame<int, int>::value, "");
   static_assert(!cslt::IsSame<int, const int>::value, "");

SFINAE with EnableIf
_____________________

.. code-block:: cpp

   #include "utilities.hpp"
   
   // Only enabled for integral types
   template <class T>
   cslt::EnableIfT<std::is_integral<T>::value, T>
   double_value(T x) {
       return x * 2;
   }
   
   // Only enabled for floating-point types
   template <class T>
   cslt::EnableIfT<std::is_floating_point<T>::value, T>
   double_value(T x) {
       return x * 2.0;
   }

Perfect Forwarding
__________________

.. code-block:: cpp

   #include "utilities.hpp"
   
   template <class T, class... Args>
   T* construct_at(T* ptr, Args&&... args) {
       return new (ptr) T(cslt::forward<Args>(args)...);
   }
   
   int main() {
       alignas(std::string) char buffer[sizeof(std::string)];
       std::string* str = construct_at(
           reinterpret_cast<std::string*>(buffer),
           "hello world"
       );
       str->~string();
   }

Tag Dispatch Pattern
____________________

.. code-block:: cpp

   #include "utilities.hpp"
   
   template <class T>
   void process_impl(T value, cslt::TrueType /* is_pointer */) {
       // Implementation for pointer types
       if (value) {
           // Dereference and process
       }
   }
   
   template <class T>
   void process_impl(T value, cslt::FalseType /* is_pointer */) {
       // Implementation for non-pointer types
       // Process value directly
   }
   
   template <class T>
   void process(T value) {
       process_impl(value, cslt::BoolConstant<std::is_pointer<T>::value>{});
   }

Detection Idiom
_______________

.. code-block:: cpp

   #include "utilities.hpp"
   
   // Detect if type has a nested ::value_type
   template <class T, class = void>
   struct HasValueType : cslt::FalseType { };
   
   template <class T>
   struct HasValueType<T, cslt::VoidT<typename T::value_type>>
       : cslt::TrueType { };
   
   // Usage
   static_assert(HasValueType<std::vector<int>>::value, "");
   static_assert(!HasValueType<int>::value, "");

Comparison with Standard Library
=================================

This implementation closely follows the C++ standard library:

============================================== ====================================
csalt++ Component                              Standard Library Equivalent
============================================== ====================================
``cslt::RemoveRef<T>``                         ``std::remove_reference<T>``
``cslt::IsLValueRef<T>``                       ``std::is_lvalue_reference<T>``
``cslt::IsSame<A, B>``                         ``std::is_same<A, B>``
``cslt::IsArray<T>``                           ``std::is_array<T>``
``cslt::IsConvertible<From, To>``              ``std::is_convertible<From, To>``
``cslt::IsConstructible<T, Args...>``          ``std::is_constructible<T, Args...>``
``cslt::IsAssignable<T, U>``                   ``std::is_assignable<T, U>``
``cslt::EnableIf<Cond, T>``                    ``std::enable_if<Cond, T>``
``cslt::move()``                               ``std::move()``
``cslt::forward()``                            ``std::forward()``
``cslt::swap()``                               ``std::swap()``
``cslt::declval()``                            ``std::declval()``
============================================== ====================================

Key Differences
_______________

* Simplified implementations focused on core functionality
* Missing some advanced traits (``RemoveConst``, ``RemoveCV``, ``Decay``, etc.)
* No C++17 ``_v`` helper variables (e.g., ``is_same_v<A, B>``)
* Control block infrastructure is part of ``utilities.hpp`` rather than separate

Performance Characteristics
===========================

Compile-Time Overhead
_____________________

All type traits and metaprogramming utilities are evaluated at compile-time:

* **Type traits**: O(1) instantiation depth, minimal compile-time cost
* **SFINAE checks**: May increase compile times for complex overload sets
* **Perfect forwarding**: No runtime overhead, generates optimal code

Runtime Overhead
________________

* **move()**, **forward()**, **swap()**: Zero runtime overhead (inline, constexpr)
* All utilities resolve to optimal assembly - equivalent to hand-written code
* **declval()**: Never actually called (only used in unevaluated contexts)

Future Enhancements
===================

Potential additions for future versions:

* Additional type manipulation traits (``RemoveConst``, ``RemoveCV``, ``AddConst``, ``Decay``)
* C++17 features (inline variables: ``is_same_v``, ``enable_if_t``)
* C++20 features (concepts-based versions)
* Additional detection idiom helpers
* More comprehensive standard library trait coverage
