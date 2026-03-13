.. _array_file:

******************
C++ Array Overview
******************

Dynamic array handling in C++ traditionally relies on ``std::vector``, which
provides convenient automatic memory management but comes with limitations in
specialized environments. The standard library implementation uses dynamic
allocation with opaque memory policies, making it unsuitable for embedded
systems, real-time applications, or projects requiring deterministic memory
behavior and MISRA C++ compliance.

The *C++ Array module* in this library provides a lightweight, allocator-aware
generic array container implemented in C++ and declared in ``array.hpp``.
Unlike ``std::vector``, this container:

* Integrates directly with custom allocator hierarchies (heap, arena, buddy, slab)
* Explicitly tracks element count, allocated capacity, and the owning allocator
* Enforces initialization through a factory pattern preventing uninitialized arrays
* Provides automatic cleanup via RAII with custom deleters
* Returns explicit success/error state using the ``Expected<T>`` pattern
* Supports in-place sorting, reversal, slicing, and cumulative transformations

The only dependencies this module has within the CSalt library are the
``allocator.hpp``, ``error.hpp``, ``iter_dir.hpp``, and ``pointers.hpp`` files,
making it suitable for integration into larger systems while maintaining minimal
coupling.

**Allocator Integration**
=========================

By integrating directly with the ``Allocator`` base class defined in
``allocator.hpp``, the array container supports multiple allocation strategies
without changing the public API:

* **HeapAllocator** — Standard heap allocation via ``operator new``
* **ArenaAllocator** — Sequential bump allocation with bulk deallocation
* **BuddyAllocator** — Power-of-two block allocation with coalescing
* **SlabAllocator** — Fixed-size object pools for uniform allocations
* **Custom allocators** — Any user-defined class inheriting from ``Allocator``

This design enables deterministic memory behavior suitable for embedded,
real-time, or safety-regulated environments where allocation patterns must
be controlled and predictable. The user can also develop their own allocators
as long as they conform to the ``Allocator`` base class.

**Factory Pattern and RAII**
============================

Unlike traditional constructors, array creation uses a static factory function
``Array::init()`` that returns ``Expected<Array<T>*>``. This design:

* Prevents creation of uninitialized or invalid arrays
* Provides explicit error handling at the point of creation
* Enables proper two-phase initialization (object allocation + buffer allocation)
* Returns typed error objects (``ArgumentError``, ``MemoryError``) with descriptive messages

Memory cleanup is automatic through the ``ArrayDeleter`` custom deleter, which
works seamlessly with ``UniquePtr`` to provide RAII semantics:

.. code-block:: cpp

   // Automatic cleanup when UniquePtr goes out of scope
   {
       cslt::HeapAllocator allocator;
       auto result = cslt::Array<int>::init(8, allocator);
       if (result.hasValue()) {
           cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
           arr->push_back(1);
           arr->push_back(2);
           arr->push_back(3);
       } // Array and buffer automatically freed here
   }

**Growth Strategy**
===================

The array grows automatically when a push operation requires more capacity.
Growth is tiered to avoid runaway allocation at large sizes while maintaining
fast ramp-up at small sizes:

+---------------------+----------------------------------+
| Current capacity    | New capacity                     |
+=====================+==================================+
| 0 elements          | 1                                |
+---------------------+----------------------------------+
| < 1,024 elements    | 2× current                       |
+---------------------+----------------------------------+
| < 8,192 elements    | 1.5× current (cap + cap/2)       |
+---------------------+----------------------------------+
| < 65,536 elements   | 1.25× current (cap + cap/4)      |
+---------------------+----------------------------------+
| ≥ 65,536 elements   | current + 256 (linear increment) |
+---------------------+----------------------------------+

A single reallocation is issued per growth event, and the array's size and
capacity are always kept consistent through the public API.

**Element Type Requirements**
=============================

The ``Array<T>`` container supports any element type ``T``, but the internal
implementation selects between two code paths at compile time based on
``std::is_trivially_copyable_v<T>``:

* **Trivially copyable T** (``int``, ``double``, plain structs with no
  user-defined constructors or destructors) — shift and copy operations use
  ``std::memmove`` / ``std::memcpy``, which the compiler can vectorise.
* **Non-trivial T** — shift and copy operations use placement-new and explicit
  destructor calls to honour the full object lifecycle.

This distinction is invisible to the caller — the public API is identical for
both cases.

**Sorting**
===========

The ``sort()`` method provides an in-place iterative quicksort with the
following properties:

* Median-of-three pivot selection to avoid worst-case behaviour on sorted input
* Insertion sort fallback for partitions smaller than 10 elements
* Tail-call optimisation keeping worst-case stack depth at O(log n)
* Direction controlled by ``cslt::Direction::FORWARD`` (ascending) or
  ``cslt::Direction::REVERSE`` (descending)
* Comparator accepts any callable with signature ``int(const T&, const T&)``

.. code-block:: cpp

   arr->sort([](const int& a, const int& b) { return (a > b) - (a < b); },
             cslt::Direction::FORWARD);

**Reversal**
============

The ``reverse()`` method reverses the array in place. Three compile-time paths
are selected based on element type and size:

* **Single-byte trivially copyable T** — delegates to ``simd_reverse_uint8()``,
  which selects the best available SIMD back-end at compile time (AVX-512,
  AVX2, AVX, SSE4.1, SSSE3, SSE2, NEON, SVE, SVE2) or a scalar fallback.
* **Multi-byte trivially copyable T** — scalar ``memcpy``-swap loop, which
  the compiler will typically auto-vectorise for common element sizes.
* **Non-trivial T** — move-swap loop that respects construction and destruction
  semantics throughout.

Array Class
===========

    .. doxygenclass:: cslt::Array
       :members:
