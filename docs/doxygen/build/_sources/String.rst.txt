.. _string_file:

*******************
C++ String Overview
*******************

String handling in C++ traditionally relies on ``std::string``, which provides
convenient automatic memory management but comes with limitations in specialized
environments. The standard library implementation uses dynamic allocation with
opaque memory policies, making it unsuitable for embedded systems, real-time
applications, or projects requiring deterministic memory behavior and MISRA C++
compliance.

The *C++ String module* in this library provides a lightweight, allocator-aware
string container implemented in C++ and declared in ``string.hpp``.
Unlike ``std::string``, this container:

* Integrates directly with custom allocator hierarchies (heap, arena, buddy, slab)
* Explicitly tracks string length, allocated capacity, and the owning allocator
* Enforces initialization through a factory pattern preventing uninitialized strings
* Provides automatic cleanup via RAII with custom deleters
* Returns explicit success/error state using the ``Expected<T>`` pattern

The only dependency this module has within the CSalt library is the ``allocator.hpp``,
``error.hpp``, and ``pointers.hpp`` files, making it suitable for integration into
larger systems while maintaining minimal coupling.

**Allocator Integration**
=========================

By integrating directly with the ``Allocator`` base class defined in
``allocator.hpp``, the string container supports multiple allocation strategies
without changing the public API:

* **HeapAllocator** — Standard heap allocation via ``operator new``
* **ArenaAllocator** — Sequential bump allocation with bulk deallocation
* **BuddyAllocator** — Power-of-two block allocation with coalescing
* **SlabAllocator** — Fixed-size object pools for uniform allocations
* **Custom allocators** — Any user-defined class inheriting from ``Allocator``

This design enables deterministic memory behavior suitable for embedded,
real-time, or safety-regulated environments where allocation patterns must
be controlled and predictable.  The user can also develop their own 
allocators as long as it conforms to the ``Allocator`` base class.

**Recommended Allocators**
==========================

Different allocators are appropriate depending on how strings are created,
resized, and destroyed:

* **HeapAllocator**

  Best for:

  * general-purpose string manipulation
  * variable-length strings with frequent resizing
  * host-side applications and utilities
  * debugging string behavior independent of allocator complexity

  Strings often reallocate and copy memory during growth. A heap allocator
  provides predictable behavior and is the easiest to reason about during
  development and testing.

* **ArenaAllocator**

  Best for:

  * short-lived string collections (e.g., parsing, tokenization)
  * append-only or immutable string usage
  * batch construction followed by bulk discard

  This is highly efficient when strings are not individually freed. However,
  repeated resizing may lead to unused memory if older buffers cannot be
  reclaimed.

* **BuddyAllocator**

  Best for:

  * dynamically sized strings with controlled fragmentation
  * systems requiring deterministic allocation patterns
  * workloads with frequent allocation and release

  Since strings often grow geometrically, a buddy allocator can help reduce
  fragmentation compared to a general heap in long-running systems.

* **PoolAllocator / SlabAllocator**

  Best for:

  * fixed-capacity strings
  * preallocated buffers
  * embedded systems with strict memory layouts

  These allocators are less suitable for dynamically growing strings, but can
  be effective when string sizes are bounded or known in advance.

**Factory Pattern and RAII**
=============================

Unlike traditional constructors, string creation uses a static factory function
``String::init()`` that returns ``Expected<String*>``. This design:

* Prevents creation of uninitialized or invalid strings
* Provides explicit error handling at the point of creation
* Enables proper two-phase initialization (object allocation + buffer allocation)
* Returns typed error objects (``ArgumentError``, ``MemoryError``) with descriptive messages

Memory cleanup is automatic through the ``StringDeleter`` custom deleter, which
works seamlessly with ``UniquePtr`` to provide RAII semantics:

.. code-block:: cpp

   // Automatic cleanup when UniquePtr goes out of scope
   {
       auto result = cslt::String::init("example", 0, allocator);
       if (result.hasValue()) {
           cslt::UniquePtr<cslt::String, cslt::StringDeleter> str(result.value());
           // Use the string...
       } // String and buffer automatically freed here
   }

String Class 
============

    .. doxygenclass:: cslt::String
       :members:
