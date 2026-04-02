.. _heap_file:

******************
C++ Heap Overview
******************

Binary heap handling in C++ is traditionally performed with
``std::priority_queue``. While this standard container provides convenient
priority-ordered access, it wraps ``std::deque`` or ``std::vector`` with an
allocator interface that is opaque and not well suited to deterministic,
embedded, real-time, or safety-regulated environments. In such systems,
explicit control over memory placement, allocation strategy, and failure
behavior is often more important than full generality.

The *C++ Heap module* in this library provides a lightweight, allocator-aware
generic binary heap container implemented in C++ and declared in ``array.hpp``.
Unlike ``std::priority_queue``, this container:

* Integrates directly with custom allocator hierarchies
* Delegates all element storage and tiered growth to ``Array<T>``
* Fixes the element type and comparator at compile time via template parameters
* Enforces initialization through a factory pattern preventing invalid state
* Provides automatic cleanup via RAII with custom deleters
* Returns explicit success/error state using the ``Expected<T>`` pattern
* Supports push, pop, peek, copy, and templated iteration via ``foreach()``

The heap is implemented on top of ``cslt::Array<T>``, which means all growth
strategy, buffer management, and element lifetime handling are inherited
directly. Changes to ``Array<T>``\'s growth policy automatically apply to
``Heap<T, Compare>`` without any duplication.

The only dependencies this module has within the CSalt library are the
``array.hpp``, ``allocator.hpp``, and ``error.hpp`` files, keeping the
container self-contained and suitable for use in larger systems with minimal
coupling.

**Comparator Convention**
=========================

The template parameter ``Compare`` determines heap ordering. It must be a
callable with signature ``bool(const T&, const T&)`` that returns ``true``
when its first argument has *higher priority* than its second. This convention
matches ``std::priority_queue`` and the standard library comparator types:

+----------------------------+--------------------------------------------+
| Comparator                 | Heap type                                  |
+============================+============================================+
| ``std::less<T>``           | Max-heap — largest element at root         |
+----------------------------+--------------------------------------------+
| ``std::greater<T>``        | Min-heap — smallest element at root        |
+----------------------------+--------------------------------------------+
| Lambda / custom functor    | Any total ordering                         |
+----------------------------+--------------------------------------------+

For ``std::less<T>`` and ``std::greater<T>`` to work, ``T`` must define
``operator<``. When a lambda or custom functor is used, the element type
needs no comparison operators at all — the ordering logic is provided
entirely by the comparator.

.. code-block:: cpp

   // Max-heap of integers — largest value at root
   cslt::Heap<int, std::less<int>>;

   // Min-heap of integers — smallest value at root
   cslt::Heap<int, std::greater<int>>;

   // Custom struct heap ordered by a field — T needs no operator<
   struct Particle { float mass; float charge; };
   auto cmp = [](const Particle& a, const Particle& b) {
       return a.mass < b.mass;   // max-heap by mass
   };
   cslt::Heap<Particle, decltype(cmp)>::init(16, true, alloc, cmp);

**Allocator Integration**
=========================

By integrating directly with the ``Allocator`` base class defined in
``allocator.hpp``, the heap container supports multiple allocation strategies
without changing the public API. The heap allocates:

* the ``Heap<T, Compare>`` object itself
* the backing ``Array<T>`` and its element buffer, through the same allocator

This makes the container compatible with the following allocator strategies:

* **HeapAllocator** — Best general-purpose choice for testing and normal host-side use
* **ArenaAllocator** — Useful when the heap has bulk lifetime semantics and is destroyed all at once
* **BuddyAllocator** — Useful when deterministic block allocation and returnable memory are important
* **PoolAllocator** — Suitable for small heaps with a bounded element count
* **SlabAllocator** — Appropriate when the element type is fixed-size and uniform
* **Custom allocators** — Any user-defined class inheriting from ``Allocator``

Because the heap delegates to ``Array<T>`` for storage, the allocator is used
for both the initial element buffer and any subsequent growth. The allocator
therefore has direct influence over:

* where the heap object itself lives
* where the element buffer lives
* whether growth allocations succeed or fail deterministically

**Recommended Allocators**
==========================

Different allocators are appropriate depending on how the heap is used:

* **HeapAllocator**

  Best for:

  * general-purpose use
  * unit testing
  * host-side applications
  * debugging heap logic independent of allocator complexity

  This is usually the best allocator to start with because failure modes are
  simple and easy to reason about.

* **ArenaAllocator**

  Best for:

  * build-then-query workflows
  * heaps used transiently during scheduling or sorting passes
  * scenarios where the full heap lifetime matches a larger arena lifetime

  Works well when the heap size is bounded and growth is rare. If the heap
  grows frequently, the arena must have sufficient capacity.

* **BuddyAllocator**

  Best for:

  * deterministic dynamic allocation
  * environments where memory should be returnable and coalesced after use
  * systems requiring more structure than a raw heap allocator

  A reasonable choice when the heap may grow and the buffer should be
  returneable on destruction.

* **PoolAllocator / SlabAllocator**

  Best for:

  * fixed-capacity heaps with a known maximum element count
  * performance-sensitive paths with many push/pop operations
  * safety-regulated environments where growth must be prevented

  These allocators pair well with ``growth = false`` in ``init()``, producing
  a bounded-capacity heap with fully deterministic allocation behavior.

**Factory Pattern and RAII**
============================

Unlike direct constructor-based creation, heap creation uses a static factory
function ``Heap::init()`` that returns ``Expected<Heap<T, Compare>*>``. This
design:

* Prevents creation of partially initialized heaps
* Ensures the backing array and heap struct are allocated through the chosen allocator
* Provides explicit error handling at the point of creation
* Returns typed error objects such as ``ArgumentError`` and ``MemoryError``

Memory cleanup is automatic through the ``HeapDeleter`` custom deleter, which
works with ``UniquePtr`` to provide RAII semantics. The deleter calls
``ArrayDeleter<T>`` on the backing array before freeing the heap struct,
ensuring all elements are correctly destroyed and all memory is returned to
the allocator in the right order:

.. code-block:: cpp

   {
       cslt::HeapAllocator allocator;
       auto result = cslt::Heap<int, std::less<int>>::init(8, true, allocator);
       if (result.hasValue()) {
           cslt::UniquePtr<cslt::Heap<int, std::less<int>>,
                           cslt::HeapDeleter<int, std::less<int>>> h(result.value());
           h->push(5);
           h->push(3);
           h->push(7);
       } // heap and backing buffer automatically freed here
   }

**Backing Storage**
===================

The heap stores its elements in a contiguous ``Array<T>`` buffer using the
standard binary heap array encoding:

* the root is at index 0
* the children of node at index ``i`` are at indices ``2i + 1`` and ``2i + 2``
* the parent of node at index ``i`` is at index ``(i - 1) / 2``

This layout means the entire heap is stored in a single contiguous allocation,
which provides good cache behaviour during sift operations. The sift helpers
access ``Array::data_`` directly through a C++ friendship declaration, bypassing
the ``Expected<T>``-returning public API to avoid overhead in tight sift loops.

When growth is enabled the buffer expands using ``Array``\'s tiered growth
strategy:

+------------------------+--------------------------------------------+
| Current capacity       | New capacity                               |
+========================+============================================+
| 0 elements             | 1                                          |
+------------------------+--------------------------------------------+
| < 1,024 elements       | 2× current                                 |
+------------------------+--------------------------------------------+
| < 8,192 elements       | 1.5× current (cap + cap/2)                 |
+------------------------+--------------------------------------------+
| < 65,536 elements      | 1.25× current (cap + cap/4)                |
+------------------------+--------------------------------------------+
| ≥ 65,536 elements      | current + 256 (linear increment)           |
+------------------------+--------------------------------------------+

When growth is disabled, ``push()`` returns ``false`` once the buffer is full,
giving the caller full control over the maximum memory footprint.

**Heap Property**
=================

The heap maintains the invariant that for every element at index ``i``,
``Compare()(element[i], element[child])`` is ``true`` for all children of
``i``. This means the root always has the highest priority element under
the stored comparator.

Two internal operations maintain this invariant:

* **Sift-up** — called after ``push()``. The new element is appended to the
  end of the backing array and bubbled upward by repeatedly swapping with its
  parent until the comparator is satisfied.

* **Sift-down** — called after ``pop()``. The last element is moved to the
  root and pushed downward by repeatedly swapping with the higher-priority
  child until the comparator is satisfied at every level.

The swap operation uses ``memcpy`` for trivially copyable ``T`` and
move-construction with explicit destructor calls for non-trivial ``T``,
consistent with the strategy used in ``Array<T>``.

**Element Type Requirements**
==============================

The ``Heap<T, Compare>`` container supports any element type ``T`` that is:

* default-constructible (required by ``Expected<T>``, which is returned by
  ``pop()`` and ``peek()``)
* copy-constructible (elements are copied into the buffer on ``push()``)
* destructible (elements are destroyed explicitly during sift swaps and on
  ``pop()`` and destruction)

The comparator type ``Compare`` must be:

* default-constructible (used when ``cmp`` is omitted from ``init()``)
* callable as ``bool(const T&, const T&)``

+----------------------------+--------------------------------------------------+
| Requirement                | Reason                                           |
+============================+==================================================+
| T default-constructible    | Expected<T> default constructor                  |
+----------------------------+--------------------------------------------------+
| T copy-constructible       | Placement-new into backing buffer on push        |
+----------------------------+--------------------------------------------------+
| T destructible             | Explicit destructor calls during sift and pop    |
+----------------------------+--------------------------------------------------+
| Compare default-ctible     | Default cmp argument in init()                   |
+----------------------------+--------------------------------------------------+
| Compare callable           | Used in _sift_up and _sift_down                  |
+----------------------------+--------------------------------------------------+

**Ordered Traversal**
=====================

The ``foreach()`` method visits elements in backing-array (heap-internal)
order, which satisfies the heap property but is not sorted order. If ordered
traversal is required, copy the heap and pop from the copy in a loop:

.. code-block:: cpp

   cslt::HeapAllocator alloc;
   auto r = cslt::Heap<int, std::less<int>>::init(8, true, alloc);
   if (!r.hasValue()) return;
   cslt::UniquePtr<cslt::Heap<int, std::less<int>>,
                   cslt::HeapDeleter<int, std::less<int>>> h(r.value());

   h->push(5);
   h->push(3);
   h->push(7);
   h->push(1);

   // Ordered traversal — copy the heap, then pop from the copy
   auto cr = cslt::Heap<int, std::less<int>>::copy(*h, alloc);
   if (!cr.hasValue()) return;
   cslt::UniquePtr<cslt::Heap<int, std::less<int>>,
                   cslt::HeapDeleter<int, std::less<int>>> tmp(cr.value());

   while (!tmp->is_empty()) {
       auto pr = tmp->pop();
       if (pr.hasValue()) {
           // elements visited in descending order: 7, 5, 3, 1
       }
   }
   // h is unchanged — still contains all 4 elements

**Use Cases**
=============

This heap implementation is particularly useful in the following scenarios:

* priority queues for task scheduling in embedded or real-time systems
* Dijkstra\'s algorithm and other graph searches requiring a priority queue
* event-driven simulations ordered by event timestamp
* top-K selection from a large dataset without full sorting
* stream processing where only the current minimum or maximum is needed
* any workload requiring O(log n) insert and O(log n) remove-root

When elements need to be accessed by index or sorted in place, ``Array<T>``
with its ``sort()`` method is usually the better choice. When the priority of
the next element to process must always be available in O(1) and insertions
arrive dynamically, the heap is the right container.

Heap Class
==========

    .. doxygenclass:: cslt::Heap
       :members:
