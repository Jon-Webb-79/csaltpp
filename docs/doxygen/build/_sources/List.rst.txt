.. _list_file:

*****************
C++ List Overview
*****************

Linked-list handling in C++ is traditionally performed with ``std::list`` or
``std::forward_list``. While these standard containers provide flexible node-
based storage, they rely on allocator behavior that is often opaque and not
well suited to deterministic, embedded, real-time, or safety-regulated
environments. In such systems, explicit control over memory placement,
allocation strategy, and failure behavior is often more important than full
generality.

The *C++ List module* in this library provides a lightweight, allocator-aware
generic singly linked list container implemented in C++ and declared in
``list.hpp``. Unlike ``std::forward_list``, this container:

* Integrates directly with custom allocator hierarchies
* Supports allocator-controlled overflow behavior
* Uses a contiguous slab region for initial node storage to improve locality
* Recycles popped slab nodes for future reuse
* Enforces initialization through a factory pattern preventing invalid state
* Provides automatic cleanup via RAII with custom deleters
* Returns explicit success/error state using the ``Expected<T>`` pattern
* Supports insertion, removal, reversal, concatenation, copying, and traversal

The list implementation is intended for environments where the developer wants
node-based semantics without surrendering control over memory layout or
allocation policy.

The only dependencies this module has within the CSalt library are the
``allocator.hpp``, ``error.hpp``, and ``pointers.hpp`` files, keeping the
container relatively self-contained and suitable for use in larger systems
with minimal coupling.

**Allocator Integration**
=========================

By integrating directly with the ``Allocator`` base class defined in
``allocator.hpp``, the list container supports multiple allocation strategies
without changing the public API. The list allocates:

* the ``SList<T>`` object itself
* the initial contiguous slab used for node storage
* optional overflow nodes when slab capacity is exhausted and overflow is enabled

This makes the container compatible with the following allocator strategies:

* **HeapAllocator** — Best general-purpose choice for testing and normal host-side use
* **ArenaAllocator** — Useful when the list has bulk lifetime semantics and is destroyed all at once
* **BuddyAllocator** — Useful when deterministic block allocation and returnable memory are important
* **PoolAllocator** — Potentially useful for overflow nodes if the pool block size matches node size
* **SlabAllocator** — A strong fit for overflow nodes when the node size is fixed and known
* **Custom allocators** — Any user-defined class inheriting from ``Allocator``

Because the list internally maintains a contiguous slab for its primary storage,
it already behaves partly like a pool-backed container. The allocator therefore
has the most influence over:

* where the list object itself lives
* where the slab lives
* how overflow nodes are obtained
* whether allocation failure is deterministic or environment-dependent

**Recommended Allocators**
==========================

Different allocators are appropriate depending on how the list is used:

* **HeapAllocator**
  
  Best for:
  
  * general-purpose use
  * unit testing
  * host-side applications
  * debugging container logic independent of allocator complexity

  This is usually the best allocator to use first because failure modes are
  simple and easy to reason about.

* **ArenaAllocator**
  
  Best for:
  
  * build-then-discard workflows
  * temporary lists used during parsing or transformation passes
  * scenarios where the full list lifetime matches a larger arena lifetime

  This works especially well when overflow is disabled or rare. If overflow
  is frequent, the arena must still have enough capacity to satisfy those
  extra node allocations.

* **BuddyAllocator**
  
  Best for:
  
  * deterministic dynamic allocation
  * environments where memory should be returnable and coalesced
  * systems requiring more structure than a heap allocator

  This is a reasonable choice when the list may grow dynamically and later
  release memory through normal destruction.

* **PoolAllocator / SlabAllocator**
  
  Best for:
  
  * overflow-heavy workloads with uniform node sizes
  * fixed-type deployments where the node footprint is stable
  * performance-sensitive paths with many push/pop operations

  These allocators can be particularly effective because list nodes are
  fixed-size objects for a given ``T``. They are less useful when only the
  primary slab is used and overflow is disabled.

**Factory Pattern and RAII**
============================

Unlike direct constructor-based creation, list creation uses a static factory
function ``SList::init()`` that returns ``Expected<SList<T>*>``. This design:

* Prevents creation of partially initialized lists
* Ensures the slab and list object are allocated together through the chosen allocator
* Provides explicit error handling at the point of creation
* Returns typed error objects such as ``ArgumentError`` and ``MemoryError``

Memory cleanup is automatic through the ``SListDeleter`` custom deleter, which
works with ``UniquePtr`` to provide RAII semantics:

.. code-block:: cpp

   {
       cslt::HeapAllocator allocator;
       auto result = cslt::SList<int>::init(8, true, allocator);
       if (result.hasValue()) {
           cslt::UniquePtr<cslt::SList<int>, cslt::SListDeleter<int>> list(result.value());
           list->push_back(1);
           list->push_back(2);
           list->push_front(0);
       } // list, slab, and any overflow nodes automatically freed here
   }

This ownership model is especially important because the list may allocate from
multiple internal sources: a slab region plus optional overflow nodes.

**Memory Model**
================

The list is implemented as a singly linked node chain, but unlike a traditional
linked list where every node is allocated independently, this implementation
starts with a *contiguous slab* of node storage.

This gives the container a hybrid memory model:

* **Primary slab storage** — a contiguous region reserved at initialization for
  up to ``N`` nodes
* **Recycled slab nodes** — popped slab nodes are returned to an internal free
  list and reused before any fresh slab slot is consumed
* **Overflow storage** — optional allocator-backed nodes used only after all
  slab capacity has been consumed or recycled capacity is unavailable

This design provides several benefits over a conventional linked list:

* improved cache locality for early inserts
* reduced allocator overhead for the first ``N`` nodes
* deterministic bounded operation when overflow is disabled
* efficient reuse of popped slab slots without external metadata

**Slab Reuse Strategy**
=======================

Unlike a monotonic arena-style node reservoir, the list tracks whether a node
belongs to the internal slab by checking whether its address lies within the
slab memory range. When a slab-backed node is removed:

* its stored value is destroyed
* the node itself is returned to an internal slab free list
* future push operations reuse that node before consuming a new slab slot

This makes the slab behave like an internal node pool.

Allocation order is therefore:

#. Reuse a recycled slab node if one is available
#. Consume the next fresh slab slot if unused slab capacity remains
#. Allocate an overflow node through the external allocator if overflow is enabled
#. Fail the operation if overflow is disabled and no slab nodes remain available

This approach gives the list predictable bounded behavior while avoiding the
surprising "write-once slab" behavior that would otherwise occur after repeated
push/pop cycles.

**Overflow Behavior**
=====================

The list can be configured in one of two modes:

* **Overflow enabled**
  
  When slab capacity is exhausted, additional nodes are allocated individually
  through the provided allocator. This allows the list to continue growing
  beyond its initial slab size.

* **Overflow disabled**
  
  The list acts as a bounded-capacity container. Once all slab nodes are in use
  and no recycled slab nodes remain available, further push operations fail.

This makes the container suitable for both:

* flexible host-side usage where growth is acceptable
* deterministic or embedded usage where fixed upper bounds are required

**Element Type Requirements**
=============================

The ``SList<T>`` container supports any element type ``T`` that is:

* copy-constructible
* destructible
* movable or copyable in contexts where values are returned from pop/get operations

Internally, nodes store the value in raw aligned storage and construct it with
placement new. This means:

* default construction of all slab entries is avoided
* each value is only constructed when a node becomes live
* each value is explicitly destroyed when the node is removed or the list is destroyed

This is especially useful for non-trivial element types such as strings or
small user-defined objects, because unused slab slots do not incur object
construction cost.

**Operations**
==============

The list supports common singly-linked-list operations including:

* ``push_back()``
* ``push_front()``
* ``push_at()``
* ``pop_back()``
* ``pop_front()``
* ``pop_at()``
* ``get()``
* ``peek_front()``
* ``peek_back()``
* ``contains()``
* ``clear()``
* ``reverse()``
* ``concat()``
* ``copy()``
* ``foreach()``

All operations preserve allocator ownership semantics and maintain consistency
between:

* live node count
* slab usage
* recycled slab capacity
* overflow node count

**Complexity Notes**
====================

Because the container is singly linked, operation complexity follows standard
linked-list expectations:

+------------------+-------------------------------+
| Operation        | Complexity                    |
+==================+===============================+
| ``push_front``   | O(1)                          |
+------------------+-------------------------------+
| ``push_back``    | O(1) with maintained tail     |
+------------------+-------------------------------+
| ``push_at``      | O(n)                          |
+------------------+-------------------------------+
| ``pop_front``    | O(1)                          |
+------------------+-------------------------------+
| ``pop_back``     | O(n)                          |
+------------------+-------------------------------+
| ``pop_at``       | O(n)                          |
+------------------+-------------------------------+
| ``get``          | O(n)                          |
+------------------+-------------------------------+
| ``contains``     | O(n)                          |
+------------------+-------------------------------+
| ``reverse``      | O(n)                          |
+------------------+-------------------------------+
| ``clear``        | O(n)                          |
+------------------+-------------------------------+

The key advantage over a conventional linked list is not asymptotic complexity,
but improved control over memory behavior and better early-node locality.

**Use Cases**
=============

This list implementation is particularly useful in the following scenarios:

* deterministic embedded software requiring bounded storage
* safety-regulated applications needing explicit allocator control
* workloads with frequent head/tail insertion but no need for random access
* temporary work queues or task chains
* graph, parser, or token structures where node semantics are useful
* applications where ``std::forward_list`` is too allocator-opaque

When random indexing is common, the array container is usually the better
choice. When node insertion/removal semantics are more important than random
access, the list container is a better fit.

List Class
==========

    .. doxygenclass:: cslt::SList
       :members:
