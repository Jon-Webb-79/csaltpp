.. _tree_file:

*********************
C++ AVL Tree Overview
*********************

Balanced ordered containers in C++ are traditionally provided by ``std::set``
and ``std::map``, or by application-specific binary tree implementations.
While these containers provide logarithmic lookup and ordered traversal, they
typically rely on allocator behavior that is opaque, implementation-defined,
and not always well suited to deterministic, embedded, real-time, or
safety-regulated environments. In such systems, explicit control over memory
placement, allocation strategy, failure behavior, and object lifetime is often
more important than full generality.

The *C++ AVL Tree module* in this library provides a lightweight,
allocator-aware generic self-balancing binary search tree container
implemented in C++ and declared in ``tree.hpp``. Unlike standard ordered tree
containers, this implementation:

* Integrates directly with custom allocator hierarchies
* Supports allocator-controlled overflow behavior
* Uses a contiguous slab region for initial node storage to improve locality
* Recycles removed slab nodes for future reuse
* Enforces initialization through a factory pattern preventing invalid state
* Provides automatic cleanup via RAII with custom deleters
* Returns explicit success/error state using the ``Expected<T>`` pattern
* Supports insertion, removal, lookup, ordered traversal, range traversal,
  copying, and introspection

The AVL tree implementation is intended for environments where the developer
wants ordered-search semantics without surrendering control over memory layout,
allocation policy, or failure behavior.

The only dependencies this module has within the CSalt library are the
``allocator.hpp``, ``error.hpp``, and ``pointers.hpp`` files, keeping the
container relatively self-contained and suitable for use in larger systems
with minimal coupling.

**Allocator Integration**
=========================

By integrating directly with the ``Allocator`` base class defined in
``allocator.hpp``, the AVL tree container supports multiple allocation
strategies without changing the public API. The tree allocates:

* the ``AVLTree<T>`` object itself
* the initial contiguous slab used for node storage
* optional overflow nodes when slab capacity is exhausted and overflow is enabled

This makes the container compatible with the following allocator strategies:

* **HeapAllocator** — Best general-purpose choice for testing and normal host-side use
* **ArenaAllocator** — Useful when the tree has bulk lifetime semantics and is destroyed all at once
* **BuddyAllocator** — Useful when deterministic block allocation and returnable memory are important
* **PoolAllocator** — Potentially useful for overflow nodes if the pool block size matches node size
* **SlabAllocator** — A strong fit for overflow nodes when the node size is fixed and known
* **Custom allocators** — Any user-defined class inheriting from ``Allocator``

Because the tree internally maintains a contiguous slab for its primary node
storage, it already behaves partly like a pool-backed ordered container. The
allocator therefore has the most influence over:

* where the tree object itself lives
* where the slab lives
* how overflow nodes are obtained
* whether allocation failure is deterministic or environment-dependent

**Recommended Allocators**
==========================

Different allocators are appropriate depending on how the tree is used:

* **HeapAllocator**
  
  Best for:
  
  * general-purpose use
  * unit testing
  * host-side applications
  * debugging tree logic independent of allocator complexity

  This is usually the best allocator to use first because failure modes are
  simple and easy to reason about.

* **ArenaAllocator**
  
  Best for:
  
  * build-then-discard workflows
  * temporary ordered structures used during parsing or transformation passes
  * scenarios where the full tree lifetime matches a larger arena lifetime

  This works especially well when overflow is disabled or rare. If overflow is
  frequent, the arena must still have enough capacity to satisfy those extra
  node allocations.

* **BuddyAllocator**
  
  Best for:
  
  * deterministic dynamic allocation
  * environments where memory should be returnable and coalesced
  * systems requiring more structure than a heap allocator

  This is a reasonable choice when the tree may grow dynamically and later
  release memory through normal destruction.

* **PoolAllocator / SlabAllocator**
  
  Best for:
  
  * overflow-heavy workloads with uniform node sizes
  * fixed-type deployments where the node footprint is stable
  * performance-sensitive paths with frequent insertion and removal

  These allocators can be particularly effective because AVL tree nodes are
  fixed-size objects for a given ``T``. They are less useful when only the
  primary slab is used and overflow is disabled.

**Factory Pattern and RAII**
============================

Unlike direct constructor-based creation, tree creation uses a static factory
function ``AVLTree::init()`` that returns ``Expected<AVLTree<T>*>``. This
design:

* Prevents creation of partially initialized trees
* Ensures the slab and tree object are allocated together through the chosen allocator
* Provides explicit error handling at the point of creation
* Returns typed error objects such as ``ArgumentError`` and ``MemoryError``

Memory cleanup is automatic through the ``AVLTreeDeleter`` custom deleter,
which works with ``UniquePtr`` to provide RAII semantics:

.. code-block:: cpp

   {
       cslt::HeapAllocator allocator;
       auto result = cslt::AVLTree<int>::init(8, allocator, true, false);
       if (result.hasValue()) {
           cslt::UniquePtr<cslt::AVLTree<int>, cslt::AVLTreeDeleter<int>> tree(result.value());
           tree->insert(10);
           tree->insert(5);
           tree->insert(20);
       } // tree object, slab, and any overflow nodes automatically freed here
   }

This ownership model is especially important because the tree may allocate from
multiple internal sources: a slab region plus optional overflow nodes.

**Memory Model**
================

The tree is implemented as a pointer-linked AVL node hierarchy, but unlike a
traditional tree where every node is allocated independently, this
implementation starts with a *contiguous slab* of node storage.

This gives the container a hybrid memory model:

* **Primary slab storage** — a contiguous region reserved at initialization for
  up to ``N`` nodes
* **Recycled slab nodes** — removed slab nodes are returned to an internal free
  list and reused before any fresh slab slot is consumed
* **Overflow storage** — optional allocator-backed nodes used only after all
  slab capacity has been consumed or recycled capacity is unavailable

This design provides several benefits over a conventional tree:

* improved cache locality for early insertions
* reduced allocator overhead for the first ``N`` nodes
* deterministic bounded operation when overflow is disabled
* efficient reuse of removed slab slots without external metadata

**Slab Reuse Strategy**
=======================

Unlike a monotonic arena-style node reservoir, the tree tracks whether a node
belongs to the internal slab by checking whether its address lies within the
slab memory range. When a slab-backed node is removed:

* its stored value is destroyed
* the node itself is returned to an internal slab free list
* future insert operations reuse that node before consuming a new slab slot

This makes the slab behave like an internal node pool.

Allocation order is therefore:

#. Reuse a recycled slab node if one is available
#. Consume the next fresh slab slot if unused slab capacity remains
#. Allocate an overflow node through the external allocator if overflow is enabled
#. Fail the operation if overflow is disabled and no slab nodes remain available

This approach gives the tree predictable bounded behavior while avoiding the
surprising "write-once slab" behavior that would otherwise occur after repeated
insert/remove cycles.

**Overflow Behavior**
=====================

The tree can be configured in one of two modes:

* **Overflow enabled**
  
  When slab capacity is exhausted, additional nodes are allocated individually
  through the provided allocator. This allows the tree to continue growing
  beyond its initial slab size.

* **Overflow disabled**
  
  The tree acts as a bounded-capacity ordered container. Once all slab nodes
  are in use and no recycled slab nodes remain available, further insert
  operations fail.

This makes the container suitable for both:

* flexible host-side usage where growth is acceptable
* deterministic or embedded usage where fixed upper bounds are required

**AVL Balancing Model**
=======================

The tree maintains the AVL invariant, meaning that for every live node, the
height difference between the left and right subtrees is kept within one. To
enforce this invariant, the implementation stores a cached height in every
node and performs rotations after insertions and removals when needed.

This gives the container the following asymptotic behavior:

* O(log n) insertion
* O(log n) removal
* O(log n) lookup
* O(1) access to cached tree height
* O(n) ordered traversal

Balancing is maintained through the standard AVL rotation cases:

* left-left
* left-right
* right-right
* right-left

This makes the container appropriate when ordered lookup and predictable
logarithmic behavior are more important than append-heavy semantics.

**Element Type Requirements**
=============================

The ``AVLTree<T>`` container supports any element type ``T`` that is:

* copy-constructible
* destructible
* movable or copyable in contexts where values are returned from remove/find/min/max

Internally, nodes store the value inline and construct it with placement new.
This means:

* default construction of all slab entries is avoided
* each value is only constructed when a node becomes live
* each value is explicitly destroyed when the node is removed or the tree is destroyed

This is especially useful for non-trivial element types such as strings or
small user-defined objects, because unused slab slots do not incur object
construction cost.

**Comparator Requirements**
===========================

Unlike unordered containers, the AVL tree relies on a comparison function to
define node ordering. The comparator type passed as the second template
parameter must behave as a three-way comparison returning:

* a negative value if ``lhs`` should sort before ``rhs``
* zero if ``lhs`` and ``rhs`` are equivalent
* a positive value if ``lhs`` should sort after ``rhs``

The default comparator follows this convention for scalar-like types.

This design mirrors the C AVL implementation and keeps insertion, removal, and
range traversal logic consistent across the C and C++ container layers.

A custom comparator can be supplied to support:

* reverse ordering
* lexicographic structure comparison
* domain-specific sort criteria
* custom numeric or string ordering policies

**Duplicate Handling**
======================

The tree supports a configurable duplicate policy selected at initialization.

* **Duplicates disabled**
  
  Inserting a value equivalent to an existing node fails with an error.

* **Duplicates enabled**
  
  Equivalent values are inserted into the right subtree of the matching node.

This allows the same implementation to be used both as a strict ordered-set
style container and as an ordered multiset-style container.

**Traversal and Range Queries**
===============================

The tree supports:

* ``foreach()`` — in-order traversal across the entire tree
* ``foreach_range()`` — in-order traversal across an inclusive value range

Because AVL trees are ordered structures, ``foreach()`` visits elements in
sorted order according to the comparator.

The range traversal is especially useful because it prunes branches that cannot
contain in-range values. This allows subset queries to avoid visiting large
unrelated portions of the tree.

This makes the container useful for:

* ordered reporting
* interval and threshold queries
* subset extraction
* sorted iteration without external sorting

**Operations**
==============

The tree supports common ordered-container operations including:

* ``insert()``
* ``remove()``
* ``contains()``
* ``find()``
* ``min()``
* ``max()``
* ``clear()``
* ``copy()``
* ``foreach()``
* ``foreach_range()``

All operations preserve allocator ownership semantics and maintain consistency
between:

* live node count
* slab usage
* recycled slab capacity
* overflow node count
* AVL height invariants

**Complexity Notes**
====================

Because the container is an AVL tree, operation complexity follows standard
balanced-binary-search-tree expectations:

+--------------------+-------------------------------+
| Operation          | Complexity                    |
+====================+===============================+
| ``insert``         | O(log n)                      |
+--------------------+-------------------------------+
| ``remove``         | O(log n)                      |
+--------------------+-------------------------------+
| ``contains``       | O(log n)                      |
+--------------------+-------------------------------+
| ``find``           | O(log n)                      |
+--------------------+-------------------------------+
| ``min``            | O(log n)                      |
+--------------------+-------------------------------+
| ``max``            | O(log n)                      |
+--------------------+-------------------------------+
| ``foreach``        | O(n)                          |
+--------------------+-------------------------------+
| ``foreach_range``  | O(log n + k) typical          |
+--------------------+-------------------------------+
| ``clear``          | O(n)                          |
+--------------------+-------------------------------+
| ``copy``           | O(n log n)                    |
+--------------------+-------------------------------+

Here, ``k`` is the number of elements visited in the requested range.

The key advantage over a conventional dynamically allocated tree is not a
different asymptotic complexity class, but improved control over memory
behavior, allocator integration, and better early-node locality.

**Use Cases**
=============

This AVL tree implementation is particularly useful in the following scenarios:

* deterministic embedded software requiring bounded ordered storage
* safety-regulated applications needing explicit allocator control
* workloads requiring ordered lookup and traversal
* interval or threshold filtering over ordered values
* temporary ordered sets built during parsing or transformation passes
* applications where standard balanced tree containers are too allocator-opaque

When random indexing is common, the array container is usually the better
choice. When ordered lookup and sorted traversal are more important than random
access, the AVL tree container is a better fit.

Tree Class
==========

    .. doxygenclass:: cslt::AVLTree
       :members:
