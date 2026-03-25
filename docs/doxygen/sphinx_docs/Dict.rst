.. _dict_file:

***********************
C++ Dictionary Overview
***********************

Dictionary handling in C++ traditionally relies on ``std::unordered_map``,
which provides convenient automatic memory management but comes with
limitations in specialized environments. The standard library implementation
uses dynamic allocation with opaque memory policies, making it unsuitable for
embedded systems, real-time applications, or projects requiring deterministic
memory behavior and MISRA C++ compliance.

The *C++ Dictionary module* in this library provides a lightweight,
allocator-aware generic hash dictionary implemented in C++ and declared in
``dict.hpp``. Unlike ``std::unordered_map``, this container:

* Integrates directly with custom allocator hierarchies (heap, arena, buddy, slab)
* Explicitly tracks entry count, bucket occupancy, and the owning allocator
* Enforces initialization through a factory pattern preventing uninitialized dicts
* Provides automatic cleanup via RAII with custom deleters
* Returns explicit success/error state using the ``Expected<T>`` pattern
* Supports deep copy, merge, and templated iteration via ``foreach()``

The only dependencies this module has within the CSalt library are the
``allocator.hpp`` and ``error.hpp`` files, making it suitable for integration
into larger systems while maintaining minimal coupling.

**Allocator Integration**
=========================

By integrating directly with the ``Allocator`` base class defined in
``allocator.hpp``, the dictionary container supports multiple allocation
strategies without changing the public API:

* **HeapAllocator** — Standard heap allocation via ``operator new``
* **ArenaAllocator** — Sequential bump allocation with bulk deallocation
* **BuddyAllocator** — Power-of-two block allocation with coalescing
* **SlabAllocator** — Fixed-size object pools for uniform allocations
* **Custom allocators** — Any user-defined class inheriting from ``Allocator``

Both the ``Dict`` struct itself and every internal node (key copy plus inline
value buffer) are allocated through the provided allocator, enabling fully
deterministic memory behaviour suitable for embedded, real-time, or
safety-regulated environments. The user can also develop their own allocators
as long as they conform to the ``Allocator`` base class.

**Factory Pattern and RAII**
=============================

Unlike traditional constructors, dictionary creation uses a static factory
function ``Dict::init()`` that returns ``Expected<Dict<K,V>*>``. This design:

* Prevents creation of uninitialized or invalid dictionaries
* Provides explicit error handling at the point of creation
* Enables proper two-phase initialization (struct allocation + bucket array allocation)
* Returns typed error objects (``ArgumentError``, ``MemoryError``) with descriptive messages

Memory cleanup is automatic through the ``DictDeleter`` custom deleter, which
works seamlessly with ``UniquePtr`` to provide RAII semantics:

.. code-block:: cpp

   // Automatic cleanup when UniquePtr goes out of scope
   {
       cslt::HeapAllocator allocator;
       auto result = cslt::Dict<int, float>::init(8, true, allocator);
       if (result.hasValue()) {
           cslt::UniquePtr<cslt::Dict<int, float>,
                           cslt::DictDeleter<int, float>> d(result.value());
           d->insert(1, 1.1f);
           d->insert(2, 2.2f);
           d->insert(3, 3.3f);
       } // Dict, all nodes, and all key copies automatically freed here
   }

**Template Type Requirements**
================================

``Dict<K, V>`` is parameterised on two independent types.

**Key type K** must satisfy ``std::is_trivially_copyable_v<K>``. This
constraint is enforced at compile time via ``static_assert``. Raw-byte
hashing operates on ``sizeof(K)`` bytes, which is only correct when the
object representation is the complete value representation — a guarantee
provided by trivial copyability. Any attempt to instantiate ``Dict`` with a
non-trivially-copyable key type will produce a clear compile-time error.

Key equality is resolved at compile time using a C++17 detection trait:

* If ``K`` provides ``operator==``, it is used for equality comparison.
* Otherwise, ``memcmp`` over ``sizeof(K)`` bytes is used as a fallback.

**Value type V** must be default-constructible, copy-constructible, and
destructible. Default-constructibility is required because ``get()`` and
``pop()`` declare a local ``Expected<V>`` which default-constructs ``V``
before a value is available. Copy-constructibility is required because values
are copy-constructed into the node's inline buffer on insert and update.
Destructibility is required because the destructor is called explicitly when
a node is freed.

+----------------------------+--------------------------------------------------+
| Requirement                | Reason                                           |
+============================+==================================================+
| K trivially copyable       | Raw-byte hashing and memcmp equality             |
+----------------------------+--------------------------------------------------+
| V default-constructible    | Expected<V> default constructor                  |
+----------------------------+--------------------------------------------------+
| V copy-constructible       | Placement-new into node inline buffer            |
+----------------------------+--------------------------------------------------+
| V destructible             | Explicit destructor call on node free            |
+----------------------------+--------------------------------------------------+

**Hash Function**
=================

The dictionary uses a MurmurHash3-inspired hash function operating on the
raw bytes of the key. The function processes the key in 4-byte blocks with
a finalisation mixing step, producing well-distributed 32-bit hashes
suitable for general-purpose use. The hash seed is fixed at ``0xdeadbeef``.

Bucket indices are computed as ``hash(key) % bucket_count``, where
``bucket_count`` is always a power of two. Because the bucket count is a
power of two a bitwise AND could replace the modulo; the modulo form is
retained for clarity.

**Growth Strategy**
===================

When growth is enabled (``growth == true`` in ``init()``), the bucket array
is resized automatically when the load factor exceeds 0.75. The growth
strategy is tiered to limit memory overhead at large sizes:

+------------------------+--------------------------------------------+
| Current bucket count   | New bucket count                           |
+========================+============================================+
| < 4,096 buckets        | 2× current                                 |
+------------------------+--------------------------------------------+
| ≥ 4,096 buckets        | current + 1,024 (linear increment)         |
+------------------------+--------------------------------------------+

The new bucket count is always rounded up to the next power of two. All
existing nodes are rehashed into the new bucket array in a single pass;
the old bucket array is freed via the allocator immediately afterwards.

When growth is disabled (``growth == false``), ``insert()`` returns
``false`` when ``hash_size() >= bucket_count()``, leaving the caller in
full control of memory allocation.

**Node Layout**
===============

Each key-value pair is stored in a single chain of nodes within a bucket.
A node consists of:

* A **node header** (``next`` pointer + ``key_data`` pointer) allocated as
  one block of ``sizeof(Node) + sizeof(V)`` bytes.
* A **key copy** — a separate allocation of ``sizeof(K)`` bytes holding a
  ``memcpy`` of the caller's key.  This ensures that the key's lifetime is
  tied to the node, not the caller's stack or buffer.
* An **inline value buffer** — the ``sizeof(V)`` bytes immediately following
  the node header in the same allocation, populated via placement new.

This layout means each insert performs exactly two allocator calls (node +
key), and each remove performs exactly two frees.

**Iteration**
=============

The ``foreach()`` method accepts any callable with signature
``void(const K&, const V&)``, including lambdas, functors, and plain function
pointers:

.. code-block:: cpp

   cslt::HeapAllocator allocator;
   auto r = cslt::Dict<int, float>::init(8, true, allocator);
   if (!r.hasValue()) return;
   cslt::UniquePtr<cslt::Dict<int, float>,
                   cslt::DictDeleter<int, float>> d(r.value());

   d->insert(1, 1.1f);
   d->insert(2, 2.2f);
   d->insert(3, 3.3f);

   // Lambda accumulator
   float total = 0.0f;
   d->foreach([&total](const int& key, const float& value) {
       total += value;
   });

Traversal order is bucket order, which is neither insertion order nor key
order. The callback must not insert, update, remove, or clear entries during
traversal.

**Copy and Merge**
==================

Two utility factory methods are provided for building new dictionaries from
existing ones.

``copy()`` creates a fully independent deep copy. All nodes are
copy-constructed into a fresh bucket array of the same capacity. A
convenience single-argument overload reuses the source dictionary's own
allocator:

.. code-block:: cpp

   // Two-argument form — explicit allocator
   auto cr = cslt::Dict<int, float>::copy(*src, allocator);

   // One-argument form — uses src's own allocator
   auto cr = cslt::Dict<int, float>::copy(*src);

``merge()`` combines two dictionaries into a new one. All entries from the
first source are inserted first, then entries from the second source are
processed according to the ``overwrite`` flag:

.. code-block:: cpp

   // overwrite == true:  second dict's values win on key conflicts
   // overwrite == false: first dict's values are preserved on conflicts
   auto mr = cslt::Dict<int, float>::merge(*a, *b, true, allocator);

Neither source dictionary is modified by ``merge()``.

**String Keys**
===============

The most common use case for a dictionary is mapping string keys to values.
``cslt::String`` cannot be used directly as a key because its private
destructor and deleted copy operations make it non-trivially copyable — a
requirement enforced by ``Dict``'s ``static_assert``.

``cslt::StringKey<N>`` solves this cleanly.  It copies the string content into
a fixed-size stack buffer of ``N`` bytes (including the null terminator),
producing a type that is trivially copyable, zero-cost to hash, and requires
no heap allocation.  Strings longer than ``N-1`` characters are silently
truncated, so choose ``N`` based on the longest key you expect to store.

+-------------------+--------------------------------------------+
| Type              | Recommended use                            |
+===================+============================================+
| StringKey<32>     | Short identifiers, sensor names, config    |
+-------------------+--------------------------------------------+
| StringKey<64>     | Dictionary words, variable names           |
+-------------------+--------------------------------------------+
| StringKey<128>    | Sentences, file names                      |
+-------------------+--------------------------------------------+
| StringKey<256>    | File paths, long descriptors               |
+-------------------+--------------------------------------------+

A type alias keeps call sites concise:

.. code-block:: cpp

   using Key64 = cslt::StringKey<64>;

**Basic string key usage:**

.. code-block:: cpp

   #include "dict.hpp"

   using Key64 = cslt::StringKey<64>;

   cslt::HeapAllocator alloc;
   auto r = cslt::Dict<Key64, float>::init(8, true, alloc);
   if (!r.hasValue()) return;

   cslt::UniquePtr<cslt::Dict<Key64, float>,
                   cslt::DictDeleter<Key64, float>> d(r.value());

   // Insert using C-string literals
   d->insert(Key64{"pi"},      3.14159f);
   d->insert(Key64{"euler"},   2.71828f);
   d->insert(Key64{"phi"},     1.61803f);

   // Look up by C-string literal
   auto gr = d->get(Key64{"pi"});
   if (gr.hasValue()) {
       float val = gr.value();   // 3.14159f
   }

   bool found   = d->has_key(Key64{"euler"});   // true
   bool missing = d->has_key(Key64{"zero"});    // false

**Constructing a key from a cslt::String:**

.. code-block:: cpp

   #include "dict.hpp"
   #include "string.hpp"

   using Key64 = cslt::StringKey<64>;

   cslt::HeapAllocator alloc;
   auto r = cslt::Dict<Key64, float>::init(8, true, alloc);
   if (!r.hasValue()) return;
   cslt::UniquePtr<cslt::Dict<Key64, float>,
                   cslt::DictDeleter<Key64, float>> d(r.value());

   d->insert(Key64{"temperature"}, 98.6f);

   // Build a lookup key from a runtime cslt::String
   auto rs = cslt::String::init("temperature", 0, alloc);
   if (rs.hasValue()) {
       cslt::UniquePtr<cslt::String, cslt::StringDeleter> s(rs.value());
       auto gr = d->get(Key64{*s});          // construct key from String
       if (gr.hasValue()) {
           float temp = gr.value();          // 98.6f
       }
   }

**Iterating over string-keyed entries:**

.. code-block:: cpp

   d->foreach([](const Key64& key, const float& value) {
       // key.c_str() returns the null-terminated string
       // key.size()  returns the logical length
   });

**Updating and removing string keys:**

.. code-block:: cpp

   d->update(Key64{"pi"}, 3.14f);          // overwrite existing value
   auto pr = d->pop(Key64{"euler"});        // remove and return value
   if (pr.hasValue()) {
       float e = pr.value();               // 2.71828f
   }

StringKey Class
===============

    .. doxygenclass:: cslt::StringKey
       :members:

Dict Class
==========

    .. doxygenclass:: cslt::Dict
       :members:
