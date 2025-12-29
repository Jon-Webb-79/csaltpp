.. _allocator_file:

********************
C Allocator Overview
********************

Memory allocation in C++ typically involves direct use of the ``new``,
and ``free`` operators. While flexible, these can incur performance
overhead, lead to fragmentation, and increase the risk of memory leaks or
dangling pointers in large applications.

The *Allocator module* in this library provides lightweight and efficient
allocation utilities implemented in pure C and declared in
``allocator.hpp``.  The only dpendency this module has within the CSalt 
library is the ``error.hpp`` file, making it suitable ifor integration 
into larger systems.

Build-time configuration flags modify behavior depending on needs:

* ``STATIC_ONLY`` — Disables all heap allocation. Only stack or static
  memory supplied by the application is permitted.

These features make the allocator suitable for environments requiring strict
determinism and safety analysis, such as embedded and real-time systems or projects 
that require compliance to MISRA C++ standards.

.. _memory_type_enum:

Enums and Data Structures 
=========================
The ``csalt++`` library uses the following enum to help 
indicate if the allocator is utilyzing statically allocated memory from 
the stack or dynamically allocated memory from the heap.

.. code-block:: c++

   enum MemType {
      ALLOC_INVALID = 0,
      STATIC = 1,
      DYNAMIC = 2
   };

.. _allocator_overview:

Allocator Overview 
==================

    .. doxygenclass:: cslt::Allocator
       :members:
       :protected-members:
       :undoc-members:
       :private-members:

.. _heap_overview:

Heap Overview 
=============

    .. doxygenclass:: cslt::HeapAllocator
       :members:
       :protected-members:
       :undoc-members:
       :private-members:

.. _arena_overview:

Arena Overview 
==============

.. _pool_overview:

Pool Overview 
=============

.. _freelist_overview:

Free List Overview 
==================

.. _buddy_overview:

Buddy Overview 
==============

.. _slab_overview:

Slab Overview 
=============
