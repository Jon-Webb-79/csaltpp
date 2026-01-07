***********************
Smart Pointers Overview
***********************

``pointers.hpp`` provides three smart pointer types that implement automatic 
memory management with RAII semantics. The library includes ``UniquePtr`` 
for exclusive ownership, ``SharedPtr`` for shared ownership with thread-safe 
reference counting, and ``WeakPtr`` for non-owning observation. All 
implementations closely mirror the C++ standard library's smart pointer behavior.

Design Philosophy
==================

The smart pointer library follows these core principles:

* **RAII Semantics**: Automatic resource management through object lifetime
* **Type-Safe**: Extensive use of SFINAE to enforce correct usage at compile-time
* **Zero-Overhead Unique Ownership**: ``UniquePtr`` has no runtime cost compared to raw pointers
* **Thread-Safe Shared Ownership**: ``SharedPtr`` uses atomic reference counting for safe concurrent access
* **Standard-Compliant**: Closely mirrors ``std::unique_ptr``, ``std::shared_ptr``, and ``std::weak_ptr``
* **Customizable Deletion**: Support for custom deleters and stateful cleanup

Library Components
==================

Smart Pointer Types
___________________

UniquePtr<T, Deleter>
~~~~~~~~~~~~~~~~~~~~~

Exclusive ownership smart pointer representing unique ownership of a dynamically allocated object.

**Key Features**:

* Zero overhead compared to raw pointers (with default deleter)
* Move-only semantics (non-copyable)
* Custom deleters by value or by reference
* Separate array specialization ``UniquePtr<T[]>``
* Converting constructors for polymorphic types
* Complete set of comparison operators

**Size**: Single pointer (with default deleter), or pointer + deleter size

SharedPtr<T>
~~~~~~~~~~~~

Shared ownership smart pointer with thread-safe reference counting.

**Key Features**:

* Atomic reference counting for thread-safety
* Multiple ``SharedPtr`` instances can own the same object
* Custom deleters (stored in control block)
* Supports ``WeakPtr`` for non-owning observation
* Converting constructors for polymorphic types
* Complete set of comparison operators

**Size**: Two pointers (object pointer + control block pointer)

WeakPtr<T>
~~~~~~~~~~

Non-owning observer that doesn't affect object lifetime.

**Key Features**:

* Observes ``SharedPtr``-managed objects without ownership
* Thread-safe promotion to ``SharedPtr`` via ``lock()``
* Automatically expires when referenced object is destroyed
* Essential for breaking reference cycles
* Supports type conversions (derived-to-base)

**Size**: Two pointers (same as ``SharedPtr``)

Supporting Components
_____________________

DefaultDelete<T>
~~~~~~~~~~~~~~~~

Default deleter that calls ``delete`` on the pointer.

.. code-block:: cpp

   template <class T>
   struct DefaultDelete {
       void operator()(T* ptr) const noexcept {
           delete ptr;
       }
   };

**Array Specialization**: ``DefaultDelete<T[]>`` calls ``delete[]``

DeleterStorage<Deleter, IsRef>
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Internal storage mechanism for deleters in ``UniquePtr``.

**Value Deleters**: Stores deleter by value
**Reference Deleters**: Stores pointer to external deleter

This allows ``UniquePtr`` to support both stateless and stateful deleters efficiently.

Helper Traits
~~~~~~~~~~~~~

**HasPointerType<D>**: Detects if deleter defines custom ``::pointer`` type

**PointerType<T, D>**: Resolves to ``D::pointer`` if available, otherwise ``T*``

These enable support for "fancy pointers" (custom pointer types).

UniquePtr Details
=================

Basic Usage
___________

.. code-block:: cpp

   #include "pointers.hpp"
   
   // Basic construction
   cslt::UniquePtr<int> p1(new int(42));
   
   // Factory function (recommended)
   auto p2 = cslt::make_unique<int>(42);
   
   // Access
   *p2 = 99;
   int value = *p2;  // 99

Ownership Semantics
___________________

``UniquePtr`` represents exclusive ownership and cannot be copied:

.. code-block:: cpp

   cslt::UniquePtr<int> p1 = cslt::make_unique<int>(42);
   
   // cslt::UniquePtr<int> p2 = p1;  // ERROR: copy is deleted
   
   cslt::UniquePtr<int> p2 = cslt::move(p1);  // OK: move transfers ownership
   // p1 is now null, p2 owns the object

Custom Deleters
_______________

Value Deleters
~~~~~~~~~~~~~~

Store the deleter by value in the ``UniquePtr``:

.. code-block:: cpp

   struct FileCloser {
       void operator()(FILE* f) const {
           if (f) fclose(f);
       }
   };
   
   cslt::UniquePtr<FILE, FileCloser> file(fopen("data.txt", "r"));
   // File automatically closed when file goes out of scope

Reference Deleters
~~~~~~~~~~~~~~~~~~

Store a reference to an external deleter:

.. code-block:: cpp

   struct LoggingDeleter {
       int count = 0;
       void operator()(int* p) {
           ++count;
           delete p;
       }
   };
   
   LoggingDeleter deleter;
   {
       cslt::UniquePtr<int, LoggingDeleter&> p(new int(42), deleter);
   }
   // deleter.count is now 1

Array Specialization
____________________

``UniquePtr<T[]>`` provides array-specific operations:

.. code-block:: cpp

   // Array construction
   cslt::UniquePtr<int[]> arr = cslt::make_unique_array<int>(100);
   
   // Array access
   arr[0] = 42;
   arr[99] = 99;
   
   // Automatic cleanup with delete[]

**Key Differences from Single-Object**:

* Uses ``operator[]`` instead of ``operator*`` and ``operator->``
* Deleter calls ``delete[]`` instead of ``delete``
* No converting constructors between array and non-array types

Type Conversions
________________

``UniquePtr`` supports converting moves from derived to base types:

.. code-block:: cpp

   struct Base { virtual ~Base() = default; };
   struct Derived : Base { };
   
   cslt::UniquePtr<Derived> derived = cslt::make_unique<Derived>();
   cslt::UniquePtr<Base> base = cslt::move(derived);
   // derived is now null, base owns the Derived object

Member Functions
________________

Constructors
~~~~~~~~~~~~

.. code-block:: cpp

   UniquePtr();                                    // Default construct (null)
   UniquePtr(nullptr_t);                           // Null pointer
   explicit UniquePtr(pointer p);                  // Take ownership of p
   UniquePtr(pointer p, const Deleter& d);         // With deleter copy
   UniquePtr(pointer p, Deleter&& d);              // With deleter move
   UniquePtr(UniquePtr&& other);                   // Move constructor
   template <class U, class D2>
   UniquePtr(UniquePtr<U, D2>&& other);            // Converting move

Observers
~~~~~~~~~

.. code-block:: cpp

   pointer get() const noexcept;                   // Get raw pointer
   Deleter& get_deleter() noexcept;                // Access deleter
   explicit operator bool() const noexcept;         // Test if non-null
   T& operator*() const noexcept;                  // Dereference
   T* operator->() const noexcept;                 // Member access
   T& operator[](size_t i) const noexcept;         // Array subscript (array specialization only)

Modifiers
~~~~~~~~~

.. code-block:: cpp

   pointer release() noexcept;                     // Release ownership, return pointer
   void reset(pointer p = pointer()) noexcept;     // Replace managed object
   void swap(UniquePtr& other) noexcept;           // Exchange contents

Comparison Operators
____________________

.. code-block:: cpp

   template <class T1, class D1, class T2, class D2>
   bool operator==(const UniquePtr<T1, D1>&, const UniquePtr<T2, D2>&);
   bool operator!=(const UniquePtr<T1, D1>&, const UniquePtr<T2, D2>&);
   bool operator<(const UniquePtr<T1, D1>&, const UniquePtr<T2, D2>&);
   bool operator<=(const UniquePtr<T1, D1>&, const UniquePtr<T2, D2>&);
   bool operator>(const UniquePtr<T1, D1>&, const UniquePtr<T2, D2>&);
   bool operator>=(const UniquePtr<T1, D1>&, const UniquePtr<T2, D2>&);
   
   // Comparisons with nullptr
   bool operator==(const UniquePtr<T, D>&, nullptr_t);
   bool operator==(nullptr_t, const UniquePtr<T, D>&);
   bool operator!=(const UniquePtr<T, D>&, nullptr_t);
   bool operator!=(nullptr_t, const UniquePtr<T, D>&);

Factory Functions
_________________

.. code-block:: cpp

   template <class T, class... Args>
   UniquePtr<T> make_unique(Args&&... args);
   
   template <class T>
   UniquePtr<T[]> make_unique_array(size_t n);

SharedPtr Details
=================

Basic Usage
___________

.. code-block:: cpp

   #include "pointers.hpp"
   
   // Basic construction
   cslt::SharedPtr<int> p1(new int(42));
   
   // Factory function (recommended)
   auto p2 = cslt::make_shared<int>(42);
   
   // Shared ownership
   auto p3 = p2;  // p2 and p3 share ownership
   
   // Reference count
   std::cout << p2.use_count();  // 2

Reference Counting
__________________

``SharedPtr`` maintains a reference count that tracks the number of ``SharedPtr`` instances sharing ownership:

.. code-block:: cpp

   cslt::SharedPtr<int> p1 = cslt::make_shared<int>(42);
   std::cout << p1.use_count();  // 1
   
   {
       auto p2 = p1;
       std::cout << p1.use_count();  // 2
       auto p3 = p1;
       std::cout << p1.use_count();  // 3
   }
   
   std::cout << p1.use_count();  // 1
   // Object destroyed when p1 goes out of scope

Thread Safety
_____________

``SharedPtr`` provides thread-safe reference counting:

**Thread-Safe Operations**:

* Copying ``SharedPtr`` instances (increments reference count atomically)
* Destroying ``SharedPtr`` instances (decrements reference count atomically)
* Reading ``use_count()`` from multiple threads
* Different ``SharedPtr`` instances managing the same object

**Not Thread-Safe**:

* Modifying the same ``SharedPtr`` object from multiple threads (e.g., concurrent ``reset()`` or ``operator=``)
* Accessing the managed object itself (requires external synchronization)

.. code-block:: cpp

   cslt::SharedPtr<Data> shared = cslt::make_shared<Data>();
   
   // Thread 1
   auto copy1 = shared;  // Safe: atomic increment
   
   // Thread 2
   auto copy2 = shared;  // Safe: atomic increment
   
   // Not safe without external lock:
   // Thread 1: shared.reset();
   // Thread 2: shared.reset();

Custom Deleters
_______________

``SharedPtr`` can use custom deleters, stored in the control block:

.. code-block:: cpp

   struct ResourceDeleter {
       void operator()(Resource* p) const {
           cleanup_resource(p);
           delete p;
       }
   };
   
   cslt::SharedPtr<Resource> res(new Resource(), ResourceDeleter{});

**Note**: Unlike ``UniquePtr``, the deleter type is not part of ``SharedPtr``'s 
type. All ``SharedPtr<T>`` instances have the same type regardless of deleter.

Type Conversions
________________

``SharedPtr`` supports conversions between related types:

.. code-block:: cpp

   struct Base { virtual ~Base() = default; };
   struct Derived : Base { };
   
   cslt::SharedPtr<Derived> derived = cslt::make_shared<Derived>();
   cslt::SharedPtr<Base> base = derived;  // Copy conversion
   // Both derived and base share ownership

Member Functions
________________

Constructors
~~~~~~~~~~~~

.. code-block:: cpp

   SharedPtr();                                    // Default construct (null)
   SharedPtr(nullptr_t);                           // Null pointer
   explicit SharedPtr(T* p);                       // Take ownership of p
   template <class Deleter>
   SharedPtr(T* p, Deleter d);                     // With custom deleter
   SharedPtr(const SharedPtr& other);              // Copy constructor
   SharedPtr(SharedPtr&& other);                   // Move constructor
   template <class U>
   SharedPtr(const SharedPtr<U>& other);           // Converting copy
   template <class U>
   SharedPtr(SharedPtr<U>&& other);                // Converting move

Observers
~~~~~~~~~

.. code-block:: cpp

   T* get() const noexcept;                        // Get raw pointer
   T& operator*() const noexcept;                  // Dereference
   T* operator->() const noexcept;                 // Member access
   explicit operator bool() const noexcept;         // Test if non-null
   size_t use_count() const noexcept;              // Get reference count
   bool unique() const noexcept;                   // Test if use_count() == 1

Modifiers
~~~~~~~~~

.. code-block:: cpp

   void reset() noexcept;                          // Release ownership
   void reset(T* p);                               // Replace with new object
   template <class Deleter>
   void reset(T* p, Deleter d);                    // Replace with new object and deleter
   void swap(SharedPtr& other) noexcept;           // Exchange contents

Comparison Operators
____________________

.. code-block:: cpp

   template <class T, class U>
   bool operator==(const SharedPtr<T>&, const SharedPtr<U>&);
   bool operator!=(const SharedPtr<T>&, const SharedPtr<U>&);
   bool operator<(const SharedPtr<T>&, const SharedPtr<U>&);
   bool operator<=(const SharedPtr<T>&, const SharedPtr<U>&);
   bool operator>(const SharedPtr<T>&, const SharedPtr<U>&);
   bool operator>=(const SharedPtr<T>&, const SharedPtr<U>&);
   
   // Comparisons with nullptr
   bool operator==(const SharedPtr<T>&, nullptr_t);
   bool operator==(nullptr_t, const SharedPtr<T>&);
   bool operator!=(const SharedPtr<T>&, nullptr_t);
   bool operator!=(nullptr_t, const SharedPtr<T>&);
   bool operator<(const SharedPtr<T>&, nullptr_t);
   bool operator<(nullptr_t, const SharedPtr<T>&);
   // ... (additional ordering comparisons with nullptr)

Factory Function
________________

.. code-block:: cpp

   template <class T, class... Args>
   SharedPtr<T> make_shared(Args&&... args);

**Note**: This implementation uses two allocations (object + control block). 
The standard library's ``std::make_shared`` uses a single allocation for better 
performance.

WeakPtr Details
===============

Basic Usage
___________

.. code-block:: cpp

   #include "pointers.hpp"
   
   cslt::SharedPtr<int> shared = cslt::make_shared<int>(42);
   cslt::WeakPtr<int> weak = shared;
   
   // Check if still valid
   if (!weak.expired()) {
       cslt::SharedPtr<int> locked = weak.lock();
       if (locked) {
           std::cout << *locked;  // 42
       }
   }

Purpose and Use Cases
_____________________

``WeakPtr`` solves several important problems:

Breaking Reference Cycles
~~~~~~~~~~~~~~~~~~~~~~~~~

Prevents memory leaks in circular references:

.. code-block:: cpp

   struct Node {
       cslt::SharedPtr<Node> next;  // Strong reference
       cslt::WeakPtr<Node> prev;    // Weak reference (breaks cycle)
   };
   
   auto n1 = cslt::make_shared<Node>();
   auto n2 = cslt::make_shared<Node>();
   n1->next = n2;
   n2->prev = n1;  // Weak pointer prevents cycle

Cache Implementation
~~~~~~~~~~~~~~~~~~~~

Allows objects to be cached without preventing cleanup:

.. code-block:: cpp

   std::map<std::string, cslt::WeakPtr<Resource>> cache;
   
   cslt::SharedPtr<Resource> get_resource(const std::string& key) {
       auto it = cache.find(key);
       if (it != cache.end()) {
           if (auto locked = it->second.lock()) {
               return locked;  // Cache hit
           }
       }
       // Cache miss, create new
       auto resource = cslt::make_shared<Resource>(key);
       cache[key] = resource;
       return resource;
   }

Observer Pattern
~~~~~~~~~~~~~~~~

Allows observers without preventing subject destruction:

.. code-block:: cpp

   class Subject {
       std::vector<cslt::WeakPtr<Observer>> observers_;
   public:
       void notify() {
           for (auto& weak : observers_) {
               if (auto obs = weak.lock()) {
                   obs->update();
               }
           }
       }
   };

Expiration
__________

``WeakPtr`` automatically expires when the referenced object is destroyed:

.. code-block:: cpp

   cslt::WeakPtr<int> weak;
   {
       auto shared = cslt::make_shared<int>(42);
       weak = shared;
       std::cout << weak.expired();  // false
   }
   // shared destroyed here
   std::cout << weak.expired();  // true
   
   auto locked = weak.lock();
   std::cout << (locked == nullptr);  // true

Thread-Safe Promotion
_____________________

``WeakPtr::lock()`` is thread-safe and atomically promotes to ``SharedPtr``:

.. code-block:: cpp

   cslt::SharedPtr<Data> shared = cslt::make_shared<Data>();
   cslt::WeakPtr<Data> weak = shared;
   
   // Thread 1
   shared.reset();  // May destroy object
   
   // Thread 2
   auto locked = weak.lock();  // Thread-safe
   // Either gets valid SharedPtr or null, never dangling pointer

The implementation uses ``compare_exchange_weak`` in a loop to atomically check 
if the object is still alive and increment the reference count if so.

Member Functions
________________

Constructors
~~~~~~~~~~~~

.. code-block:: cpp

   WeakPtr();                                      // Default construct (empty)
   WeakPtr(const SharedPtr<T>& sp);                // Construct from SharedPtr
   template <class U>
   WeakPtr(const SharedPtr<U>& sp);                // Converting construct from SharedPtr
   WeakPtr(const WeakPtr& other);                  // Copy constructor
   template <class U>
   WeakPtr(const WeakPtr<U>& other);               // Converting copy
   WeakPtr(WeakPtr&& other);                       // Move constructor
   template <class U>
   WeakPtr(WeakPtr<U>&& other);                    // Converting move

Observers
~~~~~~~~~

.. code-block:: cpp

   size_t use_count() const noexcept;              // Get strong reference count
   bool expired() const noexcept;                  // Test if object destroyed
   SharedPtr<T> lock() const noexcept;             // Try to create SharedPtr

Modifiers
~~~~~~~~~

.. code-block:: cpp

   void reset() noexcept;                          // Release weak reference
   void swap(WeakPtr& other) noexcept;             // Exchange contents

Control Block Lifetime
______________________

The control block survives as long as either strong or weak references exist:

.. code-block:: cpp

   cslt::WeakPtr<int> weak;
   {
       auto shared = cslt::make_shared<int>(42);
       weak = shared;
   }
   // Object destroyed, but control block still exists
   
   weak.reset();
   // Control block now destroyed

This ensures ``WeakPtr`` can safely detect expiration without accessing freed memory.

Usage Examples
==============

Basic Ownership Patterns
________________________

Unique Ownership
~~~~~~~~~~~~~~~~

.. code-block:: cpp

   #include "pointers.hpp"
   
   void process_data() {
       auto data = cslt::make_unique<Data>();
       data->process();
       // Automatic cleanup when function returns
   }

Shared Ownership
~~~~~~~~~~~~~~~~

.. code-block:: cpp

   #include "pointers.hpp"
   #include <vector>
   
   class Resource {
       // Large resource...
   };
   
   std::vector<cslt::SharedPtr<Resource>> workers;
   
   void distribute_work() {
       auto resource = cslt::make_shared<Resource>();
       
       for (int i = 0; i < 10; ++i) {
           workers.push_back(resource);
       }
       // Resource shared among all workers
   }

Polymorphism
____________

.. code-block:: cpp

   #include "pointers.hpp"
   #include <vector>
   
   struct Shape { virtual void draw() = 0; virtual ~Shape() = default; };
   struct Circle : Shape { void draw() override { /* ... */ } };
   struct Square : Shape { void draw() override { /* ... */ } };
   
   std::vector<cslt::UniquePtr<Shape>> shapes;
   shapes.push_back(cslt::make_unique<Circle>());
   shapes.push_back(cslt::make_unique<Square>());
   
   for (auto& shape : shapes) {
       shape->draw();  // Polymorphic call
   }

RAII Resource Management
________________________

.. code-block:: cpp

   #include "pointers.hpp"
   
   struct FileDeleter {
       void operator()(FILE* f) const {
           if (f) fclose(f);
       }
   };
   
   void process_file(const char* filename) {
       cslt::UniquePtr<FILE, FileDeleter> file(fopen(filename, "r"));
       if (!file) {
           throw std::runtime_error("Failed to open file");
       }
       
       // Use file...
       
       // Automatic close on all exit paths (return, exception, etc.)
   }

Factory Pattern
_______________

.. code-block:: cpp

   #include "pointers.hpp"
   
   struct Widget {
       virtual ~Widget() = default;
       virtual void action() = 0;
   };
   
   struct ButtonWidget : Widget {
       void action() override { /* ... */ }
   };
   
   struct SliderWidget : Widget {
       void action() override { /* ... */ }
   };
   
   cslt::UniquePtr<Widget> create_widget(const std::string& type) {
       if (type == "button") {
           return cslt::make_unique<ButtonWidget>();
       } else if (type == "slider") {
           return cslt::make_unique<SliderWidget>();
       }
       return nullptr;
   }

Doubly Linked List
__________________

.. code-block:: cpp

   #include "pointers.hpp"
   
   template <class T>
   struct Node {
       T data;
       cslt::SharedPtr<Node> next;  // Strong reference (owns)
       cslt::WeakPtr<Node> prev;    // Weak reference (observes)
       
       Node(const T& value) : data(value) {}
   };
   
   template <class T>
   class List {
       cslt::SharedPtr<Node<T>> head_;
       cslt::WeakPtr<Node<T>> tail_;
       
   public:
       void push_back(const T& value) {
           auto new_node = cslt::make_shared<Node<T>>(value);
           
           if (auto tail = tail_.lock()) {
               tail->next = new_node;
               new_node->prev = tail_;
           } else {
               head_ = new_node;
           }
           tail_ = new_node;
       }
   };

Caching System
______________

.. code-block:: cpp

   #include "pointers.hpp"
   #include <map>
   #include <string>
   
   class ResourceCache {
       std::map<std::string, cslt::WeakPtr<Resource>> cache_;
       
   public:
       cslt::SharedPtr<Resource> get(const std::string& key) {
           // Try cache first
           auto it = cache_.find(key);
           if (it != cache_.end()) {
               if (auto cached = it->second.lock()) {
                   return cached;  // Cache hit
               }
               // Expired, remove from cache
               cache_.erase(it);
           }
           
           // Cache miss, load resource
           auto resource = cslt::make_shared<Resource>(key);
           cache_[key] = resource;
           return resource;
       }
       
       void clear() {
           cache_.clear();
       }
   };

Performance Characteristics
===========================

UniquePtr Performance
_____________________

**Size**: 
* Default deleter: sizeof(T*)
* Custom value deleter: sizeof(T*) + sizeof(Deleter)
* Reference deleter: sizeof(T*)

**Operations**:
* Construction: O(1)
* Destruction: O(1) + deleter cost
* Move: O(1)
* Copy: Not available
* Comparison: O(1)

**Overhead**: Zero runtime overhead with default deleter

SharedPtr Performance
_____________________

**Size**: 
* Always 2 * sizeof(void*) (object pointer + control block pointer)

**Operations**:
* Construction: O(1) + allocation (control block)
* Copy: O(1) + atomic increment
* Destruction: O(1) + atomic decrement (+ object deletion if last)
* Move: O(1)
* use_count(): O(1) + atomic load

**Overhead**: 
* Control block allocation (separate from object)
* Atomic operations for reference counting
* Two pointers instead of one

WeakPtr Performance
___________________

**Size**: 
* Same as SharedPtr: 2 * sizeof(void*)

**Operations**:
* Construction: O(1) + atomic increment (weak count)
* Copy: O(1) + atomic increment
* Destruction: O(1) + atomic decrement
* Move: O(1)
* lock(): O(1) + atomic compare-and-swap loop
* expired(): O(1) + atomic load

**Overhead**: 
* Minimal - weak references don't affect object lifetime
* lock() may retry in highly contended scenarios

Memory Layout
_____________

UniquePtr Memory Layout (Default Deleter)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   UniquePtr<T>: [T* ptr]
   
   Heap: [T object]

SharedPtr Memory Layout
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   SharedPtr<T>: [T* ptr] [ControlBlock* cb]
   
   Heap object:  [T object]
   Control block: [atomic<size_t> strong]
                  [atomic<size_t> weak]
                  [vtable ptr]
                  [T* ptr]
                  [Deleter del]

WeakPtr Memory Layout
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   WeakPtr<T>: [T* ptr] [ControlBlock* cb]
   
   (Shares control block with SharedPtr)

Comparison with Standard Library
================================

This implementation closely follows the C++ standard library:

============================================== ====================================
csalt++ Component                              Standard Library Equivalent
============================================== ====================================
``cslt::UniquePtr<T, D>``                      ``std::unique_ptr<T, D>``
``cslt::SharedPtr<T>``                         ``std::shared_ptr<T>``
``cslt::WeakPtr<T>``                           ``std::weak_ptr<T>``
``cslt::make_unique<T>(args...)``              ``std::make_unique<T>(args...)``
``cslt::make_shared<T>(args...)``              ``std::make_shared<T>(args...)``
``cslt::DefaultDelete<T>``                     ``std::default_delete<T>``
============================================== ====================================

Key Differences
_______________

**SharedPtr / make_shared**:

* Uses two allocations (object + control block) instead of one
* Simpler implementation, slight performance cost
* Control block structure is visible in this implementation

**Missing Features**:

* ``enable_shared_from_this<T>`` - Safe sharing of ``this``
* ``weak_ptr::owner_before()`` - Ordering by ownership
* Allocator support for ``SharedPtr``
* ``unique_ptr`` hash support
* Array support for ``SharedPtr`` (C++17)

**Thread Safety**:

* Reference counting is always atomic (no separate ``atomic_shared_ptr``)
* Same thread-safety guarantees as standard library

Requirements
============

Compiler Support
________________

* C++11 or later
* Support for variadic templates
* Support for template aliases
* Support for ``constexpr`` and ``noexcept``
* Support for ``std::atomic``
* Support for deleted functions (``= delete``)

Dependencies
____________

* ``utilities.hpp`` - Type traits, move/forward, reference counting
* ``<cstddef>`` - For ``size_t``

Best Practices
==============

Prefer make_unique and make_shared
__________________________________

.. code-block:: cpp

   // Preferred
   auto p = cslt::make_unique<T>(args...);
   auto s = cslt::make_shared<T>(args...);
   
   // Avoid
   cslt::UniquePtr<T> p(new T(args...));
   cslt::SharedPtr<T> s(new T(args...));

**Reasons**:

* Exception safety (no leak if constructor throws)
* More concise
* Better performance (for ``make_shared``)

Use UniquePtr by Default
________________________

Start with ``UniquePtr`` and only use ``SharedPtr`` when shared ownership is actually needed:

.. code-block:: cpp

   // Good: clear ownership
   cslt::UniquePtr<Resource> create_resource() {
       return cslt::make_unique<Resource>();
   }
   
   // Only use SharedPtr when truly needed
   cslt::SharedPtr<Cache> global_cache = cslt::make_shared<Cache>();

Avoid Circular SharedPtr References
___________________________________

Use ``WeakPtr`` to break cycles:

.. code-block:: cpp

   // Bad: memory leak
   struct Node {
       cslt::SharedPtr<Node> next;
       cslt::SharedPtr<Node> prev;  // Circular reference!
   };
   
   // Good: cycle broken
   struct Node {
       cslt::SharedPtr<Node> next;
       cslt::WeakPtr<Node> prev;  // Weak reference
   };

Check WeakPtr Before Use
________________________

Always check ``expired()`` or test the result of ``lock()``:

.. code-block:: cpp

   cslt::WeakPtr<Resource> weak = /* ... */;
   
   // Method 1: Check expired
   if (!weak.expired()) {
       auto shared = weak.lock();
       // Use shared...
   }
   
   // Method 2: Check lock result
   if (auto shared = weak.lock()) {
       // Use shared...
   }

Don't Mix Ownership Models
__________________________

Avoid creating multiple ``SharedPtr`` instances from the same raw pointer:

.. code-block:: cpp

   // Bad: double delete!
   T* raw = new T();
   cslt::SharedPtr<T> p1(raw);
   cslt::SharedPtr<T> p2(raw);  // WRONG: two control blocks
   
   // Good: share through copy
   cslt::SharedPtr<T> p1 = cslt::make_shared<T>();
   cslt::SharedPtr<T> p2 = p1;  // Correct: shared control block

Common Pitfalls
===============

Dangling References from get()
______________________________

.. code-block:: cpp

   T* raw;
   {
       auto ptr = cslt::make_unique<T>();
       raw = ptr.get();  // Get raw pointer
   }
   // ptr destroyed, raw is now dangling!
   // *raw;  // Undefined behavior

**Solution**: Keep the smart pointer alive as long as you need access.

Forgetting to Move UniquePtr
____________________________

.. code-block:: cpp

   cslt::UniquePtr<T> create() {
       auto ptr = cslt::make_unique<T>();
       return ptr;  // Implicitly moved (OK)
   }
   
   void transfer(cslt::UniquePtr<T> dest, cslt::UniquePtr<T> src) {
       dest = src;  // ERROR: copy not allowed
       dest = cslt::move(src);  // OK
   }

Modifying Shared Object Without Synchronization
_______________________________________________

.. code-block:: cpp

   cslt::SharedPtr<int> shared = cslt::make_shared<int>(0);
   
   // Thread 1
   *shared = 42;  // Data race!
   
   // Thread 2  
   *shared = 99;  // Data race!

**Solution**: Use external synchronization (mutex) when modifying shared data.

Future Enhancements
===================

Potential additions for future versions:

* ``enable_shared_from_this<T>`` - Safe ``this`` sharing in member functions
* Single-allocation ``make_shared`` optimization
* Allocator support for custom memory allocation
* ``atomic_shared_ptr`` and ``atomic_weak_ptr`` wrappers
* Array support for ``SharedPtr`` (``SharedPtr<T[]>``)
* ``unique_ptr`` hash specialization
* C++17/20 features (deduction guides, ``std::span`` integration)
* ``observer_ptr`` - Non-owning raw pointer wrapper
