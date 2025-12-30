***********************
Error Handling Overview
***********************

The ``cslt`` library provides a robust, STL-independent error handling system designed for 
safety-critical and embedded applications. The error handling framework consists of two main 
components: a hierarchical exception class system and an ``Expected<T>`` type for 
exception-free error handling.

All error handling components are defined in ``Error.hpp`` and reside in the ``cslt`` namespace.

Error Class Overview 
====================

The ``cslt::Error`` class hierarchy provides a comprehensive, three-level taxonomy of error 
conditions designed for type-safe exception handling in C++ applications. At the base is the 
``Error`` class, which serves as the foundation for all errors in the library. This base class 
is specialized into nine category-level error classes (``ArgumentError``, ``MemoryError``, 
``StateError``, ``MathError``, ``IOError``, ``FormatError``, ``ConcurrencyError``, 
``ConfigError``, and ``GenericError``), each representing a broad class of related failures. 
These categories are further refined into 40+ specific error types that precisely describe 
individual error conditions.

This hierarchical design enables flexible error handling strategies. Applications can catch 
specific error types for fine-grained control (e.g., ``NullPointerError``), catch category-level 
errors to handle related failures uniformly (e.g., all ``ArgumentError`` types), or catch the 
base ``Error`` class to handle any library error. The hierarchy ensures that error handling can 
be as specific or general as the application requires.

Each error class provides two constructors: a default constructor that uses a predefined, 
descriptive error message, and a parameterized constructor that accepts a custom message for 
context-specific details. All error messages are stored in fixed 256-byte buffers, eliminating 
dynamic allocation and ensuring predictable memory usage—critical for embedded systems and 
safety-critical applications.

Error Class Hierarchy
---------------------

The following diagram illustrates the complete error hierarchy:

.. code-block:: text

   Error (base)
   ├── ArgumentError
   │   ├── InvalidArgError
   │   ├── NullPointerError
   │   ├── OutOfBoundsError
   │   ├── SizeMismatchError
   │   ├── UninitializedError
   │   ├── IteratorInvalidError
   │   ├── PreconditionFailError
   │   ├── PostconditionFailError
   │   └── IllegalStateError
   ├── MemoryError
   │   ├── BadAllocError
   │   ├── ReallocFailError
   │   ├── OutOfMemoryError
   │   ├── LengthOverflowError
   │   ├── CapacityOverflowError
   │   └── AlignmentError
   ├── StateError
   │   ├── StateCorruptError
   │   ├── AlreadyInitializedError
   │   ├── NotFoundError
   │   ├── EmptyError
   │   └── ConcurrentModificationError
   ├── MathError
   │   ├── DivByZeroError
   │   ├── SingularMatrixError
   │   ├── NumericOverflowError
   │   ├── DomainError
   │   └── LossOfPrecisionError
   ├── IOError
   │   ├── FileOpenError
   │   ├── FileReadError
   │   ├── FileWriteError
   │   ├── PermissionDeniedError
   │   ├── IOInterruptedError
   │   ├── IOTimeoutError
   │   ├── IOClosedError
   │   └── IOWouldBlockError
   ├── FormatError
   │   ├── TypeMismatchError
   │   ├── FormatInvalidError
   │   ├── EncodingInvalidError
   │   ├── ParsingFailedError
   │   └── ValidationFailedError
   ├── ConcurrencyError
   │   ├── LockFailedError
   │   ├── DeadlockDetectedError
   │   ├── ThreadFailError
   │   ├── CancelledError
   │   └── RaceDetectedError
   ├── ConfigError
   │   ├── ConfigInvalidError
   │   ├── UnsupportedError
   │   ├── FeatureDisabledError
   │   ├── VersionMismatchError
   │   └── ResourceExhaustedError
   └── GenericError
       ├── NotImplementedError
       ├── OperationUnavailableError
       └── UnknownError

Base Error Class
----------------

.. doxygenclass:: cslt::Error
   :members:
   :protected-members:

Category-Level Error Classes
-----------------------------

No Error 
~~~~~~~~

.. doxygenclass:: cslt::NoError
   :members:

Argument and Input Validation Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenclass:: cslt::ArgumentError
   :members:

**Specific ArgumentError Types:**

.. doxygenclass:: cslt::InvalidArgError
   :members:

.. doxygenclass:: cslt::NullPointerError
   :members:

.. doxygenclass:: cslt::OutOfBoundsError
   :members:

.. doxygenclass:: cslt::SizeMismatchError
   :members:

.. doxygenclass:: cslt::UninitializedError
   :members:

.. doxygenclass:: cslt::IteratorInvalidError
   :members:

.. doxygenclass:: cslt::PreconditionFailError
   :members:

.. doxygenclass:: cslt::PostconditionFailError
   :members:

.. doxygenclass:: cslt::IllegalStateError
   :members:

Memory Management Errors
~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenclass:: cslt::MemoryError
   :members:

**Specific MemoryError Types:**

.. doxygenclass:: cslt::BadAllocError
   :members:

.. doxygenclass:: cslt::ReallocFailError
   :members:

.. doxygenclass:: cslt::OutOfMemoryError
   :members:

.. doxygenclass:: cslt::LengthOverflowError
   :members:

.. doxygenclass:: cslt::CapacityOverflowError
   :members:

.. doxygenclass:: cslt::AlignmentError
   :members:

State and Container Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenclass:: cslt::StateError
   :members:

**Specific StateError Types:**

.. doxygenclass:: cslt::StateCorruptError
   :members:

.. doxygenclass:: cslt::AlreadyInitializedError
   :members:

.. doxygenclass:: cslt::NotFoundError
   :members:

.. doxygenclass:: cslt::EmptyError
   :members:

.. doxygenclass:: cslt::ConcurrentModificationError
   :members:

Mathematical Computation Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenclass:: cslt::MathError
   :members:

**Specific MathError Types:**

.. doxygenclass:: cslt::DivByZeroError
   :members:

.. doxygenclass:: cslt::SingularMatrixError
   :members:

.. doxygenclass:: cslt::NumericOverflowError
   :members:

.. doxygenclass:: cslt::DomainError
   :members:

.. doxygenclass:: cslt::LossOfPrecisionError
   :members:

Input/Output Errors
~~~~~~~~~~~~~~~~~~~

.. doxygenclass:: cslt::IOError
   :members:

**Specific IOError Types:**

.. doxygenclass:: cslt::FileOpenError
   :members:

.. doxygenclass:: cslt::FileReadError
   :members:

.. doxygenclass:: cslt::FileWriteError
   :members:

.. doxygenclass:: cslt::PermissionDeniedError
   :members:

.. doxygenclass:: cslt::IOInterruptedError
   :members:

.. doxygenclass:: cslt::IOTimeoutError
   :members:

.. doxygenclass:: cslt::IOClosedError
   :members:

.. doxygenclass:: cslt::IOWouldBlockError
   :members:

Format and Encoding Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenclass:: cslt::FormatError
   :members:

**Specific FormatError Types:**

.. doxygenclass:: cslt::TypeMismatchError
   :members:

.. doxygenclass:: cslt::FormatInvalidError
   :members:

.. doxygenclass:: cslt::EncodingInvalidError
   :members:

.. doxygenclass:: cslt::ParsingFailedError
   :members:

.. doxygenclass:: cslt::ValidationFailedError
   :members:

Concurrency and Synchronization Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenclass:: cslt::ConcurrencyError
   :members:

**Specific ConcurrencyError Types:**

.. doxygenclass:: cslt::LockFailedError
   :members:

.. doxygenclass:: cslt::DeadlockDetectedError
   :members:

.. doxygenclass:: cslt::ThreadFailError
   :members:

.. doxygenclass:: cslt::CancelledError
   :members:

.. doxygenclass:: cslt::RaceDetectedError
   :members:

Configuration and Environment Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenclass:: cslt::ConfigError
   :members:

**Specific ConfigError Types:**

.. doxygenclass:: cslt::ConfigInvalidError
   :members:

.. doxygenclass:: cslt::UnsupportedError
   :members:

.. doxygenclass:: cslt::FeatureDisabledError
   :members:

.. doxygenclass:: cslt::VersionMismatchError
   :members:

.. doxygenclass:: cslt::ResourceExhaustedError
   :members:

Generic and Fallback Errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. doxygenclass:: cslt::GenericError
   :members:

**Specific GenericError Types:**

.. doxygenclass:: cslt::NotImplementedError
   :members:

.. doxygenclass:: cslt::OperationUnavailableError
   :members:

.. doxygenclass:: cslt::UnknownError
   :members:

Expected Class Overview
=======================

The ``cslt::Expected<T>`` template class provides a type-safe mechanism for representing 
computations that may either succeed with a value of type ``T`` or fail with a ``cslt::Error``. 
Unlike traditional exception-based error handling, ``Expected<T>`` returns errors as values, 
making error handling explicit and predictable. This approach is particularly valuable in 
performance-critical code, embedded systems, real-time applications, and scenarios where 
exceptions are disabled or discouraged by coding standards like MISRA C++.

The ``Expected<T>`` class forces callers to explicitly acknowledge the possibility of failure 
before accessing a result. This "errors as values" paradigm eliminates hidden control flow, 
makes error propagation visible in the code, and enables the compiler to optimize more 
aggressively since there is no exception unwinding overhead. All storage is stack-based with 
no dynamic allocation, making behavior completely predictable and suitable for safety-critical 
applications.

Design Philosophy
-----------------

Unlike ``std::expected`` from C++23, ``cslt::Expected<T>`` makes several opinionated design 
choices optimized for embedded and safety-critical environments:

**Fixed Error Type**
  The error type is always ``cslt::Error`` (or any of its derived types), eliminating the need 
  for a second template parameter. This ensures consistency across the library and simplifies 
  error handling patterns.

**Static Storage**
  Both the value and error are stored simultaneously in the structure, avoiding the complexity 
  of union-based storage with placement new. While this uses more memory 
  (``sizeof(Expected<T>) == sizeof(bool) + sizeof(T) + sizeof(Error)``), it provides:
  
  - Trivial copyability (compiler-generated copy/move operations work correctly)
  - No manual memory management or placement new
  - Simple, predictable behavior
  - Full MISRA C++ compliance

**Explicit State Management**
  State transitions are performed through explicit ``setValue()`` and ``setError()`` methods 
  rather than constructor overloads, making the code's intent clearer and more auditable.

**No Monadic Operations**
  The class intentionally omits monadic operations like ``and_then()``, ``or_else()``, and 
  ``transform()`` to keep the interface minimal and predictable for embedded use cases.

Memory Layout and Performance
------------------------------

The ``Expected<T>`` class uses a straightforward memory layout:

.. code-block:: cpp

   template<typename T>
   class Expected {
   private:
       bool has_value_;    // 1 byte (+ padding)
       T value_;           // sizeof(T) bytes
       Error error_;       // 256 bytes (fixed buffer)
   };

This results in:

.. code-block:: bash

   sizeof(Expected<int>)     = 260 bytes
   sizeof(Expected<double>)  = 264 bytes
   sizeof(Expected<void*>)   = 264 bytes

While larger than a union-based approach, this design provides:

- **Zero dynamic allocation** - All memory is on the stack
- **Predictable performance** - No heap operations or cache misses
- **Simple debugging** - Both value and error visible in debugger
- **Compiler optimization** - Trivially copyable enables more optimizations

For most use cases, the memory overhead is negligible compared to the benefits of simplicity 
and safety.

Basic Usage
-----------

Creating an Expected result:

.. code-block:: cpp

   #include <cslt/Expected.hpp>
   
   cslt::Expected<int> divide(int numerator, int denominator) {
       cslt::Expected<int> result;
       
       if (denominator == 0) {
           result.setError(cslt::DivByZeroError("Cannot divide by zero"));
           return result;
       }
       
       result.setValue(numerator / denominator);
       return result;
   }

Checking and using the result:

.. code-block:: cpp

   cslt::Expected<int> result = divide(10, 2);
   
   // Method 1: Check with hasValue() / hasError()
   if (result.hasValue()) {
       printf("Result: %d\n", result.value());
   } else {
       printf("Error: %s\n", result.error().what());
   }
   
   // Method 2: Use explicit bool conversion
   if (result) {
       printf("Success: %d\n", result.value());
   } else {
       fprintf(stderr, "Failed: %s\n", result.error().what());
   }
   
   // Method 3: Use valueOr() for a safe default
   int safeValue = divide(10, 0).valueOr(-1);

Error Propagation Patterns
---------------------------

**Manual Propagation**

.. code-block:: cpp

   cslt::Expected<double> complexCalculation(int a, int b, int c) {
       cslt::Expected<int> step1 = divide(a, b);
       if (step1.hasError()) {
           cslt::Expected<double> result;
           result.setError(step1.error());
           return result;
       }
       
       cslt::Expected<int> step2 = divide(step1.value(), c);
       if (step2.hasError()) {
           cslt::Expected<double> result;
           result.setError(step2.error());
           return result;
       }
       
       cslt::Expected<double> result;
       result.setValue(static_cast<double>(step2.value()));
       return result;
   }

**Early Return Pattern**

.. code-block:: cpp

   cslt::Expected<int> processData(int* data, size_t size, size_t index) {
       cslt::Expected<int> result;
       
       if (data == nullptr) {
           result.setError(cslt::NullPointerError("Data pointer is null"));
           return result;
       }
       
       if (index >= size) {
           result.setError(cslt::OutOfBoundsError("Index out of range"));
           return result;
       }
       
       if (size == 0) {
           result.setError(cslt::EmptyError("Data array is empty"));
           return result;
       }
       
       // All checks passed, return the value
       result.setValue(data[index]);
       return result;
   }

When to Use Expected vs Exceptions
-----------------------------------

Use ``Expected<T>`` when:
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Performance is critical** - No exception unwinding overhead
- **Errors are expected** - Function commonly returns errors as part of normal operation
- **Deep call stacks** - Returning errors is faster than unwinding many stack frames
- **MISRA compliance required** - Exceptions may be forbidden
- **Embedded systems** - Predictable memory usage and no heap allocation needed
- **Real-time systems** - Deterministic execution time is required
- **Error recovery** - Caller needs detailed error information to recover

Use ``throw Error`` when:
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Errors are exceptional** - Failure indicates a bug or very rare condition
- **Simplicity matters** - Exception handling keeps error paths out of normal code flow
- **Deep stack unwinding needed** - Need automatic cleanup via destructors (RAII)
- **Multiple error points** - Many potential failure points that would clutter return checking
- **Standard C++ environment** - Exceptions enabled and expected by ecosystem
- **Library integration** - Interoperating with code that expects exceptions

Comparison Table
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Aspect
     - ``Expected<T>``
     - ``throw Error``
   * - Performance
     - Faster (no unwinding)
     - Slower (stack unwinding)
   * - Memory
     - Stack only, predictable
     - May use heap for exception objects
   * - Visibility
     - Explicit in signatures
     - Hidden (not in signature)
   * - Error handling
     - Mandatory checking
     - Can be forgotten
   * - Control flow
     - Explicit returns
     - Non-local jumps
   * - MISRA compliance
     - ✓ Compliant
     - ✗ Often forbidden
   * - Real-time systems
     - ✓ Deterministic
     - ✗ Non-deterministic
   * - Code complexity
     - More verbose
     - Cleaner normal path

API Reference
-------------

.. doxygenclass:: cslt::Expected
   :members:

Complete Example
----------------

Here's a complete example demonstrating ``Expected<T>`` in a realistic scenario:

.. code-block:: cpp

   #include <cslt/Expected.hpp>
   #include <cslt/Error.hpp>
   #include <cstdio>
   
   // Safe array access with detailed error reporting
   cslt::Expected<int> getArrayElement(int* array, size_t size, size_t index) {
       cslt::Expected<int> result;
       
       // Validate input
       if (array == nullptr) {
           result.setError(cslt::NullPointerError("Array pointer is null"));
           return result;
       }
       
       if (size == 0) {
           result.setError(cslt::EmptyError("Array is empty"));
           return result;
       }
       
       if (index >= size) {
           result.setError(cslt::OutOfBoundsError("Index exceeds array bounds"));
           return result;
       }
       
       // Success path
       result.setValue(array[index]);
       return result;
   }
   
   // Chain multiple operations
   cslt::Expected<double> calculateAverage(int* array, size_t size) {
       cslt::Expected<double> result;
       
       if (size == 0) {
           result.setError(cslt::EmptyError("Cannot average empty array"));
           return result;
       }
       
       int sum = 0;
       for (size_t i = 0; i < size; ++i) {
           cslt::Expected<int> elem = getArrayElement(array, size, i);
           if (elem.hasError()) {
               result.setError(elem.error());
               return result;
           }
           sum += elem.value();
       }
       
       result.setValue(static_cast<double>(sum) / size);
       return result;
   }
   
   int main() {
       int data[] = {10, 20, 30, 40, 50};
       
       // Example 1: Successful access
       cslt::Expected<int> elem = getArrayElement(data, 5, 2);
       if (elem.hasValue()) {
           printf("Element at index 2: %d\n", elem.value());
       } else {
           fprintf(stderr, "Error: %s\n", elem.error().what());
       }
       
       // Example 2: Out of bounds
       cslt::Expected<int> invalid = getArrayElement(data, 5, 10);
       if (invalid.hasError()) {
           fprintf(stderr, "Expected error: %s\n", invalid.error().what());
       }
       
       // Example 3: Chained operations
       cslt::Expected<double> avg = calculateAverage(data, 5);
       if (avg.hasValue()) {
           printf("Average: %.2f\n", avg.value());
       } else {
           fprintf(stderr, "Failed to calculate average: %s\n", avg.error().what());
       }
       
       // Example 4: Using valueOr for defaults
       int safeValue = getArrayElement(nullptr, 0, 0).valueOr(-1);
       printf("Safe value with default: %d\n", safeValue);
       
       return 0;
   }

Integration with Error Hierarchy
---------------------------------

``Expected<T>`` works seamlessly with the entire ``cslt::Error`` hierarchy. You can set any 
error type, and the polymorphic behavior is preserved:

.. code-block:: cpp

   cslt::Expected<double> safeSqrt(double x) {
       cslt::Expected<double> result;
       
       if (x < 0.0) {
           // Use specific error type
           result.setError(cslt::DomainError("Cannot compute sqrt of negative number"));
           return result;
       }
       
       result.setValue(sqrt(x));
       return result;
   }
   
   // Caller can catch specific or general errors
   cslt::Expected<double> res = safeSqrt(-4.0);
   if (res.hasError()) {
       // Can check error type
       printf("Error: %s\n", res.error().what());
       
       // The error object is polymorphic - it's actually a DomainError
       // but accessible through the Error interface
   }

