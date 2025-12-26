// ================================================================================
// ================================================================================
// - File:    error.hpp
// - Purpose: Describe the file purpose here
//
// Source Metadata
// - Author:  Jonathan A. Webb
// - Date:    December 24, 2025
// - Version: 1.0
// - Copyright: Copyright 2022, Jon Webb Inc.
// ================================================================================
// ================================================================================
// Include modules here

#ifndef error_HPP
#define error_HPP

#include <cstddef>
// ================================================================================ 
// ================================================================================ 

namespace cslt {

    /**
     * @brief Base class for all cslt library exceptions with fixed-size error messages.
     * 
     * The Error class provides a lightweight, STL-independent exception mechanism with
     * predictable memory usage. Error messages are stored in a fixed-size buffer to
     * avoid heap allocations, making it suitable for embedded systems and real-time
     * applications.
     * 
     * All error messages are capped at MAX_MESSAGE_LEN (256 characters). Messages
     * exceeding this length are silently truncated.
     * 
     * @par Example Usage:
     * @code
     * // Using default error message
     * try {
     *     throw cslt::Error();
     * } catch (const cslt::Error& e) {
     *     printf("Error: %s\n", e.what());  // "An error occurred"
     * }
     * 
     * // Using custom error message
     * try {
     *     throw cslt::Error("Failed to process data");
     * } catch (const cslt::Error& e) {
     *     printf("Error: %s\n", e.what());  // "Failed to process data"
     * }
     * @endcode
     */
    class Error {
    protected:
        static constexpr size_t MAX_MESSAGE_LEN = 256;
        char message[MAX_MESSAGE_LEN];
// -------------------------------------------------------------------------------- 
        /**
         * @brief Safely copies a null-terminated string to a destination buffer.
         * 
         * Copies characters from src to dest up to maxLen-1 characters, ensuring
         * the destination is always null-terminated. If src is longer than maxLen-1,
         * the string is truncated.
         * 
         * @param dest Destination buffer to copy into
         * @param src Source null-terminated string to copy from
         * @param maxLen Maximum length of destination buffer including null terminator
         */
        static void safeCopy(char* dest, const char* src, size_t maxLen);
// -------------------------------------------------------------------------------- 
        /**
         * @brief Appends a string to the end of an existing null-terminated string.
         * 
         * Appends suffix to the end of dest, respecting the maxLen buffer size.
         * If the combined length would exceed maxLen-1, the result is truncated.
         * The destination buffer is always null-terminated.
         * 
         * @param dest Destination buffer containing existing string to append to
         * @param suffix Null-terminated string to append
         * @param maxLen Maximum length of destination buffer including null terminator
         */
        static void append(char* dest, const char* suffix, size_t maxLen);
// -------------------------------------------------------------------------------- 
        /**
         * @brief Prepends a string to the beginning of an existing null-terminated string.
         * 
         * Inserts prefix before the existing content in dest, respecting the maxLen
         * buffer size. If the combined length would exceed maxLen-1, the result is
         * truncated. The destination buffer is always null-terminated.
         * 
         * @param dest Destination buffer containing existing string to prepend to
         * @param prefix Null-terminated string to insert at the beginning
         * @param maxLen Maximum length of destination buffer including null terminator
         */
        static void prepend(char* dest, const char* prefix, size_t maxLen);
// -------------------------------------------------------------------------------- 
        /**
         * @brief Composes a new string by concatenating prefix and suffix.
         * 
         * Copies prefix into dest, then appends suffix. If the combined length
         * would exceed maxLen-1, the result is truncated. The destination buffer
         * is always null-terminated. Any existing content in dest is overwritten.
         * 
         * @param dest Destination buffer to write composed string into
         * @param prefix Null-terminated string to write first
         * @param suffix Null-terminated string to append after prefix
         * @param maxLen Maximum length of destination buffer including null terminator
         */
        static void compose(char* dest, const char* prefix, const char* suffix, size_t maxLen);
// ================================================================================ 
    public:
        /**
         * @brief Constructs an Error with the default message.
         * 
         * Creates an error object with the predefined message "An error occurred".
         * Use this constructor when a generic error indication is sufficient.
         */
        Error();
// -------------------------------------------------------------------------------- 
        /**
         * @brief Constructs an Error with a custom message.
         * 
         * Creates an error object with the specified error message. If msg exceeds
         * MAX_MESSAGE_LEN-1 characters, it will be truncated to fit.
         * 
         * @param msg Null-terminated custom error message string (max 255 characters)
         */
        Error(const char* msg);
// -------------------------------------------------------------------------------- 
        /**
         * @brief Returns the error message.
         * 
         * @return Pointer to null-terminated error message string
         */
        virtual const char* what() const;
// -------------------------------------------------------------------------------- 
        /**
         * @brief Virtual destructor for proper cleanup of derived classes.
         */
        virtual ~Error();
    };
// ================================================================================ 
// ================================================================================ 

    /**
     * @brief Argument and input validation errors.
     * 
     * ArgumentError represents errors related to invalid function arguments,
     * null pointers, out-of-bounds access, and other input validation failures.
     * 
     * @par Example Usage:
     * @code
     * // Using default message
     * throw ArgumentError();  // "Invalid argument"
     * 
     * // Using custom message
     * throw ArgumentError("Index 5 is out of bounds for array of size 3");
     * @endcode
     */
    class ArgumentError : public Error {
    public:
        /**
         * @brief Constructs an ArgumentError with the default message.
         * 
         * Creates an ArgumentError with the predefined message "Invalid argument".
         */
        ArgumentError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs an ArgumentError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        ArgumentError(const char* msg);
    };

// ================================================================================
// ================================================================================ 

    /**
     * @brief Memory allocation and management errors.
     * 
     * MemoryError represents errors related to memory allocation failures,
     * reallocation issues, out-of-memory conditions, and alignment problems.
     * 
     * @par Example Usage:
     * @code
     * // Using default message
     * throw MemoryError();  // "Memory allocation failed"
     * 
     * // Using custom message
     * throw MemoryError("Failed to allocate 1024 bytes");
     * @endcode
     */
    class MemoryError : public Error {
    public:
        /**
         * @brief Constructs a MemoryError with the default message.
         * 
         * Creates a MemoryError with the predefined message "Memory allocation failed".
         */
        MemoryError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a MemoryError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        MemoryError(const char* msg);
    };
// ================================================================================
// ================================================================================ 

    /**
     * @brief State and container management errors.
     * 
     * StateError represents errors related to invalid object states, corruption,
     * initialization issues, and container operations.
     * 
     * @par Example Usage:
     * @code
     * // Using default message
     * throw StateError();  // "Invalid state"
     * 
     * // Using custom message
     * throw StateError("Container already initialized");
     * @endcode
     */
    class StateError : public Error {
    public:
        /**
         * @brief Constructs a StateError with the default message.
         * 
         * Creates a StateError with the predefined message "Invalid state".
         */
        StateError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a StateError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        StateError(const char* msg);
    };
// ================================================================================
// ================================================================================ 

    /**
     * @brief Mathematical and numerical computation errors.
     * 
     * MathError represents errors related to mathematical operations such as
     * division by zero, matrix singularities, numeric overflow, and domain errors.
     * 
     * @par Example Usage:
     * @code
     * // Using default message
     * throw MathError();  // "Mathematical error"
     * 
     * // Using custom message
     * throw MathError("Division by zero in computation");
     * @endcode
     */
    class MathError : public Error {
    public:
        /**
         * @brief Constructs a MathError with the default message.
         * 
         * Creates a MathError with the predefined message "Mathematical error".
         */
        MathError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a MathError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        MathError(const char* msg);
    };
// ================================================================================
// ================================================================================ 

    /**
     * @brief File and I/O operation errors.
     * 
     * IOError represents errors related to file operations, reading, writing,
     * permissions, timeouts, and other I/O operations.
     * 
     * @par Example Usage:
     * @code
     * // Using default message
     * throw IOError();  // "I/O operation failed"
     * 
     * // Using custom message
     * throw IOError("Failed to open file: config.txt");
     * @endcode
     */
    class IOError : public Error {
    public:
        /**
         * @brief Constructs an IOError with the default message.
         * 
         * Creates an IOError with the predefined message "I/O operation failed".
         */
        IOError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs an IOError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        IOError(const char* msg);
    };
// ================================================================================ 
// ================================================================================

    /**
     * @brief Type, format, and encoding errors.
     * 
     * FormatError represents errors related to type mismatches, invalid data formats,
     * encoding issues, parsing failures, and validation errors.
     * 
     * @par Example Usage:
     * @code
     * // Using default message
     * throw FormatError();  // "Format error"
     * 
     * // Using custom message
     * throw FormatError("Invalid JSON format at line 42");
     * @endcode
     */
    class FormatError : public Error {
    public:
        /**
         * @brief Constructs a FormatError with the default message.
         * 
         * Creates a FormatError with the predefined message "Format error".
         */
        FormatError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a FormatError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        FormatError(const char* msg);
    };
// ================================================================================ 
// ================================================================================

    /**
     * @brief Concurrency and synchronization errors.
     * 
     * ConcurrencyError represents errors related to locking, deadlocks, thread
     * operations, cancellations, and race conditions.
     * 
     * @par Example Usage:
     * @code
     * // Using default message
     * throw ConcurrencyError();  // "Concurrency error"
     * 
     * // Using custom message
     * throw ConcurrencyError("Deadlock detected in mutex acquisition");
     * @endcode
     */
    class ConcurrencyError : public Error {
    public:
        /**
         * @brief Constructs a ConcurrencyError with the default message.
         * 
         * Creates a ConcurrencyError with the predefined message "Concurrency error".
         */
        ConcurrencyError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a ConcurrencyError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        ConcurrencyError(const char* msg);
    };
// ================================================================================ 
// ================================================================================

    /**
     * @brief Configuration, policy, and environment errors.
     * 
     * ConfigError represents errors related to invalid configurations, unsupported
     * features, version mismatches, and resource exhaustion.
     * 
     * @par Example Usage:
     * @code
     * // Using default message
     * throw ConfigError();  // "Configuration error"
     * 
     * // Using custom message
     * throw ConfigError("Unsupported platform: ARM64");
     * @endcode
     */
    class ConfigError : public Error {
    public:
        /**
         * @brief Constructs a ConfigError with the default message.
         * 
         * Creates a ConfigError with the predefined message "Configuration error".
         */
        ConfigError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a ConfigError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        ConfigError(const char* msg);
    };
// ================================================================================ 
// ================================================================================

    /**
     * @brief Generic and unspecified errors.
     * 
     * GenericError represents generic fallback errors including not-implemented
     * functionality, unavailable operations, and unknown errors.
     * 
     * @par Example Usage:
     * @code
     * // Using default message
     * throw GenericError();  // "An error occurred"
     * 
     * // Using custom message
     * throw GenericError("Feature not yet implemented");
     * @endcode
     */
    class GenericError : public Error {
    public:
        /**
         * @brief Constructs a GenericError with the default message.
         * 
         * Creates a GenericError with the predefined message "An error occurred".
         */
        GenericError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a GenericError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        GenericError(const char* msg);
    };
// ================================================================================ 
// ================================================================================ 

    /**
     * @brief Invalid function argument error.
     * 
     * InvalidArgError represents an error when an invalid argument is passed
     * to a function.
     * 
     * @par Example Usage:
     * @code
     * throw InvalidArgError();  // "Invalid function argument"
     * throw InvalidArgError("Expected positive value, got -5");
     * @endcode
     */
    class InvalidArgError : public ArgumentError {
    public:
        /**
         * @brief Constructs an InvalidArgError with the default message.
         */
        InvalidArgError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs an InvalidArgError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        InvalidArgError(const char* msg);
    };
// ================================================================================ 
// ================================================================================

    /**
     * @brief Null pointer error.
     * 
     * NullPointerError represents an error when a null pointer is passed
     * where a valid pointer is required.
     * 
     * @par Example Usage:
     * @code
     * throw NullPointerError();  // "Null pointer passed"
     * throw NullPointerError("Buffer pointer is null");
     * @endcode
     */
    class NullPointerError : public ArgumentError {
    public:
        /**
         * @brief Constructs a NullPointerError with the default message.
         */
        NullPointerError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a NullPointerError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        NullPointerError(const char* msg);
    };
// ================================================================================ 
// ================================================================================

    /**
     * @brief Out of bounds access error.
     * 
     * OutOfBoundsError represents an error when an index or position is
     * outside the valid range.
     * 
     * @par Example Usage:
     * @code
     * throw OutOfBoundsError();  // "Index out of range"
     * throw OutOfBoundsError("Index 10 exceeds array size 5");
     * @endcode
     */
    class OutOfBoundsError : public ArgumentError {
    public:
        /**
         * @brief Constructs an OutOfBoundsError with the default message.
         */
        OutOfBoundsError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs an OutOfBoundsError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        OutOfBoundsError(const char* msg);
    };
// ================================================================================ 
// ================================================================================

    /**
     * @brief Size or dimension mismatch error.
     * 
     * SizeMismatchError represents an error when dimensions or sizes of
     * objects don't match as required for an operation.
     * 
     * @par Example Usage:
     * @code
     * throw SizeMismatchError();  // "Dimension/size mismatch"
     * throw SizeMismatchError("Matrix dimensions incompatible: 3x4 and 2x3");
     * @endcode
     */
    class SizeMismatchError : public ArgumentError {
    public:
        /**
         * @brief Constructs a SizeMismatchError with the default message.
         */
        SizeMismatchError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a SizeMismatchError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        SizeMismatchError(const char* msg);
    };
// ================================================================================ 
// ================================================================================

    /**
     * @brief Uninitialized element access error.
     * 
     * UninitializedError represents an error when attempting to access an
     * element or object that has not been properly initialized.
     * 
     * @par Example Usage:
     * @code
     * throw UninitializedError();  // "Uninitialized element access"
     * throw UninitializedError("Attempting to read uninitialized variable");
     * @endcode
     */
    class UninitializedError : public ArgumentError {
    public:
        /**
         * @brief Constructs an UninitializedError with the default message.
         */
        UninitializedError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs an UninitializedError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        UninitializedError(const char* msg);
    };
// ================================================================================ 
// ================================================================================

    /**
     * @brief Invalid iterator or cursor error.
     * 
     * IteratorInvalidError represents an error when an iterator or cursor
     * is dangling, invalidated, or otherwise unusable.
     * 
     * @par Example Usage:
     * @code
     * throw IteratorInvalidError();  // "Invalid iterator/cursor"
     * throw IteratorInvalidError("Iterator invalidated by container modification");
     * @endcode
     */
    class IteratorInvalidError : public ArgumentError {
    public:
        /**
         * @brief Constructs an IteratorInvalidError with the default message.
         */
        IteratorInvalidError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs an IteratorInvalidError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        IteratorInvalidError(const char* msg);
    };
// ================================================================================ 
// ================================================================================

    /**
     * @brief Precondition failure error.
     * 
     * PreconditionFailError represents an error when a required precondition
     * for an operation is not satisfied.
     * 
     * @par Example Usage:
     * @code
     * throw PreconditionFailError();  // "Precondition failed"
     * throw PreconditionFailError("Array must be sorted before binary search");
     * @endcode
     */
    class PreconditionFailError : public ArgumentError {
    public:
        /**
         * @brief Constructs a PreconditionFailError with the default message.
         */
        PreconditionFailError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a PreconditionFailError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        PreconditionFailError(const char* msg);
    };
// ================================================================================ 
// ================================================================================

    /**
     * @brief Postcondition failure error.
     * 
     * PostconditionFailError represents an error when a postcondition or
     * invariant is violated after an operation.
     * 
     * @par Example Usage:
     * @code
     * throw PostconditionFailError();  // "Postcondition failed"
     * throw PostconditionFailError("Invariant violated: size must equal capacity");
     * @endcode
     */
    class PostconditionFailError : public ArgumentError {
    public:
        /**
         * @brief Constructs a PostconditionFailError with the default message.
         */
        PostconditionFailError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a PostconditionFailError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        PostconditionFailError(const char* msg);
    };
// ================================================================================ 
// ================================================================================

    /**
     * @brief Illegal state for operation error.
     * 
     * IllegalStateError represents an error when an API call is not valid
     * for the current state of the object.
     * 
     * @par Example Usage:
     * @code
     * throw IllegalStateError();  // "Illegal state for operation"
     * throw IllegalStateError("Cannot read from closed stream");
     * @endcode
     */
    class IllegalStateError : public ArgumentError {
    public:
        /**
         * @brief Constructs an IllegalStateError with the default message.
         */
        IllegalStateError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs an IllegalStateError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        IllegalStateError(const char* msg);
    };
// ================================================================================ 
// ================================================================================ 

    /**
     * @brief Memory allocation failure error.
     * 
     * BadAllocError represents an error when malloc, calloc, or new fails
     * to allocate memory.
     * 
     * @par Example Usage:
     * @code
     * throw BadAllocError();  // "Memory allocation failed"
     * throw BadAllocError("Failed to allocate 1024 bytes for buffer");
     * @endcode
     */
    class BadAllocError : public MemoryError {
    public:
        /**
         * @brief Constructs a BadAllocError with the default message.
         */
        BadAllocError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a BadAllocError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        BadAllocError(const char* msg);
    };

// ================================================================================
    /**
     * @brief Memory reallocation failure error.
     * 
     * ReallocFailError represents an error when realloc fails to resize
     * a memory block (the original buffer remains unchanged).
     * 
     * @par Example Usage:
     * @code
     * throw ReallocFailError();  // "Memory reallocation failed"
     * throw ReallocFailError("Failed to expand buffer from 512 to 1024 bytes");
     * @endcode
     */
    class ReallocFailError : public MemoryError {
    public:
        /**
         * @brief Constructs a ReallocFailError with the default message.
         */
        ReallocFailError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a ReallocFailError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        ReallocFailError(const char* msg);
    };

// ================================================================================
    /**
     * @brief System out of memory error.
     * 
     * OutOfMemoryError represents an error when the system is completely
     * out of memory or an allocator has reached its limit.
     * 
     * @par Example Usage:
     * @code
     * throw OutOfMemoryError();  // "Out of memory"
     * throw OutOfMemoryError("System memory exhausted");
     * @endcode
     */
    class OutOfMemoryError : public MemoryError {
    public:
        /**
         * @brief Constructs an OutOfMemoryError with the default message.
         */
        OutOfMemoryError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs an OutOfMemoryError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        OutOfMemoryError(const char* msg);
    };

// ================================================================================
    /**
     * @brief Length or size arithmetic overflow error.
     * 
     * LengthOverflowError represents an error when size or length calculations
     * result in arithmetic overflow.
     * 
     * @par Example Usage:
     * @code
     * throw LengthOverflowError();  // "Length/size arithmetic overflow"
     * throw LengthOverflowError("Size calculation overflowed: SIZE_MAX exceeded");
     * @endcode
     */
    class LengthOverflowError : public MemoryError {
    public:
        /**
         * @brief Constructs a LengthOverflowError with the default message.
         */
        LengthOverflowError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a LengthOverflowError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        LengthOverflowError(const char* msg);
    };

// ================================================================================
    /**
     * @brief Capacity limit exceeded error.
     * 
     * CapacityOverflowError represents an error when a capacity policy or
     * representable limit is exceeded.
     * 
     * @par Example Usage:
     * @code
     * throw CapacityOverflowError();  // "Capacity limit exceeded"
     * throw CapacityOverflowError("Container capacity cannot exceed 65535 elements");
     * @endcode
     */
    class CapacityOverflowError : public MemoryError {
    public:
        /**
         * @brief Constructs a CapacityOverflowError with the default message.
         */
        CapacityOverflowError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a CapacityOverflowError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        CapacityOverflowError(const char* msg);
    };

// ================================================================================
    /**
     * @brief Memory alignment requirement error.
     * 
     * AlignmentError represents an error when required memory alignment
     * is not satisfied.
     * 
     * @par Example Usage:
     * @code
     * throw AlignmentError();  // "Required alignment not satisfied"
     * throw AlignmentError("Pointer must be 16-byte aligned for SIMD operations");
     * @endcode
     */
    class AlignmentError : public MemoryError {
    public:
        /**
         * @brief Constructs an AlignmentError with the default message.
         */
        AlignmentError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs an AlignmentError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        AlignmentError(const char* msg);
    };
// ================================================================================ 
// ================================================================================ 

    /**
     * @brief Internal state corruption error.
     * 
     * StateCorruptError represents an error when internal invariants are
     * violated or data corruption is detected.
     * 
     * @par Example Usage:
     * @code
     * throw StateCorruptError();  // "Internal state corrupt"
     * throw StateCorruptError("Checksum mismatch: data corruption detected");
     * @endcode
     */
    class StateCorruptError : public StateError {
    public:
        /**
         * @brief Constructs a StateCorruptError with the default message.
         */
        StateCorruptError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a StateCorruptError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        StateCorruptError(const char* msg);
    };

// ================================================================================
    /**
     * @brief Already initialized error.
     * 
     * AlreadyInitializedError represents an error when attempting to initialize
     * an object that has already been initialized (double-initialization).
     * 
     * @par Example Usage:
     * @code
     * throw AlreadyInitializedError();  // "Already initialized"
     * throw AlreadyInitializedError("Cannot reinitialize active connection");
     * @endcode
     */
    class AlreadyInitializedError : public StateError {
    public:
        /**
         * @brief Constructs an AlreadyInitializedError with the default message.
         */
        AlreadyInitializedError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs an AlreadyInitializedError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        AlreadyInitializedError(const char* msg);
    };

// ================================================================================
    /**
     * @brief Item or key not found error.
     * 
     * NotFoundError represents an error when a requested item or key
     * does not exist in a collection.
     * 
     * @par Example Usage:
     * @code
     * throw NotFoundError();  // "Item not found"
     * throw NotFoundError("Key 'username' not found in dictionary");
     * @endcode
     */
    class NotFoundError : public StateError {
    public:
        /**
         * @brief Constructs a NotFoundError with the default message.
         */
        NotFoundError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a NotFoundError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        NotFoundError(const char* msg);
    };

// ================================================================================
    /**
     * @brief Container empty error.
     * 
     * EmptyError represents an error when attempting to access or remove
     * elements from an empty container.
     * 
     * @par Example Usage:
     * @code
     * throw EmptyError();  // "Container is empty"
     * throw EmptyError("Cannot pop from empty stack");
     * @endcode
     */
    class EmptyError : public StateError {
    public:
        /**
         * @brief Constructs an EmptyError with the default message.
         */
        EmptyError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs an EmptyError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        EmptyError(const char* msg);
    };

// ================================================================================
    /**
     * @brief Concurrent modification detected error.
     * 
     * ConcurrentModificationError represents an error when a container is
     * modified during iteration (fail-fast behavior).
     * 
     * @par Example Usage:
     * @code
     * throw ConcurrentModificationError();  // "Concurrent modification detected"
     * throw ConcurrentModificationError("Container modified during iteration");
     * @endcode
     */
    class ConcurrentModificationError : public StateError {
    public:
        /**
         * @brief Constructs a ConcurrentModificationError with the default message.
         */
        ConcurrentModificationError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a ConcurrentModificationError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        ConcurrentModificationError(const char* msg);
    };
// ================================================================================ 
// ================================================================================ 

    /**
     * @brief Division by zero error.
     * 
     * DivByZeroError represents an error when attempting to divide by zero.
     * 
     * @par Example Usage:
     * @code
     * throw DivByZeroError();  // "Division by zero"
     * throw DivByZeroError("Cannot divide 10 by 0");
     * @endcode
     */
    class DivByZeroError : public MathError {
    public:
        /**
         * @brief Constructs a DivByZeroError with the default message.
         */
        DivByZeroError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a DivByZeroError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        DivByZeroError(const char* msg);
    };

// ================================================================================
    /**
     * @brief Singular or non-invertible matrix error.
     * 
     * SingularMatrixError represents an error when attempting operations
     * that require a non-singular matrix (e.g., matrix inversion, solving
     * linear systems) but the matrix is singular.
     * 
     * @par Example Usage:
     * @code
     * throw SingularMatrixError();  // "Singular/non-invertible matrix"
     * throw SingularMatrixError("Matrix determinant is zero, cannot invert");
     * @endcode
     */
    class SingularMatrixError : public MathError {
    public:
        /**
         * @brief Constructs a SingularMatrixError with the default message.
         */
        SingularMatrixError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a SingularMatrixError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        SingularMatrixError(const char* msg);
    };

// ================================================================================
    /**
     * @brief Numeric overflow or underflow error.
     * 
     * NumericOverflowError represents an error when a numeric computation
     * results in overflow (exceeds maximum) or underflow (below minimum
     * representable value).
     * 
     * @par Example Usage:
     * @code
     * throw NumericOverflowError();  // "Numeric overflow/underflow"
     * throw NumericOverflowError("Result exceeds maximum double value");
     * @endcode
     */
    class NumericOverflowError : public MathError {
    public:
        /**
         * @brief Constructs a NumericOverflowError with the default message.
         */
        NumericOverflowError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a NumericOverflowError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        NumericOverflowError(const char* msg);
    };

// ================================================================================
    /**
     * @brief Mathematical domain error.
     * 
     * DomainError represents an error when an input value is outside the
     * valid domain for a mathematical function (e.g., negative input to
     * square root, log of negative number).
     * 
     * @par Example Usage:
     * @code
     * throw DomainError();  // "Math domain error"
     * throw DomainError("Cannot compute sqrt of negative number: -4");
     * @endcode
     */
    class DomainError : public MathError {
    public:
        /**
         * @brief Constructs a DomainError with the default message.
         */
        DomainError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a DomainError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        DomainError(const char* msg);
    };

// ================================================================================
    /**
     * @brief Loss of numeric precision error.
     * 
     * LossOfPrecisionError represents an error when a computation results
     * in excessive loss of precision or numerical instability.
     * 
     * @par Example Usage:
     * @code
     * throw LossOfPrecisionError();  // "Loss of numeric precision"
     * throw LossOfPrecisionError("Ill-conditioned matrix: condition number > 1e15");
     * @endcode
     */
    class LossOfPrecisionError : public MathError {
    public:
        /**
         * @brief Constructs a LossOfPrecisionError with the default message.
         */
        LossOfPrecisionError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a LossOfPrecisionError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        LossOfPrecisionError(const char* msg);
    };
// ================================================================================ 
// ================================================================================ 

    /**
     * @brief File or handle open failure error.
     * 
     * FileOpenError represents an error when attempting to open a file
     * or handle fails.
     * 
     * @par Example Usage:
     * @code
     * throw FileOpenError();  // "Failed to open file/handle"
     * throw FileOpenError("Cannot open config.txt: file does not exist");
     * @endcode
     */
    class FileOpenError : public IOError {
    public:
        /**
         * @brief Constructs a FileOpenError with the default message.
         */
        FileOpenError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a FileOpenError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        FileOpenError(const char* msg);
    };

// ================================================================================
    /**
     * @brief File or handle read error.
     * 
     * FileReadError represents an error when reading from a file or
     * handle fails.
     * 
     * @par Example Usage:
     * @code
     * throw FileReadError();  // "Error reading from file/handle"
     * throw FileReadError("Read operation failed after 512 bytes");
     * @endcode
     */
    class FileReadError : public IOError {
    public:
        /**
         * @brief Constructs a FileReadError with the default message.
         */
        FileReadError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a FileReadError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        FileReadError(const char* msg);
    };

// ================================================================================
    /**
     * @brief File or handle write error.
     * 
     * FileWriteError represents an error when writing to a file or
     * handle fails.
     * 
     * @par Example Usage:
     * @code
     * throw FileWriteError();  // "Error writing to file/handle"
     * throw FileWriteError("Disk full: cannot write data");
     * @endcode
     */
    class FileWriteError : public IOError {
    public:
        /**
         * @brief Constructs a FileWriteError with the default message.
         */
        FileWriteError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a FileWriteError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        FileWriteError(const char* msg);
    };

// ================================================================================
    /**
     * @brief Permission denied error.
     * 
     * PermissionDeniedError represents an error when access control or
     * permission restrictions prevent an operation.
     * 
     * @par Example Usage:
     * @code
     * throw PermissionDeniedError();  // "Permission denied"
     * throw PermissionDeniedError("No write access to /etc/config");
     * @endcode
     */
    class PermissionDeniedError : public IOError {
    public:
        /**
         * @brief Constructs a PermissionDeniedError with the default message.
         */
        PermissionDeniedError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a PermissionDeniedError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        PermissionDeniedError(const char* msg);
    };

// ================================================================================
    /**
     * @brief I/O operation interrupted error.
     * 
     * IOInterruptedError represents an error when an I/O operation is
     * interrupted (e.g., by a signal like EINTR).
     * 
     * @par Example Usage:
     * @code
     * throw IOInterruptedError();  // "I/O interrupted"
     * throw IOInterruptedError("Read interrupted by signal");
     * @endcode
     */
    class IOInterruptedError : public IOError {
    public:
        /**
         * @brief Constructs an IOInterruptedError with the default message.
         */
        IOInterruptedError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs an IOInterruptedError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        IOInterruptedError(const char* msg);
    };

// ================================================================================
    /**
     * @brief I/O operation timeout error.
     * 
     * IOTimeoutError represents an error when an I/O operation exceeds
     * its timeout period.
     * 
     * @par Example Usage:
     * @code
     * throw IOTimeoutError();  // "I/O timed out"
     * throw IOTimeoutError("Network read timed out after 30 seconds");
     * @endcode
     */
    class IOTimeoutError : public IOError {
    public:
        /**
         * @brief Constructs an IOTimeoutError with the default message.
         */
        IOTimeoutError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs an IOTimeoutError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        IOTimeoutError(const char* msg);
    };

// ================================================================================
    /**
     * @brief Operation on closed stream or descriptor error.
     * 
     * IOClosedError represents an error when attempting an operation on
     * a closed file descriptor or stream.
     * 
     * @par Example Usage:
     * @code
     * throw IOClosedError();  // "Operation on closed stream/descriptor"
     * throw IOClosedError("Cannot read from closed socket");
     * @endcode
     */
    class IOClosedError : public IOError {
    public:
        /**
         * @brief Constructs an IOClosedError with the default message.
         */
        IOClosedError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs an IOClosedError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        IOClosedError(const char* msg);
    };

// ================================================================================
    /**
     * @brief Non-blocking operation would block error.
     * 
     * IOWouldBlockError represents an error when a non-blocking I/O
     * operation would block (EWOULDBLOCK/EAGAIN).
     * 
     * @par Example Usage:
     * @code
     * throw IOWouldBlockError();  // "Operation would block"
     * throw IOWouldBlockError("Socket read would block in non-blocking mode");
     * @endcode
     */
    class IOWouldBlockError : public IOError {
    public:
        /**
         * @brief Constructs an IOWouldBlockError with the default message.
         */
        IOWouldBlockError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs an IOWouldBlockError with a custom message.
         * 
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        IOWouldBlockError(const char* msg);
    };
// ================================================================================ 
// ================================================================================ 

    /**
     * @brief Type mismatch error.
     *
     * TypeMismatchError represents an error where a value or object does not
     * match the expected type.
     *
     * @par Example Usage:
     * @code
     * throw TypeMismatchError();  
     * throw TypeMismatchError("Expected integer but received string");
     * @endcode
     */
    class TypeMismatchError : public FormatError {
    public:
        /**
         * @brief Constructs a TypeMismatchError with the default message.
         */
        TypeMismatchError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a TypeMismatchError with a custom message.
         *
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        TypeMismatchError(const char* msg);
    };
// ================================================================================

    /**
     * @brief Invalid format error.
     *
     * FormatInvalidError represents an error where data does not conform
     * to the expected format or structure.
     *
     * @par Example Usage:
     * @code
     * throw FormatInvalidError();
     * throw FormatInvalidError("Malformed header detected");
     * @endcode
     */
    class FormatInvalidError : public FormatError {
    public:
        /**
         * @brief Constructs a FormatInvalidError with the default message.
         */
        FormatInvalidError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a FormatInvalidError with a custom message.
         *
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        FormatInvalidError(const char* msg);
    };
// ================================================================================

    /**
     * @brief Invalid encoding error.
     *
     * EncodingInvalidError represents an error where text or binary data
     * is not encoded using a supported or expected encoding.
     *
     * @par Example Usage:
     * @code
     * throw EncodingInvalidError();
     * throw EncodingInvalidError("UTF-8 decoding failed");
     * @endcode
     */
    class EncodingInvalidError : public FormatError {
    public:
        /**
         * @brief Constructs an EncodingInvalidError with the default message.
         */
        EncodingInvalidError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs an EncodingInvalidError with a custom message.
         *
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        EncodingInvalidError(const char* msg);
    };
// ================================================================================

    /**
     * @brief Parsing failure error.
     *
     * ParsingFailedError represents an error that occurs when structured
     * input cannot be successfully parsed.
     *
     * @par Example Usage:
     * @code
     * throw ParsingFailedError();
     * throw ParsingFailedError("JSON parsing failed at line 12");
     * @endcode
     */
    class ParsingFailedError : public FormatError {
    public:
        /**
         * @brief Constructs a ParsingFailedError with the default message.
         */
        ParsingFailedError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a ParsingFailedError with a custom message.
         *
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        ParsingFailedError(const char* msg);
    };
// ================================================================================

    /**
     * @brief Validation failure error.
     *
     * ValidationFailedError represents an error where data is syntactically
     * correct but fails semantic or rule-based validation.
     *
     * @par Example Usage:
     * @code
     * throw ValidationFailedError();
     * throw ValidationFailedError("Checksum validation failed");
     * @endcode
     */
    class ValidationFailedError : public FormatError {
    public:
        /**
         * @brief Constructs a ValidationFailedError with the default message.
         */
        ValidationFailedError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a ValidationFailedError with a custom message.
         *
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        ValidationFailedError(const char* msg);
    };
// ================================================================================ 
// ================================================================================ 

    /**
     * @brief Lock operation failure error.
     *
     * LockFailedError represents an error where a mutex, spinlock,
     * or other synchronization primitive fails to acquire or release.
     *
     * @par Example Usage:
     * @code
     * throw LockFailedError();
     * throw LockFailedError("Failed to acquire mutex");
     * @endcode
     */
    class LockFailedError : public ConcurrencyError {
    public:
        /**
         * @brief Constructs a LockFailedError with the default message.
         */
        LockFailedError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a LockFailedError with a custom message.
         *
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        LockFailedError(const char* msg);
    };
// ================================================================================

    /**
     * @brief Deadlock detection error.
     *
     * DeadlockDetectedError represents an error where a deadlock
     * condition has been detected between concurrent execution contexts.
     *
     * @par Example Usage:
     * @code
     * throw DeadlockDetectedError();
     * throw DeadlockDetectedError("Deadlock detected between worker threads");
     * @endcode
     */
    class DeadlockDetectedError : public ConcurrencyError {
    public:
        /**
         * @brief Constructs a DeadlockDetectedError with the default message.
         */
        DeadlockDetectedError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a DeadlockDetectedError with a custom message.
         *
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        DeadlockDetectedError(const char* msg);
    };
// ================================================================================

    /**
     * @brief Thread operation failure error.
     *
     * ThreadFailError represents an error where a thread could not be
     * created, joined, or otherwise managed correctly.
     *
     * @par Example Usage:
     * @code
     * throw ThreadFailError();
     * throw ThreadFailError("Thread creation failed");
     * @endcode
     */
    class ThreadFailError : public ConcurrencyError {
    public:
        /**
         * @brief Constructs a ThreadFailError with the default message.
         */
        ThreadFailError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a ThreadFailError with a custom message.
         *
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        ThreadFailError(const char* msg);
    };
// ================================================================================

    /**
     * @brief Operation cancelled error.
     *
     * CancelledError represents an error where an operation was
     * intentionally cancelled before completion.
     *
     * @par Example Usage:
     * @code
     * throw CancelledError();
     * throw CancelledError("Operation cancelled by user request");
     * @endcode
     */
    class CancelledError : public ConcurrencyError {
    public:
        /**
         * @brief Constructs a CancelledError with the default message.
         */
        CancelledError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a CancelledError with a custom message.
         *
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        CancelledError(const char* msg);
    };
// ================================================================================

    /**
     * @brief Data race detection error.
     *
     * RaceDetectedError represents an error where a data race
     * has been detected during concurrent execution.
     *
     * @par Example Usage:
     * @code
     * throw RaceDetectedError();
     * throw RaceDetectedError("Concurrent write detected on shared buffer");
     * @endcode
     */
    class RaceDetectedError : public ConcurrencyError {
    public:
        /**
         * @brief Constructs a RaceDetectedError with the default message.
         */
        RaceDetectedError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a RaceDetectedError with a custom message.
         *
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        RaceDetectedError(const char* msg);
    };
// ================================================================================ 
// ================================================================================ 

    /**
     * @brief Invalid configuration error.
     *
     * ConfigInvalidError represents an error where a configuration
     * is malformed, incomplete, or otherwise invalid.
     *
     * @par Example Usage:
     * @code
     * throw ConfigInvalidError();
     * throw ConfigInvalidError("Missing required configuration key");
     * @endcode
     */
    class ConfigInvalidError : public ConfigError {
    public:
        /**
         * @brief Constructs a ConfigInvalidError with the default message.
         */
        ConfigInvalidError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a ConfigInvalidError with a custom message.
         *
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        ConfigInvalidError(const char* msg);
    };
// ================================================================================

    /**
     * @brief Unsupported feature or platform error.
     *
     * UnsupportedError represents an error where a requested feature
     * or platform is not supported by the current build or environment.
     *
     * @par Example Usage:
     * @code
     * throw UnsupportedError();
     * throw UnsupportedError("ARM platform not supported");
     * @endcode
     */
    class UnsupportedError : public ConfigError {
    public:
        /**
         * @brief Constructs an UnsupportedError with the default message.
         */
        UnsupportedError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs an UnsupportedError with a custom message.
         *
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        UnsupportedError(const char* msg);
    };
// ================================================================================

    /**
     * @brief Feature disabled error.
     *
     * FeatureDisabledError represents an error where a feature is
     * explicitly disabled by build configuration or policy.
     *
     * @par Example Usage:
     * @code
     * throw FeatureDisabledError();
     * throw FeatureDisabledError("Feature disabled by security policy");
     * @endcode
     */
    class FeatureDisabledError : public ConfigError {
    public:
        /**
         * @brief Constructs a FeatureDisabledError with the default message.
         */
        FeatureDisabledError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a FeatureDisabledError with a custom message.
         *
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        FeatureDisabledError(const char* msg);
    };
// ================================================================================

    /**
     * @brief Version or ABI mismatch error.
     *
     * VersionMismatchError represents an error where incompatible
     * versions or ABIs are detected between components.
     *
     * @par Example Usage:
     * @code
     * throw VersionMismatchError();
     * throw VersionMismatchError("Library ABI version mismatch");
     * @endcode
     */
    class VersionMismatchError : public ConfigError {
    public:
        /**
         * @brief Constructs a VersionMismatchError with the default message.
         */
        VersionMismatchError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a VersionMismatchError with a custom message.
         *
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        VersionMismatchError(const char* msg);
    };
// ================================================================================

    /**
     * @brief Resource exhaustion error.
     *
     * ResourceExhaustedError represents an error where required
     * system or application resources have been exhausted.
     *
     * @par Example Usage:
     * @code
     * throw ResourceExhaustedError();
     * throw ResourceExhaustedError("Out of file descriptors");
     * @endcode
     */
    class ResourceExhaustedError : public ConfigError {
    public:
        /**
         * @brief Constructs a ResourceExhaustedError with the default message.
         */
        ResourceExhaustedError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a ResourceExhaustedError with a custom message.
         *
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        ResourceExhaustedError(const char* msg);
    };
// ================================================================================ 
// ================================================================================ 

    /**
     * @brief Not implemented error.
     *
     * NotImplementedError represents an error where a requested
     * operation or feature has not yet been implemented.
     *
     * @par Example Usage:
     * @code
     * throw NotImplementedError();
     * throw NotImplementedError("Serialization not yet implemented");
     * @endcode
     */
    class NotImplementedError : public GenericError {
    public:
        /**
         * @brief Constructs a NotImplementedError with the default message.
         */
        NotImplementedError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs a NotImplementedError with a custom message.
         *
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        NotImplementedError(const char* msg);
    };
// ================================================================================

    /**
     * @brief Operation unavailable error.
     *
     * OperationUnavailableError represents an error where an operation
     * cannot be performed due to current system or application state.
     *
     * @par Example Usage:
     * @code
     * throw OperationUnavailableError();
     * throw OperationUnavailableError("Service unavailable during shutdown");
     * @endcode
     */
    class OperationUnavailableError : public GenericError {
    public:
        /**
         * @brief Constructs an OperationUnavailableError with the default message.
         */
        OperationUnavailableError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs an OperationUnavailableError with a custom message.
         *
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        OperationUnavailableError(const char* msg);
    };
// ================================================================================

    /**
     * @brief Unknown error.
     *
     * UnknownError represents an error where the underlying cause
     * is unknown or could not be classified.
     *
     * @par Example Usage:
     * @code
     * throw UnknownError();
     * throw UnknownError("Unexpected failure occurred");
     * @endcode
     */
    class UnknownError : public GenericError {
    public:
        /**
         * @brief Constructs an UnknownError with the default message.
         */
        UnknownError();
// --------------------------------------------------------------------------------
        /**
         * @brief Constructs an UnknownError with a custom message.
         *
         * @param msg Custom null-terminated error message (max 255 characters)
         */
        UnknownError(const char* msg);
    };
// ================================================================================ 
// ================================================================================ 

    /**
     * @brief A type that represents either a value of type T or an Error.
     * 
     * Expected<T> stores both the value and error in a fixed-size structure.
     * No dynamic allocation is used - all storage is static.
     * 
     * @tparam T The type of the expected value
     * 
     * @par Example Usage:
     * @code
     * Expected<int> divide(int a, int b) {
     *     Expected<int> result;
     *     if (b == 0) {
     *         result.setError(DivByZeroError());
     *         return result;
     *     }
     *     result.setValue(a / b);
     *     return result;
     * }
     * 
     * Expected<int> result = divide(10, 2);
     * if (result.hasValue()) {
     *     printf("Result: %d\n", result.value());
     * } else {
     *     printf("Error: %s\n", result.error().what());
     * }
     * @endcode
     */
    template<typename T>
    class Expected {
    private:
        bool has_value_;
        T value_;
        Error error_;

    public:
        /**
         * @brief Default constructor - initializes with no value and generic error.
         */
        Expected() : has_value_(false), value_(), error_("No value") {}
        
        /**
         * @brief Sets the value and marks this as containing a value.
         * 
         * @param val The value to store
         */
        void setValue(const T& val) {
            value_ = val;
            has_value_ = true;
        }
        
        /**
         * @brief Sets the error and marks this as containing an error.
         * 
         * @param err The error to store
         */
        void setError(const Error& err) {
            error_ = err;
            has_value_ = false;
        }
        
        /**
         * @brief Checks if the Expected contains a value.
         * 
         * @return true if contains a value, false if contains an error
         */
        bool hasValue() const {
            return has_value_;
        }
        
        /**
         * @brief Checks if the Expected contains an error.
         * 
         * @return true if contains an error, false if contains a value
         */
        bool hasError() const {
            return !has_value_;
        }
        
        /**
         * @brief Gets the contained value.
         * 
         * Behavior is undefined if the Expected contains an error.
         * Use hasValue() to check before calling this method.
         * 
         * @return Reference to the contained value
         */
        T& value() {
            return value_;
        }
        
        /**
         * @brief Gets the contained value (const version).
         * 
         * Behavior is undefined if the Expected contains an error.
         * Use hasValue() to check before calling this method.
         * 
         * @return Const reference to the contained value
         */
        const T& value() const {
            return value_;
        }
        
        /**
         * @brief Gets the contained error.
         * 
         * Behavior is undefined if the Expected contains a value.
         * Use hasError() to check before calling this method.
         * 
         * @return Reference to the contained error
         */
        Error& error() {
            return error_;
        }
        
        /**
         * @brief Gets the contained error (const version).
         * 
         * Behavior is undefined if the Expected contains a value.
         * Use hasError() to check before calling this method.
         * 
         * @return Const reference to the contained error
         */
        const Error& error() const {
            return error_;
        }
        
        /**
         * @brief Gets the value or a default if error.
         * 
         * @param default_val The default value to return if this contains an error
         * @return The contained value if present, otherwise default_val
         */
        T valueOr(const T& default_val) const {
            return has_value_ ? value_ : default_val;
        }
        
        /**
         * @brief Conversion to bool (checks if has value).
         * 
         * @return true if contains a value, false if contains an error
         */
        explicit operator bool() const {
            return has_value_;
        }
    };
// ================================================================================
// ================================================================================ 
} // namespace cslt
// ================================================================================ 
// ================================================================================ 
#endif /* error_HPP */
// ================================================================================
// ================================================================================
// eof
