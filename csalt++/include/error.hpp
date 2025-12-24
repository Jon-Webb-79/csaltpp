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
} // namespace cslt
// ================================================================================ 
// ================================================================================ 
#endif /* error_HPP */
// ================================================================================
// ================================================================================
// eof
