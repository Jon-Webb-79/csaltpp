    // ================================================================================
    // ================================================================================
    // - File:    allocator.hpp
    // - Purpose: This file contains the prototypes for custom allocators as part of the 
    //            cslt namespace
    //
    // Source Metadata
    // - Author:  Jonathan A. Webb
    // - Date:    December 28, 2025
    // - Version: 1.0
    // - Copyright: Copyright 2022, Jon Webb Inc.
    // ================================================================================
    // ================================================================================
    // Include modules here

#ifndef string_HPP
#define string_HPP


#include "error.hpp"
#include "allocator.hpp"
#include "pointers.hpp"

#include <cstring>
#include <cstddef>
// ================================================================================ 
// ================================================================================ 

namespace cslt {

    /**
     * @class String
     * @brief Allocator-backed string container
     * 
     * @details Provides a string container that uses custom allocators for memory
     * management. The string and its internal buffer are both allocated through
     * the provided allocator.
     * 
     * Key features:
     * - Custom allocator support (heap, arena, buddy, slab, etc.)
     * - Null-terminated C-string compatibility
     * - Capacity management with optional pre-allocation
     * - Safe truncation when capacity is insufficient
     * - RAII-based memory management through init factory functions
     * 
     * Usage pattern:
     * - Must be initialized via static factory function init()
     * - Cannot be constructed directly (private constructor)
     * - Automatically manages both string object and buffer memory
     * - Cleaned up through custom deleter
     * 
     * @code{.cpp}
     * // Create with heap allocator
     * cslt::HeapAllocator allocator;
     * auto str_result = cslt::String::init("hello", 0, allocator);
     * 
     * if (str_result.hasValue()) {
     *     cslt::UniquePtr<cslt::String, cslt::StringDeleter> str(str_result.value());
     *     std::cout << str->c_str() << std::endl;  // prints "hello"
     *     std::cout << "length: " << str->size() << std::endl;
     *     
     *     // String is automatically cleaned up when UniquePtr goes out of scope
     * }
     * @endcode
     * 
     * @note This class cannot be instantiated directly. Use the init() factory
     *       function to create instances.
     */
    class String {
    private:
        char* str_;              ///< Internal null-terminated character buffer
        size_t len_;             ///< Logical length (excludes null terminator)
        size_t alloc_;           ///< Total allocated bytes (includes null terminator)
        Allocator* allocator_;   ///< Allocator used for memory management

        /**
         * @brief Private constructor - prevents direct instantiation
         * 
         * @param cstr Source C-string to copy
         * @param capacity_bytes Requested payload capacity (excludes null terminator)
         * @param allocator Allocator to use for memory management
         * 
         * @details This constructor is private to enforce the use of the init()
         *          factory function, which properly handles error cases and returns
         *          an Expected type.
         */
        String(const char* cstr, size_t capacity_bytes, Allocator& allocator);

        /**
         * @brief Private destructor - cleanup is handled by StringDeleter
         */
        ~String() noexcept;

        // Prevent copying and moving
        String(const String&) = delete;
        String& operator=(const String&) = delete;
        String(String&&) = delete;
        String& operator=(String&&) = delete;

    public:
        /**
         * @brief Initialize an allocator-backed string
         * 
         * @param cstr Null-terminated source C string
         * @param capacity_bytes Requested payload capacity in characters (excludes null terminator)
         * @param allocator Allocator to use for memory management
         * 
         * @return Expected<String*> containing pointer to String or error
         * 
         * @details Creates a new String instance using the provided allocator.
         * 
         * Capacity semantics:
         * - If capacity_bytes is 0, the allocation defaults to exactly the
         *   length of cstr plus space for the null terminator
         * - If capacity_bytes is non-zero, the container allocates
         *   (capacity_bytes + 1) bytes to guarantee space for the terminator
         * - If the requested capacity is smaller than the source string length,
         *   the stored string is truncated to fit and always null-terminated
         * 
         * Memory allocation:
         * - Both the String object and its internal buffer are allocated through
         *   the provided allocator
         * - The allocator reference is stored for cleanup
         * - Memory must later be released through the StringDeleter
         * 
         * @par Error conditions:
         * - ArgumentError if cstr is nullptr
         * - Propagates any allocation errors from the allocator (typically MemoryError)
         * 
         * @code{.cpp}
         * cslt::HeapAllocator allocator;
         * 
         * // Default capacity (fits exact string)
         * auto r1 = cslt::String::init("hello", 0, allocator);
         * if (r1.hasValue()) {
         *     cslt::UniquePtr<cslt::String, cslt::StringDeleter> s(r1.value());
         *     std::cout << s->c_str() << std::endl;  // "hello"
         *     std::cout << s->size() << std::endl;    // 5
         *     std::cout << s->capacity() << std::endl; // 6 (includes null)
         * }
         * 
         * // Pre-allocated capacity
         * auto r2 = cslt::String::init("hi", 100, allocator);
         * if (r2.hasValue()) {
         *     cslt::UniquePtr<cslt::String, cslt::StringDeleter> s(r2.value());
         *     std::cout << s->size() << std::endl;     // 2
         *     std::cout << s->capacity() << std::endl; // 101 (100 + null)
         * }
         * 
         * // Truncation case
         * auto r3 = cslt::String::init("hello world", 5, allocator);
         * if (r3.hasValue()) {
         *     cslt::UniquePtr<cslt::String, cslt::StringDeleter> s(r3.value());
         *     std::cout << s->c_str() << std::endl;  // "hello"
         *     std::cout << s->size() << std::endl;    // 5 (truncated)
         * }
         * @endcode
         * 
         * @see StringDeleter for cleanup semantics
         */
        static Expected<String*> init(const char* cstr, 
                                      size_t capacity_bytes, 
                                      Allocator& allocator) noexcept;

        /**
         * @brief Get the internal null-terminated C string
         * 
         * @return Pointer to null-terminated character buffer
         * 
         * @details Returns a pointer to the underlying character buffer.
         *          The pointer remains valid until the String is destroyed.
         * 
         * @code{.cpp}
         * auto r = cslt::String::init("example", 0, allocator);
         * if (r.hasValue()) {
         *     cslt::UniquePtr<cslt::String, cslt::StringDeleter> s(r.value());
         *     const char* text = s->c_str();
         *     std::cout << text << std::endl;
         * }
         * @endcode
         */
        const char* c_str() const noexcept { return str_; }

        /**
         * @brief Get the logical length of the string
         * 
         * @return Number of characters excluding the null terminator
         * 
         * @details Returns the number of characters stored in the container,
         *          not including the null terminator.
         * 
         * @code{.cpp}
         * auto r = cslt::String::init("hello", 0, allocator);
         * if (r.hasValue()) {
         *     cslt::UniquePtr<cslt::String, cslt::StringDeleter> s(r.value());
         *     std::cout << "length = " << s->size() << std::endl;  // 5
         * }
         * @endcode
         */
        size_t size() const noexcept { return len_; }

        /**
         * @brief Get the total allocated buffer size
         * 
         * @return Total allocated bytes including the null terminator
         * 
         * @details Returns the number of bytes allocated for the internal buffer,
         *          including space reserved for the null terminator.
         * 
         * @code{.cpp}
         * auto r = cslt::String::init("test", 100, allocator);
         * if (r.hasValue()) {
         *     cslt::UniquePtr<cslt::String, cslt::StringDeleter> s(r.value());
         *     std::cout << "capacity = " << s->capacity() << std::endl;  // 101
         * }
         * @endcode
         */
        size_t capacity() const noexcept { return alloc_; }

        /**
         * @brief Get the allocator used by this string
         * 
         * @return Pointer to the allocator
         * 
         * @details Returns the allocator that was used to create this string
         *          and will be used to free it.
         * 
         * @code{.cpp}
         * auto r = cslt::String::init("test", 0, allocator);
         * if (r.hasValue()) {
         *     cslt::UniquePtr<cslt::String, cslt::StringDeleter> s(r.value());
         *     Allocator* alloc = s->allocator();
         *     // Can query allocator properties...
         * }
         * @endcode
         */
        Allocator* allocator() const noexcept { return allocator_; }

        // StringDeleter needs access to private members for cleanup
        friend class StringDeleter;
    };
// ================================================================================
// ================================================================================

    /**
     * @class StringDeleter
     * @brief Custom deleter for String instances
     * 
     * @details Implements proper cleanup for String objects by:
     *          1. Freeing the internal character buffer
     *          2. Freeing the String object itself
     *          Both operations use the allocator stored in the String.
     * 
     * This deleter is used with UniquePtr to provide RAII semantics.
     * 
     * @code{.cpp}
     * cslt::HeapAllocator allocator;
     * auto r = cslt::String::init("example", 0, allocator);
     * 
     * if (r.hasValue()) {
     *     // String is automatically cleaned up when UniquePtr goes out of scope
     *     cslt::UniquePtr<cslt::String, cslt::StringDeleter> str(r.value());
     *     std::cout << str->c_str() << std::endl;
     * } // Deleter called here
     * @endcode
     */
    class StringDeleter {
    public:
        /**
         * @brief Delete a String instance
         * 
         * @param s String to delete (may be nullptr)
         * 
         * @details Frees the internal buffer and the String structure using
         *          the allocator stored within the String.
         * 
         * Safe to call with nullptr (no-op).
         */
        void operator()(String* s) const noexcept;
    };
// ================================================================================
// ================================================================================

} // namespace cslt
// ================================================================================ 
// ================================================================================ 
#endif /* STRING_HPP */
// ================================================================================
// ================================================================================
// eof
// ================================================================================
// ================================================================================
// eof
