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
// -------------------------------------------------------------------------------- 

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
// -------------------------------------------------------------------------------- 

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
// -------------------------------------------------------------------------------- 

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
// -------------------------------------------------------------------------------- 

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
// -------------------------------------------------------------------------------- 

        /**
         * @brief Concatenate a C-string to the end of this string
         * 
         * @param str Null-terminated C-string to append
         * @return true if concatenation succeeded, false on failure
         * 
         * @details Appends the contents of str to the end of the current string.
         * The function automatically handles buffer growth when necessary, using
         * the allocator's realloc() method if available, or falling back to
         * allocate-copy-free pattern.
         * 
         * Key behaviors:
         * - Always maintains null termination
         * - Detects and handles self-aliasing (when str points within current buffer)
         * - Prevents size_t overflow in length calculations
         * - Returns immediately if str is empty (no-op, returns true)
         * - Grows buffer capacity as needed
         * 
         * Buffer growth strategy:
         * - If allocator supports realloc(), uses in-place reallocation
         * - Otherwise, allocates new buffer, copies data, and frees old buffer
         * - Pre-allocating capacity via init() can reduce reallocations
         * 
         * @par Self-aliasing:
         * The function safely handles cases where str points to a substring of
         * the current buffer. It detects this condition and creates a temporary
         * copy before any reallocation occurs.
         * 
         * @par Error conditions:
         * Returns false if:
         * - str is nullptr
         * - Internal buffer is null (corrupted String)
         * - Allocator is null
         * - Length overflow would occur (len + strlen(str) + 1 > SIZE_MAX)
         * - Memory allocation fails during buffer growth
         * 
         * @par Thread safety:
         * Not thread-safe. External synchronization required for concurrent access.
         * 
         * @code{.cpp}
         * cslt::HeapAllocator allocator;
         * auto r = cslt::String::init("Hello", 0, allocator);
         * 
         * if (r.hasValue()) {
         *     cslt::UniquePtr<cslt::String, cslt::StringDeleter> str(r.value());
         *     
         *     // Basic concatenation
         *     bool success = str->concat(" world");
         *     if (success) {
         *         std::cout << str->c_str() << std::endl;  // "Hello world"
         *         std::cout << str->size() << std::endl;    // 11
         *     }
         *     
         *     // Multiple concatenations
         *     str->concat("!");
         *     str->concat(" How are you?");
         *     std::cout << str->c_str() << std::endl;  // "Hello world! How are you?"
         *     
         *     // Self-aliasing example (safe)
         *     str->concat(str->c_str());  // Doubles the string
         *     
         *     // Empty string (no-op)
         *     str->concat("");  // Returns true, no change
         * }
         * @endcode
         * 
         * @code{.cpp}
         * // Pre-allocate capacity to minimize reallocations
         * cslt::HeapAllocator allocator;
         * auto r = cslt::String::init("", 100, allocator);  // 100 bytes capacity
         * 
         * if (r.hasValue()) {
         *     cslt::UniquePtr<cslt::String, cslt::StringDeleter> path(r.value());
         *     
         *     // Multiple appends without reallocation
         *     path->concat("/usr");
         *     path->concat("/local");
         *     path->concat("/bin");
         *     
         *     std::cout << path->c_str() << std::endl;  // "/usr/local/bin"
         *     std::cout << "Size: " << path->size() << std::endl;
         *     std::cout << "Capacity: " << path->capacity() << std::endl;  // Still 101
         * }
         * @endcode
         * 
         * @code{.cpp}
         * // Error handling
         * cslt::HeapAllocator allocator;
         * auto r = cslt::String::init("test", 0, allocator);
         * 
         * if (r.hasValue()) {
         *     cslt::UniquePtr<cslt::String, cslt::StringDeleter> str(r.value());
         *     
         *     // Null pointer - returns false
         *     if (!str->concat(nullptr)) {
         *         std::cerr << "Failed to concat null pointer" << std::endl;
         *     }
         *     
         *     // String unchanged after failure
         *     std::cout << str->c_str() << std::endl;  // Still "test"
         * }
         * @endcode
         * 
         * @see concat(const String&) for String-to-String concatenation
         * @see init() for capacity pre-allocation
         */
        bool concat(const char* str) noexcept;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Concatenate another String object to the end of this string
         * 
         * @param str String object to append
         * @return true if concatenation succeeded, false on failure
         * 
         * @details Convenience overload that appends the contents of another
         * String object. This method delegates to concat(const char*) internally,
         * extracting the C-string from the provided String.
         * 
         * The source string (str) is not modified and remains independent.
         * Only the contents are copied, not ownership or allocator references.
         * 
         * @par Error conditions:
         * Returns false if:
         * - str's internal buffer is null (corrupted String)
         * - Any condition that would cause concat(const char*) to fail
         * 
         * @par Thread safety:
         * Not thread-safe. External synchronization required for concurrent access
         * to either string.
         * 
         * @code{.cpp}
         * cslt::HeapAllocator allocator;
         * 
         * auto r1 = cslt::String::init("Hello", 0, allocator);
         * auto r2 = cslt::String::init(" world", 0, allocator);
         * 
         * if (r1.hasValue() && r2.hasValue()) {
         *     cslt::UniquePtr<cslt::String, cslt::StringDeleter> str1(r1.value());
         *     cslt::UniquePtr<cslt::String, cslt::StringDeleter> str2(r2.value());
         *     
         *     // Concatenate String objects
         *     bool success = str1->concat(*str2);
         *     if (success) {
         *         std::cout << str1->c_str() << std::endl;  // "Hello world"
         *         std::cout << str2->c_str() << std::endl;  // " world" (unchanged)
         *     }
         * }
         * @endcode
         * 
         * @code{.cpp}
         * // Building a sentence from words
         * cslt::HeapAllocator allocator;
         * 
         * auto r1 = cslt::String::init("The", 50, allocator);
         * auto r2 = cslt::String::init(" quick", 0, allocator);
         * auto r3 = cslt::String::init(" brown", 0, allocator);
         * auto r4 = cslt::String::init(" fox", 0, allocator);
         * 
         * if (r1.hasValue() && r2.hasValue() && r3.hasValue() && r4.hasValue()) {
         *     cslt::UniquePtr<cslt::String, cslt::StringDeleter> sentence(r1.value());
         *     cslt::UniquePtr<cslt::String, cslt::StringDeleter> word2(r2.value());
         *     cslt::UniquePtr<cslt::String, cslt::StringDeleter> word3(r3.value());
         *     cslt::UniquePtr<cslt::String, cslt::StringDeleter> word4(r4.value());
         *     
         *     sentence->concat(*word2);
         *     sentence->concat(*word3);
         *     sentence->concat(*word4);
         *     
         *     std::cout << sentence->c_str() << std::endl;  // "The quick brown fox"
         *     std::cout << sentence->size() << std::endl;    // 19
         * }
         * @endcode
         * 
         * @code{.cpp}
         * // Mixing C-string and String concatenation
         * cslt::HeapAllocator allocator;
         * 
         * auto r1 = cslt::String::init("Hello", 0, allocator);
         * auto r2 = cslt::String::init("world", 0, allocator);
         * 
         * if (r1.hasValue() && r2.hasValue()) {
         *     cslt::UniquePtr<cslt::String, cslt::StringDeleter> str1(r1.value());
         *     cslt::UniquePtr<cslt::String, cslt::StringDeleter> str2(r2.value());
         *     
         *     str1->concat(" ");        // C-string
         *     str1->concat(*str2);      // String object
         *     str1->concat("!");        // C-string
         *     
         *     std::cout << str1->c_str() << std::endl;  // "Hello world!"
         * }
         * @endcode
         * 
         * @code{.cpp}
         * // Using different allocators (allowed but not common)
         * cslt::HeapAllocator heap_alloc;
         * cslt::ArenaAllocator arena_alloc(1024);
         * 
         * auto r1 = cslt::String::init("heap: ", 0, heap_alloc);
         * auto r2 = cslt::String::init("arena", 0, arena_alloc);
         * 
         * if (r1.hasValue() && r2.hasValue()) {
         *     cslt::UniquePtr<cslt::String, cslt::StringDeleter> str1(r1.value());
         *     cslt::UniquePtr<cslt::String, cslt::StringDeleter> str2(r2.value());
         *     
         *     // Works fine - only copies content, not allocator
         *     str1->concat(*str2);
         *     std::cout << str1->c_str() << std::endl;  // "heap: arena"
         *     
         *     // str1 still uses heap_alloc for any growth
         *     // str2 still uses arena_alloc
         * }
         * @endcode
         * 
         * @see concat(const char*) for C-string concatenation
         * @see c_str() to extract the C-string from a String object
         */
        bool concat(const String& str) noexcept;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Lexicographically compare this string with a C-string
         * 
         * @param str Null-terminated C-string to compare against
         * @return int8_t comparison result:
         *         - -1 if this string is less than str
         *         - 0 if strings are equal
         *         - 1 if this string is greater than str
         *         - -128 if either string is null (error sentinel)
         * 
         * @details Performs lexicographic comparison using unsigned byte values.
         * The comparison follows standard strcmp semantics but returns a compact
         * int8_t result with three possible values plus an error sentinel.
         * 
         * Comparison semantics:
         * - Characters are compared as unsigned bytes (0-255)
         * - Comparison proceeds left-to-right until a difference is found
         * - If one string is a prefix of the other, the shorter is considered less
         * - Null terminators are considered for the C-string parameter
         * - The String's logical length (len_) determines comparison extent
         * 
         * Return value interpretation:
         * - -1: This string sorts before str (this < str)
         * - 0: Strings are identical in content and length
         * - 1: This string sorts after str (this > str)
         * - -128: Error condition (null pointer detected)
         * 
         * @par Edge cases:
         * - Empty strings: Compare as equal if both are empty
         * - Embedded nulls in String: Comparison stops at logical length, not null
         * - C-string shorter than String: C-string is considered less
         * - C-string longer than String: String is considered less
         * 
         * @par Error conditions:
         * Returns -128 if:
         * - str parameter is nullptr
         * - Internal buffer (str_) is null (corrupted String)
         * 
         * @par Performance:
         * - O(n) where n = min(this->size(), strlen(str))
         * - Early termination on first difference
         * - Single pass, no allocation
         * 
         * @par Thread safety:
         * Thread-safe for concurrent reads if str is not being modified.
         * Not thread-safe if this String is being modified concurrently.
         * 
         * @code{.cpp}
         * cslt::HeapAllocator allocator;
         * auto r = cslt::String::init("hello", 0, allocator);
         * 
         * if (r.hasValue()) {
         *     cslt::UniquePtr<cslt::String, cslt::StringDeleter> str(r.value());
         *     
         *     // Basic comparisons
         *     int8_t cmp1 = str->compare("hello");
         *     std::cout << "Compare 'hello' vs 'hello': " << (int)cmp1 << std::endl;  // 0
         *     
         *     int8_t cmp2 = str->compare("world");
         *     std::cout << "Compare 'hello' vs 'world': " << (int)cmp2 << std::endl;  // -1
         *     
         *     int8_t cmp3 = str->compare("apple");
         *     std::cout << "Compare 'hello' vs 'apple': " << (int)cmp3 << std::endl;  // 1
         * }
         * @endcode
         * 
         * @code{.cpp}
         * // Using for sorting
         * cslt::HeapAllocator allocator;
         * 
         * std::vector<cslt::UniquePtr<cslt::String, cslt::StringDeleter>> strings;
         * 
         * auto r1 = cslt::String::init("zebra", 0, allocator);
         * auto r2 = cslt::String::init("apple", 0, allocator);
         * auto r3 = cslt::String::init("mango", 0, allocator);
         * 
         * if (r1.hasValue() && r2.hasValue() && r3.hasValue()) {
         *     strings.push_back(cslt::UniquePtr<cslt::String, cslt::StringDeleter>(r1.value()));
         *     strings.push_back(cslt::UniquePtr<cslt::String, cslt::StringDeleter>(r2.value()));
         *     strings.push_back(cslt::UniquePtr<cslt::String, cslt::StringDeleter>(r3.value()));
         *     
         *     // Sort using compare
         *     std::sort(strings.begin(), strings.end(),
         *         [](const auto& a, const auto& b) {
         *             return a->compare(b->c_str()) < 0;
         *         });
         *     
         *     for (const auto& s : strings) {
         *         std::cout << s->c_str() << std::endl;
         *     }
         *     // Output: apple, mango, zebra
         * }
         * @endcode
         * 
         * @see compare(const String&) for String-to-String comparison
         */
        int8_t compare(const char* str) const noexcept;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Lexicographically compare this string with another String object
         * 
         * @param other String object to compare against
         * @return int8_t comparison result:
         *         - -1 if this string is less than other
         *         - 0 if strings are equal
         *         - 1 if this string is greater than other
         *         - -128 if either string's buffer is null (error sentinel)
         * 
         * @details Performs lexicographic comparison of two String objects.
         * This is the preferred method for comparing String instances as it
         * uses the stored length information for optimization.
         * 
         * Comparison algorithm:
         * - Compares min(this->len_, other.len_) characters byte-by-byte
         * - Uses unsigned byte comparison (0-255)
         * - If all common characters match, compares lengths
         * - Early termination on first difference
         * 
         * Return value interpretation:
         * - -1: This string sorts before other (this < other)
         * - 0: Strings are identical in content and length
         * - 1: This string sorts after other (this > other)
         * - -128: Error condition (null internal buffer)
         * 
         * @par Advantages over C-string comparison:
         * - No need to scan for null terminator in other
         * - Handles embedded nulls correctly (compares up to logical length)
         * - More efficient for long strings with early differences
         * - Type-safe (no risk of comparing with invalid C-string)
         * 
         * @par Performance:
         * - O(n) where n = min(this->size(), other.size())
         * - No strlen() calls needed
         * - Early termination on first difference
         * - Can be SIMD-optimized (see implementation notes)
         * 
         * @par Error conditions:
         * Returns -128 if:
         * - This String's internal buffer (str_) is null
         * - Other String's internal buffer is null
         * 
         * @par Thread safety:
         * Thread-safe for concurrent reads if neither string is being modified.
         * Not thread-safe if either String is being modified concurrently.
         * 
         * @code{.cpp}
         * cslt::HeapAllocator allocator;
         * 
         * auto r1 = cslt::String::init("apple", 0, allocator);
         * auto r2 = cslt::String::init("banana", 0, allocator);
         * 
         * if (r1.hasValue() && r2.hasValue()) {
         *     cslt::UniquePtr<cslt::String, cslt::StringDeleter> str1(r1.value());
         *     cslt::UniquePtr<cslt::String, cslt::StringDeleter> str2(r2.value());
         *     
         *     int8_t cmp = str1->compare(*str2);
         *     
         *     if (cmp < 0) {
         *         std::cout << "'" << str1->c_str() << "' comes before '" 
         *                   << str2->c_str() << "'" << std::endl;
         *         // Output: 'apple' comes before 'banana'
         *     }
         * }
         * @endcode
         * 
         * @code{.cpp}
         * // Equality testing
         * cslt::HeapAllocator allocator;
         * 
         * auto r1 = cslt::String::init("test", 0, allocator);
         * auto r2 = cslt::String::init("test", 0, allocator);
         * auto r3 = cslt::String::init("Test", 0, allocator);
         * 
         * if (r1.hasValue() && r2.hasValue() && r3.hasValue()) {
         *     cslt::UniquePtr<cslt::String, cslt::StringDeleter> str1(r1.value());
         *     cslt::UniquePtr<cslt::String, cslt::StringDeleter> str2(r2.value());
         *     cslt::UniquePtr<cslt::String, cslt::StringDeleter> str3(r3.value());
         *     
         *     bool equal1 = (str1->compare(*str2) == 0);
         *     std::cout << "str1 == str2: " << equal1 << std::endl;  // true
         *     
         *     bool equal2 = (str1->compare(*str3) == 0);
         *     std::cout << "str1 == str3: " << equal2 << std::endl;  // false
         * }
         * @endcode
         * 
         * @note For SIMD-optimized implementation, see string_compare_impl.cpp
         * @see compare(const char*) for C-string comparison
         */
        int8_t compare(const String& other) const noexcept;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Reset the string to an empty state
         * 
         * @details Sets the string length to zero and null-terminates at position 0.
         * The allocated buffer is preserved and can be reused for future operations.
         * 
         * Key behaviors:
         * - Sets len_ to 0
         * - Places '\0' at position 0
         * - Preserves allocated capacity
         * - O(1) constant time operation
         * 
         * @par Performance:
         * O(1) - No memory allocation or deallocation occurs.
         * 
         * @code{.cpp}
         * cslt::HeapAllocator allocator;
         * auto r = cslt::String::init("hello world", 0, allocator);
         * 
         * if (r.hasValue()) {
         *     cslt::UniquePtr<cslt::String, cslt::StringDeleter> str(r.value());
         *     
         *     std::cout << "Before: " << str->c_str() << std::endl;  // "hello world"
         *     std::cout << "Size: " << str->size() << std::endl;      // 11
         *     
         *     str->reset();
         *     
         *     std::cout << "After: " << str->c_str() << std::endl;   // ""
         *     std::cout << "Size: " << str->size() << std::endl;      // 0
         *     std::cout << "Capacity: " << str->capacity() << std::endl;  // Unchanged
         *     
         *     // Reuse the buffer
         *     str->concat("new content");
         *     std::cout << "Reused: " << str->c_str() << std::endl;  // "new content"
         * }
         * @endcode
         * 
         * @see ~String() to free the buffer
         * @see init() to create a String with specific capacity
         */
        void reset() noexcept;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Create a deep copy using this string's allocator
         * 
         * @return Expected<String*> containing pointer to new String or error
         * 
         * @details Creates an independent copy with the same content using the same
         * allocator as the original. The copy has its own buffer and can be modified
         * independently.
         * 
         * Key behaviors:
         * - Creates new String object and buffer
         * - Uses this->allocator_ for the copy
         * - Copy capacity matches original length
         * - O(n) where n is the string length
         * 
         * @code{.cpp}
         * cslt::HeapAllocator allocator;
         * auto r = cslt::String::init("hello", 0, allocator);
         * 
         * if (r.hasValue()) {
         *     cslt::UniquePtr<cslt::String, cslt::StringDeleter> original(r.value());
         *     
         *     // Create a copy
         *     auto copy_r = original->copy();
         *     
         *     if (copy_r.hasValue()) {
         *         cslt::UniquePtr<cslt::String, cslt::StringDeleter> copy(copy_r.value());
         *         
         *         // Modify copy - original unchanged
         *         copy->concat(" world");
         *         
         *         std::cout << "Original: " << original->c_str() << std::endl;  // "hello"
         *         std::cout << "Copy: " << copy->c_str() << std::endl;          // "hello world"
         *     }
         * }
         * @endcode
         * 
         * @see copy(Allocator&) to copy using a different allocator
         */
        Expected<String*> copy() const noexcept;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Create a deep copy using a specified allocator
         * 
         * @param allocator Allocator to use for the new String
         * @return Expected<String*> containing pointer to new String or error
         * 
         * @details Creates an independent copy with the same content using the
         * specified allocator. Useful for copying strings between different
         * allocator contexts.
         * 
         * Key behaviors:
         * - Creates new String object and buffer
         * - Uses provided allocator for the copy
         * - Copy capacity matches original length
         * - O(n) where n is the string length
         * 
         * @code{.cpp}
         * cslt::HeapAllocator heap_alloc;
         * cslt::ArenaAllocator arena_alloc(1024);
         * 
         * auto r = cslt::String::init("data", 0, heap_alloc);
         * 
         * if (r.hasValue()) {
         *     cslt::UniquePtr<cslt::String, cslt::StringDeleter> heap_str(r.value());
         *     
         *     // Copy from heap to arena allocator
         *     auto copy_r = heap_str->copy(arena_alloc);
         *     
         *     if (copy_r.hasValue()) {
         *         cslt::UniquePtr<cslt::String, cslt::StringDeleter> arena_str(copy_r.value());
         *         
         *         std::cout << "Both have same content: " 
         *                   << (strcmp(heap_str->c_str(), arena_str->c_str()) == 0) 
         *                   << std::endl;  // true
         *         std::cout << "Different allocators: " 
         *                   << (heap_str->allocator() != arena_str->allocator()) 
         *                   << std::endl;  // true
         *     }
         * }
         * @endcode
         * 
         * @see copy() to copy using the same allocator
         */
        Expected<String*> copy(Allocator& allocator) const noexcept;
// -------------------------------------------------------------------------------- 

        bool is_ptr(const void* ptr) const noexcept;
// -------------------------------------------------------------------------------- 

        bool is_ptr(const void* ptr, size_t bytes) const noexcept;
// -------------------------------------------------------------------------------- 

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
