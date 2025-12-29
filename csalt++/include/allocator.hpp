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

#ifndef allocator_HPP
#define allocator_HPP

// Compile for static or dynamic memory allocation for MISRA compliance
#ifndef ARENA_ENABLE_DYNAMIC
#  ifdef STATIC_ONLY
#    define ARENA_ENABLE_DYNAMIC 0
#  else
#    define ARENA_ENABLE_DYNAMIC 1
#  endif
#endif

#if defined(STATIC_ONLY) && defined(ARENA_ENABLE_DYNAMIC) && (ARENA_ENABLE_DYNAMIC+0)!=0
#  error "STATIC_ONLY set but ARENA_ENABLE_DYNAMIC != 0"
#endif

#include "error.hpp"

#include <iostream>
#include <cstddef>
// ================================================================================ 
// ================================================================================ 

namespace cslt {

    enum MemType {
        ALLOC_INVALID = 0,
        STATIC = 1,
        DYNAMIC = 2
    };
// ================================================================================ 
// ================================================================================ 

    /**
     * @class Allocator
     * @brief Abstract base class for custom memory allocators
     * 
     * @details This class defines the interface for all custom allocators in the cslt namespace.
     * It provides core allocation/deallocation methods along with optional tracking and 
     * management capabilities. Derived classes implement specific allocation strategies
     * (heap, arena, pool, etc.).
     * 
     * The class supports:
     * - Basic allocation and deallocation with optional zeroing
     * - Aligned memory allocation
     * - Reallocation with preservation of existing data
     * - Memory tracking (size, capacity, utilization)
     * - Optional save/restore checkpointing
     * - Statistical reporting
     * 
     * @code{.cpp}
     * // Using a derived class (HeapAllocator)
     * cslt::HeapAllocator allocator;
     * 
     * // Allocate memory
     * auto result = allocator.alloc(1024, true); // 1KB zeroed
     * if (result.hasValue()) {
     *     void* ptr = result.value();
     *     // Use memory...
     *     allocator.return_element(ptr, 1024, allocator.default_alignment());
     * }
     * 
     * // Query allocator properties
     * if (allocator.owns_memory()) {
     *     size_t used = allocator.size();
     *     size_t available = allocator.remaining();
     * }
     * @endcode
     * 
     * @note This is an abstract class and cannot be instantiated directly.
     *       Use derived classes like HeapAllocator, ArenaAllocator, etc.
     */
    class Allocator {
    protected:
        /**
         * @brief Default alignment for memory allocations
         * @details Alignment value used when no specific alignment is requested.
         *          Typically set to alignof(max_align_t) for maximum portability.
         */
        size_t default_alignment_;

        /**
         * @brief Current bytes of memory in use
         * @details Tracks how many bytes are currently allocated and not yet freed.
         *          For allocators that don't track usage (like HeapAllocator), this remains 0.
         */
        size_t size_; 

        /**
         * @brief Bytes of usable memory available
         * @details The amount of memory available for user allocations, excluding any
         *          internal overhead or headers. Calculated as: alloc_ = total_alloc_ - overhead
         */
        size_t alloc_;             

         /**
          * @brief Total bytes of allocated memory
          * @details The total memory budget including overhead. This is what the user
          *          requested when creating the allocator. For heap allocators, this is 0.
          */
        size_t total_alloc_; 

        /**
         * @brief Memory type (STATIC or DYNAMIC)
         * @details Stored as uint8_t, cast to/from MemType enum.
         *          STATIC = memory allocated at compile time or from a fixed buffer
         *          DYNAMIC = memory allocated from OS heap
         */
        uint8_t mem_type_; 

        /**
         * @brief Whether allocator owns its memory
         * @details Stored as uint8_t, cast to/from bool.
         *          true = allocator owns and manages the memory buffer
         *          false = allocator is a wrapper (e.g., HeapAllocator wraps operator new)
         */
        uint8_t owns_memory_;
    public:

        /**
         * @brief Construct an Allocator
         * 
         * @param alignment Default alignment for allocations (default: alignof(max_align_t))
         * @param total_alloc Total memory budget in bytes (default: 0)
         * @param type Memory type - STATIC or DYNAMIC (default: ALLOC_INVALID)
         * @param owns_mem Whether this allocator owns its memory (default: false)
         * 
         * @details Initializes the allocator with the specified parameters. The total_alloc
         *          parameter represents the complete memory budget. Derived classes should
         *          calculate alloc_ by subtracting their overhead from total_alloc_.
         * 
         * @code{.cpp}
         * // Derived class example:
         * class ArenaAllocator : public Allocator {
         * public:
         *     ArenaAllocator(size_t total_bytes) 
         *         : Allocator(alignof(max_align_t), total_bytes, STATIC, true) 
         *     {
         *         size_t overhead = sizeof(Header);
         *         alloc_ = total_alloc_ - overhead; // Calculate usable space
         *         buffer_ = malloc(total_alloc_);
         *     }
         * };
         * @endcode
         */
        explicit Allocator(size_t alignment = alignof(max_align_t), 
                           size_t total_alloc = 0,
                           MemType type = ALLOC_INVALID, 
                           bool owns_mem = false) noexcept
            : default_alignment_(alignment), 
              size_(0),
              alloc_(0),
              total_alloc_(total_alloc),
              mem_type_(static_cast<uint8_t>(type)), 
              owns_memory_(static_cast<uint8_t>(owns_mem)) {}
    // -------------------------------------------------------------------------------- 

        /**
         * @brief Virtual destructor
         * @details Ensures proper cleanup of derived classes
         */
        virtual ~Allocator() noexcept = default;
    // -------------------------------------------------------------------------------- 

        /**
         * @brief Get the default alignment
         * 
         * @return Default alignment in bytes
         * 
         * @details Returns the alignment value that will be used for allocations
         *          when no specific alignment is requested.
         * 
         * @example
         * @code{.cpp}
         * cslt::HeapAllocator allocator(32); // 32-byte alignment
         * size_t align = allocator.default_alignment();
         * std::cout << "Default alignment: " << align << " bytes\n";
         * // Output: Default alignment: 32 bytes
         * @endcode
         */
        size_t default_alignment() const noexcept {return default_alignment_; }
    // -------------------------------------------------------------------------------- 

        /**
         * @brief Allocate memory
         * 
         * @param bytes Number of bytes to allocate
         * @param zeroed If true, zero-initialize the allocated memory (default: false)
         * @return Expected<void*> containing pointer to allocated memory or error
         * 
         * @code{.cpp}
         * cslt::HeapAllocator allocator;
         * 
         * // Allocate 256 bytes, zeroed
         * auto result = allocator.alloc(256, true);
         * if (result.hasValue()) {
         *     void* ptr = result.value();
         *     // Memory is guaranteed to be zeroed
         *     // ... use memory ...
         *     allocator.return_element(ptr, 256, allocator.default_alignment());
         * } else {
         *     // Handle allocation failure
         *     std::cerr << "Allocation failed\n";
         * }
         * @endcode
         */
        virtual Expected<void*> alloc(size_t bytes,
                                      bool zeroed = false) = 0;
    // -------------------------------------------------------------------------------- 

        /**
         * @brief Allocate aligned memory
         * 
         * @param bytes Number of bytes to allocate
         * @param alignment Required alignment in bytes (must be power of 2)
         * @param zeroed If true, zero-initialize the allocated memory (default: false)
         * @return Expected<void*> containing pointer to allocated memory or error
         * 
         * @details Allocates memory aligned to the specified boundary. The alignment
         *          must be a power of 2. If alignment is 0, uses default_alignment_.
         * 
         * @code{.cpp}
         * cslt::HeapAllocator allocator;
         * 
         * // Allocate 512 bytes aligned to 64-byte boundary
         * auto result = allocator.alloc_aligned(512, 64, false);
         * if (result.hasValue()) {
         *     void* ptr = result.value();
         *     
         *     // Verify alignment
         *     assert(reinterpret_cast<uintptr_t>(ptr) % 64 == 0);
         *     
         *     // ... use memory ...
         *     allocator.return_element(ptr, 512, 64);
         * }
         * @endcode
         */
        virtual Expected<void*> alloc_aligned(size_t bytes,
                                              size_t alignment,
                                              bool zeroed = false) = 0;
    // -------------------------------------------------------------------------------- 

        /**
         * @brief Reallocate memory
         * 
         * @param ptr Pointer to existing allocation
         * @param old_bytes Size of existing allocation in bytes
         * @param new_bytes Desired new size in bytes
         * @param zeroed If true, zero-initialize new bytes when growing (default: false)
         * @return Expected<void*> containing pointer to reallocated memory or error
         * 
         * @details Resizes an existing allocation. The existing data is preserved
         *          up to min(old_bytes, new_bytes). If growing and zeroed is true,
         *          the additional bytes are zero-initialized. The old pointer is
         *          automatically freed. The returned pointer may differ from the input.
         * 
         * @code{.cpp}
         * cslt::HeapAllocator allocator;
         * 
         * // Initial allocation
         * auto result1 = allocator.alloc(256, false);
         * void* ptr = result1.value();
         * memset(ptr, 0xAA, 256); // Fill with pattern
         * 
         * // Grow to 512 bytes, zero the new space
         * auto result2 = allocator.realloc(ptr, 256, 512, true);
         * if (result2.hasValue()) {
         *     void* new_ptr = result2.value();
         *     // First 256 bytes still contain 0xAA
         *     // Bytes 256-511 are zeroed
         *     allocator.return_element(new_ptr, 512, allocator.default_alignment());
         * }
         * @endcode
         */
        virtual Expected<void*> realloc(void* ptr,
                                        size_t old_bytes,
                                        size_t new_bytes,
                                        bool zeroed = false) = 0;
    // -------------------------------------------------------------------------------- 

        /**
         * @brief Reallocate aligned memory
         * 
         * @param ptr Pointer to existing allocation
         * @param old_bytes Size of existing allocation in bytes
         * @param new_bytes Desired new size in bytes
         * @param alignment Required alignment in bytes (must be power of 2)
         * @param zeroed If true, zero-initialize new bytes when growing (default: false)
         * @return Expected<void*> containing pointer to reallocated memory or error
         * 
         * @details Like realloc(), but maintains the specified alignment.
         * 
         * @code{.cpp}
         * cslt::HeapAllocator allocator;
         * 
         * // Initial aligned allocation
         * auto result1 = allocator.alloc_aligned(256, 64, false);
         * void* ptr = result1.value();
         * 
         * // Grow to 512 bytes, maintaining 64-byte alignment
         * auto result2 = allocator.realloc_aligned(ptr, 256, 512, 64, false);
         * if (result2.hasValue()) {
         *     void* new_ptr = result2.value();
         *     assert(reinterpret_cast<uintptr_t>(new_ptr) % 64 == 0);
         *     allocator.return_element(new_ptr, 512, 64);
         * }
         * @endcode
         */
        virtual Expected<void*> realloc_aligned(void *ptr,
                                                size_t old_bytes,
                                                size_t new_bytes,
                                                size_t alignment,
                                                bool zeroed = false) = 0;
    // -------------------------------------------------------------------------------- 

        /**
         * @brief Return/free allocated memory
         * 
         * @param ptr Pointer to memory to free
         * @param bytes Size of allocation in bytes
         * @param alignment Alignment used for the allocation
         * 
         * @details Frees memory previously allocated by this allocator.
         *          For heap allocators, this calls operator delete or free().
         *          For arena allocators, this may be a no-op or update bookkeeping.
         *          Always safe to call with nullptr (no-op).
         * 
         * @code{.cpp}
         * cslt::HeapAllocator allocator;
         * 
         * auto result = allocator.alloc_aligned(1024, 64, false);
         * void* ptr = result.value();
         * 
         * // ... use memory ...
         * 
         * // Free memory - must pass same size and alignment as allocation
         * allocator.return_element(ptr, 1024, 64);
         * 
         * // Safe to call with nullptr
         * allocator.return_element(nullptr, 0, 0); // No-op
         * @endcode
         */
        virtual void return_element(void *ptr, size_t bytes, size_t alignment) = 0;
    // -------------------------------------------------------------------------------- 

        /**
         * @brief Get memory type
         * 
         * @return MemType enum value (STATIC, DYNAMIC, or ALLOC_INVALID)
         * 
         * @details Returns whether this allocator uses static or dynamic memory.
         *          STATIC = fixed buffer allocated at compile-time or startup
         *          DYNAMIC = memory from OS heap (malloc/new)
         * 
         * @code{.cpp}
         * cslt::HeapAllocator heap_alloc;
         * cslt::ArenaAllocator arena_alloc(1024);
         * 
         * if (heap_alloc.memory_type() == cslt::DYNAMIC) {
         *     std::cout << "Heap allocator uses dynamic memory\n";
         * }
         * 
         * if (arena_alloc.memory_type() == cslt::STATIC) {
         *     std::cout << "Arena allocator uses static memory\n";
         * }
         * @endcode
         */
        MemType memory_type() const noexcept {
            return static_cast<MemType>(mem_type_);
        }
    // -------------------------------------------------------------------------------- 

        /**
         * @brief Generate statistics string
         * 
         * @param buffer Character buffer to write statistics to
         * @param buffer_size Size of buffer in bytes
         * @return true if statistics were written successfully, false otherwise
         * 
         * @details Writes human-readable statistics about the allocator to the provided
         *          buffer. Returns false if buffer is too small or null.
         * 
         * @code{.cpp}
         * cslt::HeapAllocator allocator;
         * 
         * char buffer[512];
         * if (allocator.stats(buffer, sizeof(buffer))) {
         *     printf("%s\n", buffer);
         *     // Output:
         *     // HeapAllocator Statistics:
         *     //   Type: DYNAMIC
         *     //   Default Alignment: 16 bytes
         *     //   Memory Model: System Heap (operator new/delete)
         *     //   ...
         * } else {
         *     fprintf(stderr, "Buffer too small for statistics\n");
         * }
         * @endcode
         */
        virtual bool stats(char *buffer, size_t buffer_size) const = 0;
    // -------------------------------------------------------------------------------- 

        /**
         * @brief Check if allocator owns its memory
         * 
         * @return true if allocator owns/manages its memory buffer, false otherwise
         * 
         * @details Returns whether this allocator owns its underlying memory.
         *          true = allocator allocated and manages the memory (e.g., ArenaAllocator)
         *          false = allocator is a wrapper around system allocator (e.g., HeapAllocator)
         * 
         * @code{.cpp}
         * cslt::HeapAllocator heap_alloc;
         * cslt::ArenaAllocator arena_alloc(1024);
         * 
         * if (heap_alloc.owns_memory()) {
         *     std::cout << "Heap allocator owns memory\n"; // Won't print
         * }
         * 
         * if (arena_alloc.owns_memory()) {
         *     std::cout << "Arena allocator owns memory\n"; // Will print
         * }
         * @endcode
         */
        bool owns_memory() const noexcept { 
            return static_cast<bool>(owns_memory_);
        }
    // -------------------------------------------------------------------------------- 

        /**
         * @brief Get current bytes in use
         * 
         * @return Number of bytes currently allocated and not freed
         * 
         * @details Returns the amount of memory currently in use. For allocators
         *          that don't track usage (like HeapAllocator), returns 0.
         *          For tracking allocators (like ArenaAllocator), returns actual usage.
         * 
         * @code{.cpp}
         * cslt::ArenaAllocator allocator(1024);
         * 
         * std::cout << "Initial size: " << allocator.size() << "\n"; // 0
         * 
         * auto result = allocator.alloc(256, false);
         * std::cout << "After alloc: " << allocator.size() << "\n"; // 256
         * 
         * allocator.return_element(result.value(), 256, allocator.default_alignment());
         * std::cout << "After free: " << allocator.size() << "\n"; // 0 (if supported)
         * @endcode
         */
        virtual size_t size() const noexcept { 
            return size_;
        }
    // -------------------------------------------------------------------------------- 

        /**
         * @brief Get remaining available bytes
         * 
         * @return Number of bytes still available for allocation
         * 
         * @details Returns alloc_ - size_, i.e., how much usable space is left.
         *          For allocators without fixed capacity (like HeapAllocator), returns 0.
         *          Includes underflow protection.
         * 
         * @code{.cpp}
         * cslt::ArenaAllocator allocator(1024);
         * 
         * std::cout << "Initial remaining: " << allocator.remaining() << "\n"; // ~1024
         * 
         * auto result = allocator.alloc(256, false);
         * std::cout << "After alloc: " << allocator.remaining() << "\n"; // ~768
         * 
         * // Can use remaining to check if allocation will succeed
         * if (allocator.remaining() >= 512) {
         *     auto result2 = allocator.alloc(512, false);
         * }
         * @endcode
         */
        virtual size_t remaining() const noexcept { 
            if (alloc_ > size_) {
                return alloc_ - size_;
            }
            return 0;
        }
    // -------------------------------------------------------------------------------- 

        /**
         * @brief Get total usable bytes
         * 
         * @return Total usable memory in bytes (excluding overhead)
         * 
         * @details Returns the total amount of memory available for user allocations,
         *          excluding any internal overhead or headers. This is calculated as:
         *          allocated() = total_alloc() - overhead
         *          
         *          For allocators without fixed capacity (like HeapAllocator), returns 0.
         *          For allocators with fixed capacity (like ArenaAllocator), returns the
         *          usable capacity.
         *          
         *          Relationship between memory functions:
         *          - total_alloc() = total memory including overhead
         *          - allocated() = usable memory (this function)
         *          - size() = currently used memory
         *          - remaining() = allocated() - size()
         * 
         * @code{.cpp}
         * cslt::ArenaAllocator allocator(1024); // Request 1024 total bytes
         * 
         * size_t total = allocator.total_alloc();      // 1024 (what user requested)
         * size_t usable = allocator.allocated();        // ~960 (after overhead)
         * size_t overhead = total - usable;             // ~64 (internal headers)
         * 
         * std::cout << "Total budget: " << total << " bytes\n";
         * std::cout << "Usable memory: " << usable << " bytes\n";
         * std::cout << "Overhead: " << overhead << " bytes\n";
         * 
         * // After some allocations
         * auto result = allocator.alloc(256, false);
         * 
         * std::cout << "Allocated (capacity): " << allocator.allocated() << "\n";  // ~960
         * std::cout << "Used: " << allocator.size() << "\n";                       // 256
         * std::cout << "Remaining: " << allocator.remaining() << "\n";             // ~704
         * @endcode
         * 
         * @note This differs from size() which returns currently used memory.
         *       allocated() is the total capacity, size() is current usage.
         */
        virtual size_t allocated() const noexcept {
            return alloc_;
        }
    // -------------------------------------------------------------------------------- 

        /**
         * @brief Get total allocated bytes
         * 
         * @return Total memory budget in bytes (including overhead)
         * 
         * @details Returns the total memory allocation size including any internal
         *          overhead. This is the value passed to the constructor.
         *          For heap allocators without fixed capacity, returns 0.
         * 
         * @code{.cpp}
         * cslt::ArenaAllocator allocator(1024);
         * 
         * std::cout << "Total allocation: " << allocator.total_alloc() << "\n"; // 1024
         * std::cout << "Usable memory: " << allocator.remaining() << "\n"; // < 1024
         * 
         * size_t overhead = allocator.total_alloc() - allocator.remaining();
         * std::cout << "Overhead: " << overhead << " bytes\n";
         * @endcode
         */
        virtual size_t total_alloc() const noexcept { 
            return total_alloc_;
        }
    // -------------------------------------------------------------------------------- 

        /**
         * @brief Check if pointer belongs to this allocator
         * 
         * @param ptr Pointer to check
         * @return true if pointer was allocated by this allocator, false otherwise
         * 
         * @details Checks if the given pointer was allocated by this allocator.
         *          Default implementation returns false (cannot verify).
         *          Allocators that own memory can override to provide actual verification.
         * 
         * @code{.cpp}
         * cslt::ArenaAllocator allocator(1024);
         * 
         * auto result = allocator.alloc(256, false);
         * void* ptr = result.value();
         * void* external_ptr = malloc(256);
         * 
         * if (allocator.is_ptr(ptr)) {
         *     std::cout << "ptr belongs to allocator\n";
         * }
         * 
         * if (!allocator.is_ptr(external_ptr)) {
         *     std::cout << "external_ptr does not belong to allocator\n";
         * }
         * 
         * free(external_ptr);
         * @endcode
         */
        virtual bool is_ptr(void* ptr) const { 
            (void)ptr;
            return false;  // Default: can't verify pointers
        }
    // -------------------------------------------------------------------------------- 

        /**
         * @brief Check if pointer and size are valid for this allocator
         * 
         * @param ptr Pointer to check
         * @param bytes Expected size of allocation
         * @return true if pointer with given size is valid, false otherwise
         * 
         * @details Checks if the pointer was allocated by this allocator AND
         *          if it has the specified size. Default returns false.
         *          Useful for debugging and validation.
         * 
         * @code{.cpp}
         * cslt::ArenaAllocator allocator(1024);
         * 
         * auto result = allocator.alloc(256, false);
         * void* ptr = result.value();
         * 
         * if (allocator.is_ptr_sized(ptr, 256)) {
         *     std::cout << "ptr is valid 256-byte allocation\n";
         * }
         * 
         * if (!allocator.is_ptr_sized(ptr, 512)) {
         *     std::cout << "ptr is not a 512-byte allocation\n";
         * }
         * @endcode
         */
        virtual bool is_ptr_sized(void* ptr, size_t bytes) const { 
            (void)ptr; (void)bytes;
            return false;  // Default: can't verify sized pointers
        }
    // -------------------------------------------------------------------------------- 

        /**
         * @brief Save allocator state checkpoint
         * 
         * @return Opaque checkpoint pointer, or nullptr if not supported
         * 
         * @details Saves the current state of the allocator for later restoration.
         *          Useful for implementing transactional semantics or temporary
         *          allocations. Default returns nullptr (not supported).
         *          Arena allocators typically return a position marker.
         * 
         * @code{.cpp}
         * cslt::ArenaAllocator allocator(1024);
         * 
         * // Save checkpoint
         * void* checkpoint = allocator.save();
         * 
         * // Make temporary allocations
         * auto temp1 = allocator.alloc(128, false);
         * auto temp2 = allocator.alloc(256, false);
         * 
         * // Restore to checkpoint (frees temp1 and temp2)
         * if (checkpoint) {
         *     allocator.restore(checkpoint);
         *     std::cout << "Temporary allocations rolled back\n";
         * }
         * @endcode
         */
        virtual void* save() const { 
            return nullptr;  // Default: doesn't support save/restore
        }
    // -------------------------------------------------------------------------------- 

        /**
         * @brief Restore allocator to saved checkpoint
         * 
         * @param checkpoint Checkpoint pointer from previous save() call
         * @return true if restore succeeded, false if not supported or invalid checkpoint
         * 
         * @details Restores allocator state to a previous checkpoint, effectively
         *          freeing all allocations made after that checkpoint.
         *          Default returns false (not supported).
         * 
         * @code{.cpp}
         * cslt::ArenaAllocator allocator(1024);
         * 
         * auto permanent = allocator.alloc(128, false);
         * void* checkpoint = allocator.save();
         * 
         * auto temp = allocator.alloc(256, false);
         * std::cout << "Size with temp: " << allocator.size() << "\n"; // 384
         * 
         * if (allocator.restore(checkpoint)) {
         *     std::cout << "Size after restore: " << allocator.size() << "\n"; // 128
         *     // temp is now invalid, permanent is still valid
         * }
         * @endcode
         */
        virtual bool restore(void* checkpoint) { 
            (void)checkpoint;
            return false;  // Default: doesn't support save/restore
        }
    // -------------------------------------------------------------------------------- 

        /**
         * @brief Reset allocator to initial state
         * 
         * @details Frees all allocations and resets the allocator to its initial state.
         *          Default is a no-op. Allocators that own memory override this to
         *          actually reset their state. After reset, all previous pointers are invalid.
         * 
         * @code{.cpp}
         * cslt::ArenaAllocator allocator(1024);
         * 
         * auto ptr1 = allocator.alloc(128, false);
         * auto ptr2 = allocator.alloc(256, false);
         * std::cout << "Before reset: " << allocator.size() << "\n"; // 384
         * 
         * allocator.reset();
         * std::cout << "After reset: " << allocator.size() << "\n"; // 0
         * // ptr1 and ptr2 are now INVALID - do not use!
         * 
         * // Can allocate again from fresh state
         * auto ptr3 = allocator.alloc(512, false);
         * @endcode
         */
        virtual void reset() {}
    };
// ================================================================================ 
// ================================================================================ 

#if ARENA_ENABLE_DYNAMIC

    /**
     * @class HeapAllocator
     * @brief Allocator wrapper around system heap (operator new/delete)
     * 
     * @details HeapAllocator is a simple wrapper around the system's dynamic memory
     * allocation facilities (operator new, operator delete, and platform-specific
     * aligned allocation functions). It does not own or track memory - all allocations
     * go directly to the OS heap.
     * 
     * Key characteristics:
     * - Does not own memory (owns_memory() returns false)
     * - Does not track allocations (size() returns 0)
     * - No fixed capacity (total_alloc() and allocated() return 0)
     * - Uses DYNAMIC memory type
     * - Thread-safety depends on system allocator
     * - No overhead beyond system allocator overhead
     * 
     * Use this allocator when:
     * - You need a uniform interface with other allocators
     * - You want to switch between allocator types at runtime
     * - You need alignment support beyond standard new
     * - You don't need memory tracking or limits
     * 
     * Alignment behavior:
     * - For alignments <= max_align_t: uses operator new
     * - For alignments > max_align_t: uses platform-specific functions
     *   (posix_memalign on POSIX, _aligned_malloc on Windows)
     * 
     * Creates a HeapAllocator with specified default alignment.
     * - **HeapAllocator(size_t alignment = alignof(max_align_t))** - Construct with default alignment
     * 
     * - **alloc(size_t bytes, bool zeroed = false)** - Allocate memory with default alignment
     * - **alloc_aligned(size_t bytes, size_t alignment, bool zeroed = false)** - Allocate aligned memory
     * - **realloc(void* ptr, size_t old_bytes, size_t new_bytes, bool zeroed = false)** - Resize allocation
     * - **realloc_aligned(void* ptr, size_t old_bytes, size_t new_bytes, size_t alignment, bool zeroed = false)** - Resize aligned allocation
     * - **return_element(void* ptr, size_t bytes, size_t alignment)** - Free allocated memory
     * 
     * - **default_alignment() const** - Get default alignment in bytes
     * - **memory_type() const** - Returns DYNAMIC
     * - **owns_memory() const** - Returns false (doesn't own memory)
     * - **size() const** - Returns 0 (doesn't track usage)
     * - **allocated() const** - Returns 0 (no fixed capacity)
     * - **remaining() const** - Returns 0 (no capacity limit)
     * - **total_alloc() const** - Returns 0 (no fixed allocation)
     * - **stats(char* buffer, size_t buffer_size) const** - Generate statistics string
     * 
     * - **is_ptr(void* ptr) const** - Returns false (cannot verify pointers)
     * - **is_ptr_sized(void* ptr, size_t bytes) const** - Returns false (cannot verify)
     * 
     * - **save() const** - Returns nullptr (no checkpoint support)
     * - **restore(void* checkpoint)** - Returns false (no restore support)
     * - **reset()** - No-op (cannot reset system heap)
     * 
     * @code{.c++}
     * cslt::HeapAllocator allocator;
     * 
     * // Allocate 1KB of zeroed memory
     * auto result = allocator.alloc(1024, true);
     * if (result.hasValue()) {
     *     void* ptr = result.value();
     *     // Memory is guaranteed to be zeroed
     *     // Use memory...
     *     allocator.return_element(ptr, 1024, allocator.default_alignment());
     * }
     * @endcode
     * 
     * @example Aligned allocation for SIMD
     * @code
     * cslt::HeapAllocator allocator;
     * 
     * // Allocate 512 bytes aligned to 64-byte cache line
     * auto result = allocator.alloc_aligned(512, 64, false);
     * if (result.hasValue()) {
     *     void* ptr = result.value();
     *     
     *     // Verify alignment
     *     assert(reinterpret_cast<uintptr_t>(ptr) % 64 == 0);
     *     
     *     // Can safely use SIMD instructions
     *     float* data = static_cast<float*>(ptr);
     *     // ... perform SIMD operations ...
     *     
     *     allocator.return_element(ptr, 512, 64);
     * }
     * @endcode
     * 
     * @example Reallocating a buffer
     * @code
     * cslt::HeapAllocator allocator;
     * 
     * // Start with 256 bytes
     * auto result1 = allocator.alloc(256, false);
     * void* ptr = result1.value();
     * memset(ptr, 0xAA, 256);
     * 
     * // Grow to 512 bytes, zero new space
     * auto result2 = allocator.realloc(ptr, 256, 512, true);
     * if (result2.hasValue()) {
     *     unsigned char* data = static_cast<unsigned char*>(result2.value());
     *     // First 256 bytes: 0xAA (preserved)
     *     // Next 256 bytes: 0x00 (zeroed)
     *     allocator.return_element(data, 512, allocator.default_alignment());
     * }
     * // Original ptr is now invalid - don't use it!
     * @endcode
     * 
     * @example Polymorphic usage pattern
     * @code
     * void process_data(cslt::Allocator& allocator) {
     *     auto result = allocator.alloc(256, true);
     *     if (result.hasValue()) {
     *         // Process data...
     *         allocator.return_element(result.value(), 256, 
     *                                 allocator.default_alignment());
     *     }
     * }
     * 
     * cslt::HeapAllocator heap_alloc;
     * cslt::ArenaAllocator arena_alloc(4096);
     * 
     * // Same function works with different allocators
     * process_data(heap_alloc);   // Uses system heap
     * process_data(arena_alloc);  // Uses arena
     * @endcode
     * 
     * @example Error handling
     * @code{.c++}
     * cslt::HeapAllocator allocator;
     * 
     * auto result = allocator.alloc(0, false);  // Invalid: 0 bytes
     * if (!result.hasValue()) {
     *     // Handle error - likely InvalidArgError
     *     std::cerr << "Allocation failed\n";
     * }
     * 
     * // Very large allocation might fail
     * auto result2 = allocator.alloc(SIZE_MAX, false);
     * if (!result2.hasValue()) {
     *     // Handle error - likely BadAllocError
     *     std::cerr << "Insufficient memory\n";
     * }
     * @endcode
     * 
     * @example Getting statistics
     * @code{.c++}
     * cslt::HeapAllocator allocator(32);  // 32-byte default alignment
     * 
     * char buffer[512];
     * if (allocator.stats(buffer, sizeof(buffer))) {
     *     printf("%s\n", buffer);
     *     // Outputs allocator type, alignment, and capabilities
     * }
     * @endcode
     * 
     * ### Always match allocation and deallocation:
     * @code
     * // CORRECT
     * auto r = allocator.alloc_aligned(512, 64, false);
     * allocator.return_element(r.value(), 512, 64);  // Same alignment!
     * 
     * // INCORRECT - mismatched alignment
     * auto r = allocator.alloc_aligned(512, 64, false);
     * allocator.return_element(r.value(), 512, 16);  // Wrong! Undefined behavior!
     * @endcode
     * 
     * ### Check return values:
     * @code
     * auto result = allocator.alloc(1024, false);
     * if (result.hasValue()) {
     *     // Safe to use result.value()
     * } else {
     *     // Handle allocation failure
     * }
     * @endcode
     * 
     * ### Clean up before destruction:
     * @code
     * {
     *     cslt::HeapAllocator allocator;
     *     auto result = allocator.alloc(1024, false);
     *     
     *     // ... use memory ...
     *     
     *     // MUST free before allocator is destroyed
     *     allocator.return_element(result.value(), 1024, 
     *                             allocator.default_alignment());
     * } // Safe - all memory freed
     * @endcode
     * 
     * @note This class is only available when ARENA_ENABLE_DYNAMIC is enabled.
     *       Define STATIC_ONLY to disable dynamic allocation for MISRA compliance.
     * 
     * @warning HeapAllocator does not track memory usage. Methods like size(),
     *          remaining(), and is_ptr() will return default values (0/false).
     * 
     * @warning After calling realloc() or realloc_aligned(), the original pointer
     *          is invalid and must not be used. Always use the newly returned pointer.
     * 
     * @warning Always use the same alignment value for return_element() that was
     *          used during allocation. Mismatched alignment causes undefined behavior.
     * 
     * @see Allocator Base class interface with full method documentation
     * @see ArenaAllocator For memory tracking and fixed capacity allocations
     * @see PoolAllocator For fixed-size object allocation
     */
    class HeapAllocator : public Allocator {
    public:

        /**
         * @brief Construct a HeapAllocator
         * 
         * @param alignment Default alignment for allocations (default: alignof(max_align_t))
         * 
         * @details Creates a heap allocator wrapper with the specified default alignment.
         *          The allocator has no fixed capacity and does not track allocations.
         * 
         * @example Default construction
         * @code
         * // Uses default alignment (typically 16 bytes)
         * cslt::HeapAllocator allocator;
         * std::cout << "Default alignment: " 
         *           << allocator.default_alignment() << "\n";
         * @endcode
         * 
         * @example Custom alignment
         * @code
         * // Uses 64-byte default alignment
         * cslt::HeapAllocator allocator(64);
         * 
         * // Allocations without explicit alignment use 64 bytes
         * auto result = allocator.alloc(256, false);
         * // Pointer is aligned to 64 bytes
         * @endcode
         * 
         * @note The constructor is noexcept and cannot fail.
         */
        explicit HeapAllocator(size_t alignment = alignof(max_align_t)) noexcept
            : Allocator(alignment, 0, DYNAMIC, false) {}

        /**
         * @brief Destructor
         * 
         * @details Destroys the HeapAllocator. Does not free any outstanding allocations
         *          since HeapAllocator does not track them. The user is responsible for
         *          calling return_element() on all allocated pointers before destruction.
         * 
         * @warning Destroying a HeapAllocator with outstanding allocations will cause
         *          memory leaks. Always free all allocations before destruction.
         * 
         * @example Proper cleanup
         * @code
         * {
         *     cslt::HeapAllocator allocator;
         *     
         *     auto result = allocator.alloc(1024, false);
         *     void* ptr = result.value();
         *     
         *     // Use memory...
         *     
         *     // MUST free before allocator goes out of scope
         *     allocator.return_element(ptr, 1024, allocator.default_alignment());
         * } // allocator destroyed here - safe
         * @endcode
         */
        ~HeapAllocator() noexcept override  = default;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Allocate memory from system heap
         * 
         * @param bytes Number of bytes to allocate
         * @param zeroed If true, zero-initialize the allocated memory (default: false)
         * @return Expected<void*> containing pointer to allocated memory or error
         * 
         * @details Allocates memory using operator new with the allocator's default
         *          alignment. Uses std::nothrow to avoid exceptions.
         * 
         * @throws Never throws (uses nothrow allocation)
         * 
         * @retval Expected with pointer on success
         * @retval Expected with InvalidArgError if bytes is 0
         * @retval Expected with BadAllocError if allocation fails
         * 
         * @example Basic allocation
         * @code
         * cslt::HeapAllocator allocator;
         * 
         * auto result = allocator.alloc(512, false);
         * if (result.hasValue()) {
         *     void* ptr = result.value();
         *     memset(ptr, 0xAB, 512);
         *     allocator.return_element(ptr, 512, allocator.default_alignment());
         * } else {
         *     std::cerr << "Allocation failed\n";
         * }
         * @endcode
         * 
         * @example Zeroed allocation
         * @code
         * cslt::HeapAllocator allocator;
         * 
         * auto result = allocator.alloc(1024, true);
         * if (result.hasValue()) {
         *     void* ptr = result.value();
         *     // Memory is guaranteed to be all zeros
         *     allocator.return_element(ptr, 1024, allocator.default_alignment());
         * }
         * @endcode
         * 
         * @see alloc_aligned() For custom alignment requirements
         * @see return_element() To free the allocated memory
         */
        Expected<void*> alloc(size_t bytes,
                              bool zeroed = false) override;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Allocate aligned memory from system heap
         * 
         * @param bytes Number of bytes to allocate
         * @param alignment Required alignment in bytes (must be power of 2)
         * @param zeroed If true, zero-initialize the allocated memory (default: false)
         * @return Expected<void*> containing pointer to aligned memory or error
         * 
         * @details Allocates memory aligned to the specified boundary. The alignment
         *          must be a power of 2. If alignment is 0, uses default_alignment_.
         *          
         *          Implementation strategy:
         *          - If alignment <= max_align_t: uses operator new
         *          - If alignment > max_align_t: uses posix_memalign (POSIX) or
         *            _aligned_malloc (Windows)
         * 
         * @throws Never throws (uses nothrow allocation)
         * 
         * @retval Expected with pointer on success (aligned to specified boundary)
         * @retval Expected with InvalidArgError if bytes is 0
         * @retval Expected with BadAllocError if allocation fails
         * 
         * @example Cache line alignment
         * @code
         * cslt::HeapAllocator allocator;
         * 
         * // Allocate aligned to 64-byte cache line
         * auto result = allocator.alloc_aligned(4096, 64, false);
         * if (result.hasValue()) {
         *     void* ptr = result.value();
         *     
         *     // Verify alignment
         *     assert(reinterpret_cast<uintptr_t>(ptr) % 64 == 0);
         *     
         *     allocator.return_element(ptr, 4096, 64);
         * }
         * @endcode
         * 
         * @example SIMD alignment (256-bit AVX)
         * @code
         * cslt::HeapAllocator allocator;
         * 
         * // Allocate aligned for AVX operations
         * auto result = allocator.alloc_aligned(1024, 32, true);
         * if (result.hasValue()) {
         *     float* data = static_cast<float*>(result.value());
         *     // Can safely use AVX instructions
         *     allocator.return_element(data, 1024, 32);
         * }
         * @endcode
         * 
         * @warning The alignment parameter to return_element() must match the
         *          alignment used for allocation.
         * 
         * @see alloc() For standard alignment
         * @see return_element() To free aligned memory
         */
        Expected<void*> alloc_aligned(size_t bytes,
                                      size_t alignment,
                                      bool zeroed = false) override;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Reallocate memory from system heap
         * 
         * @param ptr Pointer to existing allocation
         * @param old_bytes Size of existing allocation in bytes
         * @param new_bytes Desired new size in bytes
         * @param zeroed If true, zero-initialize new bytes when growing (default: false)
         * @return Expected<void*> containing pointer to reallocated memory or error
         * 
         * @details Resizes an existing allocation. The existing data is preserved up to
         *          min(old_bytes, new_bytes). The old pointer is automatically freed.
         *          The returned pointer may differ from the input pointer.
         *          
         *          When growing (new_bytes > old_bytes) and zeroed is true, the additional
         *          bytes beyond old_bytes are zero-initialized.
         *          
         *          Implementation: Allocates new memory, copies old data, frees old memory.
         * 
         * @throws Never throws
         * 
         * @retval Expected with new pointer on success
         * @retval Expected with NullPointerError if ptr is null
         * @retval Expected with InvalidArgError if new_bytes is 0
         * @retval Expected with ReallocFailError if new allocation fails
         * 
         * @example Growing allocation
         * @code
         * cslt::HeapAllocator allocator;
         * 
         * auto result1 = allocator.alloc(256, false);
         * void* ptr = result1.value();
         * memset(ptr, 0xAA, 256);
         * 
         * // Grow to 512 bytes, zero the new space
         * auto result2 = allocator.realloc(ptr, 256, 512, true);
         * if (result2.hasValue()) {
         *     unsigned char* data = static_cast<unsigned char*>(result2.value());
         *     // Bytes 0-255 still contain 0xAA
         *     // Bytes 256-511 are zeroed
         *     allocator.return_element(data, 512, allocator.default_alignment());
         * }
         * // Note: Original ptr is already freed, don't use it
         * @endcode
         * 
         * @example Shrinking allocation
         * @code
         * cslt::HeapAllocator allocator;
         * 
         * auto result1 = allocator.alloc(1024, false);
         * void* ptr = result1.value();
         * 
         * // Shrink to 512 bytes
         * auto result2 = allocator.realloc(ptr, 1024, 512, false);
         * if (result2.hasValue()) {
         *     void* new_ptr = result2.value();
         *     // First 512 bytes preserved
         *     allocator.return_element(new_ptr, 512, allocator.default_alignment());
         * }
         * @endcode
         * 
         * @warning After calling realloc, the original pointer is invalid. Always use
         *          the newly returned pointer.
         * 
         * @see realloc_aligned() For aligned reallocation
         * @see alloc() For initial allocation
         */
        Expected<void*> realloc(void* ptr,
                                size_t old_bytes,
                                size_t new_bytes,
                                bool zeroed = false) override;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Reallocate aligned memory from system heap
         * 
         * @param ptr Pointer to existing aligned allocation
         * @param old_bytes Size of existing allocation in bytes
         * @param new_bytes Desired new size in bytes
         * @param alignment Required alignment in bytes (must be power of 2)
         * @param zeroed If true, zero-initialize new bytes when growing (default: false)
         * @return Expected<void*> containing pointer to reallocated aligned memory or error
         * 
         * @details Like realloc(), but maintains the specified alignment. The returned
         *          pointer is guaranteed to be aligned to the specified boundary.
         * 
         * @throws Never throws
         * 
         * @retval Expected with new aligned pointer on success
         * @retval Expected with NullPointerError if ptr is null
         * @retval Expected with InvalidArgError if new_bytes is 0
         * @retval Expected with ReallocFailError if new allocation fails
         * 
         * @example Reallocate cache-aligned buffer
         * @code
         * cslt::HeapAllocator allocator;
         * 
         * // Initial 64-byte aligned allocation
         * auto result1 = allocator.alloc_aligned(512, 64, false);
         * void* ptr = result1.value();
         * 
         * // Grow while maintaining 64-byte alignment
         * auto result2 = allocator.realloc_aligned(ptr, 512, 1024, 64, false);
         * if (result2.hasValue()) {
         *     void* new_ptr = result2.value();
         *     
         *     // Verify alignment maintained
         *     assert(reinterpret_cast<uintptr_t>(new_ptr) % 64 == 0);
         *     
         *     allocator.return_element(new_ptr, 1024, 64);
         * }
         * @endcode
         * 
         * @warning The alignment parameter must match the alignment used in the
         *          original allocation.
         * 
         * @see realloc() For non-aligned reallocation
         * @see alloc_aligned() For initial aligned allocation
         */
        Expected<void*> realloc_aligned(void* ptr,
                                        size_t old_bytes,
                                        size_t new_bytes,
                                        size_t alignment,
                                        bool zeroed = false) override;
// --------------------------------------------------------------------------------

        /**
         * @brief Free memory allocated from system heap
         * 
         * @param ptr Pointer to memory to free
         * @param bytes Size of allocation in bytes (unused but required for interface)
         * @param alignment Alignment used for the allocation
         * 
         * @details Frees memory previously allocated by this allocator. The implementation
         *          uses the appropriate deallocation function based on alignment:
         *          - If alignment <= max_align_t: uses operator delete
         *          - If alignment > max_align_t: uses free() (POSIX) or _aligned_free (Windows)
         *          
         *          Safe to call with nullptr (no-op).
         * 
         * @example Basic deallocation
         * @code
         * cslt::HeapAllocator allocator;
         * 
         * auto result = allocator.alloc(1024, false);
         * void* ptr = result.value();
         * 
         * // Use memory...
         * 
         * // Free - must pass same alignment as allocation
         * allocator.return_element(ptr, 1024, allocator.default_alignment());
         * @endcode
         * 
         * @example Aligned deallocation
         * @code
         * cslt::HeapAllocator allocator;
         * 
         * auto result = allocator.alloc_aligned(512, 64, false);
         * void* ptr = result.value();
         * 
         * // Use memory...
         * 
         * // Free - must pass same alignment (64) as allocation
         * allocator.return_element(ptr, 512, 64);
         * @endcode
         * 
         * @example Safe nullptr handling
         * @code
         * cslt::HeapAllocator allocator;
         * void* ptr = nullptr;
         * 
         * // Safe - does nothing
         * allocator.return_element(ptr, 0, 0);
         * @endcode
         * 
         * @warning Always use the same alignment value that was used during allocation.
         *          Mismatched alignment can cause undefined behavior or crashes.
         * 
         * @warning Do not call return_element twice on the same pointer (double-free).
         * 
         * @note The bytes parameter is required by the interface but is not used by
         *       HeapAllocator since operator delete/free don't require size.
         */
        void return_element(void *ptr, size_t bytes, size_t alignment) override;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Generate statistics string
         * 
         * @param buffer Character buffer to write statistics to
         * @param buffer_size Size of buffer in bytes
         * @return true if statistics were written successfully, false on error
         * 
         * @details Writes human-readable statistics about the HeapAllocator to the
         *          provided buffer. Since HeapAllocator doesn't track allocations,
         *          the statistics primarily describe the allocator type and configuration.
         *          
         *          Returns false if:
         *          - buffer is nullptr
         *          - buffer_size is 0
         *          - buffer is too small for the statistics string
         * 
         * @example Print statistics
         * @code
         * cslt::HeapAllocator allocator(64);
         * 
         * char buffer[512];
         * if (allocator.stats(buffer, sizeof(buffer))) {
         *     printf("%s\n", buffer);
         *     // Output:
         *     // HeapAllocator Statistics:
         *     //   Type: DYNAMIC
         *     //   Default Alignment: 64 bytes
         *     //   Memory Model: System Heap (operator new/delete)
         *     //   Note: HeapAllocator is a wrapper around system allocator.
         *     //         It does not own or track memory; all allocations
         *     //         are managed directly by the OS heap.
         * }
         * @endcode
         * 
         * @example Error handling
         * @code
         * cslt::HeapAllocator allocator;
         * 
         * char small_buffer[10];
         * if (!allocator.stats(small_buffer, sizeof(small_buffer))) {
         *     fprintf(stderr, "Buffer too small for statistics\n");
         * }
         * @endcode
         * 
         * @note Unlike tracking allocators (e.g., ArenaAllocator), HeapAllocator
         *       statistics don't include usage information since it doesn't track
         *       allocations.
         * 
         * @see Allocator::stats() Base class documentation
         */
        bool stats(char *buffer, size_t buffer_size) const;
    };
#endif /* ARENA_ENABLE_DYNAMIC */
}
// ================================================================================ 
// ================================================================================ 
#endif /* allocator_HPP */
// ================================================================================
// ================================================================================
// eof
