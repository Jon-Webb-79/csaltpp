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
#include "pointers.hpp"

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
        virtual void return_element(void *ptr, 
                                    size_t bytes, 
                                    size_t alignment = alignof(max_align_t)) = 0;
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
        virtual bool reset(bool trim_extra_chunks = false) { 
            (void)trim_extra_chunks;
            return true;
        }
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
        void return_element(void *ptr, 
                            size_t bytes, 
                            size_t alignment = alignof(max_align_t)) override;
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
// -------------------------------------------------------------------------------- 

        bool reset(bool trim_extra_chunks = false) {
            (void)trim_extra_chunks;
            return true;
        }
    };
#endif /* ARENA_ENABLE_DYNAMIC */
// ================================================================================ 
// ================================================================================ 

    /**
     * @brief Check if a value is a power of 2
     * 
     * @param x Value to check
     * @return 1 if x is a power of 2, 0 otherwise
     * 
     * @details Returns 1 for 1, 2, 4, 8, 16, etc. Returns 0 for 0.
     * 
     * @par Example:
     * @code
     * assert(cslt::is_pow2(16) == 1);
     * assert(cslt::is_pow2(15) == 0);
     * assert(cslt::is_pow2(0) == 0);
     * @endcode
     */
    inline int is_pow2(size_t x) { 
        return x && !(x & (x - 1)); 
    }
// -------------------------------------------------------------------------------- 

    /**
     * @brief Round up to next power of 2
     * 
     * @param x Value to round up
     * @return Next power of 2 >= x, or 0 if overflow
     * 
     * @details Returns the smallest power of 2 that is >= x.
     *          Returns 0 if x is too large and would overflow.
     *          Returns 1 for x=1, 0 for x=0.
     * 
     * @par Example:
     * @code
     * assert(cslt::next_pow2(15) == 16);
     * assert(cslt::next_pow2(16) == 16);
     * assert(cslt::next_pow2(17) == 32);
     * assert(cslt::next_pow2(1) == 1);
     * assert(cslt::next_pow2(0) == 0);
     * @endcode
     */
    inline size_t next_pow2(size_t x) {
        if (x <= 1) return x ? 1 : 0;
        if (x > (SIZE_MAX >> 1)) return 0;
        x--;
        for (size_t s = 1; s < 8 * sizeof(size_t); s <<= 1) {
            x |= x >> s;
        }
        return x + 1;
    }
// ================================================================================ 
// ================================================================================ 

    class ArenaAllocator;

    struct ArenaDeleter {
        void operator()(ArenaAllocator* arena) const noexcept;
    };
// ================================================================================ 
// ================================================================================ 

    /**
     * @class ArenaAllocator
     * @brief Fast, region-based memory allocator that allocates memory in chunks
     * 
     * @details ArenaAllocator is a memory allocator that uses "arena" or "region" allocation
     *          strategy. It allocates memory sequentially from large chunks and does not
     *          support individual deallocation. All memory is freed together when the arena
     *          is reset or destroyed.
     * 
     *          Key characteristics:
     *          - Very fast allocation (just pointer bumping)
     *          - No per-allocation metadata overhead
     *          - No fragmentation
     *          - No individual deallocation (use reset() instead)
     *          - Supports checkpoints for transactional rollback
     *          - Optional dynamic growth with additional chunks
     * 
     *          Three initialization modes:
     *          - Heap(): Dynamically allocated arena that owns its memory
     *          - Stack(): Arena using a user-provided buffer (doesn't own memory)
     *          - SubArena(): Arena allocated within a parent arena (doesn't own memory)
     * 
     * @example Basic heap arena usage
     * @code
     * // Create a 64KB heap arena
     * auto result = cslt::ArenaAllocator::Heap(64 * 1024);
     * if (!result.hasValue()) {
     *     std::cerr << "Failed to create arena: " << result.error().what() << "\n";
     *     return;
     * }
     * auto arena = cslt::move(result.value());
     * 
     * // Allocate some memory
     * auto ptr1 = arena->alloc(256);
     * auto ptr2 = arena->alloc(512);
     * 
     * // Use the memory...
     * 
     * // Reset to free all at once (much faster than individual frees)
     * arena->reset();
     * 
     * // Can allocate again
     * auto ptr3 = arena->alloc(128);
     * @endcode
     * 
     * @example Stack buffer arena
     * @code
     * // Use stack memory for the arena
     * uint8_t buffer[4096];
     * auto result = cslt::ArenaAllocator::Stack(buffer, sizeof(buffer));
     * auto arena = cslt::move(result.value());
     * 
     * auto ptr = arena->alloc(128);
     * // All allocations come from 'buffer'
     * @endcode
     * 
     * @example Checkpoint and restore
     * @code
     * auto arena_result = cslt::ArenaAllocator::Heap(8192);
     * auto arena = cslt::move(arena_result.value());
     * 
     * auto permanent = arena->alloc(256);
     * 
     * // Save checkpoint before temporary allocations
     * void* checkpoint = arena->save();
     * 
     * auto temp1 = arena->alloc(128);
     * auto temp2 = arena->alloc(512);
     * 
     * // Rollback to checkpoint (frees temp1 and temp2)
     * arena->restore(checkpoint);
     * // permanent is still valid, temp1 and temp2 are now invalid
     * @endcode
     */
    class ArenaAllocator : public Allocator {
    private:
        /**
         * @brief Internal memory chunk structure for arena allocation
         * @internal
         */
        struct Chunk {
            uint8_t *chunk;     ///< Pointer to beginning of memory block
            size_t len;         ///< Used bytes in this chunk
            size_t alloc;       ///< Total allocated bytes in this chunk
            Chunk* next;        ///< Pointer to next chunk in linked list
        };
        /**
         * @brief Internal checkpoint representation
         * @internal
         */
        struct CheckpointData {
            Chunk* chunk;    ///< Chunk at checkpoint time
            uint8_t* cur;    ///< Cursor position at checkpoint time
            size_t len;      ///< Total used bytes at checkpoint time
        };
// -------------------------------------------------------------------------------- 

        uint8_t *cur_;       ///< Pointer to the next available memory slot 
        Chunk* head_;        ///< Pointer to head of memory chunk linked list 
        Chunk* tail_;        ///< Pointer to the tail of memory chunks
        size_t min_chunk_;   ///< The minimum chunk size in bytes
        uint8_t resize_;     ///< Allows resizing if true with mem_type == DYNAMIC 
// -------------------------------------------------------------------------------- 

        /**
         * @brief Find a chunk in the arena's chunk list
         * 
         * @param target Chunk to find
         * @param out_prev Output parameter for previous chunk (can be nullptr)
         * @return Pointer to found chunk, or nullptr if not found
         */
        Chunk* find_chunk_in_chain(Chunk* target, Chunk** out_prev = nullptr) const;
// -------------------------------------------------------------------------------- 

    /* Private constructor for factory methods only
     * This initializes the base Allocator with default values
     * The factory methods will then set the proper values directly
     */
        ArenaAllocator() 
            : Allocator(alignof(max_align_t), 0, ALLOC_INVALID, false),
              cur_(nullptr),
              head_(nullptr),
              tail_(nullptr),
              min_chunk_(0),
              resize_(0) {}
// ================================================================================ 

    public:
        /**
         * @brief Destructor - frees all arena memory if owned
         * 
         * @details For DYNAMIC arenas (created via Heap()), frees all additional chunks.
         *          The primary chunk (containing the ArenaAllocator object itself) is
         *          freed by ArenaDeleter. For STATIC and SUB arenas, does nothing as
         *          they don't own their memory.
         * 
         * @note All pointers allocated from this arena become invalid after destruction.
         * 
         * @example
         * @code
         * {
         *     auto arena_result = cslt::ArenaAllocator::Heap(4096);
         *     auto arena = cslt::move(arena_result.value());
         *     
         *     void* ptr = arena->alloc(256).value();
         *     // Use ptr...
         * } // arena destroyed here, all memory freed, ptr is now invalid
         * @endcode
         */
         ~ArenaAllocator() noexcept override {
                if (static_cast<bool>(owns_memory_)) {
                    // Free additional chunks (NOT the head)
                    Chunk* current = head_ ? head_->next : nullptr;
                    while (current != nullptr) {
                        Chunk* next = current->next;
                        ::operator delete(current);
                        current = next;
                    }
                }
            }
// -------------------------------------------------------------------------------- 

#if ARENA_ENABLE_DYNAMIC
        /**
         * @brief Create a heap-allocated arena that owns its memory
         * 
         * @param bytes Initial size in bytes for the arena
         * @param resize If true, arena can grow by allocating additional chunks
         * @param min_chunk_size Minimum size for additional chunks (must be power of 2, 0 = no minimum)
         * @param base_align_in Base alignment for allocations (must be power of 2, 0 = default)
         * 
         * @return Expected containing UniquePtr to arena on success, or error on failure
         * 
         * @throws Never throws - returns Expected with error on failure
         * 
         * @details Creates an arena that allocates its memory from the heap using operator new.
         *          The arena lives at the beginning of its own memory allocation. When destroyed,
         *          all memory is freed automatically via ArenaDeleter.
         *          
         *          Memory layout: [ArenaAllocator][Chunk header][data...]
         *          
         *          If `bytes` is less than `min_chunk_size`, `min_chunk_size` is used instead.
         *          If `resize` is true and the arena runs out of space, new chunks are allocated
         *          automatically. Each new chunk is at least `min_chunk_size` bytes.
         * 
         * @retval Expected with UniquePtr on success
         * @retval Expected with ArgumentError if bytes too small for arena structure
         * @retval Expected with MemoryError if allocation fails
         * @retval Expected with AlignmentError if alignment normalization fails
         * 
         * @example Basic resizable arena
         * @code
         * auto result = cslt::ArenaAllocator::Heap(4096, true, 4096, 16);
         * if (!result.hasValue()) {
         *     std::cerr << "Arena creation failed\n";
         *     return;
         * }
         * auto arena = cslt::move(result.value());
         * 
         * // Allocate until we fill the initial 4KB
         * for (int i = 0; i < 100; ++i) {
         *     arena->alloc(64);
         * }
         * // Arena automatically grows with new chunks as needed
         * @endcode
         * 
         * @example Non-resizable fixed-size arena
         * @code
         * auto result = cslt::ArenaAllocator::Heap(8192, false);
         * auto arena = cslt::move(result.value());
         * 
         * // Fill arena
         * while (arena->remaining() >= 64) {
         *     arena->alloc(64);
         * }
         * 
         * // This will fail - no space and resize disabled
         * auto fail = arena->alloc(1024);
         * assert(!fail.hasValue());
         * @endcode
         * 
         * @see Stack() For arena using user-provided buffer
         * @see SubArena() For arena within parent arena
         */
        static Expected<cslt::UniquePtr<ArenaAllocator, ArenaDeleter>>
        Heap(size_t bytes,
             bool resize = false,
             size_t min_chunk_size = 4096,
             size_t base_align_in = alignof(max_align_t));
#endif /* ARENA_ENABLE_DYNAMIC */
// -------------------------------------------------------------------------------- 

        /**
         * @brief Create an arena using a user-provided buffer
         * 
         * @param buffer Pointer to pre-allocated memory buffer
         * @param bytes Size of buffer in bytes
         * @param base_align_in Base alignment for allocations (must be power of 2, 0 = default)
         * 
         * @return Expected containing UniquePtr to arena on success, or error on failure
         * 
         * @throws Never throws - returns Expected with error on failure
         * 
         * @details Creates an arena that uses a user-provided buffer for all allocations.
         *          The arena lives at the beginning of the buffer. The arena does NOT own
         *          the buffer - the caller is responsible for buffer lifetime.
         *          
         *          Memory layout: [ArenaAllocator][Chunk header][data...]
         *          
         *          Static arenas:
         *          - Cannot resize (fixed buffer size)
         *          - Do not own memory (buffer managed externally)
         *          - Always have exactly one chunk
         *          - Faster creation (no heap allocation)
         * 
         * @retval Expected with UniquePtr on success
         * @retval Expected with ArgumentError if buffer is null or too small
         * @retval Expected with AlignmentError if alignment normalization fails
         * 
         * @warning The buffer must remain valid for the entire lifetime of the arena.
         *          Destroying the buffer while the arena exists causes undefined behavior.
         * 
         * @example Stack buffer arena
         * @code
         * uint8_t buffer[4096];
         * auto result = cslt::ArenaAllocator::Stack(buffer, sizeof(buffer));
         * auto arena = cslt::move(result.value());
         * 
         * // All allocations come from 'buffer'
         * auto ptr = arena->alloc(256);
         * 
         * // buffer must outlive arena
         * @endcode
         * 
         * @example Heap buffer arena
         * @code
         * uint8_t* buffer = new uint8_t[65536];
         * auto result = cslt::ArenaAllocator::Stack(buffer, 65536, 32);
         * auto arena = cslt::move(result.value());
         * 
         * // Use arena...
         * auto ptr = arena->alloc(1024);
         * 
         * // Arena destroyed first
         * arena.reset();
         * 
         * // Now safe to free buffer
         * delete[] buffer;
         * @endcode
         * 
         * @see Heap() For arena that owns its memory
         * @see SubArena() For arena within parent arena
         */
        static Expected<cslt::UniquePtr<ArenaAllocator, ArenaDeleter>>
        Stack(void* buffer,
              size_t bytes,
              size_t base_align_in = alignof(max_align_t));
// -------------------------------------------------------------------------------- 

        /**
         * @brief Create a sub-arena within a parent arena
         * 
         * @param parent Parent arena to allocate from
         * @param bytes Size in bytes for the sub-arena
         * @param base_align_in Base alignment for allocations (must be power of 2, 0 = default)
         * 
         * @return Expected containing UniquePtr to sub-arena on success, or error on failure
         * 
         * @throws Never throws - returns Expected with error on failure
         * 
         * @details Creates an arena that lives entirely within the memory of a parent arena.
         *          The sub-arena allocates its initial buffer from the parent and manages it
         *          independently. The sub-arena does NOT own its memory - it's managed by the
         *          parent.
         *          
         *          Memory layout in parent: [ArenaAllocator][Chunk header][data...]
         *          
         *          Sub-arenas:
         *          - Cannot resize (borrowed from parent)
         *          - Do not own memory (parent owns it)
         *          - Always have exactly one chunk
         *          - Useful for scoped/temporary allocations
         *          - Can be nested (sub-sub-arenas)
         * 
         * @retval Expected with UniquePtr on success
         * @retval Expected with ArgumentError if bytes is zero or too small
         * @retval Expected with MemoryError if parent allocation fails
         * @retval Expected with AlignmentError if alignment normalization fails
         * 
         * @warning The parent arena must outlive all its sub-arenas. Destroying the parent
         *          while sub-arenas exist causes undefined behavior.
         * 
         * @example Basic sub-arena
         * @code
         * auto parent_result = cslt::ArenaAllocator::Heap(16384);
         * auto parent = cslt::move(parent_result.value());
         * 
         * // Create sub-arena using parent's memory
         * auto sub_result = cslt::ArenaAllocator::SubArena(*parent, 4096);
         * auto sub = cslt::move(sub_result.value());
         * 
         * // Allocate from sub-arena
         * auto ptr = sub->alloc(256);
         * 
         * // Sub-arena destroyed first, memory stays in parent
         * sub.reset();
         * 
         * // Parent can still use its memory
         * parent->alloc(512);
         * @endcode
         * 
         * @example Multiple sub-arenas
         * @code
         * auto parent_result = cslt::ArenaAllocator::Heap(32768);
         * auto parent = cslt::move(parent_result.value());
         * 
         * // Create multiple independent sub-arenas
         * auto sub1 = cslt::move(cslt::ArenaAllocator::SubArena(*parent, 8192).value());
         * auto sub2 = cslt::move(cslt::ArenaAllocator::SubArena(*parent, 8192).value());
         * 
         * // Each sub-arena is independent
         * sub1->alloc(256);
         * sub2->alloc(512);
         * @endcode
         * 
         * @example Nested sub-arenas
         * @code
         * auto parent = cslt::move(cslt::ArenaAllocator::Heap(32768).value());
         * auto sub = cslt::move(cslt::ArenaAllocator::SubArena(*parent, 16384).value());
         * auto subsub = cslt::move(cslt::ArenaAllocator::SubArena(*sub, 8192).value());
         * 
         * // Three-level hierarchy works
         * parent->alloc(128);
         * sub->alloc(256);
         * subsub->alloc(512);
         * @endcode
         * 
         * @see Heap() For arena that owns its memory
         * @see Stack() For arena using user buffer
         */
        static Expected<cslt::UniquePtr<ArenaAllocator, ArenaDeleter>>
        SubArena(ArenaAllocator& parent,
                 size_t bytes,
                 size_t base_align_in = alignof(max_align_t));
// -------------------------------------------------------------------------------- 

        /**
         * @brief Allocate memory from the arena
         * 
         * @param bytes Number of bytes to allocate
         * @param zeroed If true, zero-initialize the allocated memory (default: false)
         * 
         * @return Expected containing pointer to allocated memory on success, or error on failure
         * 
         * @throws Never throws - returns Expected with error on failure
         * 
         * @details Allocates memory by bumping the arena's current pointer. Very fast operation
         *          (typically just a few pointer arithmetic operations). Memory is aligned to
         *          the arena's default alignment.
         *          
         *          If the current chunk doesn't have enough space and the arena is resizable
         *          (DYNAMIC with resize=true), a new chunk is automatically allocated.
         * 
         * @retval Expected with pointer on success
         * @retval Expected with ArgumentError if bytes is 0
         * @retval Expected with MemoryError if out of space (non-resizable) or allocation fails (resizable)
         * @retval Expected with AlignmentError if alignment check fails
         * @retval Expected with LengthOverflowError if size calculations overflow
         * 
         * @warning Allocated memory cannot be individually freed. Use reset() to free all
         *          allocations at once, or restore() to rollback to a checkpoint.
         * 
         * @example Basic allocation
         * @code
         * auto arena_result = cslt::ArenaAllocator::Heap(4096);
         * auto arena = cslt::move(arena_result.value());
         * 
         * auto result = arena->alloc(256);
         * if (result.hasValue()) {
         *     void* ptr = result.value();
         *     // Use memory...
         * }
         * @endcode
         * 
         * @example Zeroed allocation
         * @code
         * auto arena = cslt::move(cslt::ArenaAllocator::Heap(4096).value());
         * 
         * auto result = arena->alloc(1024, true);
         * uint8_t* data = static_cast<uint8_t*>(result.value());
         * // data[0..1023] are all guaranteed to be zero
         * @endcode
         * 
         * @example Handling allocation failure
         * @code
         * auto arena = cslt::move(cslt::ArenaAllocator::Heap(512, false).value());
         * 
         * // Fill arena
         * while (arena->alloc(64).hasValue()) { }
         * 
         * // This will fail - no space left
         * auto result = arena->alloc(128);
         * if (!result.hasValue()) {
         *     std::cerr << "Out of memory: " << result.error().what() << "\n";
         * }
         * @endcode
         * 
         * @see alloc_aligned() For custom alignment
         * @see realloc() To resize last allocation
         * @see reset() To free all allocations
         */
        Expected<void*> alloc(size_t bytes, bool zeroed = false) override;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Allocate aligned memory from the arena
         * 
         * @param bytes Number of bytes to allocate
         * @param alignment Required alignment in bytes (must be power of 2, 0 = use default)
         * @param zeroed If true, zero-initialize the allocated memory (default: false)
         * 
         * @return Expected containing pointer to aligned memory on success, or error on failure
         * 
         * @throws Never throws - returns Expected with error on failure
         * 
         * @details Like alloc(), but ensures the returned pointer is aligned to the specified
         *          boundary. If alignment is 0, uses the arena's default alignment. If alignment
         *          is not a power of 2, it is automatically rounded up to the next power of 2.
         *          
         *          The alignment requirement may cause padding to be inserted, which counts
         *          against the arena's capacity but isn't directly usable.
         * 
         * @retval Expected with aligned pointer on success
         * @retval Expected with ArgumentError if bytes is 0
         * @retval Expected with AlignmentError if alignment normalization fails
         * @retval Expected with MemoryError if out of space or allocation fails
         * @retval Expected with LengthOverflowError if size calculations overflow
         * 
         * @example Cache-line aligned allocation
         * @code
         * auto arena = cslt::move(cslt::ArenaAllocator::Heap(8192).value());
         * 
         * // Allocate aligned to 64-byte cache line
         * auto result = arena->alloc_aligned(256, 64);
         * void* ptr = result.value();
         * 
         * // Verify alignment
         * assert(reinterpret_cast<uintptr_t>(ptr) % 64 == 0);
         * @endcode
         * 
         * @example SIMD aligned allocation
         * @code
         * auto arena = cslt::move(cslt::ArenaAllocator::Heap(4096).value());
         * 
         * // Allocate for AVX2 operations (256-bit = 32 bytes)
         * auto result = arena->alloc_aligned(1024, 32, true);
         * float* data = static_cast<float*>(result.value());
         * // Can safely use AVX2 instructions on data
         * @endcode
         * 
         * @see alloc() For default-aligned allocation
         * @see realloc_aligned() To resize aligned allocation
         */
        Expected<void*> alloc_aligned(size_t bytes,
                                      size_t alignment,
                                      bool zeroed = false) override;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Reallocate memory (resize last allocation if possible)
         * 
         * @param ptr Pointer to existing allocation
         * @param old_bytes Size of existing allocation in bytes
         * @param new_bytes Desired new size in bytes
         * @param zeroed If true, zero-initialize new bytes when growing (default: false)
         * 
         * @return Expected containing pointer to reallocated memory on success, or error on failure
         * 
         * @throws Never throws - returns Expected with error on failure
         * 
         * @details Attempts to resize an existing allocation. If the allocation is the most
         *          recent one and there's space in the current chunk, it extends in-place
         *          (very fast). Otherwise, allocates new memory and copies the old data.
         *          
         *          The existing data is preserved up to min(old_bytes, new_bytes). When
         *          growing and zeroed=true, the additional bytes are zero-initialized.
         *          
         *          Unlike HeapAllocator, the old memory is NOT freed (arenas don't support
         *          individual deallocation). If a copy is made, the old memory becomes
         *          wasted space until reset().
         * 
         * @retval Expected with pointer to resized allocation on success
         * @retval Expected with ArgumentError if ptr is null or new_bytes is 0
         * @retval Expected with MemoryError if out of space
         * 
         * @warning After realloc, the original pointer may or may not be valid. Always
         *          use the returned pointer.
         * 
         * @example Growing allocation in-place
         * @code
         * auto arena = cslt::move(cslt::ArenaAllocator::Heap(4096).value());
         * 
         * auto ptr = arena->alloc(128).value();
         * memset(ptr, 0xAA, 128);
         * 
         * // Grow to 256 bytes (likely extends in-place)
         * auto new_ptr_result = arena->realloc(ptr, 128, 256, false);
         * uint8_t* new_ptr = static_cast<uint8_t*>(new_ptr_result.value());
         * 
         * // First 128 bytes still contain 0xAA
         * assert(new_ptr[0] == 0xAA);
         * @endcode
         * 
         * @example Growing with zeroing
         * @code
         * auto arena = cslt::move(cslt::ArenaAllocator::Heap(8192).value());
         * 
         * auto ptr = arena->alloc(256).value();
         * memset(ptr, 0xFF, 256);
         * 
         * // Grow to 512 bytes, zero new space
         * auto new_ptr = arena->realloc(ptr, 256, 512, true).value();
         * uint8_t* data = static_cast<uint8_t*>(new_ptr);
         * 
         * // Bytes 0-255: 0xFF (preserved)
         * // Bytes 256-511: 0x00 (zeroed)
         * @endcode
         * 
         * @example Shrinking allocation
         * @code
         * auto arena = cslt::move(cslt::ArenaAllocator::Heap(4096).value());
         * 
         * auto ptr = arena->alloc(1024).value();
         * 
         * // Shrink to 512 bytes
         * auto new_ptr = arena->realloc(ptr, 1024, 512, false).value();
         * // First 512 bytes preserved
         * @endcode
         * 
         * @note If realloc causes a copy, the old memory is wasted until reset().
         *       Frequent reallocs can fragment the arena.
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
         * @brief Reallocate aligned memory
         * 
         * @param ptr Pointer to existing aligned allocation
         * @param old_bytes Size of existing allocation in bytes
         * @param new_bytes Desired new size in bytes
         * @param alignment Required alignment in bytes (must be power of 2)
         * @param zeroed If true, zero-initialize new bytes when growing (default: false)
         * 
         * @return Expected containing pointer to reallocated aligned memory on success, or error
         * 
         * @throws Never throws - returns Expected with error on failure
         * 
         * @details Like realloc(), but maintains the specified alignment. The returned pointer
         *          is guaranteed to be aligned to the specified boundary.
         * 
         * @retval Expected with aligned pointer on success
         * @retval Expected with ArgumentError if ptr is null or new_bytes is 0
         * @retval Expected with AlignmentError if alignment is invalid or ptr is misaligned
         * @retval Expected with MemoryError if out of space
         * 
         * @example Reallocate cache-aligned buffer
         * @code
         * auto arena = cslt::move(cslt::ArenaAllocator::Heap(8192).value());
         * 
         * // Initial 64-byte aligned allocation
         * auto ptr = arena->alloc_aligned(512, 64).value();
         * 
         * // Grow while maintaining alignment
         * auto new_ptr = arena->realloc_aligned(ptr, 512, 1024, 64, false).value();
         * 
         * // Verify alignment maintained
         * assert(reinterpret_cast<uintptr_t>(new_ptr) % 64 == 0);
         * @endcode
         * 
         * @warning The alignment parameter must match the alignment used in the original
         *          allocation. Mismatched alignment may cause an error.
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
         * @brief Check if pointer was allocated by this arena
         * 
         * @param ptr Pointer to check
         * 
         * @return true if pointer was allocated by this arena, false otherwise
         * 
         * @details Checks if the given pointer points to memory within any of this arena's
         *          chunks. This is useful for debugging and validation.
         *          
         *          Returns false for:
         *          - nullptr
         *          - Pointers allocated by other arenas
         *          - Pointers allocated by other allocators
         *          - Invalid pointers
         * 
         * @example Validate pointer
         * @code
         * auto arena = cslt::move(cslt::ArenaAllocator::Heap(4096).value());
         * 
         * void* ptr = arena->alloc(128).value();
         * void* external = malloc(128);
         * 
         * assert(arena->is_ptr(ptr));          // true
         * assert(!arena->is_ptr(external));    // false
         * assert(!arena->is_ptr(nullptr));     // false
         * 
         * free(external);
         * @endcode
         * 
         * @example Check before using
         * @code
         * void process_data(cslt::ArenaAllocator* arena, void* data) {
         *     if (!arena->is_ptr(data)) {
         *         throw std::invalid_argument("Pointer not from this arena");
         *     }
         *     // Safe to use data...
         * }
         * @endcode
         * 
         * @see is_ptr_sized() To also validate size
         */
        bool is_ptr(void* ptr) const override;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Check if pointer and size are valid for this arena
         * 
         * @param ptr Pointer to check
         * @param bytes Expected size of allocation
         * 
         * @return true if pointer with given size is valid, false otherwise
         * 
         * @details Checks if the pointer was allocated by this arena AND if the specified
         *          size fits within the allocation. More strict than is_ptr().
         *          
         *          Returns false for:
         *          - Invalid pointers (as per is_ptr())
         *          - Valid pointers where bytes exceeds the allocated size
         *          - nullptr or bytes == 0
         * 
         * @example Validate allocation size
         * @code
         * auto arena = cslt::move(cslt::ArenaAllocator::Heap(4096).value());
         * 
         * void* ptr = arena->alloc(256).value();
         * 
         * assert(arena->is_ptr_sized(ptr, 256));   // true - exact size
         * assert(arena->is_ptr_sized(ptr, 128));   // true - subset
         * assert(!arena->is_ptr_sized(ptr, 512));  // false - too large
         * @endcode
         * 
         * @example Bounds checking
         * @code
         * void safe_write(cslt::ArenaAllocator* arena, void* buf, size_t size) {
         *     if (!arena->is_ptr_sized(buf, size)) {
         *         throw std::out_of_range("Buffer too small or invalid");
         *     }
         *     memset(buf, 0, size);  // Safe
         * }
         * @endcode
         * 
         * @see is_ptr() To check pointer only without size
         */
        bool is_ptr_sized(void* ptr, size_t bytes) const override;
// -------------------------------------------------------------------------------- 

        /**
         * @brief No-op for arena allocators (individual deallocation not supported)
         * 
         * @param ptr Pointer to memory (ignored)
         * @param bytes Size of allocation (ignored)
         * @param alignment Alignment used (ignored)
         * 
         * @details Arena allocators do not support individual deallocation. All memory is
         *          freed together via reset() or when the arena is destroyed. This method
         *          is a no-op to maintain compatibility with the Allocator interface.
         *          
         *          This design is intentional and provides:
         *          - Much faster allocation (no per-allocation metadata)
         *          - No fragmentation
         *          - Simpler implementation
         *          - Better cache locality
         * 
         * @note To free memory, use reset() to free all allocations, or restore() to
         *       rollback to a checkpoint, or let the arena be destroyed.
         * 
         * @example Demonstrating no-op behavior
         * @code
         * auto arena = cslt::move(cslt::ArenaAllocator::Heap(4096).value());
         * 
         * void* ptr = arena->alloc(256).value();
         * size_t size_before = arena->size();
         * 
         * // This does nothing
         * arena->return_element(ptr, 256);
         * 
         * // Size unchanged - memory still allocated
         * assert(arena->size() == size_before);
         * 
         * // To actually free memory:
         * arena->reset();
         * assert(arena->size() == 0);
         * @endcode
         * 
         * @example Proper cleanup pattern
         * @code
         * {
         *     auto arena = cslt::move(cslt::ArenaAllocator::Heap(8192).value());
         *     
         *     void* ptr1 = arena->alloc(512).value();
         *     void* ptr2 = arena->alloc(1024).value();
         *     
         *     // Don't call return_element - it does nothing
         *     
         *     // Option 1: Reset to reuse arena
         *     arena->reset();
         *     
         *     // Option 2: Let arena be destroyed (automatic cleanup)
         * } // All memory freed here
         * @endcode
         * 
         * @see reset() To free all allocations
         * @see restore() To rollback to checkpoint
         */
        void return_element(void *ptr, 
                            size_t bytes, 
                            size_t alignment = alignof(max_align_t)) override;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Reset arena to initial state, freeing all allocations
         * 
         * @param trim_extra_chunks If true, free additional chunks (DYNAMIC arenas only)
         * 
         * @return true on success, false if arena has no chunks
         * 
         * @details Resets the arena by marking all memory as unused. This is MUCH faster
         *          than calling free() on each individual allocation.
         *          
         *          For DYNAMIC arenas with trim_extra_chunks=true, frees all chunks except
         *          the initial chunk, reducing memory footprint. Otherwise, keeps all chunks
         *          allocated for potential reuse.
         *          
         *          For STATIC and SUB arenas, trim_extra_chunks has no effect (they can't
         *          have extra chunks).
         * 
         * @warning After reset(), ALL pointers allocated from this arena become invalid.
         *          Do not use them after calling reset().
         * 
         * @example Basic reset and reuse
         * @code
         * auto arena = cslt::move(cslt::ArenaAllocator::Heap(4096).value());
         * 
         * void* ptr1 = arena->alloc(256).value();
         * void* ptr2 = arena->alloc(512).value();
         * 
         * std::cout << "Used: " << arena->size() << "\n";  // 768+
         * 
         * // Reset - fast O(1) operation
         * arena->reset();
         * 
         * std::cout << "Used: " << arena->size() << "\n";  // 0
         * 
         * // ptr1 and ptr2 are now INVALID - do not use!
         * 
         * // Can allocate again from fresh state
         * void* ptr3 = arena->alloc(128).value();
         * @endcode
         * 
         * @example Reset with trim
         * @code
         * auto arena = cslt::move(cslt::ArenaAllocator::Heap(512, true, 512).value());
         * 
         * // Fill initial chunk
         * while (arena->remaining() >= 64) {
         *     arena->alloc(64);
         * }
         * 
         * // Trigger growth
         * arena->alloc(256);
         * 
         * size_t chunks = arena->chunk_count();  // > 1
         * 
         * // Reset and trim extra chunks
         * arena->reset(true);
         * 
         * assert(arena->chunk_count() == 1);  // Back to single chunk
         * @endcode
         * 
         * @example Frame-based allocation pattern
         * @code
         * auto arena = cslt::move(cslt::ArenaAllocator::Heap(1024 * 1024).value());
         * 
         * while (running) {
         *     // Allocate temporary data for this frame
         *     auto temp_data = arena->alloc(4096).value();
         *     process_frame(temp_data);
         *     
         *     // Free all frame allocations at once (very fast!)
         *     arena->reset();
         * }
         * @endcode
         * 
         * @see save() To create checkpoint before temporary allocations
         * @see restore() To rollback to checkpoint instead of full reset
         */
        bool reset(bool trim_extra_chunks = false) override;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Save current arena state as a checkpoint
         * 
         * @return Opaque checkpoint pointer, or nullptr on failure
         * 
         * @details Creates a checkpoint of the current arena state. The checkpoint can later
         *          be used with restore() to rollback all allocations made after the checkpoint.
         *          This is useful for transactional or scoped allocations.
         *          
         *          The checkpoint is allocated separately (not from the arena) and must be
         *          freed by restore(). If you don't call restore(), you leak the checkpoint
         *          (but not the arena memory).
         *          
         *          Returns nullptr if:
         *          - Arena has no chunks
         *          - Checkpoint allocation fails
         * 
         * @warning You must call restore() on the checkpoint to free it. Losing the
         *          checkpoint pointer without calling restore() causes a memory leak.
         * 
         * @example Transactional allocations
         * @code
         * auto arena = cslt::move(cslt::ArenaAllocator::Heap(8192).value());
         * 
         * // Make some permanent allocations
         * void* permanent1 = arena->alloc(256).value();
         * void* permanent2 = arena->alloc(512).value();
         * 
         * // Save checkpoint before temporary work
         * void* checkpoint = arena->save();
         * 
         * // Make temporary allocations
         * void* temp1 = arena->alloc(1024).value();
         * void* temp2 = arena->alloc(2048).value();
         * 
         * // Rollback - frees temp1 and temp2
         * arena->restore(checkpoint);
         * 
         * // permanent1 and permanent2 still valid
         * // temp1 and temp2 now INVALID
         * @endcode
         * 
         * @example Nested checkpoints
         * @code
         * auto arena = cslt::move(cslt::ArenaAllocator::Heap(16384).value());
         * 
         * arena->alloc(256);
         * 
         * void* cp1 = arena->save();
         * arena->alloc(512);
         * 
         * void* cp2 = arena->save();
         * arena->alloc(1024);
         * 
         * // Restore inner checkpoint first
         * arena->restore(cp2);  // Frees 1024-byte allocation
         * 
         * arena->alloc(768);
         * 
         * // Restore outer checkpoint
         * arena->restore(cp1);  // Frees 512 + 768 allocations
         * @endcode
         * 
         * @example Try-catch pattern
         * @code
         * auto arena = cslt::move(cslt::ArenaAllocator::Heap(8192).value());
         * 
         * void* checkpoint = arena->save();
         * try {
         *     // Make allocations that might fail
         *     auto data1 = arena->alloc(1024).value();
         *     risky_operation(data1);
         *     
         *     auto data2 = arena->alloc(2048).value();
         *     risky_operation(data2);
         *     
         *     // Success - keep allocations
         *     arena->restore(checkpoint);  // Must free checkpoint!
         * } catch (...) {
         *     // Failure - rollback allocations
         *     arena->restore(checkpoint);
         *     throw;
         * }
         * @endcode
         * 
         * @see restore() To restore to checkpoint
         * @see reset() To free all allocations
         */
        void* save() const override;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Restore arena to a previously saved checkpoint
         * 
         * @param checkpoint Checkpoint pointer from previous save() call
         * 
         * @return true if restore succeeded, false on error
         * 
         * @details Restores the arena to the state it was in when save() was called.
         *          All allocations made after the checkpoint are effectively freed.
         *          
         *          For DYNAMIC arenas, also frees any chunks that were allocated after
         *          the checkpoint.
         *          
         *          The checkpoint itself is freed (deleted) by this call, whether restore
         *          succeeds or fails. Do not use the checkpoint pointer after calling restore().
         *          
         *          Returns false if:
         *          - checkpoint is nullptr
         *          - Checkpoint is invalid or corrupted
         *          - Checkpoint's chunk no longer exists
         *          - Checkpoint data is out of bounds
         * 
         * @warning After restore(), ALL pointers allocated after the checkpoint become
         *          invalid. Do not use them.
         * 
         * @warning The checkpoint pointer is freed by restore(). Do not use it after
         *          calling restore(), even if restore() returns false.
         * 
         * @example Basic checkpoint/restore
         * @code
         * auto arena = cslt::move(cslt::ArenaAllocator::Heap(4096).value());
         * 
         * void* data1 = arena->alloc(256).value();
         * 
         * void* checkpoint = arena->save();
         * size_t size_at_checkpoint = arena->size();
         * 
         * void* data2 = arena->alloc(512).value();
         * void* data3 = arena->alloc(1024).value();
         * 
         * // Restore - frees data2 and data3
         * bool success = arena->restore(checkpoint);
         * assert(success);
         * assert(arena->size() == size_at_checkpoint);
         * 
         * // data1 still valid, data2 and data3 now INVALID
         * // checkpoint pointer now INVALID (freed by restore)
         * @endcode
         * 
         * @example Error handling
         * @code
         * auto arena = cslt::move(cslt::ArenaAllocator::Heap(4096).value());
         * 
         * void* checkpoint = arena->save();
         * 
         * // ... do work ...
         * 
         * if (!arena->restore(checkpoint)) {
         *     std::cerr << "Failed to restore checkpoint\n";
         *     // checkpoint is still freed even though restore failed
         * }
         * @endcode
         * 
         * @see save() To create checkpoint
         * @see reset() To free all allocations instead
         */
        bool restore(void* checkpoint) override;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Get remaining available bytes in the arena
         * 
         * @return Number of bytes available for allocation
         * 
         * @details Returns the total available space across all chunks, accounting for
         *          alignment padding. This is the sum of (chunk.alloc - chunk.len) for
         *          all chunks.
         *          
         *          For resizable DYNAMIC arenas, this shows current capacity before
         *          growth would be needed. After growth, remaining() increases.
         *          
         *          For non-resizable arenas (STATIC, SUB, or DYNAMIC with resize=false),
         *          this shows the fixed remaining capacity.
         * 
         * @example Check before allocating
         * @code
         * auto arena = cslt::move(cslt::ArenaAllocator::Heap(4096, false).value());
         * 
         * size_t needed = 1024;
         * if (arena->remaining() >= needed) {
         *     auto ptr = arena->alloc(needed).value();
         *     // Guaranteed to succeed
         * } else {
         *     std::cerr << "Not enough space\n";
         * }
         * @endcode
         * 
         * @example Monitor arena capacity
         * @code
         * auto arena = cslt::move(cslt::ArenaAllocator::Heap(8192).value());
         * 
         * std::cout << "Initial: " << arena->remaining() << " bytes\n";
         * 
         * arena->alloc(1024);
         * std::cout << "After alloc: " << arena->remaining() << " bytes\n";
         * 
         * arena->reset();
         * std::cout << "After reset: " << arena->remaining() << " bytes\n";
         * @endcode
         * 
         * @see size() Current bytes used
         * @see allocated() Total capacity
         * @see total_alloc() Total including overhead
         */
        size_t remaining() const noexcept override;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Generate human-readable statistics about the arena
         * 
         * @param buffer Character buffer to write statistics to
         * @param buffer_size Size of buffer in bytes
         * 
         * @return true if statistics were written successfully, false on error
         * 
         * @details Writes detailed arena statistics to the provided buffer, including:
         *          - Memory type (STATIC/DYNAMIC)
         *          - Used bytes
         *          - Capacity (usable bytes)
         *          - Total allocation (including overhead)
         *          - Utilization percentage
         *          - Default alignment
         *          - Chunk information
         *          - Resizable status
         *          - Memory ownership
         *          - Minimum chunk size (if applicable)
         *          
         *          Returns false if:
         *          - buffer is nullptr
         *          - buffer_size is 0
         *          - Buffer is too small for the statistics
         * 
         * @example Print arena stats
         * @code
         * auto arena = cslt::move(cslt::ArenaAllocator::Heap(4096, true, 2048).value());
         * 
         * arena->alloc(512);
         * arena->alloc(1024);
         * 
         * char buffer[1024];
         * if (arena->stats(buffer, sizeof(buffer))) {
         *     std::cout << buffer << "\n";
         *     // Output:
         *     // Arena Statistics:
         *     //   Type: DYNAMIC
         *     //   Used: 1536 bytes
         *     //   Capacity: 3800 bytes
         *     //   Total (with overhead): 4096 bytes
         *     //   Utilization: 40.4%
         *     //   Default Alignment: 16 bytes
         *     //   Chunk 1: 1536/3800 bytes
         *     //   Resizable: Yes
         *     //   Owns Memory: Yes
         *     //   Min Chunk Size: 2048 bytes
         * }
         * @endcode
         * 
         * @example Error handling
         * @code
         * auto arena = cslt::move(cslt::ArenaAllocator::Heap(4096).value());
         * 
         * char small_buffer[10];
         * if (!arena->stats(small_buffer, sizeof(small_buffer))) {
         *     std::cerr << "Buffer too small for statistics\n";
         * }
         * @endcode
         * 
         * @see size() Get current usage
         * @see allocated() Get capacity
         * @see chunk_count() Get number of chunks
         */
        bool stats(char *buffer, size_t buffer_size) const override;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Get the number of chunks in the arena
         * 
         * @return Number of chunks
         * 
         * @details Returns the number of memory chunks the arena is using. Initially 1,
         *          but can grow for resizable DYNAMIC arenas. STATIC and SUB arenas
         *          always have exactly 1 chunk.
         * 
         * @example Monitor chunk growth
         * @code
         * auto arena = cslt::move(cslt::ArenaAllocator::Heap(512, true, 512).value());
         * 
         * std::cout << "Chunks: " << arena->chunk_count() << "\n";  // 1
         * 
         * // Fill first chunk
         * while (arena->remaining() >= 64) {
         *     arena->alloc(64);
         * }
         * 
         * // Trigger growth
         * arena->alloc(256);
         * 
         * std::cout << "Chunks: " << arena->chunk_count() << "\n";  // 2
         * @endcode
         * 
         * @see reset() Can trim extra chunks
         * @see min_chunk_size() Minimum size for new chunks
         */
        size_t chunk_count() const noexcept;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Get the minimum chunk size for this arena
         * 
         * @return Minimum chunk size in bytes, or 0 if not applicable
         * 
         * @details Returns the minimum size used when allocating new chunks for resizable
         *          DYNAMIC arenas. When the arena needs to grow, new chunks are at least
         *          this size (actual size may be larger to satisfy the allocation request).
         *          
         *          Returns 0 for:
         *          - Non-resizable arenas
         *          - Arenas created without a min_chunk_size parameter
         * 
         * @example Check minimum chunk size
         * @code
         * auto arena = cslt::move(cslt::ArenaAllocator::Heap(4096, true, 8192).value());
         * 
         * std::cout << "Min chunk: " << arena->min_chunk_size() << " bytes\n";  // 8192
         * 
         * // When arena grows, new chunks will be at least 8192 bytes
         * @endcode
         * 
         * @see Heap() Set min_chunk_size at creation
         * @see chunk_count() Number of chunks
         */
        size_t min_chunk_size() const noexcept;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Enable or disable arena resizing
         * 
         * @param toggle If true, enable resizing; if false, disable resizing
         * 
         * @details Dynamically enables or disables the arena's ability to grow by allocating
         *          additional chunks. Only works for DYNAMIC arenas that own their memory.
         *          Has no effect on STATIC or SUB arenas (they cannot resize regardless).
         *          
         *          Useful for:
         *          - Enforcing strict memory limits during critical sections
         *          - Testing allocation failure handling
         *          - Preventing unbounded growth
         * 
         * @example Temporarily disable growth
         * @code
         * auto arena = cslt::move(cslt::ArenaAllocator::Heap(4096, true).value());
         * 
         * // Fill initial capacity
         * while (arena->remaining() >= 64) {
         *     arena->alloc(64);
         * }
         * 
         * // Can still grow
         * auto ptr1 = arena->alloc(1024);
         * assert(ptr1.hasValue());
         * 
         * // Disable growth
         * arena->toggle_resize(false);
         * 
         * // Fill again
         * while (arena->remaining() >= 64) {
         *     arena->alloc(64);
         * }
         * 
         * // This will fail - no growth allowed
         * auto ptr2 = arena->alloc(1024);
         * assert(!ptr2.hasValue());
         * 
         * // Re-enable growth
         * arena->toggle_resize(true);
         * 
         * // Now works again
         * auto ptr3 = arena->alloc(1024);
         * assert(ptr3.hasValue());
         * @endcode
         * 
         * @example Enforce memory budget
         * @code
         * auto arena = cslt::move(cslt::ArenaAllocator::Heap(1024 * 1024, true).value());
         * 
         * // Disable resize to enforce 1MB limit
         * arena->toggle_resize(false);
         * 
         * // Allocations will fail if they exceed 1MB total
         * while (auto ptr = arena->alloc(1024); ptr.hasValue()) {
         *     // Process...
         * }
         * std::cout << "Hit 1MB limit\n";
         * @endcode
         * 
         * @note Has no effect on STATIC arenas (user buffer) or SUB arenas (borrowed memory)
         * 
         * @see Heap() Set initial resize capability
         */
        void toggle_resize(bool toggle) noexcept;
    };
// ================================================================================ 
// ================================================================================

    /**
     * @struct ArenaDeleter
     * @brief Custom deleter for ArenaAllocator unique pointers
     * 
     * @details This deleter properly cleans up ArenaAllocator instances created by
     *          factory methods (Heap, Stack, SubArena). It:
     *          1. Calls the arena's destructor (frees additional chunks)
     *          2. Frees the base memory only if the arena owns it (DYNAMIC only)
     *          
     *          Memory ownership:
     *          - DYNAMIC (Heap): Owns memory → frees with ::operator delete
     *          - STATIC (Stack): User owns buffer → doesn't free
     *          - SUB (SubArena): Parent owns memory → doesn't free
     * 
     * @example Automatic via UniquePtr
     * @code
     * {
     *     auto arena = cslt::move(cslt::ArenaAllocator::Heap(4096).value());
     *     // Use arena...
     * } // ArenaDeleter automatically called here
     * @endcode
     * 
     * @example Manual construction (not typical)
     * @code
     * ArenaAllocator* raw_arena = // ... created via factory ...;
     * cslt::UniquePtr<ArenaAllocator, ArenaDeleter> arena(raw_arena, ArenaDeleter{});
     * // Automatic cleanup when arena goes out of scope
     * @endcode
     * 
     * @warning Never call ::operator delete directly on an ArenaAllocator created
     *          by a factory method. Always use ArenaDeleter or let UniquePtr handle it.
     * 
     * @see ArenaAllocator::Heap()
     * @see ArenaAllocator::Stack()
     * @see ArenaAllocator::SubArena()
     */
    inline void ArenaDeleter::operator()(ArenaAllocator* arena) const noexcept {
        if (!arena) return;
        
        MemType type = arena->memory_type();
        bool owns = arena->owns_memory();
        void* base = arena;
        
        // Call destructor
        arena->~ArenaAllocator();
        
        // Only free if DYNAMIC *and* owns the memory
        if (type == DYNAMIC && owns) {
            ::operator delete(base);
        }
    }
// ================================================================================ 
// ================================================================================ 

    class PoolAllocator;

    /**
     * @struct PoolDeleter
     * @brief Custom deleter for PoolAllocator unique pointers
     * 
     * @details Similar to ArenaDeleter, properly cleans up PoolAllocator instances
     *          created by factory methods. Handles ownership appropriately:
     *          - DYNAMIC pools that own their arena: frees the pool and arena
     *          - Pools with borrowed arenas: only frees the pool structure
     */
    struct PoolDeleter {
        void operator()(PoolAllocator* pool) const noexcept;
    };
// ================================================================================ 
// ================================================================================ 

    class PoolAllocator : public Allocator {
        friend struct PoolDeleter;
    private:
        /**
         * @brief Checkpoint representation for save/restore
         * @internal
         */
        struct PoolCheckpointData {
            void*    free_list;      ///< Head of free list at checkpoint time
            size_t   free_blocks;    ///< Number of blocks in free list
            uint8_t* cur;            ///< Bump pointer position in arena
            size_t   total_blocks;   ///< Total blocks available at checkpoint time
        };
        
        ArenaAllocator* arena_;           ///< Backing arena (owned or borrowed)
        bool            owns_arena_;      ///< true if pool must destroy arena
        size_t          block_size_;      ///< User-requested block size
        size_t          stride_;          ///< Actual block size (aligned)
        size_t          blocks_per_chunk_;///< Blocks allocated per growth
        uint8_t*        cur_;             ///< Next free byte in arena
        uint8_t*        end_;             ///< End of current arena slice
        void*           free_list_;       ///< Head of intrusive free list
        size_t          total_blocks_;    ///< Total blocks ever available
        size_t          free_blocks_;     ///< Blocks currently in free list
        bool            grow_enabled_;    ///< If false, fixed capacity
// -------------------------------------------------------------------------------- 

        /**
         * @brief Private constructor for factory methods
         */
        PoolAllocator()
            : Allocator(alignof(max_align_t), 0, ALLOC_INVALID, false),
              arena_(nullptr),
              owns_arena_(false),
              block_size_(0),
              stride_(0),
              blocks_per_chunk_(0),
              cur_(nullptr),
              free_list_(nullptr),
              total_blocks_(0),
              free_blocks_(0),
              grow_enabled_(false) {}
// -------------------------------------------------------------------------------- 

        /**
         * @brief Grow pool by allocating a new chunk from arena
         * @return true on success, false if growth failed or disabled
         */
        bool grow_pool();
// -------------------------------------------------------------------------------- 

        /**
         * @brief Pop a block from the free list
         * @return Pointer to block, or nullptr if free list empty
         */
        void* pop_free();
// -------------------------------------------------------------------------------- 

        /**
         * @brief Push a block onto the free list
         * @param blk Block to add to free list
         */
        void push_free(void* blk);
// ================================================================================ 

    public:
        /**
         * @brief Destructor - cleans up pool and optionally arena
         * 
         * @details If the pool owns its arena, destroys the arena. Otherwise,
         *          just cleans up pool state. All allocated blocks become invalid.
         */
        ~PoolAllocator() noexcept override;
// -------------------------------------------------------------------------------- 

#if ARENA_ENABLE_DYNAMIC
        /**
         * @brief Create a heap-allocated pool with owned arena
         * 
         * @param block_size Size of each block in bytes
         * @param blocks_per_chunk Number of blocks to allocate per growth
         * @param alignment Block alignment (0 = default, must be power of 2)
         * @param arena_initial_bytes Initial arena size
         * @param grow_enabled If true, pool can grow dynamically
         * @param prewarm If true, allocate first chunk immediately
         * 
         * @return Expected containing UniquePtr to pool on success, or error
         * 
         * @details Creates a pool that owns its own dynamic arena. The pool can
         *          optionally grow by allocating additional chunks as needed.
         *          
         *          If prewarm=true, allocates the first chunk immediately so the
         *          first allocation is guaranteed O(1). If prewarm=false and
         *          grow_enabled=false, the pool will be unusable.
         * 
         * @example Resizable pool
         * @code
         * auto pool = cslt::move(cslt::PoolAllocator::Heap(
         *     256,      // 256-byte blocks
         *     64,       // 64 blocks per chunk
         *     0,        // default alignment
         *     16384,    // 16KB initial arena
         *     true,     // can grow
         *     true      // prewarm
         * ).value());
         * 
         * // Can allocate indefinitely - pool grows as needed
         * for (int i = 0; i < 1000; ++i) {
         *     auto ptr = pool->alloc(256);
         * }
         * @endcode
         */
        static Expected<cslt::UniquePtr<PoolAllocator, PoolDeleter>>
        Heap(size_t block_size,
             size_t blocks_per_chunk,
             size_t alignment = 0,
             size_t arena_initial_bytes = 4096,
             size_t min_chunk_bytes = 4096,
             bool grow_enabled = true,
             bool prewarm = true);
#endif
// -------------------------------------------------------------------------------- 

        /**
         * @brief Create a pool using a user-provided buffer
         * 
         * @param buffer Pointer to pre-allocated memory buffer
         * @param buffer_bytes Size of buffer in bytes
         * @param block_size Size of each block in bytes
         * @param alignment Block alignment (0 = default, must be power of 2)
         * 
         * @return Expected containing UniquePtr to pool on success, or error
         * 
         * @details Creates a fixed-capacity pool using a user-provided buffer.
         *          The pool does NOT own the buffer. The buffer must remain valid
         *          for the pool's lifetime.
         *          
         *          The pool creates a static arena within the buffer, then
         *          allocates all blocks from that arena. Cannot grow.
         * 
         * @example Stack-based pool
         * @code
         * uint8_t buffer[8192];
         * auto pool = cslt::move(cslt::PoolAllocator::Stack(
         *     buffer, sizeof(buffer), 128
         * ).value());
         * 
         * // All allocations from stack buffer
         * auto ptr = pool->alloc(128);
         * @endcode
         */
        static Expected<cslt::UniquePtr<PoolAllocator, PoolDeleter>>
        Stack(void* buffer,
              size_t buffer_bytes,
              size_t block_size,
              size_t alignment = 0);
// -------------------------------------------------------------------------------- 

        /**
         * @brief Create a pool using an existing arena
         * 
         * @param arena Arena to allocate from (borrowed, not owned)
         * @param block_size Size of each block in bytes
         * @param blocks_per_chunk Number of blocks to allocate per growth
         * @param alignment Block alignment (0 = default, must be power of 2)
         * @param grow_enabled If true, pool can request more chunks from arena
         * @param prewarm If true, allocate first chunk immediately
         * 
         * @return Expected containing UniquePtr to pool on success, or error
         * 
         * @details Creates a pool that uses a borrowed arena for storage. The pool
         *          does NOT own the arena - the arena must outlive the pool.
         *          
         *          If the arena is dynamic and grow_enabled=true, the pool can
         *          request additional chunks. If the arena is static, grow_enabled
         *          is automatically disabled.
         * 
         * @example Multiple pools sharing arena
         * @code
         * auto arena = cslt::move(cslt::ArenaAllocator::Heap(128 * 1024).value());
         * 
         * // Small block pool
         * auto small_pool = cslt::move(cslt::PoolAllocator::WithArena(
         *     *arena, 64, 128
         * ).value());
         * 
         * // Large block pool
         * auto large_pool = cslt::move(cslt::PoolAllocator::WithArena(
         *     *arena, 1024, 16
         * ).value());
         * 
         * // Both share the same arena memory
         * @endcode
         */
        static Expected<cslt::UniquePtr<PoolAllocator, PoolDeleter>>
        WithArena(ArenaAllocator& arena,
                  size_t block_size,
                  size_t blocks_per_chunk,
                  size_t alignment = 0,
                  bool grow_enabled = false,
                  bool prewarm = true);
// -------------------------------------------------------------------------------- 

        /**
         * @brief Allocate a block from the pool
         * 
         * @param bytes Number of bytes to allocate (must equal block_size)
         * @param zeroed If true, zero-initialize the block
         * 
         * @return Expected containing pointer to block on success, or error
         * 
         * @details Allocates a fixed-size block. The bytes parameter must match
         *          the pool's block_size (validation for safety). First checks
         *          the free-list for recycled blocks (O(1)), then allocates from
         *          the arena if needed.
         * 
         * @retval Expected with pointer on success
         * @retval Expected with ArgumentError if bytes != block_size
         * @retval Expected with MemoryError if out of capacity
         * 
         * @example
         * @code
         * auto pool = cslt::move(cslt::PoolAllocator::Heap(256, 32).value());
         * 
         * auto ptr = pool->alloc(256);  // OK
         * // auto bad = pool->alloc(128);  // ERROR: wrong size
         * @endcode
         */
        Expected<void*> alloc(size_t bytes, bool zeroed = false) override;

        /**
         * @brief Allocate aligned block (same as alloc for pools)
         * 
         * @details For pools, alignment is fixed at creation time. This method
         *          validates the requested alignment matches the pool's stride
         *          alignment, then delegates to alloc().
         */
        Expected<void*> alloc_aligned(size_t bytes,
                                      size_t alignment,
                                      bool zeroed = false) override;

        /**
         * @brief Realloc not supported for pools
         * 
         * @return Expected with FeatureDisabledError
         * 
         * @details Pools allocate fixed-size blocks and cannot resize them.
         *          To "resize", allocate a new block, copy data, and free the old.
         */
        Expected<void*> realloc(void* ptr,
                                size_t old_bytes,
                                size_t new_bytes,
                                bool zeroed = false) override;

        /**
         * @brief Realloc aligned not supported for pools
         * 
         * @return Expected with FeatureDisabledError
         */
        Expected<void*> realloc_aligned(void* ptr,
                                        size_t old_bytes,
                                        size_t new_bytes,
                                        size_t alignment,
                                        bool zeroed = false) override;

        /**
         * @brief Return a block to the free-list
         * 
         * @param ptr Pointer to block to free
         * @param bytes Size of block (must equal block_size)
         * @param alignment Alignment (ignored, but should match pool alignment)
         * 
         * @details Returns the block to the pool's free-list for reuse. This is
         *          O(1) and the block can be immediately reallocated. The pool
         *          does NOT zero or clear the block - caller's responsibility if needed.
         * 
         * @warning Unlike arena allocators, this DOES free the block for reuse.
         *          After calling return_element, the pointer is invalid until
         *          reallocated.
         * 
         * @example
         * @code
         * auto pool = cslt::move(cslt::PoolAllocator::Heap(256, 32).value());
         * 
         * auto ptr = pool->alloc(256).value();
         * // Use ptr...
         * 
         * pool->return_element(ptr, 256);  // Back to free-list
         * // ptr now INVALID
         * 
         * auto ptr2 = pool->alloc(256).value();  // Likely reuses ptr's block
         * @endcode
         */
        void return_element(void* ptr,
                           size_t bytes,
                           size_t alignment = alignof(max_align_t)) override;

        /**
         * @brief Reset pool to initial state
         * 
         * @param trim_extra_chunks If true, free extra arena chunks (if arena supports)
         * 
         * @return true on success
         * 
         * @details Clears the free-list and resets the arena allocation pointer.
         *          All allocated blocks become invalid. If trim_extra_chunks=true
         *          and the pool owns a dynamic arena, extra chunks are freed.
         * 
         * @example
         * @code
         * auto pool = cslt::move(cslt::PoolAllocator::Heap(256, 32).value());
         * 
         * for (int i = 0; i < 100; ++i) {
         *     pool->alloc(256);
         * }
         * 
         * pool->reset();  // All blocks invalid, pool is fresh
         * @endcode
         */
        bool reset(bool trim_extra_chunks = false) override;

        /**
         * @brief Save pool state as checkpoint
         * 
         * @return Opaque checkpoint pointer, or nullptr on failure
         * 
         * @details Saves the current pool state including free-list, arena position,
         *          and block counts. Can later restore() to this point.
         * 
         * @example
         * @code
         * auto pool = cslt::move(cslt::PoolAllocator::Heap(128, 64).value());
         * 
         * auto permanent = pool->alloc(128);
         * void* checkpoint = pool->save();
         * 
         * auto temp1 = pool->alloc(128);
         * auto temp2 = pool->alloc(128);
         * 
         * pool->restore(checkpoint);  // temp1, temp2 invalid
         * @endcode
         */
        void* save() const override;

        /**
         * @brief Restore pool to saved checkpoint
         * 
         * @param checkpoint Checkpoint from previous save()
         * 
         * @return true on success, false on error
         * 
         * @details Restores pool to checkpoint state. All blocks allocated after
         *          the checkpoint become invalid. The checkpoint is freed.
         */
        bool restore(void* checkpoint) override;

        /**
         * @brief Check if pointer was allocated by this pool
         * 
         * @param ptr Pointer to check
         * 
         * @return true if pointer is from this pool's arena
         * 
         * @details Delegates to the underlying arena's is_ptr() check.
         */
        bool is_ptr(void* ptr) const override;

        /**
         * @brief Check if pointer with size is valid for this pool
         * 
         * @param ptr Pointer to check
         * @param bytes Size to validate (should be block_size)
         * 
         * @return true if pointer and size are valid
         */
        bool is_ptr_sized(void* ptr, size_t bytes) const override;

        /**
         * @brief Generate pool statistics
         * 
         * @param buffer Buffer to write statistics to
         * @param buffer_size Size of buffer
         * 
         * @return true if statistics written successfully
         * 
         * @details Writes detailed pool statistics including block size, total
         *          blocks, free blocks, utilization, and arena information.
         */
        bool stats(char* buffer, size_t buffer_size) const override;

        // ------------------------------------------------------------------------
        // Pool-Specific Methods
        // ------------------------------------------------------------------------

        /**
         * @brief Get the pool's fixed block size
         * 
         * @return Block size in bytes
         */
        size_t block_size() const noexcept { return block_size_; }

        /**
         * @brief Get the actual stride (aligned block size)
         * 
         * @return Stride in bytes
         */
        size_t stride() const noexcept { return stride_; }

        /**
         * @brief Get total number of blocks available
         * 
         * @return Total blocks (allocated + free)
         */
        size_t total_blocks() const noexcept { return total_blocks_; }

        /**
         * @brief Get number of free blocks available
         * 
         * @return Free blocks in free-list
         */
        size_t free_blocks() const noexcept { return free_blocks_; }

        /**
         * @brief Get number of allocated blocks
         * 
         * @return Blocks currently in use
         */
        size_t allocated_blocks() const noexcept {
            return total_blocks_ - free_blocks_;
        }

        /**
         * @brief Check if pool can grow
         * 
         * @return true if growth enabled
         */
        bool can_grow() const noexcept { return grow_enabled_; }

        /**
         * @brief Enable or disable pool growth
         * 
         * @param enable If true, enable growth; if false, disable
         * 
         * @details Only works if pool owns arena or arena supports growth.
         */
        void toggle_grow(bool enable) noexcept;
    };
// ================================================================================ 
// ================================================================================ 

    inline void PoolDeleter::operator()(PoolAllocator* pool) const noexcept {
        if (!pool) return;

        bool owns_arena = pool->owns_arena_;
        ArenaAllocator* arena = pool->arena_;

        // Call destructor
        pool->~PoolAllocator();

        // If pool owned its arena, the arena was created by the pool
        // and needs to be destroyed
        if (owns_arena && arena) {
            ArenaDeleter{}(arena);
        }
    }

// ================================================================================ 
// ================================================================================ 
} /* cslt namespace */
// ================================================================================ 
// ================================================================================ 
#endif /* allocator_HPP */
// ================================================================================
// ================================================================================
// eof
