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
    inline size_t next_pow2(size_t x) noexcept {
        if (x == 0) return 0;
        if (is_pow2(x)) return x;
        
        // Check if x is too large to round up to a power of 2
        // The largest power of 2 that fits in size_t is SIZE_MAX / 2 + 1
        // (which is 0x8000... with only the high bit set)
        if (x > (SIZE_MAX / 2 + 1)) {
            return 0;  // Overflow - no power of 2 can represent this
        }
        
        size_t power = 1;
        while (power < x) {
            // Check overflow before shifting
            if (power > SIZE_MAX / 2) {
                return 0;  // Would overflow on next shift
            }
            power <<= 1;
        }
        return power;
    }
// ================================================================================ 
// ================================================================================ 
#if ARENA_ENABLE_DYNAMIC
    class BuddyAllocator;
#endif 
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
        friend struct ArenaDeleter;
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
                        
        BuddyAllocator* buddy_owner_ = nullptr;  ///< Non-null only for WithBuddy arenas
        size_t          backing_bytes_ = 0;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Locate a chunk within the arena's chunk chain
         *
         * @param target Pointer to the chunk to locate
         * @param out_prev Optional output parameter that receives the previous chunk
         *                 in the chain (nullptr if target is the head or not found)
         *
         * @return Pointer to the matching chunk if found, nullptr otherwise
         *
         * @details Searches the arena's internal linked list of chunks for the specified
         *          @p target chunk. If found, returns the matching chunk pointer and,
         *          if @p out_prev is provided, also returns the preceding chunk in the
         *          chain.
         *
         *          This function performs a linear walk starting from the head chunk
         *          and compares chunk addresses directly. It does not modify allocator
         *          state and does not perform any memory allocation.
         *
         *          This helper is primarily used for:
         *          - Validating checkpoint integrity during restore()
         *          - Determining whether a saved checkpoint refers to a still-live chunk
         *          - Safely unlinking and freeing chunks beyond a checkpoint
         *
         *          If the target chunk is not part of the current arena chain (e.g.,
         *          corrupted checkpoint, stale pointer, or foreign chunk), the function
         *          returns nullptr and no state is modified.
         *
         * @note Passing a null @p target always returns nullptr.
         *
         * @note This function is for internal use only and assumes the arena's chunk
         *       list is well-formed (acyclic, properly linked).
         *
         * @par Example (internal usage in restore):
         * @code{.cpp}
         * Chunk* prev = nullptr;
         * Chunk* hit = find_chunk_in_chain(cp->chunk, &prev);
         *
         * if (!hit) {
         *     // Checkpoint refers to a chunk no longer owned by this arena
         *     return false;
         * }
         * @endcode
         *
         * @see restore() Uses this function to validate checkpoints
         * @see reset() May rely on chunk chain structure
         * @internal
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
         * @brief Create a fixed-capacity arena backed by a BuddyAllocator allocation
         *
         * @param buddy Buddy allocator to allocate the arena region from (must not be nullptr)
         * @param bytes Total bytes to request from the buddy allocator for the entire arena region
         * @param base_align_in Per-arena base alignment for allocations (0 = default)
         *
         * @return Expected containing UniquePtr to arena on success, or error on failure
         *
         * @details Performs a single allocation from @p buddy and constructs the ArenaAllocator
         *          in-place at the returned base pointer. The arena is fixed-capacity:
         *          - No resizing/growth (resize disabled)
         *          - The buddy allocator retains ownership of the backing region
         *          - Destruction returns the entire region back to the buddy allocator
         *
         *          Layout within the buddy allocation:
         *          [ArenaAllocator][padding][Chunk][padding][data...]
         *
         *          - ArenaAllocator is placed at the beginning of the buddy region.
         *          - Chunk header is aligned to alignof(Chunk).
         *          - Data region is aligned to base_align (>= alignof(max_align_t), pow2).
         *
         * @retval Expected with UniquePtr on success
         * @retval Expected with ArgumentError if buddy is null or bytes too small
         * @retval Expected with MemoryError if buddy allocation fails
         * @retval Expected with AlignmentError if alignment normalization fails
         * @retval Expected with LengthOverflowError on overflow in address arithmetic
         *
         * @warning The returned arena lives inside buddy-owned memory. Never free it with
         *          ::operator delete. Always use ArenaDeleter / UniquePtr cleanup.
         *
         * @see ArenaDeleter Returns memory to buddy for WithBuddy arenas
         */
        static Expected<cslt::UniquePtr<ArenaAllocator, ArenaDeleter>>
        WithBuddy(BuddyAllocator& buddy,
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
     * @class PoolAllocator
     * @brief Fixed-size block allocator with O(1) allocation and deallocation
     * 
     * @details PoolAllocator manages memory as a collection of uniformly-sized blocks,
     *          providing constant-time allocation and deallocation through free-list
     *          recycling. All allocations return blocks of exactly the same size,
     *          making it ideal for object pools and frequently allocated/deallocated
     *          data structures.
     * 
     *          The pool carves blocks from contiguous memory slices backed by an arena.
     *          Freed blocks are returned to a linked free-list for O(1) reuse.
     *          Optionally, the pool can request additional slices when capacity is
     *          exhausted.
     * 
     * @par Basic Usage
     * @code{.cpp}
     * // Create pool for 256-byte blocks
     * auto pool = PoolAllocator::Heap(256, 32).value();
     * 
     * // Allocate block
     * auto ptr = pool->alloc_pool(false);
     * 
     * // Return block for reuse
     * pool->return_element(ptr.value(), 256);
     * @endcode
     * 
     * @par Object Pool Pattern
     * @code{.cpp}
     * struct Particle { float x, y, z; };
     * auto pool = PoolAllocator::Heap(sizeof(Particle), 1000).value();
     * 
     * // Allocate and construct
     * void* mem = pool->alloc_pool().value();
     * Particle* p = new (mem) Particle();
     * 
     * // Destroy and return
     * p->~Particle();
     * pool->return_element(mem, sizeof(Particle));
     * @endcode
     * 
     * @note Block size must be at least sizeof(void*) for free-list management
     * @note Not thread-safe - requires external synchronization for concurrent access
     * 
     * @see Heap() Create pool with owned heap-allocated arena
     * @see WithArena() Create pool sharing an existing arena
     * @see Stack() Create pool using user-provided buffer
     */
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
         * @brief Allocate a block from the pool (base class interface - prefer alloc_pool)
         * 
         * @param bytes Number of bytes to allocate (must equal block_size, validated but ignored)
         * @param zeroed If true, zero-initialize the block
         * 
         * @return Expected containing pointer to block on success, or error
         * 
         * @note **Prefer using alloc_pool() for direct PoolAllocator usage.**
         *       This method exists to satisfy the Allocator base class interface contract.
         *       Since pools allocate fixed-size blocks, the bytes parameter is somewhat
         *       artificial - it must equal block_size but doesn't control allocation size.
         * 
         * @details Allocates a fixed-size block from the pool. First checks the free-list
         *          for recycled blocks (O(1)), then carves from the current memory slice,
         *          and finally attempts to grow the pool if needed and enabled.
         * 
         * @retval Expected with pointer on success
         * @retval Expected with ArgumentError if bytes != block_size (validation)
         * @retval Expected with CapacityOverflowError if pool full and cannot grow
         * @retval Expected with BadAllocError if growth fails
         * @retval Expected with StateCorruptError if pool state is corrupted
         * 
         * @see alloc_pool() The preferred pool-specific allocation method
         * @see return_element() To return blocks to the free-list for reuse
         * 
         * @example
         * @code
         * // When using through base class pointer (polymorphic use):
         * Allocator* allocator = pool.get();
         * auto ptr = allocator->alloc(256, false);  // Must use base interface
         * 
         * // When using PoolAllocator directly (prefer alloc_pool instead):
         * auto pool = cslt::move(cslt::PoolAllocator::Heap(256, 32).value());
         * auto ptr = pool->alloc(256, false);       // Works, but alloc_pool(false) is clearer
         * @endcode
         */
        Expected<void*> alloc(size_t bytes, bool zeroed = false) override;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Allocate a block from the pool (pool-specific interface - PREFERRED)
         * 
         * @param zeroed If true, zero-initialize the block (default: false)
         * 
         * @return Expected containing pointer to block on success, or error
         * 
         * @details **This is the recommended way to allocate from a PoolAllocator.**
         *          Unlike the base class alloc() method, this interface doesn't require
         *          a size parameter since pools always allocate fixed-size blocks of
         *          block_size bytes.
         * 
         *          Allocation strategy:
         *          1. Check free-list for recycled blocks (O(1) pop)
         *          2. If free-list empty, carve from current memory slice
         *          3. If slice exhausted and growth enabled, request new chunk
         *          4. If growth disabled or fails, return error
         * 
         * @retval Expected with pointer to allocated block (block_size bytes)
         * @retval Expected with CapacityOverflowError if pool full and cannot grow
         * @retval Expected with BadAllocError if growth fails
         * @retval Expected with StateCorruptError if pool state is corrupted
         * 
         * @note The returned block is always block_size bytes and aligned to the
         *       pool's stride alignment (set at construction).
         * 
         * @see return_element() To return blocks for reuse in the free-list
         * @see block_size() To query the pool's fixed block size
         * @see can_grow() To check if pool can expand when capacity is reached
         * 
         * @example
         * @code
         * // Create a pool for 256-byte blocks
         * auto pool = cslt::move(cslt::PoolAllocator::Heap(256, 32).value());
         * 
         * // Allocate uninitialized block
         * auto ptr1 = pool->alloc_pool();
         * if (!ptr1.hasValue()) {
         *     // Handle error
         * }
         * 
         * // Allocate zero-initialized block
         * auto ptr2 = pool->alloc_pool(true);
         * 
         * // Return block to free-list for reuse
         * pool->return_element(ptr1.value(), 256);
         * 
         * // Next allocation will reuse the freed block
         * auto ptr3 = pool->alloc_pool();  // Reuses ptr1's block (O(1))
         * @endcode
         * 
         * @example
         * @code
         * // Non-growable pool with fixed capacity
         * auto pool = cslt::move(PoolAllocator::Heap(
         *     128,    // block_size
         *     64,     // blocks_per_chunk
         *     0,      // default alignment
         *     10240,  // arena_initial_bytes
         *     4096,   // min_chunk_bytes
         *     false,  // grow_enabled = false
         *     true    // prewarm = true
         * ).value());
         * 
         * // Can allocate exactly 64 blocks
         * for (int i = 0; i < 64; ++i) {
         *     auto ptr = pool->alloc_pool();
         *     assert(ptr.hasValue());
         * }
         * 
         * // 65th allocation fails
         * auto ptr = pool->alloc_pool();
         * assert(!ptr.hasValue());  // CapacityOverflowError
         * @endcode
         */
        Expected<void*> alloc_pool(bool zeroed = false);
// -------------------------------------------------------------------------------- 

        /**
         * @brief Allocate aligned block from pool (base class interface - prefer alloc_aligned_pool)
         * 
         * @param bytes Number of bytes to allocate (must equal block_size, validated but ignored)
         * @param alignment Required alignment (must equal default_alignment, validated)
         * @param zeroed If true, zero-initialize the block
         * 
         * @return Expected containing pointer to aligned block on success, or error
         * 
         * @note **Prefer using alloc_aligned_pool() for direct PoolAllocator usage.**
         *       This method exists to satisfy the Allocator base class interface contract.
         *       For pools, both size and alignment are fixed at creation time, making
         *       these parameters redundant (though validated for safety).
         * 
         * @details For PoolAllocator, alignment is fixed at pool creation and cannot be
         *          changed per-allocation. This method validates that the requested size
         *          and alignment match the pool's configuration, then delegates to alloc().
         *          All blocks from a pool have the same size and alignment.
         * 
         * @retval Expected with pointer on success
         * @retval Expected with ArgumentError if bytes != block_size
         * @retval Expected with AlignmentError if alignment != default_alignment
         * @retval Expected with CapacityOverflowError if pool full and cannot grow
         * 
         * @see alloc_aligned_pool() The preferred pool-specific aligned allocation method
         * @see alloc_pool() Pool-specific allocation without redundant parameters
         * 
         * @example
         * @code
         * // When using through base class pointer (polymorphic use):
         * Allocator* allocator = pool.get();
         * auto ptr = allocator->alloc_aligned(256, 64, false);  // Must use base interface
         * 
         * // When using PoolAllocator directly (prefer alloc_aligned_pool instead):
         * auto pool = cslt::move(cslt::PoolAllocator::Heap(256, 32, 64).value());
         * auto ptr = pool->alloc_aligned(256, 64, false);  // Works, but verbose
         * auto ptr = pool->alloc_aligned_pool(64, false);  // Clearer - preferred
         * @endcode
         */
        Expected<void*> alloc_aligned(size_t bytes,
                                      size_t alignment,
                                      bool zeroed = false) override;
// -------------------------------------------------------------------------------- 

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
// -------------------------------------------------------------------------------- 

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
// -------------------------------------------------------------------------------- 

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
// -------------------------------------------------------------------------------- 

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
// -------------------------------------------------------------------------------- 

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
// -------------------------------------------------------------------------------- 

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
// -------------------------------------------------------------------------------- 

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
// -------------------------------------------------------------------------------- 

        /**
         * @brief Check if pointer with size is valid for this pool
         * 
         * @param ptr Pointer to check
         * @param bytes Size to validate (should be block_size)
         * 
         * @return true if pointer and size are valid
         */
        bool is_ptr_sized(void* ptr, size_t bytes) const override;
// -------------------------------------------------------------------------------- 

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
// -------------------------------------------------------------------------------- 

        /**
         * @brief Get the pool's fixed block size
         * 
         * @return Block size in bytes
         */
        size_t block_size() const noexcept { return block_size_; }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Get the actual stride (aligned block size)
         * 
         * @return Stride in bytes
         */
        size_t stride() const noexcept { return stride_; }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Get total number of blocks available
         * 
         * @return Total blocks (allocated + free)
         */
        size_t total_blocks() const noexcept { return total_blocks_; }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Get number of free blocks available
         * 
         * @return Free blocks in free-list
         */
        size_t free_blocks() const noexcept { return free_blocks_; }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Get number of allocated blocks
         * 
         * @return Blocks currently in use
         */
        size_t allocated_blocks() const noexcept {
            return total_blocks_ - free_blocks_;
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Check if pool can grow
         * 
         * @return true if growth enabled
         */
        bool can_grow() const noexcept { return grow_enabled_; }
// -------------------------------------------------------------------------------- 

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

        /**
         * @struct PoolDeleter
         * @brief Custom deleter for PoolAllocator unique pointers
         * 
         * @details This deleter properly cleans up PoolAllocator instances created by
         *          factory methods (Heap, WithArena, Stack). It handles the complex
         *          ownership relationships between pools and their backing arenas:
         * 
         *          1. Calls the pool's destructor (cleans up pool state)
         *          2. Conditionally destroys the arena based on ownership
         * 
         *          Arena ownership patterns:
         *          - **Heap()**: Pool owns arena → ArenaDeleter frees arena and memory
         *          - **WithArena()**: Pool borrows arena → Arena not deleted (caller owns it)
         *          - **Stack()**: Pool owns arena object → ArenaDeleter destructs arena
         *                         (but doesn't free buffer, user owns it)
         * 
         *          The deleter respects the `owns_arena_` flag to determine whether the
         *          pool created its own arena (Heap/Stack) or borrowed one (WithArena).
         * 
         * @note This deleter is noexcept and safe to call with nullptr.
         * 
         * @example Automatic via UniquePtr (typical usage)
         * @code
         * {
         *     auto pool = cslt::move(cslt::PoolAllocator::Heap(256, 32).value());
         *     // Use pool...
         * } // PoolDeleter automatically called here, cleans up pool and arena
         * @endcode
         * 
         * @example Heap pool (pool owns arena)
         * @code
         * {
         *     auto pool = cslt::move(PoolAllocator::Heap(256, 32).value());
         *     // owns_arena_ = true
         *     // When pool destroyed: pool destructor + ArenaDeleter frees arena
         * }
         * @endcode
         * 
         * @example WithArena pool (borrowed arena)
         * @code
         * auto arena = cslt::move(ArenaAllocator::Heap(16384).value());
         * {
         *     auto pool = cslt::move(PoolAllocator::WithArena(*arena, 256, 32).value());
         *     // owns_arena_ = false
         *     // When pool destroyed: only pool destructor called, arena untouched
         * }
         * // Arena still alive and usable
         * @endcode
         * 
         * @example Stack pool (pool owns arena object, user owns buffer)
         * @code
         * {
         *     uint8_t buffer[4096];
         *     {
         *         auto pool = cslt::move(PoolAllocator::Stack(buffer, 4096, 128).value());
         *         // owns_arena_ = true, but mem_type_ = STATIC
         *         // When pool destroyed: pool destructor + arena destructor called
         *         //                     (ArenaDeleter won't free buffer - it's STATIC)
         *     }
         *     // buffer still valid, can reuse
         * }
         * @endcode
         * 
         * @example Manual construction (not typical)
         * @code
         * PoolAllocator* raw_pool = // ... created via factory ...;
         * cslt::UniquePtr<PoolAllocator, PoolDeleter> pool(raw_pool, PoolDeleter{});
         * // Automatic cleanup when pool goes out of scope
         * @endcode
         * 
         * @warning Never call ::operator delete directly on a PoolAllocator created
         *          by a factory method. Always use PoolDeleter or let UniquePtr handle it.
         * 
         * @warning For Stack pools, ensure the buffer outlives the pool. The deleter
         *          doesn't free the buffer (user owns it), but the pool must be destroyed
         *          before the buffer goes out of scope.
         * 
         * @see PoolAllocator::Heap() Creates pool with owned arena
         * @see PoolAllocator::WithArena() Creates pool with borrowed arena
         * @see PoolAllocator::Stack() Creates pool with owned arena in user buffer
         * @see ArenaDeleter For arena cleanup details
         */
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

    class FreeListAllocator;

    /**
     * @struct FreeListDeleter
     * @brief Custom deleter for FreeListAllocator unique pointers
     * 
     * @details This deleter properly cleans up FreeListAllocator instances created by
     *          factory methods (Heap, WithArena, Stack). It handles the complex
     *          ownership relationships between freelists and their backing arenas:
     * 
     *          Arena ownership patterns:
     *          - **Heap()**: FreeList owns arena → ArenaDeleter frees arena and memory
     *          - **WithArena()**: FreeList borrows arena → Arena not deleted (caller owns it)
     *          - **Stack()**: FreeList owns arena object → ArenaDeleter destructs arena
     *                         (but doesn't free buffer, user owns it)
     * 
     * @see FreeListAllocator::Heap()
     * @see FreeListAllocator::WithArena()
     * @see FreeListAllocator::Stack()
     * @see ArenaDeleter
     */
    struct FreeListDeleter {
        void operator()(FreeListAllocator* freelist) const noexcept;
    };

// ================================================================================
// FreeListAllocator Class
// ================================================================================

    /**
     * @class FreeListAllocator
     * @brief Variable-size block allocator with automatic coalescing to reduce fragmentation
     * 
     * @details FreeListAllocator manages variable-sized allocations from a contiguous
     *          memory region using a linked list of free blocks. Unlike PoolAllocator
     *          which handles fixed-size blocks, FreeListAllocator can allocate blocks
     *          of any size up to available capacity. Freed blocks are automatically
     *          coalesced with adjacent free blocks to reduce external fragmentation.
     * 
     *          Each allocated block carries a small header storing its size and alignment
     *          offset. Free blocks maintain a linked list for efficient reuse. The
     *          allocator uses a best-fit search strategy to find suitable free blocks.
     * 
     * @par Basic Usage
     * @code{.cpp}
     * // Create freelist with 4KB capacity
     * auto freelist = FreeListAllocator::Heap(4096, 0, false).value();
     * 
     * // Allocate variable-sized blocks
     * auto ptr1 = freelist->alloc(256, false);
     * auto ptr2 = freelist->alloc(512, false);
     * auto ptr3 = freelist->alloc(128, false);
     * 
     * // Free blocks (automatically coalesces adjacent blocks)
     * freelist->return_element(ptr2.value(), 512);
     * freelist->return_element(ptr1.value(), 256);
     * @endcode
     * 
     * @par Resize Support
     * @code{.cpp}
     * auto freelist = FreeListAllocator::Heap(8192).value();
     * 
     * // Initial allocation
     * auto ptr = freelist->alloc(128, false).value();
     * 
     * // Grow allocation (allocates new block, copies data, frees old)
     * auto new_ptr = freelist->realloc(ptr, 128, 512, false);
     * 
     * // Can also resize with specific alignment
     * auto aligned = freelist->realloc_aligned(new_ptr.value(), 512, 1024, 64, true);
     * @endcode
     * 
     * @note Allocations have small per-block overhead for metadata (FreeListHeader)
     * @note Not thread-safe - requires external synchronization for concurrent access
     * @note Best suited for general-purpose allocation with varying sizes
     * 
     * @see Heap() Create freelist with owned heap-allocated arena
     * @see WithArena() Create freelist sharing an existing arena
     * @see Stack() Create freelist using user-provided buffer
     */
    class FreeListAllocator : public Allocator {
        // Friend declaration so deleter can access private members
        friend struct FreeListDeleter;

    private:
        /**
         * @brief Free block node for linked list management
         * @internal
         */
        struct FreeBlock {
            size_t size;          ///< Size of this free block
            FreeBlock* next;      ///< Next free block in list
        };

        /**
         * @brief Header stored before each allocated block
         * @internal
         */
        struct FreeListHeader {
            size_t block_size;    ///< Total size including header and padding
            size_t offset;        ///< Distance from block start to user pointer
        };

        // Member variables
        FreeBlock*       head_;          ///< Head of free list
        uint8_t*         cur_;           ///< High-water mark
        size_t           len_;           ///< Current usage
        void*            memory_;        ///< Start of memory region
        ArenaAllocator*  arena_;         ///< Parent arena
        bool             owns_arena_;    ///< Arena ownership flag

        // Private constructor (use factory methods)
        FreeListAllocator();

        // Helper methods
        static size_t min_request();
// ================================================================================ 

    public:
        /**
         * @brief Destructor for FreeListAllocator
         * 
         * @details Cleans up the freelist allocator's internal state. The destructor
         *          is intentionally minimal because memory management is handled by
         *          the FreeListDeleter custom deleter, which determines whether to
         *          destroy the backing arena based on ownership.
         * 
         *          The destructor simply nulls internal pointers to prevent dangling
         *          references. Actual memory deallocation (if any) is performed by
         *          FreeListDeleter after calling this destructor.
         * 
         * @note This destructor is called by FreeListDeleter, not directly by users.
         *       Users should let UniquePtr manage the freelist lifetime.
         * 
         * @see FreeListDeleter For the cleanup logic that follows destruction
         */ 
        ~FreeListAllocator() noexcept override;

        // Factory methods cannot be copied or moved after construction
        FreeListAllocator(const FreeListAllocator&) = delete;
        FreeListAllocator& operator=(const FreeListAllocator&) = delete;
        FreeListAllocator(FreeListAllocator&&) = delete;
        FreeListAllocator& operator=(FreeListAllocator&&) = delete;

        // ============================================================================
        // Factory Methods
        // ============================================================================

#if ARENA_ENABLE_DYNAMIC
        /**
         * @brief Create a dynamically-backed freelist allocator (PREFERRED for heap use)
         * 
         * @param bytes Minimum usable payload size (excluding metadata). If 0, defaults
         *              to a reasonable minimum (4096 bytes). Must be at least
         *              sizeof(FreeBlock).
         * @param alignment Desired alignment for allocations (0 = default = alignof(max_align_t)).
         *                  Must be power of 2. The effective alignment is always at least
         *                  alignof(max_align_t).
         * @param resize Whether the underlying arena is permitted to grow. Currently,
         *               the freelist itself remains fixed-size after construction.
         * 
         * @return Expected containing UniquePtr to freelist on success, or error
         * 
         * @retval Expected with UniquePtr<FreeListAllocator> on success
         * @retval Expected with ArgumentError if bytes < minimum size
         * @retval Expected with AlignmentError if alignment is not power of 2
         * @retval Expected with LengthOverflowError if size calculations overflow
         * @retval Expected with MemoryError if arena allocation fails
         * 
         * @details Creates a freelist with its own dedicated heap-allocated arena.
         *          The freelist owns the arena and will destroy it on cleanup via
         *          FreeListDeleter. All memory managed by the freelist is carved
         *          from this owned arena.
         * 
         *          The actual usable capacity may exceed the requested bytes,
         *          depending on how the underlying arena rounds allocations and
         *          alignment padding requirements.
         * 
         * @note The freelist owns the arena. Call reset() on the UniquePtr to
         *       release all associated memory.
         * 
         * @note Requires ARENA_ENABLE_DYNAMIC to be enabled at compile time.
         * 
         * @warning All pointers obtained from this freelist become invalid once
         *          the freelist is destroyed.
         * 
         * @par Example - Basic heap freelist
         * @code{.cpp}
         * #include "FreeListAllocator.hpp"
         * 
         * // Create 8KB freelist
         * auto result = cslt::FreeListAllocator::Heap(8192, 0, false);
         * if (!result.hasValue()) {
         *     // Handle error
         *     std::cerr << result.error().what() << std::endl;
         *     return;
         * }
         * 
         * auto freelist = cslt::move(result.value());
         * 
         * // Allocate variable-sized blocks
         * auto ptr1 = freelist->alloc(256, false);
         * auto ptr2 = freelist->alloc(512, true);  // Zero-initialized
         * auto ptr3 = freelist->alloc(128, false);
         * 
         * // Use allocations...
         * 
         * // Free blocks (will be coalesced if adjacent)
         * freelist->return_element(ptr1.value(), 256);
         * freelist->return_element(ptr3.value(), 128);
         * 
         * // Freelist and arena automatically destroyed when freelist goes out of scope
         * @endcode
         * 
         * @par Example - Custom alignment
         * @code{.cpp}
         * // Create freelist with 64-byte alignment
         * auto freelist = cslt::FreeListAllocator::Heap(4096, 64, false).value();
         * 
         * // All allocations will be at least 64-byte aligned
         * auto ptr = freelist->alloc(256, false);
         * assert(reinterpret_cast<uintptr_t>(ptr.value()) % 64 == 0);
         * @endcode
         * 
         * @see WithArena() For sharing an arena between multiple allocators
         * @see Stack() For stack/embedded scenarios with user-provided buffers
         * @see FreeListDeleter For cleanup behavior
         */
        static Expected<UniquePtr<FreeListAllocator, FreeListDeleter>>
        Heap(size_t bytes,
             size_t alignment = 0,
             bool resize = false);
#endif
// -------------------------------------------------------------------------------- 

        /**
         * @brief Create a freelist using a borrowed arena (PREFERRED for shared arena)
         * 
         * @param arena Reference to existing arena (must outlive freelist). The arena
         *              is borrowed, not owned. Multiple freelists can share the same
         *              arena.
         * @param bytes Usable payload size to allocate from arena (excluding metadata).
         *              Must be at least sizeof(FreeBlock).
         * @param alignment Desired alignment for allocations (0 = default = alignof(max_align_t)).
         *                  Must be power of 2.
         * 
         * @return Expected containing UniquePtr to freelist on success, or error
         * 
         * @retval Expected with UniquePtr<FreeListAllocator> on success
         * @retval Expected with ArgumentError if bytes < minimum size
         * @retval Expected with AlignmentError if alignment is not power of 2
         * @retval Expected with LengthOverflowError if size calculations overflow
         * @retval Expected with MemoryError if arena cannot supply requested memory
         * 
         * @details Creates a freelist within an existing arena's memory space.
         *          The freelist does NOT own the arena - the caller or another
         *          component is responsible for the arena's lifetime. Multiple
         *          allocators can share the same arena for efficient memory use.
         * 
         *          The freelist inherits its memory type (STATIC/DYNAMIC) from the
         *          parent arena. When the freelist is destroyed, the arena remains
         *          valid and usable by other allocators.
         * 
         * @note Arena must remain valid for the freelist's entire lifetime.
         * 
         * @note The freelist does NOT own the arena. Destroying the freelist
         *       does not affect the arena.
         * 
         * @warning Destroying the arena while freelists still reference it results
         *          in undefined behavior.
         * 
         * @par Example - Multiple freelists sharing one arena
         * @code{.cpp}
         * #include "FreeListAllocator.hpp"
         * #include "ArenaAllocator.hpp"
         * 
         * // Create parent arena
         * auto arena = cslt::ArenaAllocator::Heap(64 * 1024).value();
         * 
         * // Create multiple freelists sharing the arena
         * auto freelist1 = cslt::FreeListAllocator::WithArena(*arena, 8192, 0).value();
         * auto freelist2 = cslt::FreeListAllocator::WithArena(*arena, 4096, 0).value();
         * auto freelist3 = cslt::FreeListAllocator::WithArena(*arena, 2048, 0).value();
         * 
         * // All freelists allocate from the same 64KB arena
         * auto ptr1 = freelist1->alloc(512, false);
         * auto ptr2 = freelist2->alloc(256, false);
         * auto ptr3 = freelist3->alloc(128, false);
         * 
         * // Freelists can be destroyed independently
         * freelist1.reset();
         * freelist2.reset();
         * 
         * // Arena still valid for freelist3
         * auto ptr4 = freelist3->alloc(64, false);
         * 
         * // Arena destroyed last (after all freelists)
         * @endcode
         * 
         * @par Example - Mixing allocator types
         * @code{.cpp}
         * auto arena = cslt::ArenaAllocator::Heap(128 * 1024).value();
         * 
         * // Mix different allocator types on same arena
         * auto freelist = cslt::FreeListAllocator::WithArena(*arena, 16384, 0).value();
         * auto pool = cslt::PoolAllocator::WithArena(*arena, 256, 64, 0, true, true).value();
         * 
         * // Both share the same underlying memory efficiently
         * auto var_sized = freelist->alloc(1024, false);
         * auto fixed_sized = pool->alloc_pool(false);
         * @endcode
         * 
         * @see Heap() For standalone freelists with owned arenas
         * @see Stack() For stack-based freelists
         */
        static Expected<UniquePtr<FreeListAllocator, FreeListDeleter>>
        WithArena(ArenaAllocator& arena,
                  size_t bytes,
                  size_t alignment = 0);
// -------------------------------------------------------------------------------- 

        /**
         * @brief Create a freelist over a user-supplied buffer (PREFERRED for embedded)
         * 
         * @param buffer Pointer to user-owned memory buffer. Must not be NULL.
         *               The buffer must remain valid for the freelist's entire lifetime.
         * @param buffer_bytes Total size of buffer in bytes. Must be large enough to
         *                     contain FreeListAllocator header, arena header, and at
         *                     least one FreeBlock.
         * @param alignment Required alignment for allocations (0 = default = alignof(max_align_t)).
         *                  Must be power of 2.
         * 
         * @return Expected containing UniquePtr to freelist on success, or error
         * 
         * @retval Expected with UniquePtr<FreeListAllocator> on success
         * @retval Expected with ArgumentError if buffer is NULL or buffer_bytes == 0
         * @retval Expected with ArgumentError if buffer too small for structures
         * @retval Expected with AlignmentError if alignment is not power of 2
         * @retval Expected with MemoryError if arena initialization fails
         * 
         * @details Constructs a freelist entirely within a caller-provided memory buffer.
         *          No heap allocation is performed. The freelist does NOT take ownership
         *          of the buffer; the caller is responsible for ensuring that the buffer
         *          remains valid for the full lifetime of the freelist.
         * 
         *          Internally, this creates a non-owning STATIC arena over the supplied
         *          buffer and then carves the freelist allocator from that arena. The
         *          freelist owns the arena object but not the underlying buffer.
         * 
         * @note The freelist does NOT own the buffer. Destroying or resetting the
         *       freelist does not free the buffer.
         * 
         * @note Well suited for embedded systems, scratch allocators, or environments
         *       where dynamic allocation is restricted.
         * 
         * @warning Buffer must outlive the freelist. Destroying the buffer before
         *          the freelist results in undefined behavior.
         * 
         * @par Example - Stack-based freelist
         * @code{.cpp}
         * #include "FreeListAllocator.hpp"
         * 
         * void process_data() {
         *     uint8_t buffer[4096];  // Stack-allocated buffer
         *     
         *     auto result = cslt::FreeListAllocator::Stack(buffer, sizeof(buffer), 0);
         *     if (!result.hasValue()) {
         *         // Handle error
         *         return;
         *     }
         *     
         *     auto freelist = cslt::move(result.value());
         *     
         *     // Allocate from stack buffer
         *     auto ptr1 = freelist->alloc(256, false);
         *     auto ptr2 = freelist->alloc(512, false);
         *     
         *     // Use allocations...
         *     
         *     freelist->return_element(ptr1.value(), 256);
         *     
         *     // Freelist automatically cleaned up at scope exit
         * }  // Buffer still valid after freelist destruction
         * @endcode
         * 
         * @par Example - Embedded system with static buffer
         * @code{.cpp}
         * // Global or static buffer
         * static uint8_t g_memory_pool[8192];
         * 
         * void init_allocator() {
         *     auto freelist = cslt::FreeListAllocator::Stack(
         *         g_memory_pool, 
         *         sizeof(g_memory_pool),
         *         16  // 16-byte alignment
         *     ).value();
         *     
         *     // Use freelist for temporary allocations
         *     void* temp = freelist->alloc(128, true);
         *     // ... process ...
         *     freelist->return_element(temp, 128);
         *     
         *     // Reset for reuse
         *     freelist->reset();
         * }
         * @endcode
         * 
         * @par Example - Heap-allocated buffer (user controls lifetime)
         * @code{.cpp}
         * uint8_t* buffer = new uint8_t[16384];
         * 
         * {
         *     auto freelist = cslt::FreeListAllocator::Stack(buffer, 16384, 0).value();
         *     
         *     // Use freelist...
         *     auto ptr = freelist->alloc(1024, false);
         *     
         * }  // Freelist destroyed, but buffer remains valid
         * 
         * // Buffer can be reused or freed
         * delete[] buffer;
         * @endcode
         * 
         * @see Heap() For heap-allocated freelists
         * @see WithArena() For sharing arenas
         */
        static Expected<UniquePtr<FreeListAllocator, FreeListDeleter>>
        Stack(void* buffer,
              size_t buffer_bytes,
              size_t alignment = 0);
// -------------------------------------------------------------------------------- 

        /**
         * @brief Allocate variable-size block from freelist
         * 
         * @param bytes Number of bytes to allocate. Must be greater than 0.
         * @param zeroed If true, zero-initialize the allocated block before returning.
         * 
         * @return Expected containing pointer to allocated block on success, or error
         * 
         * @retval Expected with void* pointer on success
         * @retval Expected with ArgumentError if bytes == 0
         * @retval Expected with AlignmentError if alignment validation fails
         * @retval Expected with CapacityOverflowError if no suitable free block available
         * 
         * @details Allocates a block of the requested size from the freelist using
         *          a first-fit search strategy. The allocator searches the free list
         *          for the first block large enough to satisfy the request, accounting
         *          for alignment padding and header overhead.
         * 
         *          Internally, this delegates to alloc_aligned() using the freelist's
         *          default alignment. Each allocated block carries a small header
         *          (FreeListHeader) storing its total size and offset for proper
         *          deallocation.
         * 
         *          If a free block is larger than needed, it may be split into an
         *          allocated portion and a new smaller free block. If the remainder
         *          would be too small (< sizeof(FreeBlock)), the entire block is
         *          consumed to avoid creating unusable fragments.
         * 
         * @note The actual memory consumed may exceed 'bytes' due to:
         *       - FreeListHeader (stored before user pointer)
         *       - Alignment padding
         *       - Full block consumption when remainder is too small
         * 
         * @note Allocated blocks must be returned via return_element() for reuse.
         *       Adjacent freed blocks are automatically coalesced to reduce
         *       fragmentation.
         * 
         * @warning Passing bytes larger than available capacity will fail.
         *          Check remaining() before large allocations if needed.
         * 
         * @par Example - Basic allocation
         * @code{.cpp}
         * auto freelist = cslt::FreeListAllocator::Heap(8192, 0, false).value();
         * 
         * // Allocate different sizes
         * auto ptr1 = freelist->alloc(256, false);
         * if (!ptr1.hasValue()) {
         *     std::cerr << "Allocation failed: " << ptr1.error().what() << std::endl;
         *     return;
         * }
         * 
         * auto ptr2 = freelist->alloc(512, true);  // Zero-initialized
         * auto ptr3 = freelist->alloc(128, false);
         * 
         * // Use allocations...
         * uint8_t* data = static_cast<uint8_t*>(ptr1.value());
         * data[0] = 42;
         * 
         * // Free when done
         * freelist->return_element(ptr1.value(), 256);
         * freelist->return_element(ptr2.value(), 512);
         * freelist->return_element(ptr3.value(), 128);
         * @endcode
         * 
         * @par Example - Checking capacity
         * @code{.cpp}
         * auto freelist = cslt::FreeListAllocator::Heap(4096, 0, false).value();
         * 
         * size_t large_size = 8192;
         * if (freelist->remaining() >= large_size) {
         *     auto ptr = freelist->alloc(large_size, false);
         *     // Will succeed
         * } else {
         *     std::cerr << "Insufficient space: need " << large_size 
         *               << ", have " << freelist->remaining() << std::endl;
         * }
         * @endcode
         * 
         * @par Example - Zero-initialized allocation
         * @code{.cpp}
         * auto freelist = cslt::FreeListAllocator::Heap(4096, 0, false).value();
         * 
         * // Allocate zero-initialized buffer
         * auto result = freelist->alloc(1024, true);
         * if (result.hasValue()) {
         *     uint8_t* buffer = static_cast<uint8_t*>(result.value());
         *     // All 1024 bytes guaranteed to be 0
         *     assert(buffer[0] == 0);
         *     assert(buffer[1023] == 0);
         * }
         * @endcode
         * 
         * @see alloc_aligned() For allocations with specific alignment requirements
         * @see return_element() To free allocated blocks
         * @see realloc() To resize existing allocations
         * @see remaining() To check available capacity
         */
        Expected<void*> alloc(size_t bytes, bool zeroed = false) override;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Allocate variable-size block with specific alignment
         * 
         * @param bytes Number of bytes to allocate. Must be greater than 0.
         * @param alignment Required alignment for the returned pointer. If 0, uses
         *                  freelist's default alignment. Must be power of 2.
         *                  The effective alignment is always at least the freelist's
         *                  default alignment.
         * @param zeroed If true, zero-initialize the allocated block before returning.
         * 
         * @return Expected containing aligned pointer on success, or error
         * 
         * @retval Expected with void* pointer (aligned to requested alignment) on success
         * @retval Expected with ArgumentError if bytes == 0
         * @retval Expected with AlignmentError if alignment is not power of 2
         * @retval Expected with CapacityOverflowError if no suitable free block available
         * 
         * @details Allocates a block of the requested size with the specified alignment.
         *          The allocator searches the free list for a block large enough to
         *          accommodate:
         *          - FreeListHeader metadata
         *          - Alignment padding to satisfy the alignment requirement
         *          - The requested user bytes
         * 
         *          The returned pointer is guaranteed to be aligned to at least the
         *          requested alignment (or the freelist's default alignment, whichever
         *          is greater). Alignment padding increases internal fragmentation but
         *          ensures correct data alignment for hardware requirements.
         * 
         *          If the requested alignment is 0 or less than the freelist's default,
         *          the default alignment is used. Non-power-of-two alignments are
         *          rounded up to the next power of 2.
         * 
         * @note Stricter alignment may consume significantly more memory due to
         *       padding. For example, a 64-byte allocation with 256-byte alignment
         *       may consume up to ~320 bytes total.
         * 
         * @note The allocated block must be freed via return_element(), not free().
         * 
         * @warning Requesting alignment stricter than the freelist's base alignment
         *          increases fragmentation and reduces effective capacity.
         * 
         * @par Example - Cache-line aligned allocation
         * @code{.cpp}
         * auto freelist = cslt::FreeListAllocator::Heap(16384, 0, false).value();
         * 
         * // Allocate 256 bytes aligned to 64-byte cache line
         * auto result = freelist->alloc_aligned(256, 64, false);
         * if (!result.hasValue()) {
         *     std::cerr << "Aligned allocation failed" << std::endl;
         *     return;
         * }
         * 
         * void* ptr = result.value();
         * 
         * // Verify alignment
         * assert(reinterpret_cast<uintptr_t>(ptr) % 64 == 0);
         * 
         * // Use aligned memory...
         * 
         * freelist->return_element(ptr, 256);
         * @endcode
         * 
         * @par Example - SIMD-friendly allocation
         * @code{.cpp}
         * auto freelist = cslt::FreeListAllocator::Heap(8192, 0, false).value();
         * 
         * // Allocate buffer for AVX operations (32-byte alignment)
         * auto buffer = freelist->alloc_aligned(1024, 32, true);
         * 
         * if (buffer.hasValue()) {
         *     float* simd_data = static_cast<float*>(buffer.value());
         *     
         *     // Safe for AVX intrinsics
         *     // __m256 vec = _mm256_load_ps(simd_data);
         *     
         *     freelist->return_element(buffer.value(), 1024);
         * }
         * @endcode
         * 
         * @par Example - Page-aligned allocation
         * @code{.cpp}
         * auto freelist = cslt::FreeListAllocator::Heap(65536, 0, false).value();
         * 
         * // Allocate page-aligned memory (4KB pages)
         * auto result = freelist->alloc_aligned(8192, 4096, false);
         * 
         * if (result.hasValue()) {
         *     void* page_aligned = result.value();
         *     
         *     // Pointer is aligned to 4KB boundary
         *     assert(reinterpret_cast<uintptr_t>(page_aligned) % 4096 == 0);
         *     
         *     // Could be used for mmap-like operations or DMA
         *     
         *     freelist->return_element(page_aligned, 8192);
         * }
         * @endcode
         * 
         * @par Example - Mixing alignments
         * @code{.cpp}
         * auto freelist = cslt::FreeListAllocator::Heap(16384, 16, false).value();
         * 
         * // Default alignment (16 bytes)
         * auto ptr1 = freelist->alloc(256, false);
         * 
         * // Custom alignment (128 bytes)
         * auto ptr2 = freelist->alloc_aligned(256, 128, false);
         * 
         * // Alignment less than default - uses default (16 bytes)
         * auto ptr3 = freelist->alloc_aligned(256, 8, false);  // Actually 16-byte aligned
         * 
         * // Verify alignments
         * assert(reinterpret_cast<uintptr_t>(ptr1.value()) % 16 == 0);
         * assert(reinterpret_cast<uintptr_t>(ptr2.value()) % 128 == 0);
         * assert(reinterpret_cast<uintptr_t>(ptr3.value()) % 16 == 0);
         * @endcode
         * 
         * @see alloc() For allocations with default alignment
         * @see realloc_aligned() To resize with specific alignment
         * @see return_element() To free allocated blocks
         */
        Expected<void*> alloc_aligned(size_t bytes,
                                      size_t alignment,
                                      bool zeroed = false) override;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Resize an existing allocation
         * 
         * @param ptr Existing pointer previously returned by alloc() or alloc_aligned().
         *            May be nullptr, in which case this behaves like alloc(new_bytes, zeroed).
         * @param old_bytes Size of the existing allocation in bytes. The freelist does
         *                  not track user sizes internally; caller must provide accurate value.
         * @param new_bytes Desired new size in bytes. Must be greater than 0.
         * @param zeroed If true and the allocation grows, the newly added region
         *               [old_bytes, new_bytes) is zero-initialized.
         * 
         * @return Expected containing pointer to resized allocation on success, or error
         * 
         * @retval Expected with void* pointer on success (may be same as ptr or different)
         * @retval Expected with ArgumentError if new_bytes == 0
         * @retval Expected with CapacityOverflowError if growth fails due to insufficient space
         * 
         * @details Resizes an allocation following standard realloc() semantics:
         * 
         *          **NULL pointer**: Behaves like alloc(new_bytes, zeroed)
         * 
         *          **Shrink or same size** (new_bytes <= old_bytes):
         *          - Returns the original pointer unchanged
         *          - No memory is freed or reallocated
         *          - Freelist does not support in-place shrinking
         * 
         *          **Grow** (new_bytes > old_bytes):
         *          1. Allocates new block of new_bytes
         *          2. Copies first old_bytes from old block to new block
         *          3. Zero-fills [old_bytes, new_bytes) if zeroed == true
         *          4. Returns old block to freelist via return_element()
         *          5. Returns pointer to new block
         * 
         *          The freelist never performs in-place growth; all expansions require
         *          allocate-copy-free semantics. This ensures proper alignment and
         *          allows coalescing of the old block.
         * 
         * @note Growth always allocates a new block - the returned pointer will be
         *       different from the original when new_bytes > old_bytes.
         * 
         * @note The caller must track old_bytes accurately. Incorrect values may
         *       result in data corruption (too small) or invalid memory access (too large).
         * 
         * @note Old pointer becomes invalid after successful growth. Do not use it.
         * 
         * @warning Passing incorrect old_bytes can corrupt data or crash. The freelist
         *          cannot validate the size parameter.
         * 
         * @par Example - Growing a buffer
         * @code{.cpp}
         * auto freelist = cslt::FreeListAllocator::Heap(8192, 0, false).value();
         * 
         * // Initial allocation
         * auto result = freelist->alloc(128, false);
         * if (!result.hasValue()) {
         *     return;
         * }
         * 
         * void* ptr = result.value();
         * memcpy(ptr, "Hello", 5);  // Store some data
         * 
         * // Need more space - grow to 512 bytes
         * auto new_result = freelist->realloc(ptr, 128, 512, false);
         * if (!new_result.hasValue()) {
         *     std::cerr << "Realloc failed: " << new_result.error().what() << std::endl;
         *     freelist->return_element(ptr, 128);  // Free original on failure
         *     return;
         * }
         * 
         * void* new_ptr = new_result.value();
         * // First 128 bytes copied, "Hello" still there
         * assert(memcmp(new_ptr, "Hello", 5) == 0);
         * // ptr is now invalid, use new_ptr
         * 
         * freelist->return_element(new_ptr, 512);
         * @endcode
         * 
         * @par Example - Growing with zero-fill
         * @code{.cpp}
         * auto freelist = cslt::FreeListAllocator::Heap(4096, 0, false).value();
         * 
         * // Allocate 64 bytes
         * void* ptr = freelist->alloc(64, false).value();
         * uint8_t* data = static_cast<uint8_t*>(ptr);
         * data[0] = 42;
         * 
         * // Grow to 256 bytes, zero new region
         * auto new_ptr = freelist->realloc(ptr, 64, 256, true).value();
         * uint8_t* new_data = static_cast<uint8_t*>(new_ptr);
         * 
         * // Old data preserved
         * assert(new_data[0] == 42);
         * 
         * // New region zeroed
         * assert(new_data[64] == 0);
         * assert(new_data[255] == 0);
         * 
         * freelist->return_element(new_ptr, 256);
         * @endcode
         * 
         * @par Example - Shrinking (no-op)
         * @code{.cpp}
         * auto freelist = cslt::FreeListAllocator::Heap(4096, 0, false).value();
         * 
         * void* ptr = freelist->alloc(512, false).value();
         * 
         * // Try to shrink to 256 bytes
         * auto result = freelist->realloc(ptr, 512, 256, false);
         * 
         * // Returns same pointer (no reallocation)
         * assert(result.value() == ptr);
         * 
         * // Still need to free with original size
         * freelist->return_element(ptr, 512);
         * @endcode
         * 
         * @par Example - NULL pointer (behaves like alloc)
         * @code{.cpp}
         * auto freelist = cslt::FreeListAllocator::Heap(4096, 0, false).value();
         * 
         * // NULL pointer - allocates new block
         * auto result = freelist->realloc(nullptr, 0, 256, true);
         * 
         * // Equivalent to alloc(256, true)
         * assert(result.hasValue());
         * 
         * freelist->return_element(result.value(), 256);
         * @endcode
         * 
         * @par Example - Dynamic array growth pattern
         * @code{.cpp}
         * auto freelist = cslt::FreeListAllocator::Heap(16384, 0, false).value();
         * 
         * void* array = nullptr;
         * size_t capacity = 0;
         * size_t count = 0;
         * 
         * // Add elements, growing as needed
         * for (int i = 0; i < 100; ++i) {
         *     if (count >= capacity) {
         *         // Grow by 2x
         *         size_t new_capacity = capacity == 0 ? 16 : capacity * 2;
         *         size_t new_bytes = new_capacity * sizeof(int);
         *         size_t old_bytes = capacity * sizeof(int);
         *         
         *         auto result = freelist->realloc(array, old_bytes, new_bytes, false);
         *         if (!result.hasValue()) {
         *             break;  // Out of memory
         *         }
         *         
         *         array = result.value();
         *         capacity = new_capacity;
         *     }
         *     
         *     static_cast<int*>(array)[count++] = i;
         * }
         * 
         * // Cleanup
         * if (array) {
         *     freelist->return_element(array, capacity * sizeof(int));
         * }
         * @endcode
         * 
         * @see alloc() For initial allocations
         * @see realloc_aligned() To resize with specific alignment
         * @see return_element() To free allocations
         */
        Expected<void*> realloc(void* ptr,
                                size_t old_bytes,
                                size_t new_bytes,
                                bool zeroed = false) override;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Resize an existing allocation with specific alignment
         * 
         * @param ptr Existing pointer previously returned by alloc() or alloc_aligned().
         *            May be nullptr, in which case this behaves like alloc_aligned().
         * @param old_bytes Size of the existing allocation in bytes. Must be accurate.
         * @param new_bytes Desired new size in bytes. Must be greater than 0.
         * @param alignment Required alignment for the new allocation. If 0, uses
         *                  freelist's default alignment. Must be power of 2.
         * @param zeroed If true and the allocation grows, the newly added tail region
         *               [old_bytes, new_bytes) is zero-initialized.
         * 
         * @return Expected containing aligned pointer on success, or error
         * 
         * @retval Expected with void* pointer (aligned to requested alignment) on success
         * @retval Expected with ArgumentError if new_bytes == 0
         * @retval Expected with AlignmentError if alignment is not power of 2
         * @retval Expected with CapacityOverflowError if growth fails
         * 
         * @details Resizes an allocation with alignment-aware semantics:
         * 
         *          **NULL pointer**: Behaves like alloc_aligned(new_bytes, alignment, zeroed)
         * 
         *          **Shrink or same size** (new_bytes <= old_bytes):
         *          - Returns the original pointer unchanged
         *          - No reallocation occurs (even if alignment differs)
         * 
         *          **Grow** (new_bytes > old_bytes):
         *          1. Allocates new aligned block of new_bytes with requested alignment
         *          2. Copies first old_bytes from old block to new block
         *          3. Zero-fills [old_bytes, new_bytes) if zeroed == true
         *          4. Returns old block to freelist
         *          5. Returns pointer to new aligned block
         * 
         *          If the original block was allocated with weaker alignment than
         *          requested, the new block will necessarily move to satisfy the
         *          stricter alignment requirement.
         * 
         * @note Growth always allocates a new block with the requested alignment.
         *       The returned pointer will differ from the original when growing.
         * 
         * @note The effective alignment is max(requested, freelist_default_alignment).
         * 
         * @warning If the original allocation had strict alignment and you request
         *          weaker alignment on realloc, the new block may not preserve the
         *          original alignment. Specify alignment explicitly if needed.
         * 
         * @par Example - Growing with alignment preserved
         * @code{.cpp}
         * auto freelist = cslt::FreeListAllocator::Heap(16384, 0, false).value();
         * 
         * // Allocate 128 bytes with 64-byte alignment
         * auto result = freelist->alloc_aligned(128, 64, false);
         * void* ptr = result.value();
         * 
         * // Verify alignment
         * assert(reinterpret_cast<uintptr_t>(ptr) % 64 == 0);
         * 
         * // Grow to 512 bytes, preserving 64-byte alignment
         * auto new_result = freelist->realloc_aligned(ptr, 128, 512, 64, false);
         * void* new_ptr = new_result.value();
         * 
         * // New block also 64-byte aligned
         * assert(reinterpret_cast<uintptr_t>(new_ptr) % 64 == 0);
         * 
         * freelist->return_element(new_ptr, 512);
         * @endcode
         * 
         * @par Example - SIMD buffer growth
         * @code{.cpp}
         * auto freelist = cslt::FreeListAllocator::Heap(32768, 0, false).value();
         * 
         * // Start with small AVX-aligned buffer
         * void* buffer = freelist->alloc_aligned(256, 32, true).value();
         * float* simd_buffer = static_cast<float*>(buffer);
         * 
         * // Fill with data
         * for (int i = 0; i < 64; ++i) {
         *     simd_buffer[i] = static_cast<float>(i);
         * }
         * 
         * // Need more space - grow to 1024 bytes
         * auto new_buffer = freelist->realloc_aligned(
         *     buffer, 256, 1024, 32, true  // Preserve 32-byte alignment, zero new region
         * ).value();
         * 
         * float* new_simd = static_cast<float*>(new_buffer);
         * 
         * // Old data preserved
         * assert(new_simd[0] == 0.0f);
         * assert(new_simd[63] == 63.0f);
         * 
         * // New region zeroed
         * assert(new_simd[64] == 0.0f);
         * 
         * freelist->return_element(new_buffer, 1024);
         * @endcode
         * 
         * @par Example - Changing alignment on realloc
         * @code{.cpp}
         * auto freelist = cslt::FreeListAllocator::Heap(16384, 0, false).value();
         * 
         * // Start with default alignment
         * void* ptr = freelist->alloc(128, false).value();
         * 
         * // Grow and require stricter alignment (128-byte)
         * auto new_ptr = freelist->realloc_aligned(ptr, 128, 512, 128, false).value();
         * 
         * // New allocation is 128-byte aligned
         * assert(reinterpret_cast<uintptr_t>(new_ptr) % 128 == 0);
         * 
         * freelist->return_element(new_ptr, 512);
         * @endcode
         * 
         * @par Example - Page-aligned buffer expansion
         * @code{.cpp}
         * auto freelist = cslt::FreeListAllocator::Heap(65536, 0, false).value();
         * 
         * // Allocate page-aligned buffer for DMA
         * void* dma_buffer = freelist->alloc_aligned(4096, 4096, false).value();
         * 
         * // Need to expand DMA buffer
         * auto expanded = freelist->realloc_aligned(
         *     dma_buffer, 4096, 8192, 4096, false  // Maintain page alignment
         * );
         * 
         * if (expanded.hasValue()) {
         *     void* new_dma = expanded.value();
         *     
         *     // Still page-aligned for DMA operations
         *     assert(reinterpret_cast<uintptr_t>(new_dma) % 4096 == 0);
         *     
         *     freelist->return_element(new_dma, 8192);
         * }
         * @endcode
         * 
         * @par Example - NULL pointer with alignment
         * @code{.cpp}
         * auto freelist = cslt::FreeListAllocator::Heap(8192, 0, false).value();
         * 
         * // NULL pointer - allocates new aligned block
         * auto result = freelist->realloc_aligned(nullptr, 0, 512, 64, true);
         * 
         * // Equivalent to alloc_aligned(512, 64, true)
         * assert(result.hasValue());
         * assert(reinterpret_cast<uintptr_t>(result.value()) % 64 == 0);
         * 
         * freelist->return_element(result.value(), 512);
         * @endcode
         * 
         * @see realloc() For resizing with default alignment
         * @see alloc_aligned() For initial aligned allocations
         * @see return_element() To free allocations
         */
        Expected<void*> realloc_aligned(void* ptr,
                                        size_t old_bytes,
                                        size_t new_bytes,
                                        size_t alignment,
                                        bool zeroed = false) override;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Return an allocated block to the freelist for reuse
         * 
         * @param ptr Pointer to block previously allocated from this freelist.
         *            Must have been returned by alloc(), alloc_aligned(), realloc(),
         *            or realloc_aligned(). Must not be NULL.
         * @param bytes Size parameter (unused for freelists, for interface compatibility).
         *              FreeListAllocator stores size in the block header.
         * @param alignment Alignment parameter (unused for freelists, for interface compatibility).
         * 
         * @details Returns an allocated block to the freelist, making it available for
         *          future allocations. The freelist automatically coalesces the freed
         *          block with adjacent free blocks to reduce external fragmentation.
         * 
         *          The method performs extensive validation:
         *          1. Pointer is within freelist memory region
         *          2. Valid FreeListHeader exists before the pointer
         *          3. Block size and offset are sane
         *          4. Block fits within freelist bounds
         * 
         *          The freed block is inserted into the free list in address order,
         *          then coalesced with adjacent blocks:
         *          - **Forward coalescing**: Merges with next block if adjacent
         *          - **Backward coalescing**: Merges with previous block if adjacent
         * 
         *          Coalescing prevents fragmentation by combining small free blocks
         *          into larger contiguous regions.
         * 
         * @note The bytes and alignment parameters are ignored. FreeListAllocator
         *       retrieves size information from the block's header.
         * 
         * @note This method is silent on invalid pointers - it returns without error
         *       if validation fails. This is defensive programming to prevent crashes.
         * 
         * @note Double-freeing the same pointer is detected and ignored safely.
         * 
         * @warning After calling this method, ptr becomes invalid and must not be
         *          dereferenced. Any attempt to use the pointer results in undefined
         *          behavior.
         * 
         * @warning Passing a pointer from a different allocator results in undefined
         *          behavior. Only return pointers allocated from this freelist.
         * 
         * @par Example - Basic free and reuse
         * @code{.cpp}
         * auto freelist = cslt::FreeListAllocator::Heap(4096, 0, false).value();
         * 
         * // Allocate block
         * auto ptr = freelist->alloc(256, false).value();
         * 
         * // Use the allocation
         * memset(ptr, 0x42, 256);
         * 
         * // Return to freelist
         * freelist->return_element(ptr, 256);  // bytes parameter is ignored
         * 
         * // ptr is now invalid - do not use it
         * 
         * // Next allocation may reuse the freed block
         * auto ptr2 = freelist->alloc(256, false).value();
         * // ptr2 might equal ptr (block was reused)
         * @endcode
         * 
         * @par Example - Coalescing demonstration
         * @code{.cpp}
         * auto freelist = cslt::FreeListAllocator::Heap(8192, 0, false).value();
         * 
         * // Allocate three adjacent blocks
         * auto ptr1 = freelist->alloc(256, false).value();
         * auto ptr2 = freelist->alloc(256, false).value();
         * auto ptr3 = freelist->alloc(256, false).value();
         * 
         * std::cout << "Free blocks: " << count_free_blocks(freelist) << std::endl;
         * // Output: Free blocks: 1 (one large remaining block)
         * 
         * // Free middle block
         * freelist->return_element(ptr2, 256);
         * // Output: Free blocks: 2 (middle block + remaining)
         * 
         * // Free first block - coalesces with middle
         * freelist->return_element(ptr1, 256);
         * // Output: Free blocks: 2 (first+middle merged, remaining)
         * 
         * // Free last block - coalesces all three
         * freelist->return_element(ptr3, 256);
         * // Output: Free blocks: 1 (all merged back into one large block)
         * @endcode
         * 
         * @par Example - Safe double-free protection
         * @code{.cpp}
         * auto freelist = cslt::FreeListAllocator::Heap(4096, 0, false).value();
         * 
         * auto ptr = freelist->alloc(128, false).value();
         * 
         * // Free once
         * freelist->return_element(ptr, 128);
         * 
         * // Accidentally free again - silently ignored (no crash)
         * freelist->return_element(ptr, 128);  // Safe, but still a bug in caller code
         * 
         * // Best practice: null out pointer after freeing
         * ptr = nullptr;
         * @endcode
         * 
         * @par Example - RAII wrapper for automatic cleanup
         * @code{.cpp}
         * template<typename T>
         * class FreeListPtr {
         *     FreeListAllocator* freelist_;
         *     T* ptr_;
         *     size_t size_;
         * 
         * public:
         *     FreeListPtr(FreeListAllocator* fl, T* p, size_t sz)
         *         : freelist_(fl), ptr_(p), size_(sz) {}
         *     
         *     ~FreeListPtr() {
         *         if (ptr_) {
         *             freelist_->return_element(ptr_, size_);
         *         }
         *     }
         *     
         *     T* get() { return ptr_; }
         *     T* operator->() { return ptr_; }
         * };
         * 
         * // Usage
         * auto freelist = cslt::FreeListAllocator::Heap(4096, 0, false).value();
         * {
         *     auto result = freelist->alloc(256, false);
         *     FreeListPtr<uint8_t> managed(freelist.get(), 
         *                                   static_cast<uint8_t*>(result.value()),
         *                                   256);
         *     
         *     // Use managed.get()...
         *     
         * }  // Automatically freed here
         * @endcode
         * 
         * @par Example - Tracking allocations for bulk free
         * @code{.cpp}
         * auto freelist = cslt::FreeListAllocator::Heap(16384, 0, false).value();
         * 
         * std::vector<std::pair<void*, size_t>> allocations;
         * 
         * // Make several allocations
         * allocations.push_back({freelist->alloc(128, false).value(), 128});
         * allocations.push_back({freelist->alloc(256, false).value(), 256});
         * allocations.push_back({freelist->alloc(512, false).value(), 512});
         * 
         * // Use allocations...
         * 
         * // Bulk free
         * for (const auto& [ptr, size] : allocations) {
         *     freelist->return_element(ptr, size);
         * }
         * @endcode
         * 
         * @see alloc() For allocating blocks
         * @see alloc_aligned() For aligned allocations
         * @see reset() To free all allocations at once
         * @see is_ptr() To validate pointers before freeing
         */
        void return_element(void* ptr,
                           size_t bytes = 0,
                           size_t alignment = 0) override;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Reset freelist to initial empty state
         * 
         * @param trim Unused for freelists (for interface compatibility). Freelists
         *             do not trim or release memory back to the system.
         * 
         * @return true if reset succeeded, false if freelist is not initialized
         * 
         * @details Resets the freelist to its initial pristine state, as if freshly
         *          created. All allocation state is cleared and the entire usable
         *          region is rebuilt as a single large free block.
         * 
         *          The reset operation:
         *          1. Clears usage accounting (len_ = 0, size_ = 0)
         *          2. Resets high-water mark (cur_ = memory_)
         *          3. Destroys free list
         *          4. Creates single free block spanning entire region
         * 
         *          This is an O(1) operation regardless of how many allocations exist
         *          or how fragmented the freelist has become. No memory is freed back
         *          to the operating system; the freelist retains its capacity.
         * 
         * @note All outstanding allocations become invalid immediately. Any pointers
         *       obtained before reset() must not be used after reset().
         * 
         * @note The freelist is immediately ready for new allocations after reset().
         *       There is no fragmentation - the entire capacity is available as one
         *       contiguous block.
         * 
         * @note No memory is released. The backing arena remains unchanged.
         * 
         * @warning Invalidates ALL outstanding pointers. Ensure no code holds
         *          pointers across a reset() call.
         * 
         * @par Example - Per-frame allocation pattern
         * @code{.cpp}
         * auto freelist = cslt::FreeListAllocator::Heap(65536, 0, false).value();
         * 
         * while (game_running) {
         *     // Frame allocations
         *     auto vertex_buffer = freelist->alloc(8192, false);
         *     auto index_buffer = freelist->alloc(4096, false);
         *     auto uniform_buffer = freelist->alloc(256, false);
         *     
         *     // Render frame using buffers...
         *     render_frame(vertex_buffer.value(), 
         *                  index_buffer.value(),
         *                  uniform_buffer.value());
         *     
         *     // Fast cleanup - all frame allocations freed instantly
         *     freelist->reset();
         *     
         *     // Ready for next frame with zero fragmentation
         * }
         * @endcode
         * 
         * @par Example - Request/response processing
         * @code{.cpp}
         * auto freelist = cslt::FreeListAllocator::Heap(16384, 0, false).value();
         * 
         * void handle_request(const Request& req) {
         *     // Clear previous request's allocations
         *     freelist->reset();
         *     
         *     // Allocate buffers for this request
         *     void* parse_buffer = freelist->alloc(4096, false).value();
         *     void* response_buffer = freelist->alloc(8192, false).value();
         *     
         *     // Process request...
         *     
         *     // No manual cleanup needed - next request will reset()
         * }
         * @endcode
         * 
         * @par Example - Temporary computation workspace
         * @code{.cpp}
         * auto freelist = cslt::FreeListAllocator::Heap(32768, 0, false).value();
         * 
         * void complex_computation() {
         *     // Many temporary allocations
         *     std::vector<void*> temp_buffers;
         *     
         *     for (int i = 0; i < 100; ++i) {
         *         auto temp = freelist->alloc(256, false);
         *         if (temp.hasValue()) {
         *             temp_buffers.push_back(temp.value());
         *         }
         *     }
         *     
         *     // Use temporary buffers...
         *     
         *     // Fast cleanup - O(1) instead of freeing 100 individual blocks
         *     freelist->reset();
         * }
         * @endcode
         * 
         * @par Example - Testing/benchmarking reset
         * @code{.cpp}
         * auto freelist = cslt::FreeListAllocator::Heap(4096, 0, false).value();
         * 
         * // Allocate and fragment the freelist
         * std::vector<void*> ptrs;
         * for (int i = 0; i < 10; ++i) {
         *     auto ptr = freelist->alloc(128, false);
         *     if (ptr.hasValue()) {
         *         ptrs.push_back(ptr.value());
         *     }
         * }
         * 
         * // Free every other block (create fragmentation)
         * for (size_t i = 0; i < ptrs.size(); i += 2) {
         *     freelist->return_element(ptrs[i], 128);
         * }
         * 
         * std::cout << "Before reset - used: " << freelist->used() << std::endl;
         * std::cout << "Before reset - fragmented" << std::endl;
         * 
         * // Reset to pristine state
         * bool ok = freelist->reset();
         * assert(ok);
         * 
         * std::cout << "After reset - used: " << freelist->used() << std::endl;  // 0
         * std::cout << "After reset - capacity: " << freelist->capacity() << std::endl;
         * std::cout << "After reset - no fragmentation" << std::endl;
         * 
         * // All pointers in ptrs are now invalid
         * @endcode
         * 
         * @par Example - Reusing freelist across operations
         * @code{.cpp}
         * auto freelist = cslt::FreeListAllocator::Heap(8192, 0, false).value();
         * 
         * // Operation 1
         * {
         *     auto data = freelist->alloc(2048, false).value();
         *     process_data(data);
         *     // Don't bother freeing individually
         * }
         * 
         * // Clean slate for operation 2
         * freelist->reset();
         * 
         * // Operation 2
         * {
         *     auto buffer = freelist->alloc(4096, false).value();
         *     process_buffer(buffer);
         * }
         * 
         * // Clean slate for operation 3
         * freelist->reset();
         * 
         * // Continue reusing...
         * @endcode
         * 
         * @see return_element() To free individual blocks
         * @see remaining() To check available capacity after reset
         * @see used() To verify reset cleared allocations (should be 0)
         */
        bool reset(bool trim = false) override;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Validate whether a pointer belongs to this freelist
         * 
         * @param ptr Pointer to validate. May be NULL.
         * 
         * @return true if pointer is a valid allocation from this freelist, false otherwise
         * 
         * @details Performs comprehensive validation to determine if a pointer was
         *          allocated by this freelist and is still valid. The validation checks:
         * 
         *          1. Pointer is not NULL
         *          2. Pointer is within the freelist's memory region
         *          3. Pointer has room for FreeListHeader before it
         *          4. Valid header exists with sane block_size and offset
         *          5. Reconstructed block boundaries fit within memory region
         *          6. Pointer lies within the reconstructed block
         * 
         *          This method cannot distinguish between currently allocated blocks
         *          and blocks that have been freed (returned to free list). It only
         *          validates that the pointer could have come from this freelist based
         *          on memory layout and metadata.
         * 
         * @note Returns false for NULL pointers without error.
         * 
         * @note Returns false for pointers from other allocators.
         * 
         * @note Cannot detect if a pointer has been freed - only validates that it
         *       could be a valid freelist pointer based on structure.
         * 
         * @note Useful for defensive programming and debugging, but not foolproof
         *       against all forms of pointer corruption.
         * 
         * @see is_ptr_sized() To additionally validate available size
         * @see return_element() Uses validation internally before freeing
         */
        bool is_ptr(void* ptr) const override;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Validate pointer and verify it has at least the requested size
         * 
         * @param ptr Pointer to validate. May be NULL.
         * @param bytes Minimum number of bytes required to be available at the pointer.
         * 
         * @return true if pointer is valid and has at least bytes available, false otherwise
         * 
         * @details Extends is_ptr() validation with size checking. First validates
         *          that the pointer is a plausible freelist allocation using the same
         *          checks as is_ptr(). Then additionally verifies:
         * 
         *          1. Requested bytes is greater than 0
         *          2. Block's user data region is large enough (user_data_size >= bytes)
         *          3. ptr + bytes does not exceed freelist memory bounds
         * 
         *          The user data size is calculated as (block_size - offset), representing
         *          the actual space available to the user after accounting for the header
         *          and alignment padding.
         * 
         *          This method is useful for bounds checking before writing to a buffer,
         *          ensuring that the allocation is large enough to hold the intended data.
         * 
         * @note Returns false if bytes == 0 (zero-size check is invalid).
         * 
         * @note Returns false if ptr fails basic is_ptr() validation.
         * 
         * @note The check is conservative - it validates against the block's actual
         *       capacity, not just the user's original allocation request.
         * 
         * @warning Does not prevent buffer overruns within the validated size. It only
         *          checks that the block is large enough, not that subsequent writes
         *          will be bounds-safe.
         * 
         * @see is_ptr() For basic pointer validation without size check
         */
        bool is_ptr_sized(void* ptr, size_t bytes) const override;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Generate diagnostic statistics report for the freelist
         * 
         * @param buffer Destination character buffer for the report. Must not be NULL.
         * @param buffer_size Size of buffer in bytes. Must be greater than 0.
         * 
         * @return true if report was successfully generated, false on error
         * 
         * @retval true Report generated successfully (may be truncated if buffer too small)
         * @retval false Invalid parameters (buffer is NULL or buffer_size is 0)
         * 
         * @details Generates a human-readable, multi-line text report describing the
         *          internal state of the freelist allocator. The report includes:
         * 
         *          - **Type**: STATIC or DYNAMIC (inherited from backing arena)
         *          - **Ownership**: Whether the freelist owns its arena
         *          - **Memory accounting**: Used bytes, remaining bytes, capacity
         *          - **Overhead**: Total size including headers
         *          - **Utilization**: Percentage of capacity currently in use
         *          - **Alignment**: Base alignment for allocations
         *          - **Free list**: Enumeration of all free blocks with addresses and sizes
         * 
         *          The report is written using internal `_buf_appendf()` utility which
         *          safely handles buffer overflow. Output is always null-terminated as
         *          long as buffer_size >= 1.
         * 
         *          This method is useful for debugging, logging, performance analysis,
         *          and verification during development. It provides insight into
         *          fragmentation, capacity utilization, and memory layout.
         * 
         * @note If the buffer is too small, the report will be truncated but still
         *       null-terminated and safe to use.
         * 
         * @note This method does not allocate memory - all output uses the provided buffer.
         * 
         * @warning Large freelists with many free blocks may produce output exceeding
         *          typical buffer sizes. Consider using a 2KB or larger buffer for
         *          detailed reports.
         * 
         * @par Example - Basic statistics report
         * @code{.cpp}
         * auto freelist = cslt::FreeListAllocator::Heap(8192, 0, false).value();
         * 
         * // Make some allocations
         * auto ptr1 = freelist->alloc(512, false).value();
         * auto ptr2 = freelist->alloc(1024, false).value();
         * auto ptr3 = freelist->alloc(256, false).value();
         * 
         * // Free one to create fragmentation
         * freelist->return_element(ptr2, 1024);
         * 
         * // Generate report
         * char buffer[2048];
         * if (freelist->stats(buffer, sizeof(buffer))) {
         *     std::cout << buffer << std::endl;
         * } else {
         *     std::cerr << "Failed to generate stats" << std::endl;
         * }
         * @endcode
         * 
         * @par Example Output
         * @code{.txt}
         * FreeListAllocator Statistics:
         *   Type: DYNAMIC
         *   Owns arena: yes
         *   Used (accounted): 1872 bytes
         *   Remaining: 6224 bytes
         *   Capacity (usable region): 8096 bytes
         *   Total (with header/overhead): 8288 bytes
         *   Utilization: 23.1%
         *   Base alignment: 16 bytes
         *   Free block 1: 0x7f8a4c002400, 1024 bytes
         *   Free block 2: 0x7f8a4c002c00, 5200 bytes
         *   Free blocks: 2, total free bytes (raw): 6224
         * @endcode
         * 
         * @par Example - Comparing before/after reset
         * @code{.cpp}
         * auto freelist = cslt::FreeListAllocator::Heap(4096, 0, false).value();
         * 
         * // Create fragmentation
         * std::vector<void*> ptrs;
         * for (int i = 0; i < 5; ++i) {
         *     auto ptr = freelist->alloc(256, false);
         *     if (ptr.hasValue()) ptrs.push_back(ptr.value());
         * }
         * 
         * // Free alternating blocks
         * for (size_t i = 0; i < ptrs.size(); i += 2) {
         *     freelist->return_element(ptrs[i], 256);
         * }
         * 
         * char buffer[2048];
         * 
         * // Before reset
         * std::cout << "=== BEFORE RESET ===" << std::endl;
         * freelist->stats(buffer, sizeof(buffer));
         * std::cout << buffer << std::endl;
         * 
         * // Reset
         * freelist->reset();
         * 
         * // After reset
         * std::cout << "=== AFTER RESET ===" << std::endl;
         * freelist->stats(buffer, sizeof(buffer));
         * std::cout << buffer << std::endl;
         * @endcode
         * 
         * @par Example Output - Before/After Reset
         * @code{.txt}
         * === BEFORE RESET ===
         * FreeListAllocator Statistics:
         *   Type: DYNAMIC
         *   Owns arena: yes
         *   Used (accounted): 768 bytes
         *   Remaining: 3328 bytes
         *   Capacity (usable region): 4096 bytes
         *   Total (with header/overhead): 4288 bytes
         *   Utilization: 18.8%
         *   Base alignment: 16 bytes
         *   Free block 1: 0x7f8a4c002000, 256 bytes
         *   Free block 2: 0x7f8a4c002200, 256 bytes
         *   Free block 3: 0x7f8a4c002400, 256 bytes
         *   Free block 4: 0x7f8a4c002a00, 2560 bytes
         *   Free blocks: 4, total free bytes (raw): 3328
         * 
         * === AFTER RESET ===
         * FreeListAllocator Statistics:
         *   Type: DYNAMIC
         *   Owns arena: yes
         *   Used (accounted): 0 bytes
         *   Remaining: 4096 bytes
         *   Capacity (usable region): 4096 bytes
         *   Total (with header/overhead): 4288 bytes
         *   Utilization: 0.0%
         *   Base alignment: 16 bytes
         *   Free block 1: 0x7f8a4c002000, 4096 bytes
         *   Free blocks: 1, total free bytes (raw): 4096
         * @endcode
         * 
         * @par Example - Logging to file
         * @code{.cpp}
         * auto freelist = cslt::FreeListAllocator::Heap(16384, 0, false).value();
         * 
         * // Perform operations...
         * 
         * // Log statistics to file
         * char buffer[4096];
         * if (freelist->stats(buffer, sizeof(buffer))) {
         *     std::ofstream log("freelist_stats.txt");
         *     log << "Freelist Statistics at " << current_timestamp() << std::endl;
         *     log << buffer << std::endl;
         *     log.close();
         * }
         * @endcode
         * 
         * @see remaining() For just available capacity
         * @see used() For just current usage
         * @see capacity() For just total capacity
         */
        bool stats(char* buffer, size_t buffer_size) const override;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Get the number of bytes available for new allocations
         * 
         * @return Number of bytes remaining for allocation
         * 
         * @details Returns the number of bytes currently available for new allocations,
         *          calculated as (capacity - used). This represents the logical free
         *          space, accounting for all overhead including headers and alignment
         *          padding that has been consumed by existing allocations.
         * 
         *          Note that the actual usable space for a new allocation may be less
         *          than this value due to:
         *          - Fragmentation (free space split across multiple small blocks)
         *          - Alignment requirements for the new allocation
         *          - Per-allocation header overhead (FreeListHeader)
         * 
         * @note This is a logical capacity measure, not a guarantee that an allocation
         *       of this size will succeed. Use is_ptr_sized() to validate actual
         *       allocation feasibility.
         * 
         * @note The value equals capacity() immediately after construction or reset().
         * 
         * @see capacity() For total usable capacity
         * @see used() For currently consumed bytes
         */
        size_t remaining() const noexcept;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Get the number of bytes currently consumed by allocations
         * 
         * @return Number of bytes currently in use (including internal overhead)
         * 
         * @details Returns the total number of bytes consumed by current allocations,
         *          including all internal overhead such as:
         *          - FreeListHeader metadata for each allocated block
         *          - Alignment padding between headers and user data
         *          - Full block consumption when remainders are too small to split
         * 
         *          This is the internal accounting value (len_) which tracks the total
         *          size charged for all outstanding allocations. It increases with each
         *          alloc() call and decreases with each return_element() call.
         * 
         *          The value may be significantly larger than the sum of user-requested
         *          allocation sizes due to internal overhead. For example, a 256-byte
         *          allocation might consume 280+ bytes when accounting for header and
         *          alignment padding.
         * 
         * @note This is the total block size consumed, not just user-visible bytes.
         *       It includes all metadata and padding.
         * 
         * @note Returns 0 immediately after construction or reset().
         * 
         * @note The relationship: used() + remaining() == capacity() always holds.
         * 
         * @see remaining() For available bytes
         * @see capacity() For total capacity
         * @see stats() For detailed usage breakdown
         */
        size_t used() const;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Check whether the freelist owns its backing arena
         * 
         * @return true if the freelist owns the arena, false if arena is borrowed
         * 
         * @details Returns the ownership status of the freelist's backing arena, which
         *          determines cleanup behavior when the freelist is destroyed.
         * 
         *          **Ownership by factory method:**
         *          - **Heap()**: Returns true - freelist owns the arena (DYNAMIC)
         *          - **WithArena()**: Returns false - arena is borrowed
         *          - **Stack()**: Returns true - freelist owns the arena object (STATIC)
         * 
         *          When owns_arena() returns true, the FreeListDeleter will destroy
         *          the arena when the freelist is destroyed:
         *          - For DYNAMIC arenas (Heap): Arena is deleted
         *          - For STATIC arenas (Stack): Arena destructor called, buffer not freed
         * 
         *          When owns_arena() returns false (WithArena), the arena outlives the
         *          freelist and remains valid for use by other allocators.
         * 
         * @note This indicates ownership of the arena OBJECT, not necessarily the
         *       underlying memory buffer. Stack() freelists own their arena object
         *       but not the user-provided buffer.
         * 
         * @note This value is set at construction time and never changes.
         * 
         * @see Heap() Creates freelist with owned arena
         * @see WithArena() Creates freelist with borrowed arena
         * @see Stack() Creates freelist with owned arena object over user buffer
         * @see FreeListDeleter For cleanup behavior based on ownership
         */
        bool owns_arena() const;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Save allocator state (not supported for freelists)
         * 
         * @return Always returns nullptr
         * 
         * @details This method is implemented to satisfy the Allocator base class
         *          interface contract but is not supported for FreeListAllocator.
         * 
         *          Unlike ArenaAllocator and PoolAllocator which support checkpointing,
         *          FreeListAllocator cannot provide meaningful checkpoint/restore
         *          functionality because:
         *          - Free blocks are managed via a linked list with pointers
         *          - Allocation metadata is interleaved with user data
         *          - Restoring would require tracking all allocation headers
         *          - No clear way to invalidate user pointers to restored allocations
         * 
         *          For bulk cleanup of all allocations, use reset() instead, which
         *          returns the freelist to its initial empty state.
         * 
         * @note This is a no-op implementation. Always returns nullptr.
         * 
         * @note If checkpoint/restore functionality is needed, consider using
         *       ArenaAllocator or PoolAllocator instead.
         * 
         * @see restore() Corresponding restore method (also unsupported)
         * @see reset() To clear all allocations and return to initial state
         */
        void* save() const override { return nullptr; }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Restore allocator state (not supported for freelists)
         * 
         * @param checkpoint Checkpoint pointer (ignored, must be from save())
         * 
         * @return Always returns false
         * 
         * @details This method is implemented to satisfy the Allocator base class
         *          interface contract but is not supported for FreeListAllocator.
         * 
         *          FreeListAllocator cannot support checkpoint/restore semantics because:
         *          - The free list structure uses embedded pointers that would be invalidated
         *          - Allocation headers contain metadata that cannot be simply discarded
         *          - User pointers to "restored away" allocations would become dangling
         *          - No efficient way to track which allocations to invalidate
         * 
         *          For resetting the allocator state, use reset() which clears all
         *          allocations and returns the freelist to pristine condition.
         * 
         * @note This is a no-op implementation. The checkpoint parameter is ignored
         *       and the method always returns false.
         * 
         * @note Attempting to use checkpoints from other allocator types will fail
         *       safely (returns false) but should be avoided.
         * 
         * @see save() Corresponding save method (also unsupported)
         * @see reset() To clear all allocations and start fresh
         */
        bool restore(void* checkpoint) override {
            (void)checkpoint;
            return false;
        }
    };
// ================================================================================ 
// ================================================================================ 

    /**
     * @brief Custom deleter for FreeListAllocator UniquePtr cleanup
     * 
     * @param freelist Pointer to FreeListAllocator to delete. May be nullptr.
     * 
     * @details This custom deleter is invoked when a UniquePtr<FreeListAllocator>
     *          goes out of scope or is explicitly reset. It handles proper cleanup
     *          of the freelist and its backing arena based on ownership semantics.
     * 
     *          **Cleanup sequence:**
     *          1. Extract ownership information (owns_arena, arena, mem_type)
     *          2. Call ~FreeListAllocator() destructor
     *          3. Conditionally clean up arena based on ownership and type:
     * 
     *          **Heap() freelists** (owns_arena=true, mem_type=DYNAMIC):
     *          - Calls ArenaDeleter{}(arena)
     *          - Arena and all memory is freed back to heap
     * 
     *          **WithArena() freelists** (owns_arena=false):
     *          - No arena cleanup performed
     *          - Arena remains valid for other allocators
     * 
     *          **Stack() freelists** (owns_arena=true, mem_type=STATIC):
     *          - Calls arena->~ArenaAllocator() destructor
     *          - Does NOT free buffer (user owns it)
     *          - Arena object destroyed, buffer remains valid
     * 
     *          This deleter ensures memory is properly released for heap-allocated
     *          arenas while preserving user-owned buffers for stack-based freelists
     *          and borrowed arenas for shared freelists.
     * 
     * @note This is a noexcept function - no exceptions are thrown during cleanup.
     * 
     * @note Null pointer is safely handled (early return, no crash).
     * 
     * @note The freelist destructor is always called before arena cleanup to ensure
     *       proper cleanup ordering.
     * 
     * @note Users should never call this directly - it is automatically invoked by
     *       UniquePtr when the freelist goes out of scope.
     * 
     * @see Heap() Creates freelist with owned DYNAMIC arena (arena will be freed)
     * @see WithArena() Creates freelist with borrowed arena (arena not touched)
     * @see Stack() Creates freelist with owned STATIC arena (destructor only)
     * @see ~FreeListAllocator() Freelist destructor called before arena cleanup
     */
    inline void FreeListDeleter::operator()(FreeListAllocator* freelist) const noexcept {
        if (!freelist) return;
        
        bool owns_arena = freelist->owns_arena_;
        ArenaAllocator* arena = freelist->arena_;
        MemType mem_type = static_cast<MemType>(freelist->mem_type_);
        
        // Call freelist destructor
        freelist->~FreeListAllocator();
        
        // Only delete arena if:
        // 1. Freelist owns it AND
        // 2. It's DYNAMIC (heap-allocated)
        // For STATIC arenas (Stack freelists), the arena is in the user buffer
        if (owns_arena && arena && mem_type == DYNAMIC) {
            ArenaDeleter{}(arena);
        }
        // For STATIC freelists, we still need to call arena destructor
        // but NOT delete it, since it's in the user buffer
        else if (owns_arena && arena && mem_type == STATIC) {
            arena->~ArenaAllocator();
        }
    }
// ================================================================================ 
// ================================================================================ 

#if ARENA_ENABLE_DYNAMIC

    struct BuddyDeleter {
        void operator()(BuddyAllocator* buddy) const noexcept;
    };
// ================================================================================ 

    /**
     * @class BuddyAllocator
     * @brief Binary buddy memory allocator with power-of-two block management
     * 
     * @details BuddyAllocator implements the binary buddy memory allocation algorithm,
     *          which organizes memory into power-of-two sized blocks. When a block is
     *          freed, it attempts to coalesce (merge) with its "buddy" block to form
     *          larger free blocks, reducing fragmentation.
     * 
     *          All allocations are rounded up to the nearest power-of-two size
     *          (including a 16-byte header). Memory is obtained directly from the OS
     *          via mmap() (POSIX) or VirtualAlloc() (Windows). The pool size is fixed
     *          at creation time.
     * 
     *          Two blocks are "buddies" if they have the same size, are adjacent in
     *          memory, and their combined offset satisfies: offset XOR (2^order).
     *          When both buddies are free, they merge into a larger block, potentially
     *          triggering recursive coalescing up the size hierarchy.
     * 
     *          Performance: O(log n) for allocation, deallocation, and coalescing,
     *          where n is the number of size levels. Memory overhead is ~16 bytes per
     *          allocation for the header.
     * 
     * @note BuddyAllocator is not thread-safe. External synchronization required.
     * @note Pool size is fixed at creation - use reset() to clear or recreate for
     *       a different size.
     * @note alloc() does not guarantee custom alignment - use alloc_aligned() for
     *       specific alignment requirements.
     * 
     * @warning Do not mix pointers from different allocators.
     * @warning All allocations become invalid when BuddyAllocator is destroyed.
     * 
     * @par Basic Usage:
     * @code{.cpp}
     * // Create a 64KB buddy allocator with 64-byte minimum blocks
     * auto buddy = BuddyAllocator::Heap(65536, 64, 0).value();
     * 
     * // Allocate 256 bytes
     * auto ptr = buddy->alloc(256, false).value();
     * 
     * // Use memory
     * memset(ptr, 0, 256);
     * 
     * // Free (automatic coalescing)
     * buddy->return_element(ptr);
     * 
     * // Allocator destroyed automatically when buddy goes out of scope
     * @endcode
     * 
     * @par Aligned Allocation:
     * @code{.cpp}
     * auto buddy = BuddyAllocator::Heap(1024 * 1024, 64, 0).value();
     * 
     * // Allocate 512 bytes with 256-byte alignment, zero-initialized
     * auto ptr = buddy->alloc_aligned(512, 256, true).value();
     * 
     * // Verify alignment
     * assert(reinterpret_cast<uintptr_t>(ptr) % 256 == 0);
     * 
     * // Grow allocation
     * auto new_ptr = buddy->realloc_aligned(ptr, 512, 1024, 256, false);
     * if (new_ptr.hasValue()) {
     *     buddy->return_element(new_ptr.value());
     * }
     * @endcode
     * 
     * @par Monitoring:
     * @code{.cpp}
     * auto buddy = BuddyAllocator::Heap(8192, 64, 0).value();
     * 
     * // Make allocations
     * auto p1 = buddy->alloc(128, false).value();
     * auto p2 = buddy->alloc(256, false).value();
     * 
     * // Query state
     * size_t used = buddy->size();              // Currently allocated
     * size_t free = buddy->remaining();         // Available
     * size_t max = buddy->largest_block();      // Largest contiguous
     * 
     * // Detect fragmentation
     * if (free > max) {
     *     std::cout << "Fragmented: " << (free - max) << " bytes\n";
     * }
     * 
     * // Generate report
     * char buffer[2048];
     * buddy->stats(buffer, sizeof(buffer));
     * std::cout << buffer;
     * 
     * buddy->return_element(p1);
     * buddy->return_element(p2);
     * @endcode
     * 
     * @par Coalescing:
     * @code{.cpp}
     * auto buddy = BuddyAllocator::Heap(4096, 64, 0).value();
     * 
     * // Allocate adjacent blocks
     * auto p1 = buddy->alloc(128, false).value();
     * auto p2 = buddy->alloc(128, false).value();
     * auto p3 = buddy->alloc(128, false).value();
     * 
     * // Free in order - blocks coalesce automatically
     * buddy->return_element(p1);
     * buddy->return_element(p2);
     * buddy->return_element(p3);
     * 
     * // Largest block increased due to coalescing
     * @endcode
     * 
     * @see Heap() Primary factory method
     * @see alloc() Basic allocation
     * @see alloc_aligned() Aligned allocation
     * @see realloc() Resize allocation
     * @see return_element() Free and coalesce
     * @see reset() Bulk cleanup
     * @see stats() Diagnostics
     */ 
    class BuddyAllocator : public Allocator {
    private:
        struct BuddyBlock {
            BuddyBlock* next;    ///< Next block in free list
        };

        struct BuddyHeader {
            uint32_t order;       ///< log2(block_size)
            size_t   block_offset; ///< Offset from pool base
        };

        // Member variables
        void*         base_;           ///< OS-backed memory pool
        BuddyBlock**  free_lists_;     ///< Array of free lists by level
        size_t        pool_size_;      ///< Total pool size (power of 2)
        size_t        base_align_;     ///< Minimum alignment guarantee
        size_t        user_offset_;    ///< Offset from block to user pointer
        uint32_t      min_order_;      ///< log2(min_block_size)
        uint32_t      max_order_;      ///< log2(pool_size)
        uint32_t      num_levels_;     ///< Number of free list levels

        // Private constructor (use factory methods)
        BuddyAllocator();

        // Helper methods
        static uint32_t ilog2(size_t x);
// -------------------------------------------------------------------------------- 

        //static size_t next_pow2(size_t x);
// -------------------------------------------------------------------------------- 

        uint32_t order_to_level(uint32_t order) const;
// -------------------------------------------------------------------------------- 

        uint32_t level_to_order(uint32_t level) const;
// -------------------------------------------------------------------------------- 

        int32_t find_nonempty_level(uint32_t desired_level) const;
// -------------------------------------------------------------------------------- 

        void freelist_push(BuddyBlock** head, BuddyBlock* block);
// -------------------------------------------------------------------------------- 

        bool freelist_remove(BuddyBlock** head, BuddyBlock* block);
// -------------------------------------------------------------------------------- 

        BuddyBlock* freelist_find(BuddyBlock* head, void* addr) const;
// -------------------------------------------------------------------------------- 

        static void* os_alloc(size_t size);
// -------------------------------------------------------------------------------- 

        static void os_free(void* ptr, size_t size);
// -------------------------------------------------------------------------------- 

        bool ptr_in_pool_(const void* p) const noexcept;
// ================================================================================ 

    public:
        /**
         * @brief Destructor for BuddyAllocator
         * 
         * @details The destructor is intentionally minimal and does NOT free the OS-backed
         *          memory pool or free-lists array. Cleanup is handled by the custom
         *          BuddyDeleter when the UniquePtr goes out of scope.
         * 
         *          BuddyAllocator instances are always managed via UniquePtr with
         *          BuddyDeleter, which performs the actual resource cleanup in this order:
         *          1. Free OS-backed memory pool (via os_free)
         *          2. Free free-lists array (via delete[])
         *          3. Call this destructor
         *          4. Free BuddyAllocator structure (via ::operator delete)
         * 
         * @note Users never call this destructor directly. The BuddyDeleter handles all
         *       cleanup automatically when the UniquePtr is destroyed or reset.
         * 
         * @note All outstanding allocations become invalid when the BuddyAllocator is
         *       destroyed. Accessing freed pointers results in undefined behavior.
         * 
         * @see BuddyDeleter Custom deleter that performs resource cleanup
         * @see Heap() Factory method that creates the UniquePtr with BuddyDeleter
         */
        ~BuddyAllocator() noexcept override;
// -------------------------------------------------------------------------------- 

        BuddyAllocator(const BuddyAllocator&) = delete;
        BuddyAllocator& operator=(const BuddyAllocator&) = delete;
        BuddyAllocator(BuddyAllocator&&) = delete;
        BuddyAllocator& operator=(BuddyAllocator&&) = delete;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Create an OS-backed buddy allocator with specified pool and block sizes
         * 
         * @param pool_size Total memory pool size in bytes (will be rounded up to power of 2)
         * @param min_block_size Minimum allocation block size in bytes (will be rounded up to power of 2)
         * @param base_align Default alignment for allocations (0 = alignof(max_align_t), will be rounded up to power of 2)
         * 
         * @return Expected containing UniquePtr<BuddyAllocator, BuddyDeleter> on success,
         *         or error on failure
         * 
         * @details Creates a buddy allocator with memory obtained directly from the operating
         *          system via mmap() on POSIX systems or VirtualAlloc() on Windows.
         * 
         *          **Initialization Process:**
         *          1. Validates and normalizes parameters (rounds to powers of 2)
         *          2. Allocates OS memory pool
         *          3. Initializes free-list array (one per size level)
         *          4. Places initial large free block at top level
         *          5. Returns UniquePtr with BuddyDeleter for automatic cleanup
         * 
         *          **Parameter Normalization:**
         *          - pool_size: Rounded up to next power of 2 (e.g., 5000 → 8192)
         *          - min_block_size: Rounded up to next power of 2 (e.g., 100 → 128)
         *          - base_align: Rounded up to next power of 2 (e.g., 48 → 64)
         *          - min_block_size adjusted to hold header + alignment if needed
         * 
         *          **Size Levels:**
         *          The number of free-list levels is: log2(pool_size) - log2(min_block_size) + 1
         *          Example: pool=4096, min=64 → log2(4096) - log2(64) + 1 = 12 - 6 + 1 = 7 levels
         * 
         *          **Memory Overhead:**
         *          - OS pool: pool_size bytes
         *          - Free-lists array: num_levels * sizeof(void*)
         *          - BuddyAllocator structure: ~128 bytes
         * 
         * @throws ArgumentError If pool_size or min_block_size is zero
         * @throws ArgumentError If min_block_size > pool_size (after rounding)
         * @throws AlignmentError If base_align is invalid after normalization
         * @throws MemoryError If OS memory allocation fails
         * @throws CapacityOverflowError If arithmetic overflow occurs during initialization
         * 
         * @note The returned allocator is wrapped in a UniquePtr with BuddyDeleter, which
         *       automatically frees all resources (OS memory, free-lists, structure) when
         *       the UniquePtr is destroyed.
         * 
         * @note The pool size is fixed for the lifetime of the allocator. It cannot be
         *       resized. Use reset() to clear allocations or destroy and recreate for a
         *       different size.
         * 
         * @note base_align affects the minimum block size calculation but does NOT guarantee
         *       that alloc() returns aligned pointers. Use alloc_aligned() for specific
         *       alignment requirements.
         * 
         * @warning All parameters must be representable as size_t. Extremely large values
         *          may cause overflow errors.
         * 
         * @par Basic Example:
         * @code{.cpp}
         * // Create 64KB allocator with 64-byte minimum blocks
         * auto result = BuddyAllocator::Heap(65536, 64, 0);
         * 
         * if (!result.hasValue()) {
         *     std::cerr << "Error: " << result.error().message() << "\n";
         *     return 1;
         * }
         * 
         * auto buddy = cslt::move(result.value());
         * 
         * // Use allocator...
         * auto ptr = buddy->alloc(256, false);
         * 
         * // Automatic cleanup when buddy goes out of scope
         * @endcode
         * 
         * @par Power-of-2 Rounding Example:
         * @code{.cpp}
         * // Request non-power-of-2 sizes
         * auto buddy = BuddyAllocator::Heap(5000, 100, 48).value();
         * 
         * // Actual sizes after rounding:
         * // pool_size: 5000 → 8192 (next power of 2)
         * // min_block_size: 100 → 128 (next power of 2)
         * // base_align: 48 → 64 (next power of 2)
         * 
         * // Verify actual sizes
         * size_t min = buddy->min_block_size();  // 128
         * size_t max = buddy->max_block_size();  // 8192
         * @endcode
         * 
         * @par Error Handling Example:
         * @code{.cpp}
         * // Invalid: min_block_size > pool_size
         * auto result1 = BuddyAllocator::Heap(1024, 4096, 0);
         * EXPECT_FALSE(result1.hasValue());
         * 
         * // Invalid: zero pool size
         * auto result2 = BuddyAllocator::Heap(0, 64, 0);
         * EXPECT_FALSE(result2.hasValue());
         * 
         * // Invalid: zero min_block_size
         * auto result3 = BuddyAllocator::Heap(4096, 0, 0);
         * EXPECT_FALSE(result3.hasValue());
         * 
         * // Valid: all parameters will be normalized
         * auto result4 = BuddyAllocator::Heap(3000, 50, 12);
         * EXPECT_TRUE(result4.hasValue());
         * @endcode
         * 
         * @par Large Allocator Example:
         * @code{.cpp}
         * // Create 16MB allocator with 256-byte minimum blocks
         * auto buddy = BuddyAllocator::Heap(16 * 1024 * 1024, 256, 0).value();
         * 
         * // Can allocate up to ~16MB (minus overhead)
         * auto large = buddy->alloc(8 * 1024 * 1024, false);
         * EXPECT_TRUE(large.hasValue());
         * 
         * buddy->return_element(large.value());
         * @endcode
         * 
         * @see BuddyDeleter Custom deleter for automatic resource cleanup
         * @see min_block_size() Query minimum block size
         * @see max_block_size() Query maximum block size
         * @see reset() Clear all allocations
         */
        static Expected<UniquePtr<BuddyAllocator, BuddyDeleter>>
        Heap(size_t pool_size,
             size_t min_block_size,
             size_t base_align = 0);
// -------------------------------------------------------------------------------- 

        /**
         * @brief Allocate memory from the buddy allocator
         * 
         * @param bytes Number of bytes to allocate (must be > 0)
         * @param zeroed If true, zero-initialize the allocated memory
         * 
         * @return Expected containing pointer to allocated memory on success, or error on failure
         * 
         * @details Allocates memory using the buddy allocation algorithm. The actual block
         *          size will be rounded up to the nearest power of 2 that can accommodate
         *          both the requested bytes and the internal 16-byte BuddyHeader.
         * 
         *          **Allocation Process:**
         *          1. Calculate total size needed: bytes + sizeof(BuddyHeader)
         *          2. Round to next power of 2 (minimum: min_block_size)
         *          3. Find free block of appropriate size (or larger)
         *          4. Split larger blocks if necessary (recursive division by 2)
         *          5. Place header at block start, return pointer after header
         *          6. Optionally zero-initialize the user portion
         * 
         *          **Block Sizing Example:**
         *          - Request 100 bytes → 100 + 16 (header) = 116 → rounds to 128 bytes
         *          - Request 256 bytes → 256 + 16 (header) = 272 → rounds to 512 bytes
         *          - Request 1000 bytes → 1000 + 16 (header) = 1016 → rounds to 1024 bytes
         * 
         *          **Alignment:**
         *          The returned pointer has natural alignment based on block position in
         *          the pool. It is NOT guaranteed to match base_align from Heap().
         *          Use alloc_aligned() for specific alignment requirements.
         * 
         *          **Internal Fragmentation:**
         *          Power-of-2 rounding creates internal fragmentation. A 100-byte request
         *          wastes ~12 bytes within the 128-byte block (after accounting for header).
         * 
         * @note The returned pointer is NOT the start of the block. The block starts
         *       16 bytes earlier with the BuddyHeader.
         * 
         * @note The actual usable space may be larger than requested due to power-of-2
         *       rounding, but you should only use the requested number of bytes.
         * 
         * @throws ArgumentError If bytes is zero
         * @throws CapacityOverflowError If bytes + header causes arithmetic overflow
         * @throws MemoryError If no free blocks available or request exceeds pool capacity
         * 
         * @par Basic Example:
         * @code{.cpp}
         * auto buddy = BuddyAllocator::Heap(8192, 64, 0).value();
         * 
         * // Allocate 256 bytes (will use 512-byte block: 256 + 16 header → rounds to 512)
         * auto ptr_result = buddy->alloc(256, false);
         * 
         * if (ptr_result.hasValue()) {
         *     void* ptr = ptr_result.value();
         *     
         *     // Use memory
         *     memset(ptr, 42, 256);
         *     
         *     // Free when done
         *     buddy->return_element(ptr);
         * } else {
         *     std::cerr << "Allocation failed: " << ptr_result.error().message() << "\n";
         * }
         * @endcode
         * 
         * @par Zero-Initialization Example:
         * @code{.cpp}
         * auto buddy = BuddyAllocator::Heap(4096, 64, 0).value();
         * 
         * // Allocate 512 bytes, zero-initialized
         * auto ptr = buddy->alloc(512, true).value();
         * 
         * // All bytes are guaranteed to be zero
         * uint8_t* data = static_cast<uint8_t*>(ptr);
         * assert(data[0] == 0);
         * assert(data[511] == 0);
         * 
         * buddy->return_element(ptr);
         * @endcode
         * 
         * @par Multiple Allocations:
         * @code{.cpp}
         * auto buddy = BuddyAllocator::Heap(16384, 64, 0).value();
         * 
         * std::vector<void*> ptrs;
         * 
         * // Allocate until exhaustion
         * for (int i = 0; i < 100; ++i) {
         *     auto ptr = buddy->alloc(128, false);
         *     if (!ptr.hasValue()) {
         *         break;  // Pool exhausted
         *     }
         *     ptrs.push_back(ptr.value());
         * }
         * 
         * std::cout << "Allocated " << ptrs.size() << " blocks\n";
         * 
         * // Free all
         * for (void* ptr : ptrs) {
         *     buddy->return_element(ptr);
         * }
         * @endcode
         * 
         * @see alloc_aligned() Allocation with specific alignment
         * @see realloc() Resize existing allocation
         * @see return_element() Free allocated memory
         * @see largest_block() Check maximum allocatable size
         */
        Expected<void*> alloc(size_t bytes, bool zeroed) override;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Allocate aligned memory from the buddy allocator
         * 
         * @param bytes Number of bytes to allocate (must be > 0)
         * @param alignment Required alignment in bytes (0 = alignof(max_align_t), will be rounded up to power of 2)
         * @param zeroed If true, zero-initialize the allocated memory
         * 
         * @return Expected containing pointer to aligned memory on success, or error on failure
         * 
         * @details Allocates memory with a specific alignment guarantee. The allocation
         *          process is similar to alloc(), but allocates a larger block to ensure
         *          the user pointer can be positioned at the requested alignment.
         * 
         *          **Allocation Process:**
         *          1. Normalize alignment to power of 2 (e.g., 48 → 64)
         *          2. Calculate total: bytes + header + (alignment - 1) for worst-case padding
         *          3. Round to next power of 2 (minimum: min_block_size)
         *          4. Find and allocate block (with splitting if needed)
         *          5. Find aligned position within block
         *          6. Place header immediately before aligned position
         *          7. Return aligned pointer
         * 
         *          **Alignment Guarantee:**
         *          The returned pointer is GUARANTEED to satisfy:
         *          reinterpret_cast<uintptr_t>(ptr) % alignment == 0
         * 
         *          **Block Sizing with Alignment:**
         *          - Request 256 bytes, 64-byte align → 256 + 16 + 63 = 335 → 512 bytes
         *          - Request 512 bytes, 256-byte align → 512 + 16 + 255 = 783 → 1024 bytes
         * 
         *          **Overhead:**
         *          Aligned allocations use more memory than alloc() due to:
         *          - Larger block size to accommodate alignment padding
         *          - Potential unused space before and after user data
         * 
         * @note alignment=0 uses the platform's natural alignment (alignof(max_align_t))
         * 
         * @note Non-power-of-2 alignments are automatically rounded up. For example,
         *       requesting 48-byte alignment will be rounded to 64 bytes.
         * 
         * @note The header is placed immediately before the aligned user pointer, NOT
         *       at the block start (unlike alloc()).
         * 
         * @throws ArgumentError If bytes is zero
         * @throws AlignmentError If alignment is invalid after normalization
         * @throws CapacityOverflowError If bytes + header + alignment causes overflow
         * @throws MemoryError If no free blocks available, insufficient space for alignment,
         *                     or request exceeds pool capacity
         * 
         * @par Basic Alignment Example:
         * @code{.cpp}
         * auto buddy = BuddyAllocator::Heap(16384, 64, 0).value();
         * 
         * // Allocate 512 bytes with 256-byte alignment
         * auto ptr_result = buddy->alloc_aligned(512, 256, false);
         * 
         * if (ptr_result.hasValue()) {
         *     void* ptr = ptr_result.value();
         *     
         *     // Verify alignment
         *     uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
         *     assert(addr % 256 == 0);
         *     
         *     // Use memory
         *     memcpy(ptr, data, 512);
         *     
         *     buddy->return_element(ptr);
         * }
         * @endcode
         * 
         * @par SIMD/Cache-Line Alignment:
         * @code{.cpp}
         * auto buddy = BuddyAllocator::Heap(32768, 64, 0).value();
         * 
         * // Allocate for SIMD operations (32-byte alignment)
         * auto simd_ptr = buddy->alloc_aligned(1024, 32, true).value();
         * 
         * // Safe for AVX operations
         * __m256* vec = reinterpret_cast<__m256*>(simd_ptr);
         * // ... SIMD code ...
         * 
         * // Allocate for cache-line alignment (64 bytes)
         * auto cache_ptr = buddy->alloc_aligned(2048, 64, false).value();
         * 
         * buddy->return_element(simd_ptr);
         * buddy->return_element(cache_ptr);
         * @endcode
         * 
         * @par Page Alignment Example:
         * @code{.cpp}
         * auto buddy = BuddyAllocator::Heap(1024 * 1024, 64, 0).value();
         * 
         * // Allocate page-aligned memory (4096 bytes)
         * auto page_ptr = buddy->alloc_aligned(8192, 4096, false);
         * 
         * if (page_ptr.hasValue()) {
         *     void* ptr = page_ptr.value();
         *     
         *     // Guaranteed page-aligned
         *     assert(reinterpret_cast<uintptr_t>(ptr) % 4096 == 0);
         *     
         *     // Useful for memory mapping, DMA, etc.
         *     
         *     buddy->return_element(ptr);
         * }
         * @endcode
         * 
         * @par Non-Power-of-2 Alignment:
         * @code{.cpp}
         * auto buddy = BuddyAllocator::Heap(8192, 64, 0).value();
         * 
         * // Request 48-byte alignment (not power of 2)
         * auto ptr = buddy->alloc_aligned(256, 48, false).value();
         * 
         * // Actual alignment will be 64 (next power of 2)
         * uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
         * assert(addr % 64 == 0);
         * 
         * buddy->return_element(ptr);
         * @endcode
         * 
         * @par Zero-Initialization with Alignment:
         * @code{.cpp}
         * auto buddy = BuddyAllocator::Heap(16384, 64, 0).value();
         * 
         * // Allocate aligned and zeroed
         * auto ptr = buddy->alloc_aligned(1024, 128, true).value();
         * 
         * // Both alignment AND zero-initialization guaranteed
         * assert(reinterpret_cast<uintptr_t>(ptr) % 128 == 0);
         * 
         * uint8_t* data = static_cast<uint8_t*>(ptr);
         * for (size_t i = 0; i < 1024; ++i) {
         *     assert(data[i] == 0);
         * }
         * 
         * buddy->return_element(ptr);
         * @endcode
         * 
         * @see alloc() Basic allocation without alignment guarantee
         * @see realloc_aligned() Resize with alignment preserved
         * @see return_element() Free allocated memory
         */
        Expected<void*> alloc_aligned(size_t bytes, size_t alignment, bool zeroed) override;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Resize an existing allocation
         * 
         * @param ptr Pointer to existing allocation (or nullptr)
         * @param old_bytes Current size of the allocation in bytes
         * @param new_bytes Desired new size in bytes
         * @param zeroed If true, zero-initialize any newly allocated space (when growing)
         * 
         * @return Expected containing pointer to resized allocation on success, or error on failure
         * 
         * @details Resizes an existing allocation to a new size. The behavior depends on
         *          whether the allocation is growing, shrinking, or being created/freed.
         * 
         *          **Special Cases:**
         *          - realloc(nullptr, 0, n) → Equivalent to alloc(n)
         *          - realloc(ptr, old, 0) → Frees ptr, returns nullptr (success)
         *          - realloc(ptr, 0, n) → Error (old_bytes must be non-zero with valid ptr)
         * 
         *          **Shrinking (new_bytes <= current block capacity):**
         *          - Returns same pointer (no reallocation)
         *          - Block is NOT split or resized (buddy system doesn't support shrinking)
         *          - If zeroed=true and new_bytes > old_bytes, zeros the extra space
         *          - Very fast (O(1)) - no memory movement
         * 
         *          **Growing (new_bytes > current block capacity):**
         *          - Allocates new larger block via alloc()
         *          - Copies min(old_bytes, usable_old) bytes to new block
         *          - Frees old block (triggers coalescing)
         *          - Returns new pointer (old pointer becomes invalid)
         *          - If new allocation fails, old pointer remains valid
         * 
         *          **Block Capacity:**
         *          The usable capacity of a block is determined by its order (power-of-2 size)
         *          minus the header. For example, a block with order=8 (256 bytes) has
         *          240 bytes usable (256 - 16 byte header).
         * 
         *          **Data Preservation:**
         *          When growing, all data from the old allocation is copied to the new one.
         *          The copy size is min(old_bytes, actual_usable_capacity) to prevent
         *          reading beyond the old allocation.
         * 
         * @note The returned pointer may be different from the input pointer, even when
         *       shrinking. Always use the returned pointer.
         * 
         * @note When growing fails, the old pointer remains valid and unchanged. This
         *       provides transactional semantics - the operation either succeeds completely
         *       or leaves the allocation in its original state.
         * 
         * @note old_bytes is used to determine how much data to preserve when growing.
         *       Providing an incorrect value may result in data loss or reading invalid memory.
         * 
         * @throws ArgumentError If ptr is non-null but old_bytes is zero
         * @throws ArgumentError If ptr is nullptr and new_bytes is zero
         * @throws CapacityOverflowError If new_bytes causes arithmetic overflow
         * @throws MemoryError If growing and no free blocks available
         * 
         * @par Growing Example:
         * @code{.cpp}
         * auto buddy = BuddyAllocator::Heap(8192, 64, 0).value();
         * 
         * // Initial allocation: 128 bytes
         * auto ptr = buddy->alloc(128, false).value();
         * memset(ptr, 42, 128);
         * 
         * // Grow to 512 bytes
         * auto new_ptr_result = buddy->realloc(ptr, 128, 512, false);
         * 
         * if (new_ptr_result.hasValue()) {
         *     void* new_ptr = new_ptr_result.value();
         *     
         *     // Data preserved (first 128 bytes still contain 42)
         *     uint8_t* data = static_cast<uint8_t*>(new_ptr);
         *     assert(data[0] == 42);
         *     assert(data[127] == 42);
         *     
         *     // New space available (bytes 128-511)
         *     memset(data + 128, 0, 384);
         *     
         *     buddy->return_element(new_ptr);
         * } else {
         *     // Old pointer still valid on failure
         *     buddy->return_element(ptr);
         * }
         * @endcode
         * 
         * @par Shrinking Example:
         * @code{.cpp}
         * auto buddy = BuddyAllocator::Heap(4096, 64, 0).value();
         * 
         * // Allocate 512 bytes (gets a power-of-2 block)
         * auto ptr = buddy->alloc(512, false).value();
         * 
         * // Shrink to 256 bytes
         * auto result = buddy->realloc(ptr, 512, 256, false);
         * 
         * // Returns same pointer (no reallocation needed)
         * assert(result.value() == ptr);
         * 
         * // Still using same block, just treating it as smaller
         * buddy->return_element(result.value());
         * @endcode
         * 
         * @par Realloc from nullptr:
         * @code{.cpp}
         * auto buddy = BuddyAllocator::Heap(4096, 64, 0).value();
         * 
         * void* ptr = nullptr;
         * 
         * // Realloc from nullptr acts as alloc
         * auto result = buddy->realloc(ptr, 0, 256, true);
         * 
         * if (result.hasValue()) {
         *     ptr = result.value();
         *     
         *     // Memory is zero-initialized
         *     uint8_t* data = static_cast<uint8_t*>(ptr);
         *     assert(data[0] == 0);
         *     
         *     buddy->return_element(ptr);
         * }
         * @endcode
         * 
         * @par Realloc to Zero (Free):
         * @code{.cpp}
         * auto buddy = BuddyAllocator::Heap(4096, 64, 0).value();
         * 
         * auto ptr = buddy->alloc(256, false).value();
         * 
         * // Realloc to zero frees the memory
         * auto result = buddy->realloc(ptr, 256, 0, false);
         * 
         * // Returns nullptr (success), ptr is now invalid
         * assert(result.hasValue());
         * assert(result.value() == nullptr);
         * @endcode
         * 
         * @par Growth with Zero-Fill:
         * @code{.cpp}
         * auto buddy = BuddyAllocator::Heap(8192, 64, 0).value();
         * 
         * auto ptr = buddy->alloc(128, false).value();
         * memset(ptr, 'A', 128);
         * 
         * // Grow to 512 with zero-fill
         * auto new_ptr = buddy->realloc(ptr, 128, 512, true).value();
         * 
         * uint8_t* data = static_cast<uint8_t*>(new_ptr);
         * 
         * // Old data preserved
         * assert(data[0] == 'A');
         * assert(data[127] == 'A');
         * 
         * // New space zeroed
         * assert(data[128] == 0);
         * assert(data[511] == 0);
         * 
         * buddy->return_element(new_ptr);
         * @endcode
         * 
         * @see alloc() Initial allocation
         * @see realloc_aligned() Realloc with alignment preserved
         * @see return_element() Free memory
         */
        Expected<void*> realloc(void* ptr, size_t old_bytes, size_t new_bytes, bool zeroed) override;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Resize an existing allocation while preserving alignment
         * 
         * @param ptr Pointer to existing allocation (or nullptr)
         * @param old_bytes Current size of the allocation in bytes
         * @param new_bytes Desired new size in bytes
         * @param alignment Required alignment in bytes (0 = alignof(max_align_t), will be rounded up to power of 2)
         * @param zeroed If true, zero-initialize any newly allocated space (when growing)
         * 
         * @return Expected containing pointer to resized aligned allocation on success, or error on failure
         * 
         * @details Resizes an existing allocation while maintaining alignment guarantees.
         *          Behaves similarly to realloc() but with additional alignment handling.
         * 
         *          **Special Cases:**
         *          - realloc_aligned(nullptr, 0, n, a) → Equivalent to alloc_aligned(n, a)
         *          - realloc_aligned(ptr, old, 0, a) → Frees ptr, returns nullptr (success)
         *          - realloc_aligned(ptr, 0, n, a) → Error (old_bytes must be non-zero)
         * 
         *          **In-Place Reuse (no reallocation):**
         *          Occurs when ALL of these conditions are met:
         *          - new_bytes fits in current block capacity
         *          - ptr already satisfies the requested alignment
         *          Returns same pointer, very fast (O(1))
         * 
         *          **Reallocation Required:**
         *          Occurs when ANY of these is true:
         *          - new_bytes exceeds current block capacity
         *          - ptr doesn't satisfy requested alignment (alignment changed)
         *          Allocates new block with alloc_aligned(), copies data, frees old block
         * 
         *          **Alignment Changes:**
         *          You can change alignment during realloc. For example, reallocating
         *          from 32-byte to 64-byte alignment will trigger a new allocation even
         *          if the size fits in the current block.
         * 
         *          **Data Preservation:**
         *          When reallocating, copies min(old_bytes, actual_usable_capacity) bytes
         *          to preserve all data from the old allocation.
         * 
         * @note alignment is normalized to a power of 2. Requesting 48 will be rounded to 64.
         * 
         * @note The returned pointer may be different from the input pointer. Always use
         *       the returned pointer and consider the old pointer invalid.
         * 
         * @note When growing fails, the old pointer remains valid and unchanged.
         * 
         * @note Changing alignment (even to a less strict value) may trigger reallocation.
         * 
         * @throws ArgumentError If ptr is non-null but old_bytes is zero
         * @throws AlignmentError If alignment is invalid after normalization
         * @throws CapacityOverflowError If new_bytes + alignment causes overflow
         * @throws MemoryError If growing and no free blocks available or insufficient space for alignment
         * 
         * @par Growing with Same Alignment:
         * @code{.cpp}
         * auto buddy = BuddyAllocator::Heap(16384, 64, 0).value();
         * 
         * // Allocate 256 bytes with 128-byte alignment
         * auto ptr = buddy->alloc_aligned(256, 128, false).value();
         * memset(ptr, 'X', 256);
         * 
         * // Grow to 1024 bytes, keep 128-byte alignment
         * auto new_ptr_result = buddy->realloc_aligned(ptr, 256, 1024, 128, false);
         * 
         * if (new_ptr_result.hasValue()) {
         *     void* new_ptr = new_ptr_result.value();
         *     
         *     // Alignment preserved
         *     assert(reinterpret_cast<uintptr_t>(new_ptr) % 128 == 0);
         *     
         *     // Data preserved
         *     uint8_t* data = static_cast<uint8_t*>(new_ptr);
         *     assert(data[0] == 'X');
         *     
         *     buddy->return_element(new_ptr);
         * }
         * @endcode
         * 
         * @par In-Place Reuse:
         * @code{.cpp}
         * auto buddy = BuddyAllocator::Heap(8192, 64, 0).value();
         * 
         * // Allocate with 64-byte alignment (gets larger block for padding)
         * auto ptr = buddy->alloc_aligned(256, 64, false).value();
         * 
         * // Shrink with same alignment
         * auto result = buddy->realloc_aligned(ptr, 256, 128, 64, false);
         * 
         * // Same pointer returned (in-place reuse)
         * if (result.hasValue()) {
         *     assert(result.value() == ptr);
         *     buddy->return_element(result.value());
         * }
         * @endcode
         * 
         * @par Changing Alignment:
         * @code{.cpp}
         * auto buddy = BuddyAllocator::Heap(16384, 64, 0).value();
         * 
         * // Start with 32-byte alignment
         * auto ptr = buddy->alloc_aligned(512, 32, false).value();
         * memset(ptr, 'A', 512);
         * 
         * // Change to 256-byte alignment (same size)
         * auto new_ptr = buddy->realloc_aligned(ptr, 512, 512, 256, false).value();
         * 
         * // Different pointer (reallocation occurred due to alignment change)
         * // But data is preserved
         * uint8_t* data = static_cast<uint8_t*>(new_ptr);
         * assert(data[0] == 'A');
         * assert(reinterpret_cast<uintptr_t>(new_ptr) % 256 == 0);
         * 
         * buddy->return_element(new_ptr);
         * @endcode
         * 
         * @par Realloc SIMD Buffer:
         * @code{.cpp}
         * auto buddy = BuddyAllocator::Heap(32768, 64, 0).value();
         * 
         * // Start with AVX alignment (32 bytes)
         * auto ptr = buddy->alloc_aligned(1024, 32, true).value();
         * 
         * // Process data with AVX...
         * __m256* vec = reinterpret_cast<__m256*>(ptr);
         * // ... SIMD operations ...
         * 
         * // Grow buffer for AVX-512 (64-byte alignment)
         * auto new_ptr = buddy->realloc_aligned(ptr, 1024, 2048, 64, false).value();
         * 
         * // Now suitable for AVX-512
         * __m512* vec512 = reinterpret_cast<__m512*>(new_ptr);
         * // ... AVX-512 operations ...
         * 
         * buddy->return_element(new_ptr);
         * @endcode
         * 
         * @par Growth with Zero-Fill and Alignment:
         * @code{.cpp}
         * auto buddy = BuddyAllocator::Heap(16384, 64, 0).value();
         * 
         * auto ptr = buddy->alloc_aligned(256, 128, false).value();
         * memset(ptr, 'B', 256);
         * 
         * // Grow to 1024 bytes with zero-fill, maintain alignment
         * auto new_ptr = buddy->realloc_aligned(ptr, 256, 1024, 128, true).value();
         * 
         * uint8_t* data = static_cast<uint8_t*>(new_ptr);
         * 
         * // Old data preserved
         * assert(data[0] == 'B');
         * assert(data[255] == 'B');
         * 
         * // New space zeroed (implementation zeros from new_bytes onward in new block)
         * // Note: Exact zeroing behavior may vary based on block positioning
         * 
         * // Alignment maintained
         * assert(reinterpret_cast<uintptr_t>(new_ptr) % 128 == 0);
         * 
         * buddy->return_element(new_ptr);
         * @endcode
         * 
         * @par Realloc from nullptr with Alignment:
         * @code{.cpp}
         * auto buddy = BuddyAllocator::Heap(8192, 64, 0).value();
         * 
         * void* ptr = nullptr;
         * 
         * // Acts as alloc_aligned
         * auto result = buddy->realloc_aligned(ptr, 0, 512, 256, true);
         * 
         * if (result.hasValue()) {
         *     ptr = result.value();
         *     
         *     // Aligned and zeroed
         *     assert(reinterpret_cast<uintptr_t>(ptr) % 256 == 0);
         *     
         *     buddy->return_element(ptr);
         * }
         * @endcode
         * 
         * @see alloc_aligned() Initial aligned allocation
         * @see realloc() Realloc without alignment requirements
         * @see return_element() Free memory
         */
        Expected<void*> realloc_aligned(void* ptr, size_t old_bytes, size_t new_bytes,
                                     size_t alignment, bool zeroed) override;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Free an allocated block and trigger buddy coalescing
         * 
         * @param ptr Pointer to free (returned from alloc, alloc_aligned, realloc, or realloc_aligned)
         * @param bytes Ignored (present for interface compatibility)
         * @param alignment Ignored (present for interface compatibility)
         * 
         * @details Frees a previously allocated block and attempts to coalesce (merge) it
         *          with its buddy block to form larger free blocks. This is the core
         *          deallocation mechanism of the buddy allocator.
         * 
         *          **Deallocation Process:**
         *          1. Validate pointer is not null (null is silently ignored)
         *          2. Read BuddyHeader located 16 bytes before the user pointer
         *          3. Validate header (order within valid range, offset within pool)
         *          4. Calculate block start from header's block_offset
         *          5. Update accounting (decrease size_)
         *          6. Begin coalescing process
         * 
         *          **Coalescing Algorithm:**
         *          The allocator recursively merges blocks with their buddies:
         *          1. Calculate buddy offset: current_offset XOR (2^current_order)
         *          2. Check if buddy is in the free list at this level
         *          3. If buddy is free:
         *             - Remove buddy from free list
         *             - Merge with current block (use lower address)
         *             - Increase order (double the size)
         *             - Repeat at next level
         *          4. If buddy is NOT free (still allocated):
         *             - Stop coalescing
         *             - Insert current block into appropriate free list
         * 
         *          **Coalescing Example:**
         *          @code
         *          Pool: 1024 bytes (order 10)
         *          
         *          Free block at offset 0, order 7 (128 bytes):
         *            Buddy at offset: 0 XOR 128 = 128
         *            If buddy at 128 is free:
         *              Merge → 256-byte block at offset 0, order 8
         *              
         *            New buddy at: 0 XOR 256 = 256
         *            If buddy at 256 is free:
         *              Merge → 512-byte block at offset 0, order 9
         *              
         *            Continue until buddy not free or reach max_order
         *          @endcode
         * 
         *          **Performance:**
         *          O(log n) where n is the number of size levels, due to recursive
         *          coalescing up the buddy tree.
         * 
         * @note The bytes and alignment parameters are ignored by BuddyAllocator.
         *       They exist only for base class interface compatibility. The allocator
         *       determines block size from the internal header.
         * 
         * @note Passing nullptr is safe and does nothing (similar to free(nullptr)).
         * 
         * @note Invalid pointers (wrong allocator, already freed, corrupted header)
         *       are handled silently. The function returns without error but may not
         *       free the memory correctly.
         * 
         * @note Double-free results in undefined behavior. The allocator may detect
         *       some cases via header validation but cannot catch all scenarios.
         * 
         * @note After this call, ptr becomes invalid. Accessing it results in
         *       undefined behavior.
         * 
         * @warning Do not free pointers from a different BuddyAllocator instance.
         *          Each allocator manages its own distinct pool.
         * 
         * @warning Do not modify the memory before the returned pointer. The header
         *          region must remain intact for proper deallocation.
         * 
         * @par Basic Usage:
         * @code{.cpp}
         * auto buddy = BuddyAllocator::Heap(4096, 64, 0).value();
         * 
         * auto ptr = buddy->alloc(256, false).value();
         * 
         * // Use memory...
         * memset(ptr, 0, 256);
         * 
         * // Free when done (automatic coalescing)
         * buddy->return_element(ptr);
         * 
         * // ptr is now invalid - do not use!
         * @endcode
         * 
         * @par Coalescing Demonstration:
         * @code{.cpp}
         * auto buddy = BuddyAllocator::Heap(4096, 64, 0).value();
         * 
         * // Allocate three adjacent blocks
         * auto p1 = buddy->alloc(128, false).value();
         * auto p2 = buddy->alloc(128, false).value();
         * auto p3 = buddy->alloc(128, false).value();
         * 
         * size_t largest_before = buddy->largest_block();
         * 
         * // Free all three - they coalesce into larger block
         * buddy->return_element(p1);
         * buddy->return_element(p2);
         * buddy->return_element(p3);
         * 
         * size_t largest_after = buddy->largest_block();
         * 
         * // Largest block increased due to coalescing
         * assert(largest_after > largest_before);
         * @endcode
         * 
         * @par Free Order Doesn't Matter:
         * @code{.cpp}
         * auto buddy = BuddyAllocator::Heap(8192, 64, 0).value();
         * 
         * // Allocate four blocks
         * auto p1 = buddy->alloc(128, false).value();
         * auto p2 = buddy->alloc(128, false).value();
         * auto p3 = buddy->alloc(128, false).value();
         * auto p4 = buddy->alloc(128, false).value();
         * 
         * // Free in random order - coalescing still works
         * buddy->return_element(p3);
         * buddy->return_element(p1);
         * buddy->return_element(p4);
         * buddy->return_element(p2);
         * 
         * // All blocks coalesced back
         * @endcode
         * 
         * @par NULL Pointer Handling:
         * @code{.cpp}
         * auto buddy = BuddyAllocator::Heap(4096, 64, 0).value();
         * 
         * void* ptr = nullptr;
         * 
         * // Safe - does nothing
         * buddy->return_element(ptr);
         * 
         * // Conditional freeing is safe
         * auto maybe_ptr = buddy->alloc(256, false);
         * if (maybe_ptr.hasValue()) {
         *     ptr = maybe_ptr.value();
         *     // ... use ptr ...
         * }
         * 
         * // Safe even if allocation failed (ptr is nullptr)
         * buddy->return_element(ptr);
         * @endcode
         * 
         * @par Fragmentation Reduction:
         * @code{.cpp}
         * auto buddy = BuddyAllocator::Heap(8192, 64, 0).value();
         * 
         * // Allocate many blocks (creates fragmentation)
         * std::vector<void*> ptrs;
         * for (int i = 0; i < 30; ++i) {
         *     auto ptr = buddy->alloc(128, false);
         *     if (ptr.hasValue()) {
         *         ptrs.push_back(ptr.value());
         *     }
         * }
         * 
         * // Free every other block
         * for (size_t i = 0; i < ptrs.size(); i += 2) {
         *     buddy->return_element(ptrs[i]);
         * }
         * 
         * size_t fragmented = buddy->largest_block();
         * 
         * // Free remaining blocks - coalescing reduces fragmentation
         * for (size_t i = 1; i < ptrs.size(); i += 2) {
         *     buddy->return_element(ptrs[i]);
         * }
         * 
         * size_t after_coalesce = buddy->largest_block();
         * 
         * // Fragmentation reduced through coalescing
         * assert(after_coalesce > fragmented);
         * @endcode
         * 
         * @see alloc() Allocate memory
         * @see alloc_aligned() Allocate aligned memory
         * @see reset() Bulk deallocation of all blocks
         * @see largest_block() Check largest free block after coalescing
         */
        void return_element(void* ptr, size_t bytes = 0, size_t alignment = 0) override;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Reset allocator to initial state, freeing all allocations
         * 
         * @param trim Ignored (present for interface compatibility)
         * 
         * @return true on success, false if allocator is not properly initialized
         * 
         * @details Resets the buddy allocator to its pristine initial state by clearing
         *          all free lists and placing a single large free block spanning the
         *          entire pool at the top level.
         * 
         *          **Reset Process:**
         *          1. Validate allocator is initialized (base, pool_size, num_levels valid)
         *          2. Clear all free lists (set all levels to nullptr)
         *          3. Create initial free block at pool base
         *          4. Place block in top-level free list (max_order)
         *          5. Reset accounting (size_ = 0)
         * 
         *          **Effect:**
         *          - All previous allocations become invalid
         *          - Memory pool returns to single large free block
         *          - All fragmentation eliminated
         *          - Allocator ready for reuse
         * 
         *          **Performance:**
         *          O(n) where n is the number of free list levels, but typically very fast
         *          since it just clears an array and sets a few pointers.
         * 
         *          **Use Cases:**
         *          - Bulk cleanup faster than freeing individually
         *          - Per-frame allocations in game engines
         *          - Request/response cycle allocations in servers
         *          - Temporary computation scratch space
         *          - Test cleanup between test cases
         * 
         * @note The trim parameter is ignored by BuddyAllocator. The pool size is fixed
         *       and cannot be trimmed. The parameter exists for interface compatibility.
         * 
         * @note All outstanding pointers become invalid after reset(). Accessing them
         *       results in undefined behavior.
         * 
         * @note The OS-backed memory pool is NOT freed. reset() only clears the internal
         *       free list structure. To free OS memory, destroy the BuddyAllocator.
         * 
         * @note reset() is much faster than individually freeing many allocations,
         *       especially when there are hundreds or thousands of active allocations.
         * 
         * @warning After reset(), ALL pointers obtained from this allocator become invalid.
         *          Ensure no code will access these pointers after calling reset().
         * 
         * @par Basic Reset:
         * @code{.cpp}
         * auto buddy = BuddyAllocator::Heap(8192, 64, 0).value();
         * 
         * // Make many allocations
         * for (int i = 0; i < 50; ++i) {
         *     buddy->alloc(128, false);
         * }
         * 
         * size_t used_before = buddy->size();
         * assert(used_before > 0);
         * 
         * // Reset to initial state
         * bool ok = buddy->reset();
         * assert(ok);
         * 
         * // All allocations gone
         * size_t used_after = buddy->size();
         * assert(used_after == 0);
         * 
         * // Allocator ready for reuse
         * auto ptr = buddy->alloc(256, false);
         * assert(ptr.hasValue());
         * @endcode
         * 
         * @par Per-Frame Allocations:
         * @code{.cpp}
         * auto buddy = BuddyAllocator::Heap(1024 * 1024, 64, 0).value();
         * 
         * while (game_running) {
         *     // Frame start - allocator is clean
         *     
         *     // Allocate temporary data for this frame
         *     auto render_data = buddy->alloc(10000, false).value();
         *     auto physics_data = buddy->alloc(5000, false).value();
         *     auto ai_data = buddy->alloc(8000, false).value();
         *     
         *     // Process frame...
         *     render_frame(render_data);
         *     update_physics(physics_data);
         *     update_ai(ai_data);
         *     
         *     // Frame end - bulk cleanup
         *     buddy->reset();
         *     
         *     // All frame allocations invalidated
         *     // Ready for next frame
         * }
         * @endcode
         * 
         * @par Request/Response Cycle:
         * @code{.cpp}
         * auto buddy = BuddyAllocator::Heap(256 * 1024, 64, 0).value();
         * 
         * void handle_request(const Request& req) {
         *     // Parse request (may allocate parsing buffers)
         *     auto parsed = parse_request(buddy, req);
         *     
         *     // Process request (may allocate temporary structures)
         *     auto response = process(buddy, parsed);
         *     
         *     // Send response
         *     send_response(response);
         *     
         *     // Cleanup all request-scoped allocations
         *     buddy->reset();
         *     
         *     // Ready for next request
         * }
         * @endcode
         * 
         * @par Reset vs Individual Frees:
         * @code{.cpp}
         * auto buddy = BuddyAllocator::Heap(16384, 64, 0).value();
         * 
         * std::vector<void*> ptrs;
         * 
         * // Allocate 500 blocks
         * for (int i = 0; i < 500; ++i) {
         *     auto ptr = buddy->alloc(128, false);
         *     if (ptr.hasValue()) {
         *         ptrs.push_back(ptr.value());
         *     }
         * }
         * 
         * // Option 1: Free individually (slow - 500 operations with coalescing)
         * // for (void* ptr : ptrs) {
         * //     buddy->return_element(ptr);
         * // }
         * 
         * // Option 2: Reset (fast - O(levels) operation)
         * buddy->reset();
         * 
         * // reset() is MUCH faster for bulk cleanup
         * @endcode
         * 
         * @par Test Cleanup:
         * @code{.cpp}
         * class BuddyTest : public ::testing::Test {
         * protected:
         *     UniquePtr<BuddyAllocator, BuddyDeleter> buddy;
         *     
         *     void SetUp() override {
         *         buddy = BuddyAllocator::Heap(8192, 64, 0).value();
         *     }
         *     
         *     void TearDown() override {
         *         // Clean up between tests
         *         buddy->reset();
         *     }
         * };
         * 
         * TEST_F(BuddyTest, Test1) {
         *     auto ptr = buddy->alloc(256, false);
         *     // Test logic...
         *     // No need to manually free - TearDown resets
         * }
         * 
         * TEST_F(BuddyTest, Test2) {
         *     // Starts with clean allocator thanks to reset()
         *     auto ptr = buddy->alloc(512, false);
         *     // Test logic...
         * }
         * @endcode
         * 
         * @par Temporary Computation Space:
         * @code{.cpp}
         * auto buddy = BuddyAllocator::Heap(512 * 1024, 64, 0).value();
         * 
         * void matrix_multiply(Matrix& result, const Matrix& a, const Matrix& b) {
         *     // Allocate temporary workspace
         *     auto temp = buddy->alloc(a.rows * b.cols * sizeof(double), true).value();
         *     
         *     // Perform computation using temp buffer
         *     // ...
         *     
         *     // Copy result
         *     memcpy(result.data, temp, result.size);
         *     
         *     // Cleanup all temporary allocations
         *     buddy->reset();
         * }
         * @endcode
         * 
         * @see return_element() Free individual allocation
         * @see size() Check current usage
         * @see remaining() Check available space
         */
        bool reset(bool trim = false) override;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Check whether a pointer refers to a valid allocation from this allocator
         *
         * @param ptr Pointer to validate
         *
         * @return true if @p ptr is a valid user pointer returned by this BuddyAllocator,
         *         false otherwise
         *
         * @details Performs a non-throwing validation of a pointer against this allocator's
         *          internal state. The function verifies that:
         *
         *          - @p ptr is non-null
         *          - @p ptr lies within the allocator's managed memory pool
         *          - The internal BuddyHeader immediately preceding @p ptr is readable
         *          - The header encodes a valid block order for this allocator
         *          - The block offset and size are consistent with the pool boundaries
         *          - @p ptr lies within the bounds of the referenced allocation block
         *
         *          This function is designed to be **memory-safe**: it first performs a
         *          range check to ensure that reading the internal header cannot cause
         *          undefined behavior or a segmentation fault, even if @p ptr is invalid
         *          or foreign.
         *
         * @note This function does NOT check whether the block is currently allocated or
         *       free; it only validates that the pointer is structurally consistent with
         *       a block that could have been allocated by this allocator.
         *
         * @note Passing a pointer obtained from a different allocator instance will
         *       return false.
         *
         * @note Passing an interior pointer (i.e., not the original user pointer returned
         *       by alloc/alloc_aligned/realloc*) will return false.
         *
         * @note The function is intentionally side-effect free and does not modify
         *       allocator state.
         *
         * @warning A return value of true does not guarantee that the pointer has not
         *          already been freed or that the underlying memory has not been reused.
         *          Use-after-free remains undefined behavior.
         *
         * @par Typical Usage:
         * @code{.cpp}
         * void* p = buddy->alloc(256, false).value();
         *
         * assert(buddy->is_ptr(p));          // valid
         * assert(!buddy->is_ptr(p + 8));     // interior pointer
         *
         * buddy->return_element(p);
         * assert(buddy->is_ptr(p));          // may still return true (use-after-free!)
         * @endcode
         *
         * @see is_ptr_sized() Pointer validation with size constraint
         * @see return_element() Free memory previously allocated by this allocator
         */
        bool is_ptr(void* ptr) const override;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Check whether a pointer refers to a valid allocation of at least a given size
         *
         * @param ptr Pointer to validate
         * @param bytes Minimum required usable size in bytes
         *
         * @return true if @p ptr is a valid user pointer returned by this BuddyAllocator
         *         and the underlying allocation can accommodate at least @p bytes,
         *         false otherwise
         *
         * @details Extends is_ptr() by additionally verifying that the allocation
         *          referenced by @p ptr has sufficient usable capacity for @p bytes.
         *
         *          Validation steps include:
         *          - All structural checks performed by is_ptr()
         *          - Determination of the allocation block size from the internal header
         *          - Verification that @p bytes does not exceed the block's usable payload
         *            (block size minus header)
         *
         *          This function is **memory-safe** and will not dereference invalid
         *          memory, even if @p ptr is foreign, corrupted, or out of bounds.
         *
         * @note The @p bytes parameter represents a *logical size requirement* and is
         *       not required to match the size originally requested during allocation.
         *
         * @note As with is_ptr(), this function does NOT detect use-after-free. A pointer
         *       that was previously freed may still return true if its header remains
         *       intact and has not been reused.
         *
         * @note Passing @p bytes = 0 will return true for any valid pointer.
         *
         * @par Typical Usage:
         * @code{.cpp}
         * auto p = buddy->alloc(256, false).value();
         *
         * assert(buddy->is_ptr_sized(p, 128));   // fits
         * assert(buddy->is_ptr_sized(p, 256));   // fits
         * assert(!buddy->is_ptr_sized(p, 512));  // exceeds block capacity
         *
         * buddy->return_element(p);
         * @endcode
         *
         * @see is_ptr() Basic pointer validation
         * @see alloc() Allocate memory
         * @see alloc_aligned() Allocate aligned memory
         */
        bool is_ptr_sized(void* ptr, size_t bytes) const override;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Generate a human-readable summary of allocator state
         *
         * @param buffer Destination buffer to receive the formatted statistics text
         * @param buffer_size Size of @p buffer in bytes
         *
         * @return true if statistics were successfully written to @p buffer,
         *         false if an error occurred (e.g., invalid buffer or insufficient space)
         *
         * @details Writes a textual report describing the current state of the
         *          BuddyAllocator into the user-provided buffer. The report includes:
         *
         *          - Total pool size
         *          - Minimum and maximum block sizes
         *          - Currently allocated bytes
         *          - Remaining free capacity
         *          - Largest available free block
         *          - Overall utilization percentage
         *          - Per-level free list statistics (block count and free bytes)
         *
         *          The output is intended for diagnostics, debugging, and monitoring.
         *          The exact formatting is implementation-defined but stable enough
         *          for human consumption.
         *
         *          This function performs no allocations and does not modify allocator
         *          state.
         *
         * @note If @p buffer is null or @p buffer_size is zero, the function returns
         *       false and no output is produced.
         *
         * @note If the buffer is too small to hold the full report, the function
         *       returns false. Partial output may have been written to @p buffer.
         *
         * @note The statistics represent a snapshot at the time of the call and may
         *       become stale immediately in multi-threaded environments.
         *
         * @par Typical Usage:
         * @code{.cpp}
         * char buf[2048];
         *
         * if (buddy->stats(buf, sizeof(buf))) {
         *     std::puts(buf);
         * } else {
         *     std::cerr << "Failed to generate allocator stats\n";
         * }
         * @endcode
         *
         * @see remaining() Query available capacity
         * @see largest_block() Query largest contiguous free block
         */
        bool stats(char* buffer, size_t buffer_size) const override;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Query the number of bytes currently available for allocation
         *
         * @return Number of free bytes remaining in the allocator's memory pool
         *
         * @details Returns the difference between the total pool size and the number
         *          of bytes currently allocated. This represents the *aggregate*
         *          free capacity and does NOT account for fragmentation.
         *
         *          As a result, it is possible for remaining() to return a non-zero
         *          value even when a large allocation cannot be satisfied due to
         *          fragmentation.
         *
         *          This function is constant-time and does not modify allocator state.
         *
         * @note To determine the maximum single allocation that can currently succeed,
         *       use largest_block() instead.
         *
         * @par Example:
         * @code{.cpp}
         * auto used = buddy->size();
         * auto free = buddy->remaining();
         *
         * std::cout << "Used: " << used
         *           << ", Remaining: " << free << '\n';
         * @endcode
         *
         * @see size() Query currently allocated bytes
         * @see largest_block() Query largest contiguous free block
         */
        size_t remaining() const noexcept override;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Save allocator state (unsupported for BuddyAllocator)
         *
         * @return Always returns nullptr
         *
         * @details This function is part of the Allocator base-class interface but is
         *          intentionally unsupported by BuddyAllocator.
         *
         *          Buddy allocators manage complex free-list and coalescing state that
         *          is not trivially serializable or restorable without significant
         *          overhead. As a result, checkpointing is not provided.
         *
         * @note Calling this function always returns nullptr and has no side effects.
         *
         * @note Code that relies on save()/restore() semantics should not use
         *       BuddyAllocator.
         *
         * @see restore() Corresponding restore operation (also unsupported)
         */
        void* save() const override { return nullptr; }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Restore allocator state from a checkpoint (unsupported for BuddyAllocator)
         *
         * @param checkpoint Ignored
         *
         * @return Always returns false
         *
         * @details This function is part of the Allocator base-class interface but is
         *          intentionally unsupported by BuddyAllocator.
         *
         *          Because save() does not produce a valid checkpoint, restore() always
         *          fails and performs no action.
         *
         * @note The allocator state is unchanged by this call.
         *
         * @note Code that requires allocator checkpointing should use an allocator
         *       that explicitly supports save/restore semantics (e.g., arena allocators).
         *
         * @see save() Corresponding save operation (unsupported)
         * @see reset() Clear all allocations and return allocator to initial state
         */
        bool restore(void* checkpoint) override {
            (void)checkpoint;
            return false;
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Query the size of the largest contiguous free block
         *
         * @return Size in bytes of the largest currently available free block,
         *         or 0 if no free blocks are available
         *
         * @details Scans the allocator's free lists from the largest block size
         *          down to the smallest and returns the size of the first non-empty
         *          free list encountered.
         *
         *          This value represents the **maximum single allocation** that can
         *          currently succeed (ignoring internal header and alignment overhead).
         *
         *          Unlike remaining(), which reports total free capacity, this function
         *          accounts for **fragmentation**. A large remaining() value does not
         *          guarantee that a large allocation can succeed if free memory is
         *          split across smaller blocks.
         *
         *          The function runs in O(L) time, where L is the number of size levels
         *          (log₂(pool_size / min_block_size)).
         *
         * @note The returned size is the raw block size managed by the buddy system.
         *       The actual maximum allocatable user payload may be smaller due to
         *       internal headers and alignment requirements.
         *
         * @note The result may change immediately after any allocation or deallocation.
         *
         * @par Example:
         * @code{.cpp}
         * auto buddy = BuddyAllocator::Heap(8192, 64, 0).value();
         *
         * auto p1 = buddy->alloc(128, false);
         * auto p2 = buddy->alloc(128, false);
         *
         * size_t largest = buddy->largest_block();
         *
         * if (largest < 1024) {
         *     std::cout << "Fragmentation limits large allocations\n";
         * }
         * @endcode
         *
         * @see remaining() Total free capacity
         * @see alloc() Allocation behavior
         * @see stats() Detailed free-list diagnostics
         */
        size_t largest_block() const noexcept;
// -------------------------------------------------------------------------------- 

        /**
         * @brief Query the minimum block size managed by the allocator
         *
         * @return Minimum allocation block size in bytes
         *
         * @details Returns the smallest block size (in bytes) that this BuddyAllocator
         *          can manage internally. All allocations are rounded up to at least
         *          this size (after accounting for internal headers and alignment).
         *
         *          The value is determined during allocator creation and is always
         *          a power of two.
         *
         *          Requests smaller than this size will still consume at least one
         *          block of this size.
         *
         * @note The minimum block size may be larger than the value originally
         *       requested at creation time, due to rounding to a power of two and
         *       ensuring space for internal headers and alignment.
         *
         * @par Example:
         * @code{.cpp}
         * auto buddy = BuddyAllocator::Heap(4096, 100, 0).value();
         *
         * // Requested min block size: 100 bytes
         * // Actual min block size (rounded): 128 bytes
         * assert(buddy->min_block_size() == 128);
         * @endcode
         *
         * @see max_block_size() Maximum possible block size
         * @see Heap() Allocator creation and normalization rules
         */
        size_t min_block_size() const noexcept {
            return (size_t)1 << min_order_;
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Query the maximum block size managed by the allocator
         *
         * @return Maximum allocation block size in bytes
         *
         * @details Returns the size of the largest block that the allocator can manage,
         *          which corresponds to the full memory pool size.
         *
         *          This value represents the theoretical upper bound for a single
         *          allocation before considering fragmentation, headers, and alignment
         *          overhead.
         *
         *          The value is always a power of two and is fixed for the lifetime of
         *          the allocator.
         *
         * @note A request of exactly max_block_size() bytes may not succeed if
         *       internal headers or alignment padding are required. Use largest_block()
         *       to determine the maximum allocation that can succeed at a given moment.
         *
         * @par Example:
         * @code{.cpp}
         * auto buddy = BuddyAllocator::Heap(5000, 64, 0).value();
         *
         * // Pool rounded up: 5000 -> 8192
         * assert(buddy->max_block_size() == 8192);
         * @endcode
         *
         * @see min_block_size() Minimum allocation block size
         * @see largest_block() Largest currently available block
         * @see remaining() Aggregate free capacity
         */
        size_t max_block_size() const noexcept {
            return (size_t)1 << max_order_;
        }

        friend struct BuddyDeleter;
    };
// ================================================================================ 
// ================================================================================ 

    /**
     * @brief Custom deleter for BuddyAllocator UniquePtr cleanup
     *
     * @param buddy Pointer to BuddyAllocator to delete. May be nullptr.
     *
     * @details This custom deleter is invoked when a
     *          UniquePtr<BuddyAllocator, BuddyDeleter> goes out of scope or is
     *          explicitly reset. It performs complete and ordered cleanup of all
     *          resources owned by a BuddyAllocator instance.
     *
     *          BuddyAllocator always owns all of its internal resources. Unlike
     *          other allocator types, it does not support borrowed memory pools
     *          or user-provided backing storage. As a result, cleanup behavior
     *          is unconditional and uniform.
     *
     *          **Cleanup sequence:**
     *          1. Release the OS-backed memory pool used for allocations
     *             (via BuddyAllocator::os_free)
     *          2. Free the free-lists array allocated during initialization
     *          3. Invoke the BuddyAllocator destructor
     *          4. Free the BuddyAllocator object itself (via ::operator delete)
     *
     *          This ordering guarantees that all allocator-internal state remains
     *          valid for the duration of the destructor and that no memory leaks
     *          occur.
     *
     * @note This deleter is noexcept and never throws exceptions during cleanup.
     *
     * @note Passing a null pointer is safe and results in an immediate no-op.
     *
     * @note All outstanding allocations become invalid once this deleter is
     *       invoked. Accessing previously allocated memory after destruction
     *       results in undefined behavior.
     *
     * @note Users must not call this function directly. It is automatically
     *       invoked by UniquePtr when the BuddyAllocator goes out of scope or
     *       is reset.
     *
     * @warning Pointers returned by this BuddyAllocator must not be freed or
     *          accessed after the allocator has been destroyed.
     *
     * @see Heap() Factory method that creates a BuddyAllocator wrapped in a
     *      UniquePtr with BuddyDeleter
     * @see ~BuddyAllocator() Destructor invoked as part of cleanup
     */
    inline void BuddyDeleter::operator()(BuddyAllocator* buddy) const noexcept {
        if (!buddy) return;
        
        // Free OS-backed memory pool
        if (buddy->base_ && buddy->pool_size_) {
            BuddyAllocator::os_free(buddy->base_, buddy->pool_size_);
        }
        
        // Free free-lists array
        if (buddy->free_lists_) {
            delete[] buddy->free_lists_;
        }
        
        // Call destructor
        buddy->~BuddyAllocator();
        
        // Free the BuddyAllocator structure itself
        ::operator delete(buddy);
    }

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

        // Capture teardown metadata BEFORE destructor
        BuddyAllocator* buddy = arena->buddy_owner_;
        size_t backing_bytes  = arena->backing_bytes_;

        MemType type = arena->memory_type();
        bool owns    = arena->owns_memory();

        // Run destructor first (frees extra chunks if any)
        arena->~ArenaAllocator();

        // Buddy-backed arena: return entire region to buddy, NOT operator delete
        if (buddy && backing_bytes) {
            buddy->return_element(arena, backing_bytes, alignof(ArenaAllocator));
            return;
        }

        // Heap arena: free base allocation
        if (type == DYNAMIC && owns) {
            ::operator delete(static_cast<void*>(arena));
        }
    }
// ================================================================================ 
// ================================================================================ 

    class SlabAllocator;

    struct SlabDeleter {
        void operator()(SlabAllocator* slab) const noexcept;
    };

    class SlabAllocator : public Allocator {
        friend struct SlabDeleter;
    private:
        BuddyAllocator* buddy_;               // backing buddy allocator (non-owning)

        size_t obj_size_;                     // user-visible object size
        size_t slot_size_;                    // internal stride >= obj_size and >= sizeof(Slot)
        size_t align_;                        // slot alignment (pow2)

        size_t slab_bytes_;                   // per-page allocation size
        size_t page_hdr_bytes_;               // aligned sizeof(Page)
        size_t objs_per_slab_;                // slots per page

        // Tracking
        size_t len_bytes_;                    // logical payload bytes in use (obj_size * live objects)

        struct Slot { Slot* next; };          // intrusive freelist node
        struct Page { Page* next; };          // page header at beginning of each page
                                              
        Page* pages_;
        Slot* free_list_;
// -------------------------------------------------------------------------------- 

        explicit SlabAllocator(BuddyAllocator& buddy) noexcept;
// -------------------------------------------------------------------------------- 

        static size_t align_up(size_t x, size_t a) noexcept;
// -------------------------------------------------------------------------------- 

        bool grow_();                         // allocate a page and populate free list
// -------------------------------------------------------------------------------- 

        Page* find_page_(const void* ptr) const noexcept;
// ================================================================================ 
    public:
        ~SlabAllocator() noexcept override;
// -------------------------------------------------------------------------------- 

        static Expected<cslt::UniquePtr<SlabAllocator, SlabDeleter>>
        WithBuddy(BuddyAllocator& buddy,
                  size_t obj_size,
                  size_t align = 0,
                  size_t slab_bytes_hint = 0);  // FIXED: size_t not slize_t
// -------------------------------------------------------------------------------- 
 
        Expected<void*> alloc(size_t bytes, bool zeroed = false) override;
// -------------------------------------------------------------------------------- 

        Expected<void*> alloc_aligned(size_t bytes, size_t alignment, bool zeroed = false) override;
// -------------------------------------------------------------------------------- 

        Expected<void*> realloc(void* ptr, size_t old_bytes, size_t new_bytes, bool zeroed = false) override;
// -------------------------------------------------------------------------------- 

        Expected<void*> realloc_aligned(void* ptr, size_t old_bytes, size_t new_bytes,
                                        size_t alignment, bool zeroed = false) override;
// -------------------------------------------------------------------------------- 

        void return_element(void* ptr, size_t bytes = 0, size_t alignment = 0) override;  // FIXED: defaults
// -------------------------------------------------------------------------------- 

        bool stats(char* buffer, size_t buffer_size) const override;
// -------------------------------------------------------------------------------- 

        bool reset(bool trim = false) override;  // FIXED: consistent naming
// -------------------------------------------------------------------------------- 

        void* save() const override { return nullptr; }
// -------------------------------------------------------------------------------- 

        bool restore(void* checkpoint) override { 
            (void)checkpoint; 
            return false; 
        }
// -------------------------------------------------------------------------------- 

        bool is_ptr(void* ptr) const override;
// -------------------------------------------------------------------------------- 

        bool is_ptr_sized(void* ptr, size_t bytes) const override;
// -------------------------------------------------------------------------------- 

        size_t remaining() const noexcept override;  // ADDED: base class requirement
// -------------------------------------------------------------------------------- 

        size_t stride() const noexcept { return slot_size_; }
// -------------------------------------------------------------------------------- 

        size_t total_blocks() const noexcept;
// -------------------------------------------------------------------------------- 

        size_t free_blocks() const noexcept;
// -------------------------------------------------------------------------------- 

        size_t in_use_blocks() const noexcept;
    };
// ================================================================================ 
// ================================================================================ 

    inline void SlabDeleter::operator()(SlabAllocator* slab) const noexcept {
        if (!slab) return;
        
        // Save the buddy pointer before calling destructor
        BuddyAllocator* buddy = slab->buddy_;
        
        // Calculate the size that was allocated for the slab structure
        size_t slab_struct_bytes = SlabAllocator::align_up(sizeof(SlabAllocator), 
                                                            alignof(max_align_t));
        
        // Call destructor to free pages
        slab->~SlabAllocator();
        
        // Now free the SlabAllocator structure itself back to buddy
        // CRITICAL: Use return_element, NOT operator delete!
        if (buddy) {
            buddy->return_element(static_cast<void*>(slab), 
                                 slab_struct_bytes, 
                                 alignof(max_align_t));
        }
    } 
#endif /* ARENA_ENABLE_DYNAMIC */
// ================================================================================ 
// ================================================================================
} /* cslt namespace */
// ================================================================================ 
// ================================================================================ 
#endif /* allocator_HPP */
// ================================================================================
// ================================================================================
// eof
