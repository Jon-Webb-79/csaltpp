// ================================================================================
// ================================================================================
// - File:    allocator.cpp
// - Purpose: This file contains the implementation for custom allocators as part 
//            of the cslt namespace
//
// Source Metadata
// - Author:  Jonathan A. Webb
// - Date:    December 28, 2025
// - Version: 1.0
// - Copyright: Copyright 2022, Jon Webb Inc.
// ================================================================================
// ================================================================================
// Include modules here

#include "allocator.hpp"

#include <cstring>
#include <cstdarg>
#include <new>
#include <cstdint>
#ifdef _WIN32
    #include <malloc.h>  // for _aligned_malloc/_aligned_free
#else
    #include <stdlib.h>  // for posix_memalign/free
#endif

// ================================================================================ 
// ================================================================================ 

// In allocator.cpp or as class static members
static constexpr size_t k_growth_limit = 4 * 1024 * 1024;  // 4MB
static constexpr size_t k_max_chunk = 64 * 1024 * 1024;    // 64MB

namespace cslt {

    // Helper function for formatted appending
    static bool _buf_appendf(char *buffer,
                            size_t buffer_size,
                            size_t *p_offset,
                            const char *fmt, ...) {
        if ((buffer == nullptr) || (p_offset == nullptr) || (fmt == nullptr)) {
            return false;
        }
        size_t const offset = *p_offset;
        if (offset > buffer_size) {
            return false;
        }
        size_t const remaining = buffer_size - offset;
        if (remaining == 0U) {
            return false;
        }
        va_list args;
        va_start(args, fmt);
        int const n = vsnprintf(&buffer[offset], remaining, fmt, args);
        va_end(args);
        if (n < 0) {
            return false;
        }
        if ((size_t)n >= remaining) {
            return false;
        }
        *p_offset = offset + (size_t)n;
        return true;
    }
// -------------------------------------------------------------------------------- 

    static size_t normalize_alignment(size_t a) {
        if (a == 0) return alignof(max_align_t);
        if (a < alignof(max_align_t)) a = alignof(max_align_t);
        return a;
    }
// ================================================================================ 
// ================================================================================ 

#if ARENA_ENABLE_DYNAMIC
    Expected<void*> HeapAllocator::alloc(size_t bytes, bool zeroed) {
        Expected<void*> result;
        if (bytes == 0) {
            result.setError(InvalidArgError()); 
            return result;
        }

        void *ptr = nullptr;
        ptr = ::operator new(bytes);

        if (!ptr) {
            result.setError(BadAllocError());
            return result;
        }
        if (zeroed) {
            memset(ptr, 0, bytes);
        }
        result.setValue(ptr);
        return result;
    }
// -------------------------------------------------------------------------------- 

    Expected<void*> HeapAllocator::alloc_aligned(size_t bytes,
                                                 size_t alignment,
                                                 bool zeroed) {
        Expected<void*> result;
        if (bytes == 0) {
            result.setError(InvalidArgError());
            return result;
        }

        size_t a = normalize_alignment(alignment ? alignment : default_alignment_);
        void *ptr = nullptr;

        // Only use platform-specific alignment if we need more than max_align_t
        if (a <= alignof(max_align_t)) {
            // Standard alignment - use operator new
            ptr = ::operator new(bytes, std::nothrow);
        } else {
            // Special alignment needed - use platform-specific aligned allocation
            #ifdef _WIN32
                ptr = _aligned_malloc(bytes, a);
            #else
                if (posix_memalign(&ptr, a, bytes) != 0) {
                    ptr = nullptr;
                }
            #endif
        }

        if (!ptr) {
            result.setError(BadAllocError());
            return result;
        }
        if (zeroed) {
            memset(ptr, 0, bytes);
        }
        result.setValue(ptr);
        return result;
    }
// -------------------------------------------------------------------------------- 

    Expected<void*> HeapAllocator::realloc(void* ptr,
                                           size_t old_bytes,
                                           size_t new_bytes,
                                           bool zeroed) {
        Expected<void*> result;
        
        if (!ptr) {
            result.setError(NullPointerError());
            return result;
        }
        if (new_bytes == 0) {
            result.setError(InvalidArgError());
            return result;
        }

        Expected<void*> new_ptr = alloc(new_bytes, false);
        if (!new_ptr.hasValue()) {
            result.setError(ReallocFailError());
            return result;
        }
        
        // Copy old data to new allocation
        if (old_bytes > 0) {
            size_t copy_size = (old_bytes < new_bytes) ? old_bytes : new_bytes;
            memcpy(new_ptr.value(), ptr, copy_size);
        }
        
        // Zero the extra bytes if needed
        if (zeroed && new_bytes > old_bytes) {
            memset(static_cast<char*>(new_ptr.value()) + old_bytes, 0, new_bytes - old_bytes);
        }
        
        return_element(ptr, old_bytes, default_alignment_);
        result.setValue(new_ptr.value());
        return result;
    }
    // -------------------------------------------------------------------------------- 

    Expected<void*> HeapAllocator::realloc_aligned(void* ptr,  
                                                   size_t old_bytes,
                                                   size_t new_bytes,
                                                   size_t alignment,
                                                   bool zeroed) {
        Expected<void*> result;
            
        if (!ptr) {
            result.setError(NullPointerError());
            return result;
        }
        if (new_bytes == 0) {
            result.setError(InvalidArgError());
            return result;
        }
        
        size_t a = normalize_alignment(alignment ? alignment : default_alignment_);

        Expected<void*> new_ptr = alloc_aligned(new_bytes, a, false);
        if (!new_ptr.hasValue()) {
            result.setError(ReallocFailError());
            return result;
        }
        
        // Copy old data to new allocation
        if (old_bytes > 0) {
            size_t copy_size = (old_bytes < new_bytes) ? old_bytes : new_bytes;
            memcpy(new_ptr.value(), ptr, copy_size);
        }
        
        // Zero the extra bytes if needed
        if (zeroed && new_bytes > old_bytes) {
            memset(static_cast<char*>(new_ptr.value()) + old_bytes, 0, new_bytes - old_bytes);
        }
        
        return_element(ptr, old_bytes, alignment);
        result.setValue(new_ptr.value());
        return result;
    }
// -------------------------------------------------------------------------------- 

    void HeapAllocator::return_element(void *ptr, 
                                       size_t bytes, 
                                       size_t alignment) {
        (void)bytes;
        if (!ptr) {
            return;  // Nothing to free
        }
        
        size_t a = normalize_alignment(alignment);
        
        // Use matching deallocation for the allocation method
        if (a <= alignof(max_align_t)) {
            // Was allocated with ::operator new
            ::operator delete(ptr);
        } else {
            // Was allocated with platform-specific aligned allocation
            #ifdef _WIN32
                _aligned_free(ptr);
            #else
                free(ptr);  // posix_memalign uses regular free
            #endif
        }
    }
// -------------------------------------------------------------------------------- 

    bool HeapAllocator::stats(char *buffer, size_t buffer_size) const {
        size_t offset = 0U;
        
        if ((buffer == nullptr) || (buffer_size == 0U)) {
            return false;
        }
        
        if (!_buf_appendf(buffer, buffer_size, &offset, "%s", "HeapAllocator Statistics:\n")) {
            return false;
        }
        
        if (!_buf_appendf(buffer, buffer_size, &offset,
                         "  Type: %s\n", "DYNAMIC")) {
            return false;
        }
        
        if (!_buf_appendf(buffer, buffer_size, &offset,
                         "  Default Alignment: %zu bytes\n", default_alignment_)) {
            return false;
        }
        
        if (!_buf_appendf(buffer, buffer_size, &offset,
                         "  Memory Model: %s\n", "System Heap (operator new/delete)")) {
            return false;
        }
        
        if (!_buf_appendf(buffer, buffer_size, &offset,
                         "  Note: HeapAllocator is a wrapper around system allocator.\n")) {
            return false;
        }
        
        if (!_buf_appendf(buffer, buffer_size, &offset,
                         "        It does not own or track memory; all allocations\n")) {
            return false;
        }
        
        if (!_buf_appendf(buffer, buffer_size, &offset,
                         "        are managed directly by the OS heap.\n")) {
            return false;
        }
        
        return true;
    }
#endif /* ARENA_ENABLE_DYNAMIC */
// ================================================================================ 
// ================================================================================ 

    static inline size_t _align_up_size(size_t x, size_t a) {
        /* a must be power-of-two */
        return (x + (a - 1)) & ~(a - 1);
    }
// -------------------------------------------------------------------------------- 

    static inline uintptr_t _align_up_uintptr(uintptr_t p, size_t a) {
        return (p + (a - 1)) & ~(a - 1);
    }
// -------------------------------------------------------------------------------- 

    static inline size_t _pad_up(uintptr_t p, size_t a) {
        size_t const mask = a - 1u;
        return (size_t)(((p + mask) & ~mask) - p);
    }
// -------------------------------------------------------------------------------- 

    static inline size_t _mul_div_ceil(size_t x, size_t mul, size_t div) {
        /* assumes div > 0, caller ensures small constants like 2 or 3 */
        size_t q = x / div;
        size_t r = x % div;
        size_t hi = r * mul;
        size_t add = (hi + (div - 1u)) / div;
        size_t t = q * mul;
        size_t y = t + add;
        /* very defensive overflow clamp */
        if (y < t) { return SIZE_MAX; }
        return y;
    }
// -------------------------------------------------------------------------------- 

#if ARENA_ENABLE_DYNAMIC 
    static size_t _next_chunk_size(size_t prev_data_alloc, size_t need, size_t align, size_t min_chunk) {
        /* meet the request at minimum */
        size_t grow = (need > prev_data_alloc) ? need : prev_data_alloc;

        /* geometric target: 2x until growth_limit, then 1.5x */
        size_t doubled = (prev_data_alloc <= (SIZE_MAX / 2u)) ? (prev_data_alloc << 1) : SIZE_MAX;
        size_t onefive = _mul_div_ceil(prev_data_alloc, 3u, 2u); /* 1.5x */

        size_t target = (prev_data_alloc < k_growth_limit) ? doubled : onefive;
        if (target > grow) { grow = target; }

        /* clamp to floor/ceiling */
        if (grow < min_chunk) { grow = min_chunk; }
        if (grow > k_max_chunk) { grow = k_max_chunk; }

        /* align capacity to alignment to keep chunk->chunk naturally aligned */
        grow = _align_up_size(grow, align);

        /* final safety: ensure capacity still covers need after rounding */
        if (grow < need) { grow = need; }

        return grow;
    }
#endif /* ARENA_ENABLE_DYNAMIC */
// -------------------------------------------------------------------------------- 

    ArenaAllocator::Chunk* ArenaAllocator::find_chunk_in_chain(Chunk* target, 
                                                                Chunk** out_prev) const {
        if (!target) {
            return nullptr;
        }
        
        Chunk* prev = nullptr;
        for (Chunk const* cur = head_; cur; cur = cur->next) {
            if (cur == target) {
                if (out_prev) {
                    *out_prev = prev;
                }
                return const_cast<Chunk*>(cur);  // FIXED: return cur, not cur_
            }
            prev = const_cast<Chunk*>(cur);  // FIXED: prev = cur, not cur_
        }
        
        return nullptr;
    }
// ================================================================================ 

#if ARENA_ENABLE_DYNAMIC 
    Expected<cslt::UniquePtr<ArenaAllocator, ArenaDeleter>>
    ArenaAllocator::Heap(size_t bytes,
                         bool resize,
                         size_t min_chunk_size,
                         size_t base_align_in) {
        Expected<cslt::UniquePtr<ArenaAllocator, ArenaDeleter>> result;
        
        // Normalize min_chunk (0 allowed)
        size_t min_chunk = min_chunk_size;
        if (min_chunk && !is_pow2(min_chunk)) {
            min_chunk = next_pow2(min_chunk);
            if (!min_chunk) {
                result.setError(PreconditionFailError("Arena Chunk size does not conform to power of 2"));
                return result;
            }
        }
        
        // Normalize base alignment; enforce ABI floor
        size_t base_align = base_align_in ? base_align_in : alignof(max_align_t);
        if (!is_pow2(base_align)) {
            base_align = next_pow2(base_align);
            if (!base_align) {
                result.setError(AlignmentError("Arena Alignment Fail in Dynamic Initialization"));
                return result;
            }
        }
        if (base_align < alignof(max_align_t)) {
            base_align = alignof(max_align_t);
        }
        
        // Ensure base_align can accommodate ArenaAllocator itself
        size_t arena_align = alignof(ArenaAllocator);
        if (base_align < arena_align) {
            base_align = arena_align;
        }
        
        // Initial total buffer size (must fit ArenaAllocator + Chunk + data)
        size_t total = bytes;
        if (min_chunk && total < min_chunk) {
            total = min_chunk;
        }
        size_t min_required = sizeof(ArenaAllocator) + sizeof(Chunk);
        if (total < min_required) {
            result.setError(ArgumentError("Total Arena size does not fit arena + chunk structure"));
            return result;
        }
        
        // Allocate memory
        void *base = ::operator new(total, std::nothrow);
        if (!base) {
            result.setError(MemoryError("Arena Dynamic Memory Allocation Failed"));
            return result;
        }
        
        uintptr_t const b = reinterpret_cast<uintptr_t>(base);
        
        // Layout: [ArenaAllocator][padding][Chunk][padding][data...]
        
        // 1. ArenaAllocator at the beginning
        uintptr_t p_arena = b;
        uintptr_t arena_end = p_arena + sizeof(ArenaAllocator);
        
        if (arena_end < p_arena) {
            ::operator delete(base);
            result.setError(LengthOverflowError("Overflow in arena calculation"));
            return result;
        }
        
        // 2. Chunk follows ArenaAllocator (aligned to base_align)
        uintptr_t p_chunk = _align_up_uintptr(arena_end, base_align);
        uintptr_t chunk_end = p_chunk + sizeof(Chunk);
        
        if (chunk_end < p_chunk || chunk_end > b + total) {
            ::operator delete(base);
            result.setError(LengthOverflowError("Overflow in chunk calculation"));
            return result;
        }
        
        // 3. Data starts after Chunk (aligned to base_align)
        uintptr_t p_data = _align_up_uintptr(chunk_end, base_align);
        if (p_data > b + total) {
            ::operator delete(base);
            result.setError(AlignmentError("Alignment exceeds allocation"));
            return result;
        }
        
        size_t usable = static_cast<size_t>((b + total) - p_data);
        if (!usable) {
            ::operator delete(base);
            result.setError(MemoryError("No usable memory after alignment"));
            return result;
        }
        
        // Initialize chunk
        Chunk *h = reinterpret_cast<Chunk*>(p_chunk);
        h->chunk = reinterpret_cast<uint8_t*>(p_data);
        h->len = 0;
        h->alloc = usable;
        h->next = nullptr;
        
        // Use placement new to construct ArenaAllocator in the allocated memory
        ArenaAllocator *arena = new (base) ArenaAllocator();
        
        // Initialize arena members
        arena->head_ = h;
        arena->tail_ = h;
        arena->cur_ = reinterpret_cast<uint8_t*>(p_data);
        arena->min_chunk_ = min_chunk;
        arena->resize_ = static_cast<uint8_t>(resize);
        arena->size_ = 0;
        arena->alloc_ = usable;
        arena->total_alloc_ = total;
        arena->default_alignment_ = base_align;
        arena->mem_type_ = static_cast<uint8_t>(DYNAMIC);
        arena->owns_memory_ = static_cast<uint8_t>(true);
       
        cslt::UniquePtr<ArenaAllocator, ArenaDeleter> ptr(arena, ArenaDeleter{});
        result.setValue(cslt::move(ptr));
        return result;
    }
#endif /* ARENA_ENABLE_DYNAMIC */
// -------------------------------------------------------------------------------- 

    Expected<cslt::UniquePtr<ArenaAllocator, ArenaDeleter>>
    ArenaAllocator::Stack(void* buffer,
                          size_t bytes,
                          size_t base_align_in) {
        Expected<cslt::UniquePtr<ArenaAllocator, ArenaDeleter>> result;
        
        // Validate buffer
        if (!buffer) {
            result.setError(ArgumentError("Static arena buffer cannot be null"));
            return result;
        }
        
        // Normalize base alignment; enforce ABI floor
        size_t base_align = base_align_in ? base_align_in : alignof(max_align_t);
        if (!is_pow2(base_align)) {
            base_align = next_pow2(base_align);
            if (!base_align) {
                result.setError(AlignmentError("Arena Alignment Fail in Static Initialization"));
                return result;
            }
        }
        if (base_align < alignof(max_align_t)) {
            base_align = alignof(max_align_t);
        }
        
        // Ensure base_align can accommodate ArenaAllocator itself
        size_t arena_align = alignof(ArenaAllocator);
        if (base_align < arena_align) {
            base_align = arena_align;
        }
        
        // Check minimum size
        size_t min_required = sizeof(ArenaAllocator) + sizeof(Chunk);
        if (bytes < min_required) {
            result.setError(ArgumentError("Static buffer too small for arena + chunk structure"));
            return result;
        }
        
        uintptr_t const b = reinterpret_cast<uintptr_t>(buffer);
        
        // Layout: [ArenaAllocator][padding][Chunk][padding][data...]
        
        // 1. ArenaAllocator at the beginning
        uintptr_t p_arena = b;
        uintptr_t arena_end = p_arena + sizeof(ArenaAllocator);
        
        if (arena_end < p_arena) {
            result.setError(LengthOverflowError("Overflow in arena calculation"));
            return result;
        }
        
        // 2. Chunk follows ArenaAllocator (aligned to base_align)
        uintptr_t p_chunk = _align_up_uintptr(arena_end, base_align);
        uintptr_t chunk_end = p_chunk + sizeof(Chunk);
        
        if (chunk_end < p_chunk || chunk_end > b + bytes) {
            result.setError(LengthOverflowError("Overflow in chunk calculation"));
            return result;
        }
        
        // 3. Data starts after Chunk (aligned to base_align)
        uintptr_t p_data = _align_up_uintptr(chunk_end, base_align);
        if (p_data > b + bytes) {
            result.setError(AlignmentError("Alignment exceeds buffer size"));
            return result;
        }
        
        size_t usable = static_cast<size_t>((b + bytes) - p_data);
        if (!usable) {
            result.setError(MemoryError("No usable memory after alignment"));
            return result;
        }
        
        // Initialize chunk
        Chunk *h = reinterpret_cast<Chunk*>(p_chunk);
        h->chunk = reinterpret_cast<uint8_t*>(p_data);
        h->len = 0;
        h->alloc = usable;
        h->next = nullptr;
        
        // Use placement new to construct ArenaAllocator at the start of buffer
        ArenaAllocator *arena = new (buffer) ArenaAllocator();
        
        // Initialize arena members
        arena->head_ = h;
        arena->tail_ = h;
        arena->cur_ = reinterpret_cast<uint8_t*>(p_data);
        arena->min_chunk_ = 0;  // Static arenas don't resize
        arena->resize_ = static_cast<uint8_t>(false);
        arena->size_ = 0;
        arena->alloc_ = usable;
        arena->total_alloc_ = bytes;
        arena->default_alignment_ = base_align;
        arena->mem_type_ = static_cast<uint8_t>(STATIC);
        arena->owns_memory_ = static_cast<uint8_t>(false);
       
        cslt::UniquePtr<ArenaAllocator, ArenaDeleter> ptr(arena, ArenaDeleter{});
        result.setValue(cslt::move(ptr));
        return result;
    }
// -------------------------------------------------------------------------------- 

    Expected<cslt::UniquePtr<ArenaAllocator, ArenaDeleter>>
    ArenaAllocator::SubArena(ArenaAllocator& parent,
                             size_t bytes,
                             size_t base_align_in) {
        Expected<cslt::UniquePtr<ArenaAllocator, ArenaDeleter>> result;
        
        // Validate bytes
        if (bytes == 0) {
            result.setError(ArgumentError("Sub-arena size cannot be zero"));
            return result;
        }
        
        // Normalize base alignment; enforce ABI floor
        size_t base_align = base_align_in ? base_align_in : alignof(max_align_t);
        if (!is_pow2(base_align)) {
            base_align = next_pow2(base_align);
            if (!base_align) {
                result.setError(AlignmentError("Arena Alignment Fail in Sub-Arena Initialization"));
                return result;
            }
        }
        if (base_align < alignof(max_align_t)) {
            base_align = alignof(max_align_t);
        }
        
        // Ensure base_align can accommodate ArenaAllocator itself
        size_t arena_align = alignof(ArenaAllocator);
        if (base_align < arena_align) {
            base_align = arena_align;
        }
        
        // Calculate total required size
        size_t total = bytes;
        size_t min_required = sizeof(ArenaAllocator) + sizeof(Chunk);
        if (total < min_required) {
            result.setError(ArgumentError("Sub-arena size too small for arena + chunk structure"));
            return result;
        }
        
        // Allocate from parent arena (aligned to arena_align for ArenaAllocator)
        auto alloc_result = parent.alloc_aligned(total, arena_align, false);
        if (!alloc_result.hasValue()) {
            result.setError(MemoryError("Parent arena allocation failed for sub-arena"));
            return result;
        }
        void* buffer = alloc_result.value();
        
        uintptr_t const b = reinterpret_cast<uintptr_t>(buffer);
        
        // Layout: [ArenaAllocator][padding][Chunk][padding][data...]
        
        // 1. ArenaAllocator at the beginning
        uintptr_t p_arena = b;
        uintptr_t arena_end = p_arena + sizeof(ArenaAllocator);
        
        if (arena_end < p_arena) {
            result.setError(LengthOverflowError("Overflow in arena calculation"));
            return result;
        }
        
        // 2. Chunk follows ArenaAllocator (aligned to base_align)
        uintptr_t p_chunk = _align_up_uintptr(arena_end, base_align);
        uintptr_t chunk_end = p_chunk + sizeof(Chunk);
        
        if (chunk_end < p_chunk || chunk_end > b + total) {
            result.setError(LengthOverflowError("Overflow in chunk calculation"));
            return result;
        }
        
        // 3. Data starts after Chunk (aligned to base_align)
        uintptr_t p_data = _align_up_uintptr(chunk_end, base_align);
        if (p_data > b + total) {
            result.setError(AlignmentError("Alignment exceeds allocation"));
            return result;
        }
        
        size_t usable = static_cast<size_t>((b + total) - p_data);
        if (!usable) {
            result.setError(MemoryError("No usable memory after alignment"));
            return result;
        }
        
        // Initialize chunk
        Chunk *h = reinterpret_cast<Chunk*>(p_chunk);
        h->chunk = reinterpret_cast<uint8_t*>(p_data);
        h->len = 0;
        h->alloc = usable;
        h->next = nullptr;
        
        // Use placement new to construct ArenaAllocator in parent's memory
        ArenaAllocator *arena = new (buffer) ArenaAllocator();
        
        // Initialize arena members
        arena->head_ = h;
        arena->tail_ = h;
        arena->cur_ = reinterpret_cast<uint8_t*>(p_data);
        arena->min_chunk_ = 0;  // Sub-arenas don't resize
        arena->resize_ = static_cast<uint8_t>(false);
        arena->size_ = 0;
        arena->alloc_ = usable;
        arena->total_alloc_ = total;
        arena->default_alignment_ = base_align;
        arena->mem_type_ = static_cast<uint8_t>(parent.memory_type());
        arena->owns_memory_ = static_cast<uint8_t>(false);
       
        cslt::UniquePtr<ArenaAllocator, ArenaDeleter> ptr(arena, ArenaDeleter{});
        result.setValue(cslt::move(ptr));
        return result;
    }
// -------------------------------------------------------------------------------- 

    Expected<void*> ArenaAllocator::alloc(size_t bytes, bool zeroed) {
        Expected<void*> result;
        
        // Validate input
        if (bytes == 0) {
            result.setError(ArgumentError());
            return result;
        }
        
        size_t const a = default_alignment_;
        
        // Validate alignment (should always be valid from constructor, but defensive)
        if (a == 0 || (a & (a - 1)) != 0) {
            result.setError(AlignmentError());
            return result;
        }
        
        Chunk *tail = tail_;
        if (tail == nullptr) {
            result.setError(MemoryError());
            return result;
        }
        
        uintptr_t const cur = reinterpret_cast<uintptr_t>(cur_);
        size_t const pad = _pad_up(cur, a);
        
        // Check for overflow in pad + bytes
        if (bytes > (SIZE_MAX - pad)) {
            result.setError(LengthOverflowError());
            return result;
        }
        
        size_t const need = pad + bytes;
        size_t const avail = (alloc_ >= size_) ? (alloc_ - size_) : 0;
        
        // Fast path: fits in current tail
        if (avail >= need) {
            uint8_t *p = reinterpret_cast<uint8_t*>(cur + pad);  // a-aligned
            cur_ = p + bytes;
            tail->len += need;  // charge pad + bytes
            size_ += need;
            
            if (zeroed) {
                memset(p, 0, bytes);
            }
            
            result.setValue(p);
            return result;
        }
        
        // No space in current chunk - need to grow
#if ARENA_ENABLE_DYNAMIC        
        // Check if growth is allowed
        if (static_cast<MemType>(mem_type_) == STATIC || !resize_) {
            result.setError(MemoryError("Arena out of memory and resize disabled"));
            return result;
        }
        
        // Calculate new chunk size
        size_t const grow_data = _next_chunk_size(tail->alloc, need, a, min_chunk_);
        if (grow_data == 0) {
            result.setError(LengthOverflowError("Chunk size calculation overflow"));
            return result;
        }
        
        // Allocate new chunk
        size_t chunk_total = _align_up_size(sizeof(Chunk), a) + grow_data;
        void *base = ::operator new(chunk_total, std::nothrow);
        if (!base) {
            result.setError(MemoryError("Failed to allocate new arena chunk"));
            return result;
        }
        
        // Initialize new chunk
        uintptr_t b = reinterpret_cast<uintptr_t>(base);
        uintptr_t p_chunk = b;
        uintptr_t chunk_end = p_chunk + sizeof(Chunk);
        uintptr_t p_data = _align_up_uintptr(chunk_end, a);
        size_t usable = static_cast<size_t>((b + chunk_total) - p_data);
        
        Chunk *nc = reinterpret_cast<Chunk*>(p_chunk);
        nc->chunk = reinterpret_cast<uint8_t*>(p_data);
        nc->len = bytes;
        nc->alloc = usable;
        nc->next = nullptr;
        
        // Link new chunk as tail
        tail->next = nc;
        tail_ = nc;
        
        // Update accounting
        alloc_ += nc->alloc;
        total_alloc_ += chunk_total;
        
        // First allocation from fresh chunk: base is a-aligned -> no pad needed
        void *p = nc->chunk;
        cur_ = nc->chunk + bytes;
        size_ += bytes;
        
        if (zeroed) {
            memset(p, 0, bytes);
        }
        
        result.setValue(p);
        return result;
#endif /* ARENA_ENABLE_DYNAMIC */
        result.setError(MemoryError("Arena out of memory and resize disabled"));
        return result;
    }
// -------------------------------------------------------------------------------- 

    Expected<void*> ArenaAllocator::alloc_aligned(size_t bytes,
                                                  size_t alignment,
                                                  bool zeroed) {
        Expected<void*> result;
        
        // Validate input
        if (bytes == 0) {
            result.setError(ArgumentError("Cannot allocate 0 bytes"));
            return result;
        }
        
        // Normalize alignment (0 means use default)
        size_t a = alignment ? alignment : default_alignment_;
        
        // Validate alignment is power of 2
        if (!is_pow2(a)) {
            result.setError(AlignmentError("Alignment must be power of 2"));
            return result;
        }
        
        Chunk *tail = tail_;
        if (tail == nullptr) {
            result.setError(MemoryError("Arena not properly initialized"));
            return result;
        }
        
        uintptr_t const cur = reinterpret_cast<uintptr_t>(cur_);
        size_t const pad = _pad_up(cur, a);
        
        // Check for overflow in pad + bytes
        if (bytes > (SIZE_MAX - pad)) {
            result.setError(LengthOverflowError("Allocation size overflow"));
            return result;
        }
        
        size_t const need = pad + bytes;
        size_t const avail = (alloc_ >= size_) ? (alloc_ - size_) : 0;
        
        // Fast path: fits in current tail
        if (avail >= need) {
            uint8_t *p = reinterpret_cast<uint8_t*>(cur + pad);  // a-aligned
            cur_ = p + bytes;
            tail->len += need;  // charge pad + bytes
            size_ += need;
            
            if (zeroed) {
                memset(p, 0, bytes);
            }
            
            result.setValue(p);
            return result;
        }
        
        // No space in current chunk - need to grow
#if ARENA_ENABLE_DYNAMIC 
        // Check if growth is allowed
        if (static_cast<MemType>(mem_type_) == STATIC || !resize_) {
            result.setError(MemoryError("Arena out of memory and resize disabled"));
            return result;
        }
        
        // Calculate new chunk size
        size_t const grow_data = _next_chunk_size(tail->alloc, need, a, min_chunk_);
        if (grow_data == 0) {
            result.setError(LengthOverflowError("Chunk size calculation overflow"));
            return result;
        }
        
        // Allocate new chunk with extra space for alignment
        size_t chunk_total = _align_up_size(sizeof(Chunk), a) + grow_data + a;
        void *base = ::operator new(chunk_total, std::nothrow);
        if (!base) {
            result.setError(MemoryError("Failed to allocate new arena chunk"));
            return result;
        }
        
        // Initialize new chunk
        uintptr_t b = reinterpret_cast<uintptr_t>(base);
        uintptr_t p_chunk = b;
        uintptr_t chunk_end = p_chunk + sizeof(Chunk);
        uintptr_t p_data = _align_up_uintptr(chunk_end, a);
        size_t usable = static_cast<size_t>((b + chunk_total) - p_data);
        
        Chunk *nc = reinterpret_cast<Chunk*>(p_chunk);
        nc->chunk = reinterpret_cast<uint8_t*>(p_data);
        nc->len = bytes;
        nc->alloc = usable;
        nc->next = nullptr;
        
        // Link new chunk as tail
        tail->next = nc;
        tail_ = nc;
        
        // Update accounting
        alloc_ += nc->alloc;
        total_alloc_ += chunk_total;
        
        // First allocation from fresh chunk: base is a-aligned -> no pad needed
        void *p = nc->chunk;
        cur_ = nc->chunk + bytes;
        size_ += bytes;
        
        if (zeroed) {
            memset(p, 0, bytes);
        }
        
        result.setValue(p);
        return result;
#endif /* ARENA_ENABLE_DYNAMIC */
        result.setError(MemoryError("Arena out of memory and resize disabled"));
        return result;
    }
// -------------------------------------------------------------------------------- 

    Expected<void*> ArenaAllocator::realloc(void* ptr,
                                            size_t old_bytes,
                                            size_t new_bytes,
                                            bool zeroed) {
        Expected<void*> result;
        
        // Validate input
        if (!ptr) {
            result.setError(ArgumentError("Cannot realloc null pointer"));
            return result;
        }
        if (new_bytes == 0) {
            result.setError(ArgumentError("Cannot realloc to 0 bytes"));
            return result;
        }
        
        // Special case: if shrinking, we could just return the same pointer
        // But for simplicity and consistency, we always allocate new space
        
        // Check if this is the most recent allocation and we can extend in-place
        // This is an optimization for the common case of growing the last allocation
        uintptr_t ptr_addr = reinterpret_cast<uintptr_t>(ptr);
        uintptr_t cur_addr = reinterpret_cast<uintptr_t>(cur_);
        
        // Check if ptr + old_bytes == cur_ (this is the last allocation)
        if (ptr_addr + old_bytes == cur_addr) {
            // Try to extend in-place
            size_t additional = new_bytes - old_bytes;
            size_t const avail = (alloc_ >= size_) ? (alloc_ - size_) : 0;
            
            if (avail >= additional) {
                // Can extend in-place!
                cur_ = reinterpret_cast<uint8_t*>(ptr_addr + new_bytes);
                tail_->len += additional;
                size_ += additional;
                
                // Zero the new bytes if requested
                if (zeroed && new_bytes > old_bytes) {
                    memset(static_cast<uint8_t*>(ptr) + old_bytes, 0, additional);
                }
                
                result.setValue(ptr);
                return result;
            }
        }
        
        // Cannot extend in-place - allocate new memory
        Expected<void*> new_ptr = alloc(new_bytes, false);
        if (!new_ptr.hasValue()) {
            result.setError(new_ptr.error());
            return result;
        }
        
        // Copy old data
        if (old_bytes > 0) {
            size_t copy_size = (old_bytes < new_bytes) ? old_bytes : new_bytes;
            memcpy(new_ptr.value(), ptr, copy_size);
        }
        
        // Zero new bytes if requested and growing
        if (zeroed && new_bytes > old_bytes) {
            memset(static_cast<uint8_t*>(new_ptr.value()) + old_bytes, 0, new_bytes - old_bytes);
        }
        
        // Note: old memory is NOT freed in arena (it becomes wasted space until reset)
        
        result.setValue(new_ptr.value());
        return result;
    }
// -------------------------------------------------------------------------------- 

    Expected<void*> ArenaAllocator::realloc_aligned(void* ptr,
                                                    size_t old_bytes,
                                                    size_t new_bytes,
                                                    size_t alignment,
                                                    bool zeroed) {
        Expected<void*> result;
        
        // Validate input
        if (!ptr) {
            result.setError(ArgumentError("Cannot realloc null pointer"));
            return result;
        }
        if (new_bytes == 0) {
            result.setError(ArgumentError("Cannot realloc to 0 bytes"));
            return result;
        }
        
        // Normalize alignment
        size_t a = alignment ? alignment : default_alignment_;
        
        // Validate alignment
        if (!is_pow2(a)) {
            result.setError(AlignmentError("Alignment must be power of 2"));
            return result;
        }
        
        // Check if this is the most recent allocation and we can extend in-place
        uintptr_t ptr_addr = reinterpret_cast<uintptr_t>(ptr);
        uintptr_t cur_addr = reinterpret_cast<uintptr_t>(cur_);
        
        // Verify pointer is already aligned (it should be from original alloc_aligned)
        if ((ptr_addr % a) != 0) {
            result.setError(AlignmentError("Pointer is not aligned to specified alignment"));
            return result;
        }
        
        // Check if ptr + old_bytes == cur_ (this is the last allocation)
        if (ptr_addr + old_bytes == cur_addr) {
            // Try to extend in-place
            size_t additional = new_bytes - old_bytes;
            size_t const avail = (alloc_ >= size_) ? (alloc_ - size_) : 0;
            
            if (avail >= additional) {
                // Can extend in-place!
                cur_ = reinterpret_cast<uint8_t*>(ptr_addr + new_bytes);
                tail_->len += additional;
                size_ += additional;
                
                // Zero the new bytes if requested
                if (zeroed && new_bytes > old_bytes) {
                    memset(static_cast<uint8_t*>(ptr) + old_bytes, 0, additional);
                }
                
                result.setValue(ptr);
                return result;
            }
        }
        
        // Cannot extend in-place - allocate new aligned memory
        Expected<void*> new_ptr = alloc_aligned(new_bytes, a, false);
        if (!new_ptr.hasValue()) {
            result.setError(new_ptr.error());
            return result;
        }
        
        // Copy old data
        if (old_bytes > 0) {
            size_t copy_size = (old_bytes < new_bytes) ? old_bytes : new_bytes;
            memcpy(new_ptr.value(), ptr, copy_size);
        }
        
        // Zero new bytes if requested and growing
        if (zeroed && new_bytes > old_bytes) {
            memset(static_cast<uint8_t*>(new_ptr.value()) + old_bytes, 0, new_bytes - old_bytes);
        }
        
        result.setValue(new_ptr.value());
        return result;
    }
// -------------------------------------------------------------------------------- 

    bool ArenaAllocator::is_ptr(void* ptr) const {
        if (!ptr) {
            return false;
        }
        
        uintptr_t const p = reinterpret_cast<uintptr_t>(ptr);
        
        // Fast check: tail first (most recent allocations likely here)
        Chunk const* c = tail_;
        if (c && c->chunk && c->len <= c->alloc) {
            uintptr_t s = reinterpret_cast<uintptr_t>(c->chunk);
            uintptr_t e = s + c->len;  // end is exclusive
            
            // Guard against overflow and check range
            if (e >= s && p >= s && p < e) {
                return true;
            }
        }
        
        // Walk remaining chunks
        for (Chunk const* cur = head_; cur; cur = cur->next) {
            if (!cur->chunk) {
                continue;
            }
            
            // Defensive clamp: use min of len and alloc
            size_t used = cur->len;
            if (used > cur->alloc) {
                used = cur->alloc;
            }
            
            uintptr_t s = reinterpret_cast<uintptr_t>(cur->chunk);
            uintptr_t e = s + used;
            
            // Overflow guard
            if (e < s) {
                continue;
            }
            
            // Check if pointer is in range [s, e)
            if (p >= s && p < e) {
                return true;
            }
        }
        
        return false;
    }
// -------------------------------------------------------------------------------- 

    bool ArenaAllocator::is_ptr_sized(void* ptr, size_t bytes) const {
        if (!ptr || bytes == 0) {
            return false;
        }
        
        uintptr_t const p = reinterpret_cast<uintptr_t>(ptr);
        uintptr_t const pe = p + bytes;  // end (exclusive)
        
        // Overflow check
        if (pe < p) {
            return false;
        }
        
        // Tail fast-path (most recent allocations likely here)
        Chunk const* c = tail_;
        if (c && c->chunk && c->len <= c->alloc) {
            uintptr_t const s = reinterpret_cast<uintptr_t>(c->chunk);
            uintptr_t const ue = s + c->len;
            
            // Guard overflow and check if entire range fits
            if (ue >= s && p >= s && pe <= ue) {
                return true;
            }
        }
        
        // Walk remaining chunks
        for (Chunk const* cur = head_; cur; cur = cur->next) {
            if (!cur->chunk) {
                continue;
            }
            
            // Defensive clamp if corrupted
            size_t used = cur->len;
            if (used > cur->alloc) {
                used = cur->alloc;
            }
            
            uintptr_t const s = reinterpret_cast<uintptr_t>(cur->chunk);
            uintptr_t const ue = s + used;  // end (exclusive)
            
            // Overflow guard in chunk arithmetic
            if (ue < s) {
                continue;
            }
            
            // Check if entire range [p, pe) fits within [s, ue)
            if (p >= s && pe <= ue) {
                return true;
            }
        }
        
        return false;
    }
// -------------------------------------------------------------------------------- 

    void ArenaAllocator::return_element(void *ptr, 
                                        size_t bytes, 
                                        size_t alignment) {
        (void)ptr;
        (void)bytes;
        (void) alignment;
    }
// -------------------------------------------------------------------------------- 

    bool ArenaAllocator::reset(bool trim_extra_chunks) {
        // Defensive: if no head, reset to empty state and return false
        if (!head_) {
            cur_ = nullptr;
            size_ = 0;
            tail_ = nullptr;
            return false;
        }
        
        // Zero usage counters on all chunks
        for (Chunk *cur = head_; cur; cur = cur->next) {
            cur->len = 0;
        }
        size_ = 0;
        
        // Trim extra chunks if requested and allowed
        if (trim_extra_chunks && static_cast<MemType>(mem_type_) == DYNAMIC) {
            // Calculate header size rounded to alignment
            size_t const hdr_rounded = _align_up_size(sizeof(Chunk), default_alignment_);
            
            // Free all chunks after head
            Chunk *to_free = head_->next;
            while (to_free) {
                Chunk *next = to_free->next;
                
                // Subtract this chunk's contribution from total allocation
                size_t contrib = hdr_rounded + to_free->alloc;
                if (total_alloc_ >= contrib) {
                    total_alloc_ -= contrib;
                } else {
                    total_alloc_ = 0;  // Defensive clamp
                }
                
                // IMPORTANT: Free the chunk header pointer (owns the whole block)
                // NOT to_free->chunk which is interior to the allocation
                ::operator delete(to_free);
                
                to_free = next;
            }
            
            // Detach and normalize to a single head chunk
            head_->next = nullptr;
            tail_ = head_;
            cur_ = head_->chunk;
            
            // Usable capacity now equals the head's data capacity
            alloc_ = head_->alloc;
            
            // total_alloc_ already adjusted above
        } else {
            // Keep all chunks allocated: zero usage but preserve capacity/footprint
            tail_ = tail_ ? tail_ : head_;
            cur_ = tail_->chunk ? tail_->chunk : head_->chunk;
            
            // alloc_ and total_alloc_ unchanged
        }
        
        return true;
    }
// -------------------------------------------------------------------------------- 

    void* ArenaAllocator::save() const {
        if (!tail_) {
            return nullptr;
        }
        
        // Allocate checkpoint data
        CheckpointData* cp = new (std::nothrow) CheckpointData;
        if (!cp) {
            return nullptr;
        }
        
        // Save current state
        cp->chunk = tail_;
        cp->cur = cur_;
        cp->len = size_;
        
        return static_cast<void*>(cp);
    }
// -------------------------------------------------------------------------------- 

    bool ArenaAllocator::restore(void* checkpoint) {
        if (!checkpoint) {
            return false;
        }
        
        // Unpack the checkpoint
        CheckpointData* cp = static_cast<CheckpointData*>(checkpoint);
        
        // Empty checkpoint is a no-op (success)
        if (!cp->chunk) {
            delete cp;
            return true;
        }
        
        // Validate that the checkpoint's chunk still exists in the chain
        Chunk* prev = nullptr;
        Chunk* hit = find_chunk_in_chain(cp->chunk, &prev);
        if (!hit) {
            delete cp;
            return false;
        }
        
        // Validate the chunk has a valid data region
        if (!cp->chunk->chunk) {
            delete cp;
            return false;
        }
        
        // Validate the checkpoint cursor is within bounds
        uintptr_t chunk_start = reinterpret_cast<uintptr_t>(cp->chunk->chunk);
        uintptr_t cursor_pos = reinterpret_cast<uintptr_t>(cp->cur);
        uintptr_t chunk_end = chunk_start + cp->chunk->alloc;  // exclusive
        
        // Check for overflow and valid cursor position
        if (chunk_end < chunk_start || cursor_pos < chunk_start || cursor_pos > chunk_end) {
            delete cp;
            return false;
        }
        
        // For DYNAMIC arenas: free all chunks after the checkpoint chunk
        if (static_cast<MemType>(mem_type_) == DYNAMIC) {
            size_t const hdr_rounded = _align_up_size(sizeof(Chunk), default_alignment_);
            
            Chunk* to_free = cp->chunk->next;
            while (to_free) {
                Chunk* next = to_free->next;
                
                // Subtract this chunk's contribution from total_alloc_
                size_t contrib = hdr_rounded + to_free->alloc;
                if (total_alloc_ >= contrib) {
                    total_alloc_ -= contrib;
                } else {
                    total_alloc_ = 0;  // Defensive clamp
                }
                
                ::operator delete(to_free);
                to_free = next;
            }
            
            // Detach the freed chunks from the list
            cp->chunk->next = nullptr;
        }
        // For STATIC arenas: can't free chunks, just validate they exist
        // (validation already happened via find_chunk_in_chain)
        
        // Update the tail chunk's used length to match the checkpoint cursor
        cp->chunk->len = static_cast<size_t>(cursor_pos - chunk_start);
        
        // Update arena state
        tail_ = cp->chunk;
        cur_ = cp->cur;
        
        // Recompute accounting for the remaining chain
        size_t alignment = default_alignment_;
        
        // Validate alignment is a power of two
        if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
            delete cp;
            return false;
        }
        
        // Walk the chain and recompute totals
        size_t total_used = 0;
        size_t total_cap = 0;
        
        for (Chunk* k = head_; k; k = k->next) {
            // Clamp used to allocation (defensive)
            size_t used = (k->len <= k->alloc) ? k->len : k->alloc;
            
            total_used += used;
            total_cap += k->alloc;
        }
        
        size_ = total_used;
        alloc_ = total_cap;
        
        // For STATIC arenas, recompute total_alloc_
        // For DYNAMIC, we already adjusted it during chunk freeing
        if (static_cast<MemType>(mem_type_) == STATIC) {
            size_t total_foot = 0;
            for (Chunk* k = head_; k; k = k->next) {
                total_foot += _align_up_size(sizeof(Chunk), alignment) + k->alloc;
            }
            total_alloc_ = total_foot;
        }
        
        // Clean up checkpoint data
        delete cp;
        return true;
    }
// -------------------------------------------------------------------------------- 

    size_t ArenaAllocator::remaining() const noexcept {
        if (!head_) {
            return 0;
        }
        
        size_t total_alloc = 0;
        size_t total_used = 0;
        
        for (Chunk* cur = head_; cur; cur = cur->next) {
            total_alloc += cur->alloc;
            
            // Clamp used to allocated (defensive)
            size_t used = (cur->len <= cur->alloc) ? cur->len : cur->alloc;
            total_used += used;
        }
        
        if (total_alloc > total_used) {
            return total_alloc - total_used;
        }
        return 0;
    }
// -------------------------------------------------------------------------------- 

    bool ArenaAllocator::stats(char *buffer, size_t buffer_size) const {
        size_t offset = 0;
        
        if (!buffer || buffer_size == 0) {
            return false;
        }
        
        if (!_buf_appendf(buffer, buffer_size, &offset, "%s", "Arena Statistics:\n")) {
            return false;
        }
        
        // Memory type
        const char* type_str = (static_cast<MemType>(mem_type_) == STATIC) ? "STATIC" : "DYNAMIC";
        if (!_buf_appendf(buffer, buffer_size, &offset,
                         "  Type: %s\n", type_str)) {
            return false;
        }
        
        // Used bytes
        if (!_buf_appendf(buffer, buffer_size, &offset,
                         "  Used: %zu bytes\n", size_)) {
            return false;
        }
        
        // Capacity
        if (!_buf_appendf(buffer, buffer_size, &offset,
                         "  Capacity: %zu bytes\n", alloc_)) {
            return false;
        }
        
        // Total with overhead
        if (!_buf_appendf(buffer, buffer_size, &offset,
                         "  Total (with overhead): %zu bytes\n", total_alloc_)) {
            return false;
        }
        
        // Utilization with divide-by-zero guard
        if (alloc_ == 0) {
            if (!_buf_appendf(buffer, buffer_size, &offset,
                             "%s", "  Utilization: N/A (capacity is 0)\n")) {
                return false;
            }
        } else {
            double const util = (100.0 * static_cast<double>(size_)) / static_cast<double>(alloc_);
            if (!_buf_appendf(buffer, buffer_size, &offset,
                             "  Utilization: %.1f%%\n", util)) {
                return false;
            }
        }
        
        // Default alignment
        if (!_buf_appendf(buffer, buffer_size, &offset,
                         "  Default Alignment: %zu bytes\n", default_alignment_)) {
            return false;
        }
        
        // List chunks
        int chunk_num = 0;
        Chunk const* current = head_;
        while (current) {
            chunk_num++;
            if (!_buf_appendf(buffer, buffer_size, &offset,
                             "  Chunk %d: %zu/%zu bytes\n",
                             chunk_num, current->len, current->alloc)) {
                return false;
            }
            current = current->next;
        }
        
        // Resizable status
        const char* resize_str = resize_ ? "Yes" : "No";
        if (!_buf_appendf(buffer, buffer_size, &offset,
                         "  Resizable: %s\n", resize_str)) {
            return false;
        }
        
        // Ownership status
        const char* owns_str = static_cast<bool>(owns_memory_) ? "Yes" : "No";
        if (!_buf_appendf(buffer, buffer_size, &offset,
                         "  Owns Memory: %s\n", owns_str)) {
            return false;
        }
        
        // Min chunk size (if applicable)
        if (min_chunk_ > 0) {
            if (!_buf_appendf(buffer, buffer_size, &offset,
                             "  Min Chunk Size: %zu bytes\n", min_chunk_)) {
                return false;
            }
        }
        
        return true;
    }
// -------------------------------------------------------------------------------- 

    size_t ArenaAllocator::chunk_count() const noexcept {
        size_t count = 0;
        Chunk const* current = head_;
        while (current) {
            count++;
            current = current->next;
        }
        return count;
    }
    // -------------------------------------------------------------------------------- 

    size_t ArenaAllocator::min_chunk_size() const noexcept {
        return min_chunk_;
    }
    // -------------------------------------------------------------------------------- 

    void ArenaAllocator::toggle_resize(bool toggle) noexcept {
        // Cannot toggle resize for static arenas
        if (static_cast<MemType>(mem_type_) == STATIC) {
            return;
        }
        
        // Cannot toggle resize for sub-arenas (borrowed memory)
        // Sub-arenas are always fixed-size by design
        if (!static_cast<bool>(owns_memory_)) {
            return;
        }
        
        // Only dynamic arenas that own their memory can have resize toggled
        resize_ = static_cast<uint8_t>(toggle ? 1 : 0);
    }
// ================================================================================ 
// ================================================================================ 

    bool PoolAllocator::grow_pool() {
        if (!grow_enabled_ || !arena_) {
            return false;
        }

        // Calculate bytes needed for this chunk
        size_t bytes = stride_ * blocks_per_chunk_;

        // Allocate from arena
        auto result = arena_->alloc_aligned(bytes, stride_, false);
        if (!result.hasValue()) {
            return false;
        }

        uint8_t* base = static_cast<uint8_t*>(result.value());
        cur_ = base;
        end_ = base + bytes;
        total_blocks_ += blocks_per_chunk_;

        // Update accounting
        size_ += bytes;

        return true;
    } 
// -------------------------------------------------------------------------------- 

    void* PoolAllocator::pop_free() {
        void* blk = free_list_;
        if (blk) {
            // Read next pointer from first word of block
            free_list_ = *static_cast<void**>(blk);
            free_blocks_--;
        }
        return blk;
    }
// -------------------------------------------------------------------------------- 

    void PoolAllocator::push_free(void* blk) {
        if (!blk) return;

        // Write current free_list head into first word of block
        *static_cast<void**>(blk) = free_list_;
        free_list_ = blk;
        free_blocks_++;
    }
// ================================================================================ 

    PoolAllocator::~PoolAllocator() noexcept {
        // Pool destructor doesn't need to do much - the PoolDeleter
        // handles arena cleanup if needed
        // Just clear our state
        free_list_ = nullptr;
        cur_ = nullptr;
        arena_ = nullptr;
    }
// -------------------------------------------------------------------------------- 

#if ARENA_ENABLE_DYNAMIC
    Expected<cslt::UniquePtr<PoolAllocator, PoolDeleter>>
    PoolAllocator::Heap(size_t block_size,
                        size_t blocks_per_chunk,
                        size_t alignment,
                        size_t arena_initial_bytes,
                        size_t min_chunk_bytes,
                        bool grow_enabled,
                        bool prewarm) {
        Expected<cslt::UniquePtr<PoolAllocator, PoolDeleter>> result;

        // Validate inputs
        if (block_size == 0 || blocks_per_chunk == 0) {
            result.setError(InvalidArgError("Block size and blocks_per_chunk must be > 0"));
            return result;
        }

        if (arena_initial_bytes == 0) {
            result.setError(InvalidArgError("Arena initial bytes must be > 0"));
            return result;
        }

        // Fixed pools must be prewarmed
        if (!grow_enabled && !prewarm) {
            result.setError(InvalidArgError("Fixed-capacity pools must be prewarmed"));
            return result;
        }

        if (alignment != 0u && (alignment & (alignment - 1u)) != 0u) {
            result.setError(AlignmentError("Pool Heap Alignment not set properly!"));
            return result;
        }

        // Normalize alignment (use LOCAL variable, not stride_)
        size_t eff_align = alignment ? alignment : alignof(max_align_t);
        if (eff_align < alignof(void*)) {
            eff_align = alignof(void*);
        }

        // Calculate stride (LOCAL variable)
        size_t stride = _align_up_size(block_size, eff_align);
        if (stride < sizeof(void*)) {
            stride = sizeof(void*);
        }

        // const size_t slice_bytes = stride * blocks_per_chunk;

        /* Arena base alignment should never be less than max_align_t */
        size_t arena_base_align = eff_align;
        if (arena_base_align < alignof(max_align_t)) {
            arena_base_align = alignof(max_align_t);
        }

        // Create the arena
        auto arena_result = ArenaAllocator::Heap(
            arena_initial_bytes,
            grow_enabled,  // Arena resize matches pool growth
            min_chunk_bytes,
            arena_base_align
        );

        if (!arena_result.hasValue()) {
            result.setError(arena_result.error());
            return result;
        }

        auto arena_ptr = cslt::move(arena_result.value());
        ArenaAllocator* arena = arena_ptr.get();

        // Allocate pool header from the arena
        auto pool_header = arena->alloc_aligned(sizeof(PoolAllocator), alignof(PoolAllocator), false);
        if (!pool_header.hasValue()) {
            result.setError(pool_header.error());
            return result;
        }

        // Construct pool in-place
        PoolAllocator* pool = new (pool_header.value()) PoolAllocator();

        // Release arena from UniquePtr - pool now owns it
        arena_ptr.release();

        // Initialize pool members (NOW assign to pool->stride_)
        pool->arena_ = arena;
        pool->owns_arena_ = true;
        pool->block_size_ = block_size;
        pool->stride_ = stride;  // Assign local variable to member
        pool->blocks_per_chunk_ = blocks_per_chunk;
        pool->cur_ = nullptr;
        pool->end_ = nullptr;
        pool->free_list_ = nullptr;
        pool->total_blocks_ = 0;
        pool->free_blocks_ = 0;
        pool->grow_enabled_ = grow_enabled;

        // Base class initialization
        pool->default_alignment_ = arena_base_align;
        pool->mem_type_ = static_cast<uint8_t>(DYNAMIC);
        pool->owns_memory_ = static_cast<uint8_t>(true);
        pool->size_ = 0;
        pool->alloc_ = 0;
        pool->total_alloc_ = arena_initial_bytes;

        // Prewarm if requested
        if (prewarm) {
            // Calculate bytes needed for initial chunk
            size_t bytes = stride * blocks_per_chunk;
            
            // Allocate from arena directly (not via grow_pool which checks grow_enabled)
            auto prewarm_result = arena->alloc_aligned(bytes, stride, false);
            if (!prewarm_result.hasValue()) {
                pool->~PoolAllocator();
                ArenaDeleter{}(arena);
                result.setError(MemoryError("Failed to prewarm pool"));
                return result;
            }
            
            uint8_t* base = static_cast<uint8_t*>(prewarm_result.value());
            pool->cur_ = base;
            pool->end_ = base + bytes;
            pool->total_blocks_ = blocks_per_chunk;
            pool->size_ = bytes;
            pool->alloc_ = pool->total_blocks_ * stride;
        }

        cslt::UniquePtr<PoolAllocator, PoolDeleter> ptr(pool, PoolDeleter{});
        result.setValue(cslt::move(ptr));
        return result;
    }
#endif
// -------------------------------------------------------------------------------- 

    Expected<cslt::UniquePtr<PoolAllocator, PoolDeleter>>
    PoolAllocator::Stack(void* buffer,
                         size_t buffer_bytes,
                         size_t block_size,
                         size_t alignment) {
        Expected<cslt::UniquePtr<PoolAllocator, PoolDeleter>> result;

        // Validate inputs
        if (!buffer) {
            result.setError(ArgumentError("Buffer cannot be null"));
            return result;
        }

        if (buffer_bytes == 0 || block_size == 0) {
            result.setError(ArgumentError("Buffer size and block size must be > 0"));
            return result;
        }

        // Validate alignment
        if (alignment != 0u && (alignment & (alignment - 1u)) != 0u) {
            result.setError(AlignmentError("Pool Stack Alignment not set properly!"));
            return result;
        }

        // Normalize alignment (LOCAL variable)
        size_t eff_align = alignment ? alignment : alignof(max_align_t);
        if (eff_align < alignof(void*)) {
            eff_align = alignof(void*);
        }

        // Calculate stride (LOCAL variable)
        size_t stride = _align_up_size(block_size, eff_align);
        if (stride < sizeof(void*)) {
            stride = sizeof(void*);
        }

        // Arena base alignment should be >= max_align_t
        size_t arena_base_align = eff_align;
        if (arena_base_align < alignof(max_align_t)) {
            arena_base_align = alignof(max_align_t);
        }

        // Create the owned static arena (header lives in caller buffer)
        auto arena_result = ArenaAllocator::Stack(buffer, buffer_bytes, arena_base_align);
        if (!arena_result.hasValue()) {
            result.setError(arena_result.error());
            return result;
        }

        auto arena_ptr = cslt::move(arena_result.value());
        ArenaAllocator* arena = arena_ptr.get();

        // Allocate pool header inside the arena
        auto pool_header = arena->alloc_aligned(sizeof(PoolAllocator), alignof(PoolAllocator), false);
        if (!pool_header.hasValue()) {
            result.setError(pool_header.error());
            return result;
        }

        // Construct pool in-place
        PoolAllocator* pool = new (pool_header.value()) PoolAllocator();

        // Release arena from UniquePtr - pool now manages it
        arena_ptr.release();

        // ============================================================================
        // CHANGED: Calculate how many blocks fit, accounting for alignment padding
        // ============================================================================
        size_t remaining = arena->remaining();
        
        // When we call arena->alloc_aligned(bytes, stride, false), the arena may need
        // up to (stride - 1) bytes of padding to align the allocation.
        // We must reserve this padding to avoid the allocation failing.
        size_t max_padding = stride > 0 ? (stride - 1) : 0;
        
        if (remaining <= max_padding) {
            pool->~PoolAllocator();
            ArenaDeleter{}(arena);
            result.setError(MemoryError("Buffer too small for even one block"));
            return result;
        }
        
        // Calculate usable space after reserving for worst-case alignment padding
        size_t usable = remaining - max_padding;
        size_t blocks = usable / stride;
        // ============================================================================

        if (blocks == 0) {
            pool->~PoolAllocator();
            ArenaDeleter{}(arena);
            result.setError(MemoryError("Buffer too small for even one block"));
            return result;
        }

        // Initialize pool members
        pool->arena_ = arena;
        pool->owns_arena_ = true;   // Pool owns the arena object
        pool->block_size_ = block_size;
        pool->stride_ = stride;
        pool->blocks_per_chunk_ = 0;  // Not used for static pools
        pool->cur_ = nullptr;
        pool->end_ = nullptr;
        pool->free_list_ = nullptr;
        pool->total_blocks_ = 0;
        pool->free_blocks_ = 0;
        pool->grow_enabled_ = false;  // Static pools never grow

        // Base class initialization
        pool->default_alignment_ = eff_align;
        pool->mem_type_ = static_cast<uint8_t>(STATIC);
        pool->owns_memory_ = static_cast<uint8_t>(false);  // User owns buffer
        pool->size_ = 0;
        pool->alloc_ = blocks * stride;
        pool->total_alloc_ = buffer_bytes;

        // Prewarm (required for static pools - allocate all available blocks)
        size_t bytes = blocks * stride;
        auto slice = arena->alloc_aligned(bytes, stride, false);
        if (!slice.hasValue()) {
            pool->~PoolAllocator();
            ArenaDeleter{}(arena);
            result.setError(slice.error());
            return result;
        }

        uint8_t* base = static_cast<uint8_t*>(slice.value());
        pool->cur_ = base;
        pool->end_ = base + bytes;
        pool->total_blocks_ = blocks;
        pool->size_ = bytes;

        cslt::UniquePtr<PoolAllocator, PoolDeleter> ptr(pool, PoolDeleter{});
        result.setValue(cslt::move(ptr));
        return result;
    }
// // -------------------------------------------------------------------------------- 

    Expected<cslt::UniquePtr<PoolAllocator, PoolDeleter>>
    PoolAllocator::WithArena(ArenaAllocator& arena,
                             size_t block_size,
                             size_t blocks_per_chunk,
                             size_t alignment,
                             bool grow_enabled,
                             bool prewarm) {
        Expected<cslt::UniquePtr<PoolAllocator, PoolDeleter>> result;

        // Validate inputs
        if (block_size == 0 || blocks_per_chunk == 0) {
            result.setError(InvalidArgError("Block size and blocks_per_chunk must be > 0"));
            return result;
        }

        // Validate alignment
        if (alignment != 0u && (alignment & (alignment - 1u)) != 0u) {
            result.setError(AlignmentError("Pool WithArena Alignment not set properly!"));
            return result;
        }

        // Normalize alignment (LOCAL variable)
        size_t eff_align = alignment ? alignment : alignof(max_align_t);
        if (eff_align < alignof(void*)) {
            eff_align = alignof(void*);
        }

        // Calculate stride (LOCAL variable)
        size_t stride = _align_up_size(block_size, eff_align);
        if (stride < sizeof(void*)) {
            stride = sizeof(void*);
        }

        // Allocate pool header from arena
        auto pool_header = arena.alloc_aligned(sizeof(PoolAllocator), alignof(PoolAllocator), false);
        if (!pool_header.hasValue()) {
            result.setError(pool_header.error());
            return result;
        }

        // Construct pool in-place
        PoolAllocator* pool = new (pool_header.value()) PoolAllocator();

        // Initialize pool members
        pool->arena_ = &arena;
        pool->owns_arena_ = false;  // Borrowed arena
        pool->block_size_ = block_size;
        pool->stride_ = stride;
        pool->blocks_per_chunk_ = blocks_per_chunk;
        pool->cur_ = nullptr;
        pool->end_ = nullptr;
        pool->free_list_ = nullptr;
        pool->total_blocks_ = 0;
        pool->free_blocks_ = 0;
        pool->grow_enabled_ = grow_enabled;

        // Base class initialization
        pool->default_alignment_ = eff_align;
        pool->mem_type_ = static_cast<uint8_t>(arena.memory_type());
        pool->owns_memory_ = static_cast<uint8_t>(false);
        pool->size_ = 0;
        pool->alloc_ = 0;
        pool->total_alloc_ = 0;

        // Prewarm if requested
        if (prewarm) {
            // Calculate bytes needed for initial chunk
            size_t bytes = stride * blocks_per_chunk;
            
            // Allocate from arena directly
            auto prewarm_result = arena.alloc_aligned(bytes, stride, false);
            if (!prewarm_result.hasValue()) {
                pool->~PoolAllocator();
                result.setError(MemoryError("Failed to prewarm pool"));
                return result;
            }
            
            uint8_t* base = static_cast<uint8_t*>(prewarm_result.value());
            pool->cur_ = base;
            pool->end_ = base + bytes;
            pool->total_blocks_ = blocks_per_chunk;
            pool->size_ = bytes;
            pool->alloc_ = pool->total_blocks_ * stride;
        }

        cslt::UniquePtr<PoolAllocator, PoolDeleter> ptr(pool, PoolDeleter{});
        result.setValue(cslt::move(ptr));
        return result;
    }
// -------------------------------------------------------------------------------- 

    Expected<void*> PoolAllocator::alloc_aligned(size_t bytes,
                                                 size_t alignment,
                                                 bool zeroed) {
        Expected<void*> result;

        // Validate size matches block size
        if (bytes != block_size_) {
            result.setError(ArgumentError("Allocation size must equal pool block size"));
            return result;
        }

        // Validate alignment matches pool alignment
        size_t eff_align = alignment ? alignment : default_alignment_;
        if (eff_align != default_alignment_) {
            result.setError(AlignmentError("Pool alignment is fixed at creation time"));
            return result;
        }

        // Delegate to regular alloc
        return alloc(block_size_, zeroed);
    }
// -------------------------------------------------------------------------------- 

    Expected<void*> PoolAllocator::alloc_aligned_pool(size_t alignment, bool zeroed) {
        if (alignment != default_alignment_) {
            Expected<void*> result;
            result.setError(ArgumentError("Alignment must be equal to pool alignment"));
            return result;
        }
        return alloc_aligned(block_size_, alignment, zeroed);
    }
// -------------------------------------------------------------------------------- 

    Expected<void*> PoolAllocator::alloc(size_t bytes, bool zeroed) {
        (void) bytes;
        Expected<void*> result;
        void* blk = nullptr;
        
        // 1.) Try to pop from free list first
        void* ptr = pop_free();
        if (ptr) {
            if (zeroed) {
                memset(ptr, 0, block_size_);
            }
            result.setValue(ptr);
            return result;
        }
        
        // 2.) Carve from the current slice
        if (cur_ != end_) {
            blk = cur_;
            cur_ += stride_;
            if (zeroed) {
                memset(blk, 0, block_size_);
            }
            result.setValue(blk);
            return result;
        }
        
        // 3.) At slice end -> attempt to grow (if allowed)
        if (!grow_enabled_) {
            result.setError(CapacityOverflowError("Pool out of capacity and cannot grow"));
            return result;
        }
        
        bool ok = grow_pool();
        if (!ok) {
            result.setError(BadAllocError("Pool cannot alloc more memory"));
            return result;
        }
        
        if (cur_ >= end_) {
            result.setError(StateCorruptError("Pool state corrupted"));
            return result;
        }
        
        blk = cur_;
        cur_ += stride_;
        if (zeroed) {
            memset(blk, 0, block_size_);
        }
        result.setValue(blk);
        return result;
    }
// -------------------------------------------------------------------------------- 

    Expected<void*> PoolAllocator::alloc_pool(bool zeroed) {
        return alloc(10, zeroed);
    }
// -------------------------------------------------------------------------------- 

    Expected<void*> PoolAllocator::realloc(void* ptr,
                                           size_t old_bytes,
                                           size_t new_bytes,
                                           bool zeroed) {
        (void)ptr;
        (void)old_bytes;
        (void)new_bytes;
        (void)zeroed;

        Expected<void*> result;
        result.setError(FeatureDisabledError("Pool allocators do not support realloc"));
        return result;
    }
// -------------------------------------------------------------------------------- 

    Expected<void*> PoolAllocator::realloc_aligned(void* ptr,
                                                   size_t old_bytes,
                                                   size_t new_bytes,
                                                   size_t alignment,
                                                   bool zeroed) {
        (void)ptr;
        (void)old_bytes;
        (void)new_bytes;
        (void)alignment;
        (void)zeroed;

        Expected<void*> result;
        result.setError(FeatureDisabledError("Pool allocators do not support realloc"));
        return result;
    }
// -------------------------------------------------------------------------------- 

    void PoolAllocator::return_element(void* ptr, size_t bytes, size_t alignment) {
        (void)alignment;

        if (!ptr) return;

        // Validate size
        if (bytes != block_size_) {
            // In debug builds, this is a serious error
            // In release, we'll just ignore it
            return;
        }

        // Validate pointer belongs to this pool
        if (!is_ptr(ptr)) {
            return;
        }

        // Push to free list
        push_free(ptr);
    }
// -------------------------------------------------------------------------------- 

    bool PoolAllocator::reset(bool trim_extra_chunks) {
        // Clear free list
        free_list_ = nullptr;
        free_blocks_ = 0;

        // Reset arena if we own it
        if (owns_arena_ && arena_) {
            return arena_->reset(trim_extra_chunks);
        }

        // If we don't own the arena, we can't reset it
        // Just clear our state
        cur_ = nullptr;
        total_blocks_ = 0;
        size_ = 0;

        return true;
    }
// -------------------------------------------------------------------------------- 

    void* PoolAllocator::save() const {
        if (!arena_) {
            return nullptr;
        }

        // Allocate checkpoint data
        PoolCheckpointData* cp = new (std::nothrow) PoolCheckpointData;
        if (!cp) {
            return nullptr;
        }

        // Save current state
        cp->free_list = free_list_;
        cp->free_blocks = free_blocks_;
        cp->cur = cur_;
        cp->total_blocks = total_blocks_;

        return static_cast<void*>(cp);
    }
// -------------------------------------------------------------------------------- 

    bool PoolAllocator::restore(void* checkpoint) {
        if (!checkpoint) {
            return false;
        }

        PoolCheckpointData* cp = static_cast<PoolCheckpointData*>(checkpoint);

        // Restore state
        free_list_ = cp->free_list;
        free_blocks_ = cp->free_blocks;
        cur_ = cp->cur;
        total_blocks_ = cp->total_blocks;

        // Update size accounting
        size_t allocated = total_blocks_ - free_blocks_;
        size_ = allocated * stride_;

        // Clean up checkpoint
        delete cp;
        return true;
    }
// -------------------------------------------------------------------------------- 

    bool PoolAllocator::is_ptr(void* ptr) const {
        if (!arena_) {
            return false;
        }
        return arena_->is_ptr(ptr);
    }
// -------------------------------------------------------------------------------- 

    bool PoolAllocator::is_ptr_sized(void* ptr, size_t bytes) const {
        if (bytes != block_size_) {
            return false;
        }
        return is_ptr(ptr);
    }
// -------------------------------------------------------------------------------- 

    bool PoolAllocator::stats(char* buffer, size_t buffer_size) const {
        size_t offset = 0;

        if (!buffer || buffer_size == 0) {
            return false;
        }

        if (!_buf_appendf(buffer, buffer_size, &offset, "Pool Allocator Statistics:\n")) {
            return false;
        }

        // Memory type
        const char* type_str = (static_cast<MemType>(mem_type_) == STATIC) ? "STATIC" : "DYNAMIC";
        if (!_buf_appendf(buffer, buffer_size, &offset, "  Type: %s\n", type_str)) {
            return false;
        }

        // Block information
        if (!_buf_appendf(buffer, buffer_size, &offset, "  Block Size: %zu bytes\n", block_size_)) {
            return false;
        }

        if (!_buf_appendf(buffer, buffer_size, &offset, "  Stride (aligned): %zu bytes\n", stride_)) {
            return false;
        }

        // Block counts
        size_t allocated = total_blocks_ - free_blocks_;
        if (!_buf_appendf(buffer, buffer_size, &offset, "  Total Blocks: %zu\n", total_blocks_)) {
            return false;
        }

        if (!_buf_appendf(buffer, buffer_size, &offset, "  Allocated Blocks: %zu\n", allocated)) {
            return false;
        }

        if (!_buf_appendf(buffer, buffer_size, &offset, "  Free Blocks: %zu\n", free_blocks_)) {
            return false;
        }

        // Utilization
        if (total_blocks_ > 0) {
            double util = (100.0 * static_cast<double>(allocated)) / static_cast<double>(total_blocks_);
            if (!_buf_appendf(buffer, buffer_size, &offset, "  Utilization: %.1f%%\n", util)) {
                return false;
            }
        }

        // Memory usage
        size_t total_memory = total_blocks_ * stride_;
        size_t used_memory = allocated * stride_;
        if (!_buf_appendf(buffer, buffer_size, &offset, "  Total Memory: %zu bytes\n", total_memory)) {
            return false;
        }

        if (!_buf_appendf(buffer, buffer_size, &offset, "  Used Memory: %zu bytes\n", used_memory)) {
            return false;
        }

        // Growth information
        const char* grow_str = grow_enabled_ ? "Yes" : "No";
        if (!_buf_appendf(buffer, buffer_size, &offset, "  Can Grow: %s\n", grow_str)) {
            return false;
        }

        if (blocks_per_chunk_ > 0) {
            if (!_buf_appendf(buffer, buffer_size, &offset, "  Blocks Per Chunk: %zu\n", blocks_per_chunk_)) {
                return false;
            }
        }

        // Arena ownership
        const char* owns_str = owns_arena_ ? "Yes" : "No";
        if (!_buf_appendf(buffer, buffer_size, &offset, "  Owns Arena: %s\n", owns_str)) {
            return false;
        }

        return true;
    }
// -------------------------------------------------------------------------------- 

    void PoolAllocator::toggle_grow(bool enable) noexcept {
        // Can only enable growth if we own a dynamic arena
        if (!owns_arena_) {
            return;
        }

        if (static_cast<MemType>(mem_type_) == STATIC) {
            return;
        }

        grow_enabled_ = enable;
    }
// ================================================================================ 
// ================================================================================ 
} /* cslt namespace */
// ================================================================================
// ================================================================================
// eof
