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

    constexpr size_t DEFAULT_FREELIST_SIZE = 4096;

    // Minimum allocation size (must fit a FreeBlock)
    size_t FreeListAllocator::min_request() {
        return DEFAULT_FREELIST_SIZE;
    }

// ================================================================================
// FreeListAllocator Constructor/Destructor
// ================================================================================

    FreeListAllocator::FreeListAllocator()
        : head_(nullptr)
        , cur_(nullptr)
        , len_(0)
        , memory_(nullptr)
        , arena_(nullptr)
        , owns_arena_(false) {
    }

    FreeListAllocator::~FreeListAllocator() noexcept {
        // Destructor doesn't need to do much - the FreeListDeleter
        // handles arena cleanup if needed
        head_ = nullptr;
        cur_ = nullptr;
        memory_ = nullptr;
        arena_ = nullptr;
    }

// ================================================================================
// Factory Methods
// ================================================================================

#if ARENA_ENABLE_DYNAMIC
    Expected<cslt::UniquePtr<FreeListAllocator, FreeListDeleter>>
    FreeListAllocator::Heap(size_t bytes,
                            size_t alignment,
                            bool resize) {
        Expected<cslt::UniquePtr<FreeListAllocator, FreeListDeleter>> result;

        // Validate inputs - bytes must be at least DEFAULT_FREELIST_SIZE
        if (bytes < DEFAULT_FREELIST_SIZE) {
            bytes = DEFAULT_FREELIST_SIZE;  // Default to 4096
        }
        
        if (bytes < min_request()) {
            result.setError(ArgumentError("Bytes must be at least minimum freelist size"));
            return result;
        }

        // Validate and normalize alignment
        size_t eff_align;
        if (alignment != 0u && (alignment & (alignment - 1u)) != 0u) {
            result.setError(AlignmentError("Alignment must be power of 2"));
            return result;
        }
        
        eff_align = alignment ? alignment : alignof(max_align_t);
        if (eff_align < alignof(max_align_t)) {
            eff_align = alignof(max_align_t);
        }

        // Compute minimum user-space required:
        // [aligned FreeListAllocator] + [at least one FreeBlock] + [payload bytes]
        size_t struct_size_aligned = _align_up_size(sizeof(FreeListAllocator), eff_align);
        size_t min_free_region = sizeof(FreeBlock);
        size_t requested_payload = bytes;

        // Overflow guard
        if (struct_size_aligned > SIZE_MAX - min_free_region ||
            (struct_size_aligned + min_free_region) > SIZE_MAX - requested_payload) {
            result.setError(LengthOverflowError("Size calculation overflow"));
            return result;
        }

        size_t min_total_user = struct_size_aligned + min_free_region + requested_payload;

        // Create owned arena (dynamic)
        size_t min_chunk = 0;  // Use arena's default
        
        auto arena_result = ArenaAllocator::Heap(
            min_total_user,
            resize,
            min_chunk,
            eff_align
        );

        if (!arena_result.hasValue()) {
            result.setError(arena_result.error());
            return result;
        }

        auto arena_ptr = cslt::move(arena_result.value());
        ArenaAllocator* arena = arena_ptr.get();

        // Determine actual usable bytes exposed by arena
        size_t available = arena->remaining();
        if (available < (struct_size_aligned + min_free_region)) {
            result.setError(MemoryError("Arena too small for freelist structures"));
            return result;
        }

        // Carve a single contiguous region from the arena for everything
        auto mem_result = arena->alloc(available, false);
        if (!mem_result.hasValue()) {
            result.setError(mem_result.error());
            return result;
        }

        void* base = mem_result.value();

        // FreeListAllocator at the beginning
        FreeListAllocator* fl = new (base) FreeListAllocator();

        // Release arena from UniquePtr - freelist now owns it
        arena_ptr.release();

        // Usable memory starts after the aligned FreeListAllocator
        void* memory = static_cast<uint8_t*>(base) + struct_size_aligned;
        size_t usable_size = available - struct_size_aligned;

        // Initialize freelist members
        fl->memory_ = memory;
        fl->cur_ = static_cast<uint8_t*>(memory);
        fl->len_ = 0;
        fl->arena_ = arena;
        fl->owns_arena_ = true;

        // Base class initialization
        fl->default_alignment_ = eff_align;
        fl->mem_type_ = static_cast<uint8_t>(DYNAMIC);
        fl->owns_memory_ = static_cast<uint8_t>(true);
        fl->size_ = 0;
        fl->alloc_ = usable_size;       // Usable space for allocations
        fl->total_alloc_ = bytes;       // Total requested by user

        // Initialize with one large free block spanning entire usable region
        fl->head_ = static_cast<FreeBlock*>(memory);
        fl->head_->size = usable_size;
        fl->head_->next = nullptr;

        cslt::UniquePtr<FreeListAllocator, FreeListDeleter> ptr(fl, FreeListDeleter{});
        result.setValue(cslt::move(ptr));
        return result;
    }
#endif // ARENA_ENABLE_DYNAMIC
// ================================================================================ 
// ================================================================================ 

    Expected<cslt::UniquePtr<FreeListAllocator, FreeListDeleter>>
    FreeListAllocator::Stack(void* buffer,
                             size_t buffer_bytes,
                             size_t alignment) {
        Expected<cslt::UniquePtr<FreeListAllocator, FreeListDeleter>> result;

        // Validate inputs
        if (!buffer) {
            result.setError(ArgumentError("Buffer cannot be null"));
            return result;
        }

        if (buffer_bytes == 0) {
            result.setError(ArgumentError("Buffer size must be > 0"));
            return result;
        }

        // Must at least fit control structures in the caller buffer
        if (buffer_bytes < (sizeof(FreeListAllocator) + sizeof(FreeBlock))) {
            result.setError(ArgumentError("Buffer too small for freelist structures"));
            return result;
        }

        // Validate and normalize alignment
        if (alignment != 0u && (alignment & (alignment - 1u)) != 0u) {
            result.setError(AlignmentError("Alignment must be power of 2"));
            return result;
        }

        size_t eff_align = alignment ? alignment : alignof(max_align_t);
        if (eff_align < alignof(max_align_t)) {
            eff_align = alignof(max_align_t);
        }

        // Create static arena over user buffer (arena header lives in buffer)
        auto arena_result = ArenaAllocator::Stack(buffer, buffer_bytes, eff_align);
        if (!arena_result.hasValue()) {
            result.setError(arena_result.error());
            return result;
        }

        auto arena_ptr = cslt::move(arena_result.value());
        ArenaAllocator* arena = arena_ptr.get();

        // Space for freelist header inside arena
        size_t fl_hdr = _align_up_size(sizeof(FreeListAllocator), eff_align);

        // Use arena usable capacity (data capacity, not total footprint)
        size_t arena_bytes = arena->allocated();
        if (arena_bytes < (fl_hdr + sizeof(FreeBlock))) {
            result.setError(ArgumentError("Arena capacity insufficient after overhead"));
            return result;
        }

        // Calculate usable space for freelist region
        size_t usable_size = arena_bytes - fl_hdr;
        size_t total_needed = fl_hdr + usable_size;

        // Carve everything (freelist header + remaining usable memory) in one shot
        auto mem_result = arena->alloc(total_needed, false);
        if (!mem_result.hasValue()) {
            result.setError(mem_result.error());
            return result;
        }

        void* base = mem_result.value();

        // Construct freelist in-place
        FreeListAllocator* fl = new (base) FreeListAllocator();

        // Release arena from UniquePtr - freelist now manages it
        arena_ptr.release();

        // Usable region starts after aligned freelist header
        void* region = static_cast<uint8_t*>(base) + fl_hdr;

        // Initialize freelist members
        fl->memory_ = region;
        fl->cur_ = static_cast<uint8_t*>(region);
        fl->len_ = 0;
        fl->arena_ = arena;
        fl->owns_arena_ = true;  // Owns the arena object (but not the buffer)

        // Base class initialization
        fl->default_alignment_ = eff_align;
        fl->mem_type_ = static_cast<uint8_t>(STATIC);
        fl->owns_memory_ = static_cast<uint8_t>(false);  // User owns buffer
        fl->size_ = 0;
        fl->alloc_ = usable_size;
        fl->total_alloc_ = total_needed;

        // Initialize with one large free block
        fl->head_ = static_cast<FreeBlock*>(region);
        fl->head_->size = usable_size;
        fl->head_->next = nullptr;

        cslt::UniquePtr<FreeListAllocator, FreeListDeleter> ptr(fl, FreeListDeleter{});
        result.setValue(cslt::move(ptr));
        return result;
    }
// -------------------------------------------------------------------------------- 

    Expected<cslt::UniquePtr<FreeListAllocator, FreeListDeleter>>
    FreeListAllocator::WithArena(ArenaAllocator& arena,
                                 size_t bytes,
                                 size_t alignment) {
        Expected<cslt::UniquePtr<FreeListAllocator, FreeListDeleter>> result;

        // Validate inputs
        if (bytes < DEFAULT_FREELIST_SIZE) {
            bytes = DEFAULT_FREELIST_SIZE;
        }

        // Validate and normalize alignment
        if (alignment != 0u && (alignment & (alignment - 1u)) != 0u) {
            result.setError(AlignmentError("Alignment must be power of 2"));
            return result;
        }

        size_t eff_align = alignment ? alignment : alignof(max_align_t);
        if (eff_align < alignof(max_align_t)) {
            eff_align = alignof(max_align_t);
        }

        // Compute struct and usable sizes (both aligned)
        size_t struct_size = _align_up_size(sizeof(FreeListAllocator), eff_align);
        size_t usable_size = _align_up_size(bytes, eff_align);

        if (usable_size < sizeof(FreeBlock)) {
            usable_size = sizeof(FreeBlock);
        }

        // Overflow guard: total_alloc = struct_size + usable_size
        if (struct_size > (SIZE_MAX - usable_size)) {
            result.setError(LengthOverflowError("Size calculation overflow"));
            return result;
        }

        size_t total_alloc = struct_size + usable_size;

        // Single allocation from arena for everything
        auto alloc_result = arena.alloc(total_alloc, false);
        if (!alloc_result.hasValue()) {
            result.setError(alloc_result.error());
            return result;
        }

        void* base = alloc_result.value();

        // FreeListAllocator struct at the beginning
        FreeListAllocator* fl = new (base) FreeListAllocator();

        // Usable memory starts after the struct region (already aligned)
        void* memory = static_cast<uint8_t*>(base) + struct_size;

        // Initialize freelist members
        fl->memory_ = memory;
        fl->cur_ = static_cast<uint8_t*>(memory);
        fl->len_ = 0;
        fl->arena_ = &arena;
        fl->owns_arena_ = false;  // Borrowed arena

        // Base class initialization
        fl->default_alignment_ = eff_align;
        fl->mem_type_ = static_cast<uint8_t>(arena.memory_type());  // Inherit from arena
        fl->owns_memory_ = static_cast<uint8_t>(false);  // Doesn't own arena
        fl->size_ = 0;
        fl->alloc_ = usable_size;
        fl->total_alloc_ = total_alloc;

        // Initialize with one large free block
        fl->head_ = static_cast<FreeBlock*>(memory);
        fl->head_->size = usable_size;
        fl->head_->next = nullptr;

        cslt::UniquePtr<FreeListAllocator, FreeListDeleter> ptr(fl, FreeListDeleter{});
        result.setValue(cslt::move(ptr));
        return result;
    }
// ================================================================================
// Allocator Interface Implementation - Stubs for now
// ================================================================================

    Expected<void*> FreeListAllocator::alloc(size_t bytes, bool zeroed) {
        Expected<void*> result;
        
        if (bytes == 0) {
            result.setError(ArgumentError("Cannot allocate 0 bytes"));
            return result;
        }
        
        // Use default alignment
        return alloc_aligned(bytes, default_alignment_, zeroed);
    }
// -------------------------------------------------------------------------------- 

    Expected<void*> FreeListAllocator::alloc_aligned(size_t bytes,
                                                      size_t alignment,
                                                      bool zeroed) {
        Expected<void*> result;
        
        if (bytes == 0) {
            result.setError(ArgumentError("Cannot allocate 0 bytes"));
            return result;
        }
        
        // Normalize alignment (0 means use default)
        size_t eff_align = alignment ? alignment : default_alignment_;
        
        // Validate alignment is power of 2
        if ((eff_align & (eff_align - 1u)) != 0u) {
            result.setError(AlignmentError("Alignment must be power of 2"));
            return result;
        }
        
        const size_t header_size = sizeof(FreeListHeader);
        
        // Overflow guard for: user_addr = align_up(block_addr + header_size, eff_align)
        // and user_end = user_addr + bytes
        if (bytes > SIZE_MAX - header_size - (eff_align - 1u)) {
            result.setError(CapacityOverflowError("Allocation size overflow"));
            return result;
        }
        
        // Search free list for suitable block
        FreeBlock** current = &head_;
        
        while (*current) {
            FreeBlock* block = *current;
            uintptr_t block_addr = reinterpret_cast<uintptr_t>(block);
            
            // Defensive overflow check for block_end
            if (block->size > static_cast<size_t>(UINTPTR_MAX - block_addr)) {
                result.setError(CapacityOverflowError("Block size overflow"));
                return result;
            }
            uintptr_t block_end = block_addr + static_cast<uintptr_t>(block->size);
            
            // Calculate where user pointer would be (after header and alignment)
            uintptr_t after_header = block_addr + static_cast<uintptr_t>(header_size);
            uintptr_t user_addr = _align_up_uintptr(after_header, eff_align);
            
            // Defensive overflow check for user_end
            if (static_cast<uintptr_t>(bytes) > (UINTPTR_MAX - user_addr)) {
                result.setError(CapacityOverflowError("User region overflow"));
                return result;
            }
            uintptr_t user_end = user_addr + static_cast<uintptr_t>(bytes);
            
            // Does this block have enough space?
            if (user_end > block_end) {
                current = &block->next;
                continue;
            }
            
            // Block is large enough - calculate sizes
            size_t offset = static_cast<size_t>(user_addr - block_addr);
            size_t used_size = static_cast<size_t>(user_end - block_addr);
            size_t remaining = block->size - used_size;
            
            size_t block_size_for_hdr;
            
            // Should we split the block?
            if (remaining >= sizeof(FreeBlock)) {
                // Split block: front portion used, remainder stays free
                FreeBlock* new_block = reinterpret_cast<FreeBlock*>(
                    reinterpret_cast<uint8_t*>(block) + used_size
                );
                new_block->size = remaining;
                new_block->next = block->next;
                
                block->size = used_size;
                *current = new_block;
                
                block_size_for_hdr = used_size;
            } else {
                // Consume entire block (remaining too small to split)
                block_size_for_hdr = block->size;
                *current = block->next;
            }
            
            // Store allocation metadata in header
            uint8_t* user_ptr = reinterpret_cast<uint8_t*>(user_addr);
            FreeListHeader* hdr = reinterpret_cast<FreeListHeader*>(user_ptr - header_size);
            
            hdr->block_size = block_size_for_hdr;
            hdr->offset = offset;
            
            // Account for full block consumption
            len_ += block_size_for_hdr;
            size_ += block_size_for_hdr;
            
            // Update high-water mark
            uint8_t* block_used_end = reinterpret_cast<uint8_t*>(block) + block_size_for_hdr;
            if (block_used_end > cur_) {
                cur_ = block_used_end;
            }
            
            // Zero if requested
            if (zeroed) {
                memset(user_ptr, 0, bytes);
            }
            
            result.setValue(user_ptr);
            return result;
        }
        
        // No block large enough
        result.setError(CapacityOverflowError("No suitable free block available"));
        return result;
    }
// -------------------------------------------------------------------------------- 

    Expected<void*> FreeListAllocator::realloc(void* ptr,
                                               size_t old_bytes,
                                               size_t new_bytes,
                                               bool zeroed) {
        Expected<void*> result;
        
        // Reject zero-size realloc requests
        if (new_bytes == 0) {
            result.setError(ArgumentError("Cannot realloc to 0 bytes"));
            return result;
        }
        
        // NULL ptr behaves like alloc
        if (!ptr) {
            return alloc(new_bytes, zeroed);
        }
        
        // Shrink or same size: keep pointer (freelist semantics - no shrinking)
        if (new_bytes <= old_bytes) {
            result.setValue(ptr);
            return result;
        }
        
        // Grow: allocate new, copy, optionally zero tail, then free old
        auto alloc_result = alloc(new_bytes, false);  // Don't zero yet
        if (!alloc_result.hasValue()) {
            // Propagate the allocation failure
            return alloc_result;
        }
        
        void* new_ptr = alloc_result.value();
        
        // Copy old data
        memcpy(new_ptr, ptr, old_bytes);
        
        // Zero new region if requested
        if (zeroed) {
            memset(static_cast<uint8_t*>(new_ptr) + old_bytes, 0, new_bytes - old_bytes);
        }
        
        // Return old block to freelist
        return_element(ptr, old_bytes, 0);
        
        result.setValue(new_ptr);
        return result;
    }
// -------------------------------------------------------------------------------- 

    Expected<void*> FreeListAllocator::realloc_aligned(void* ptr,
                                                        size_t old_bytes,
                                                        size_t new_bytes,
                                                        size_t alignment,
                                                        bool zeroed) {
        Expected<void*> result;
        
        // Validate requested size
        if (new_bytes == 0) {
            result.setError(ArgumentError("Cannot realloc to 0 bytes"));
            return result;
        }
        
        // NULL ptr behaves like aligned alloc
        if (!ptr) {
            return alloc_aligned(new_bytes, alignment, zeroed);
        }
        
        // Shrink or same size: keep pointer (no shrink performed)
        if (new_bytes <= old_bytes) {
            result.setValue(ptr);
            return result;
        }
        
        // Grow with requested alignment
        auto alloc_result = alloc_aligned(new_bytes, alignment, false);  // Don't zero yet
        if (!alloc_result.hasValue()) {
            // Propagate alignment/capacity failures
            return alloc_result;
        }
        
        void* new_ptr = alloc_result.value();
        
        // Copy old data
        memcpy(new_ptr, ptr, old_bytes);
        
        // Zero new tail region if requested
        if (zeroed) {
            memset(static_cast<uint8_t*>(new_ptr) + old_bytes, 0, new_bytes - old_bytes);
        }
        
        // Return old block to freelist
        return_element(ptr, old_bytes, 0);
        
        result.setValue(new_ptr);
        return result;
    }
// -------------------------------------------------------------------------------- 

    void FreeListAllocator::return_element(void* ptr, size_t bytes, size_t alignment) {
        // bytes and alignment are unused for freelists (interface compatibility)
        (void)bytes;
        (void)alignment;
        
        if (!ptr) {
            return;
        }
        
        const size_t header_size = sizeof(FreeListHeader);
        
        uint8_t* user_ptr = static_cast<uint8_t*>(ptr);
        uint8_t* mem_start8 = static_cast<uint8_t*>(memory_);
        uint8_t* mem_end8 = mem_start8 + alloc_;
        
        // Basic bounds: user pointer must be inside region and leave room for header
        if (user_ptr < mem_start8 + header_size || user_ptr > mem_end8) {
            return;  // Invalid pointer
        }
        
        // Header sits immediately before user pointer
        FreeListHeader* hdr = reinterpret_cast<FreeListHeader*>(user_ptr - header_size);
        
        size_t block_size = hdr->block_size;
        size_t offset = hdr->offset;
        
        // Reconstruct block start
        uint8_t* block_start = user_ptr - offset;
        
        uintptr_t block_addr = reinterpret_cast<uintptr_t>(block_start);
        uintptr_t mem_start = reinterpret_cast<uintptr_t>(memory_);
        uintptr_t mem_end = mem_start + alloc_;
        
        // Sanity checks on block size and bounds
        if (block_size < sizeof(FreeBlock) || block_size > alloc_) {
            return;  // Invalid block size
        }
        
        if (block_addr < mem_start || block_addr + block_size > mem_end) {
            return;  // Block out of bounds
        }
        
        // Also ensure offset is sane (block_size must cover offset + header at least)
        if (offset > block_size) {
            return;  // Invalid offset
        }
        
        // Accounting: we charged block_size on alloc, so undo exactly that
        if (len_ < block_size) {
            return;  // Underflow - shouldn't happen
        }
        
        len_ -= block_size;
        size_ -= block_size;
        
        // Turn region back into a free block
        FreeBlock* block = reinterpret_cast<FreeBlock*>(block_start);
        block->size = block_size;
        
        // Insert into free list in address order
        FreeBlock* prev = nullptr;
        FreeBlock* curr = head_;
        
        while (curr && curr < block) {
            prev = curr;
            curr = curr->next;
        }
        
        block->next = curr;
        if (prev) {
            prev->next = block;
        } else {
            head_ = block;
        }
        
        // Coalesce with next block if adjacent
        if (block->next) {
            uint8_t* block_end = reinterpret_cast<uint8_t*>(block) + block->size;
            if (block_end == reinterpret_cast<uint8_t*>(block->next)) {
                // Merge with next
                block->size += block->next->size;
                block->next = block->next->next;
            }
        }
        
        // Coalesce with previous block if adjacent
        if (prev) {
            uint8_t* prev_end = reinterpret_cast<uint8_t*>(prev) + prev->size;
            if (prev_end == reinterpret_cast<uint8_t*>(block)) {
                // Merge with prev
                prev->size += block->size;
                prev->next = block->next;
            }
        }
    }
// -------------------------------------------------------------------------------- 

    bool FreeListAllocator::reset(bool trim) {
        // trim parameter unused for freelists
        (void)trim;
        
        if (!memory_ || alloc_ == 0) {
            // Freelist not properly initialized or already torn down
            return false;
        }
        
        // Reset accounting
        cur_ = static_cast<uint8_t*>(memory_);
        len_ = 0;
        size_ = 0;
        
        // Recreate a single large free block covering the entire region
        head_ = static_cast<FreeBlock*>(memory_);
        head_->size = alloc_;
        head_->next = nullptr;
        
        return true;
    }
// -------------------------------------------------------------------------------- 

    bool FreeListAllocator::is_ptr(void* ptr) const {
        if (!ptr) {
            return false;
        }
        
        const size_t header_size = sizeof(FreeListHeader);
        
        uintptr_t mem_start = reinterpret_cast<uintptr_t>(memory_);
        uintptr_t mem_end = mem_start + alloc_;
        uintptr_t ptr_addr = reinterpret_cast<uintptr_t>(ptr);
        
        // Pointer must be within the managed region and leave room for the header
        if (ptr_addr < mem_start + header_size || ptr_addr > mem_end) {
            return false;
        }
        
        // Header sits immediately before the user pointer
        const uint8_t* user_ptr = static_cast<const uint8_t*>(ptr);
        const FreeListHeader* hdr = reinterpret_cast<const FreeListHeader*>(user_ptr - header_size);
        
        size_t block_size = hdr->block_size;
        size_t offset = hdr->offset;
        
        // Reconstruct block start
        const uint8_t* block_start = user_ptr - offset;
        uintptr_t block_addr = reinterpret_cast<uintptr_t>(block_start);
        
        // Basic sanity on offset and size
        // block_size must be large enough to cover the offset and at least a FreeBlock
        if (offset > block_size) {
            return false;
        }
        
        if (block_size < sizeof(FreeBlock) || block_size > alloc_) {
            return false;
        }
        
        // block_start must be within the freelist region
        if (block_addr < mem_start || block_addr >= mem_end) {
            return false;
        }
        
        // Check that the full block fits within the region, overflow-safe:
        // block_addr + block_size <= mem_end  <=>  block_size <= mem_end - block_addr
        if (block_size > static_cast<size_t>(mem_end - block_addr)) {
            return false;
        }
        
        // Also check that ptr lies within [block_start, block_start + block_size)
        if (ptr_addr < block_addr || ptr_addr >= block_addr + block_size) {
            return false;
        }
        
        return true;
    }
// -------------------------------------------------------------------------------- 

    bool FreeListAllocator::is_ptr_sized(void* ptr, size_t bytes) const {
        if (!ptr || bytes == 0) {
            return false;
        }
        
        // First check if it's at least a plausible freelist pointer
        if (!is_ptr(ptr)) {
            return false;
        }
        
        const size_t header_size = sizeof(FreeListHeader);
        const uint8_t* user_ptr = static_cast<const uint8_t*>(ptr);
        
        const FreeListHeader* hdr = reinterpret_cast<const FreeListHeader*>(user_ptr - header_size);
        
        size_t block_size = hdr->block_size;
        size_t offset = hdr->offset;
        
        if (offset > block_size) {
            return false;
        }
        
        // User data size = block_size - offset
        size_t user_data_size = block_size - offset;
        
        // Requested size must fit within the user data region
        if (bytes > user_data_size) {
            return false;
        }
        
        // Also verify that ptr + bytes doesn't run off the freelist region
        uintptr_t ptr_addr = reinterpret_cast<uintptr_t>(ptr);
        uintptr_t mem_start = reinterpret_cast<uintptr_t>(memory_);
        uintptr_t mem_end = mem_start + alloc_;
        
        // Overflow-safe: ptr_addr + bytes <= mem_end  <=>  bytes <= mem_end - ptr_addr
        if (bytes > static_cast<size_t>(mem_end - ptr_addr)) {
            return false;
        }
        
        return true;
    }
// -------------------------------------------------------------------------------- 

    bool FreeListAllocator::stats(char* buffer, size_t buffer_size) const {
        size_t offset = 0;
        
        if (!buffer || buffer_size == 0) {
            return false;
        }
        
        if (!_buf_appendf(buffer, buffer_size, &offset, "%s", "FreeListAllocator Statistics:\n")) {
            return false;
        }
        
        // Type / ownership information
        const char* type_str = "UNKNOWN";
        switch (static_cast<MemType>(mem_type_)) {
            case STATIC:  type_str = "STATIC";  break;
            case DYNAMIC: type_str = "DYNAMIC"; break;
            default:      type_str = "UNKNOWN"; break;
        }
        
        if (!_buf_appendf(buffer, buffer_size, &offset, "  Type: %s\n", type_str)) {
            return false;
        }
        
        if (!_buf_appendf(buffer, buffer_size, &offset,
                         "  Owns arena: %s\n", owns_arena_ ? "yes" : "no")) {
            return false;
        }
        
        // Basic accounting
        size_t used = len_;                    // Current usage
        size_t capacity = alloc_;              // Usable capacity
        size_t total = total_alloc_;           // Total with overhead
        size_t remaining_bytes = capacity > used ? capacity - used : 0;
        
        if (!_buf_appendf(buffer, buffer_size, &offset,
                         "  Used (accounted): %zu bytes\n", used)) {
            return false;
        }
        
        if (!_buf_appendf(buffer, buffer_size, &offset,
                         "  Remaining: %zu bytes\n", remaining_bytes)) {
            return false;
        }
        
        if (!_buf_appendf(buffer, buffer_size, &offset,
                         "  Capacity (usable region): %zu bytes\n", capacity)) {
            return false;
        }
        
        if (!_buf_appendf(buffer, buffer_size, &offset,
                         "  Total (with header/overhead): %zu bytes\n", total)) {
            return false;
        }
        
        // Utilization of the usable freelist region
        if (capacity == 0) {
            if (!_buf_appendf(buffer, buffer_size, &offset,
                             "%s", "  Utilization: N/A (capacity is 0)\n")) {
                return false;
            }
        } else {
            double util = (100.0 * static_cast<double>(used)) / static_cast<double>(capacity);
            if (!_buf_appendf(buffer, buffer_size, &offset,
                             "  Utilization: %.1f%%\n", util)) {
                return false;
            }
        }
        
        // Alignment info
        if (!_buf_appendf(buffer, buffer_size, &offset,
                         "  Base alignment: %zu bytes\n", default_alignment_)) {
            return false;
        }
        
        // Free list layout
        const FreeBlock* current = head_;
        int block_count = 0;
        size_t free_bytes = 0;
        
        while (current) {
            block_count++;
            free_bytes += current->size;
            
            if (!_buf_appendf(buffer, buffer_size, &offset,
                             "  Free block %d: %p, %zu bytes\n",
                             block_count,
                             static_cast<const void*>(current),
                             current->size)) {
                return false;
            }
            
            current = current->next;
        }
        
        if (!_buf_appendf(buffer, buffer_size, &offset,
                         "  Free blocks: %d, total free bytes (raw): %zu\n",
                         block_count, free_bytes)) {
            return false;
        }
        
        return true;
    }
// ================================================================================
// FreeList-Specific Query Methods
// ================================================================================

    size_t FreeListAllocator::remaining() const noexcept {
        return alloc_ - size_;
    }

    size_t FreeListAllocator::used() const {
        return len_;
    }

    bool FreeListAllocator::owns_arena() const {
        return owns_arena_;
    }
// ================================================================================ 
// ================================================================================ 

// Platform-specific includes
#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#else
    #include <sys/mman.h>
    #include <unistd.h>
#endif

// ================================================================================
// OS Memory Allocation (Platform-Specific)
// ================================================================================

#ifdef _WIN32
    void* BuddyAllocator::os_alloc(size_t size) {
        return VirtualAlloc(NULL, size, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    }
// -------------------------------------------------------------------------------- 

    void BuddyAllocator::os_free(void* ptr, size_t size) {
        (void)size;
        if (ptr) {
            VirtualFree(ptr, 0, MEM_RELEASE);
        }
    }
#else
// -------------------------------------------------------------------------------- 

    void* BuddyAllocator::os_alloc(size_t size) {
        void* p = mmap(NULL, size, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        return (p == MAP_FAILED) ? nullptr : p;
    }
// -------------------------------------------------------------------------------- 

    void BuddyAllocator::os_free(void* ptr, size_t size) {
        if (ptr && size) {
            munmap(ptr, size);
        }
    }
#endif
// ================================================================================
// Helper Functions
// ================================================================================

    uint32_t BuddyAllocator::ilog2(size_t x) {
        uint32_t r = 0;
        while (x > 1) {
            x >>= 1;
            r++;
        }
        return r;
    }
// -------------------------------------------------------------------------------- 

    size_t BuddyAllocator::next_pow2(size_t x) {
        if (x == 0) return 0;
        if ((x & (x - 1)) == 0) return x;  // Already power of 2
        
        // Find position of highest set bit
        size_t power = 1;
        while (power < x) {
            if (power > SIZE_MAX / 2) return 0;  // Overflow
            power <<= 1;
        }
        return power;
    }
// -------------------------------------------------------------------------------- 

    uint32_t BuddyAllocator::order_to_level(uint32_t order) const {
        return order - min_order_;
    }
// -------------------------------------------------------------------------------- 

    uint32_t BuddyAllocator::level_to_order(uint32_t level) const {
        return min_order_ + level;
    }
// -------------------------------------------------------------------------------- 

    int32_t BuddyAllocator::find_nonempty_level(uint32_t desired_level) const {
        for (uint32_t lvl = desired_level; lvl < num_levels_; ++lvl) {
            if (free_lists_[lvl] != nullptr) {
                return static_cast<int32_t>(lvl);
            }
        }
        return -1;  // None available
    }
// -------------------------------------------------------------------------------- 

    void BuddyAllocator::freelist_push(BuddyBlock** head, BuddyBlock* block) {
        block->next = *head;
        *head = block;
    }
// -------------------------------------------------------------------------------- 

    bool BuddyAllocator::freelist_remove(BuddyBlock** head, BuddyBlock* block) {
        BuddyBlock* prev = nullptr;
        BuddyBlock* cur = *head;
        
        while (cur) {
            if (cur == block) {
                if (prev) {
                    prev->next = cur->next;
                } else {
                    *head = cur->next;
                }
                return true;
            }
            prev = cur;
            cur = cur->next;
        }
        return false;
    }
// -------------------------------------------------------------------------------- 

    BuddyAllocator::BuddyBlock* BuddyAllocator::freelist_find(BuddyBlock* head, void* addr) const {
        while (head) {
            if (static_cast<void*>(head) == addr) {
                return head;
            }
            head = head->next;
        }
        return nullptr;
    }
// -------------------------------------------------------------------------------- 

    bool BuddyAllocator::ptr_in_pool_(const void* p) const noexcept {
        if (!base_ || pool_size_ == 0) return false;
        auto b = static_cast<const uint8_t*>(base_);
        auto x = static_cast<const uint8_t*>(p);
        // Ensure header-at-(p-Header) lies in pool
        return (x >= b + sizeof(BuddyHeader)) && (x < b + pool_size_);
    }

// ================================================================================
// Constructor/Destructor
// ================================================================================

    BuddyAllocator::BuddyAllocator()
        : base_(nullptr)
        , free_lists_(nullptr)
        , pool_size_(0)
        , base_align_(0)
        , user_offset_(0)
        , min_order_(0)
        , max_order_(0)
        , num_levels_(0)
    {
    }
// -------------------------------------------------------------------------------- 

    BuddyAllocator::~BuddyAllocator() noexcept {
        // Cleanup is handled by BuddyDeleter
    }
// ================================================================================
// Factory Method: Heap
// ================================================================================

    Expected<UniquePtr<BuddyAllocator, BuddyDeleter>>
    BuddyAllocator::Heap(size_t pool_size,
                         size_t min_block_size,
                         size_t base_align) {
        Expected<UniquePtr<BuddyAllocator, BuddyDeleter>> result;
        
        // Validate inputs
        if (pool_size == 0 || min_block_size == 0) {
            result.setError(ArgumentError("Pool size and min block size must be non-zero"));
            return result;
        }
        
        // Allocate BuddyAllocator structure
        void* mem = ::operator new(sizeof(BuddyAllocator), std::nothrow);
        if (!mem) {
            result.setError(MemoryError("Failed to allocate BuddyAllocator structure"));
            return result;
        }
        
        BuddyAllocator* buddy = new (mem) BuddyAllocator();
        
        // Normalize base alignment
        if (base_align == 0) {
            base_align = alignof(max_align_t);
        }
        
        // Round up to power of 2
        if ((base_align & (base_align - 1)) != 0) {
            size_t next = next_pow2(base_align);
            if (next == 0) {
                buddy->~BuddyAllocator();
                ::operator delete(mem);
                result.setError(CapacityOverflowError("Alignment too large"));
                return result;
            }
            base_align = next;
        }
        
        // Validate alignment
        if (base_align == 0 || (base_align & (base_align - 1)) != 0) {
            buddy->~BuddyAllocator();
            ::operator delete(mem);
            result.setError(AlignmentError("Invalid alignment"));
            return result;
        }
        
        // Compute header-aligned user offset
        size_t header_size = sizeof(BuddyHeader);
        
        // Overflow guard
        if (header_size > SIZE_MAX - (base_align - 1)) {
            buddy->~BuddyAllocator();
            ::operator delete(mem);
            result.setError(CapacityOverflowError("User offset calculation overflow"));
            return result;
        }
        
        size_t user_offset = (header_size + (base_align - 1)) & ~(base_align - 1);
        
        // Ensure min_block_size can hold header + alignment
        if (min_block_size < user_offset) {
            min_block_size = user_offset;
        }
        
        // Round sizes up to powers of two
        size_t min_blk = next_pow2(min_block_size);
        if (min_blk == 0) {
            buddy->~BuddyAllocator();
            ::operator delete(mem);
            result.setError(CapacityOverflowError("Min block size too large"));
            return result;
        }
        
        size_t pool = next_pow2(pool_size);
        if (pool == 0) {
            buddy->~BuddyAllocator();
            ::operator delete(mem);
            result.setError(CapacityOverflowError("Pool size too large"));
            return result;
        }
        
        // Validate min block <= pool
        if (min_blk > pool) {
            buddy->~BuddyAllocator();
            ::operator delete(mem);
            result.setError(ArgumentError("Min block size exceeds pool size"));
            return result;
        }
        
        uint32_t min_order = ilog2(min_blk);
        uint32_t max_order = ilog2(pool);
        uint32_t num_levels = (max_order - min_order) + 1;
        
        if (num_levels == 0) {
            buddy->~BuddyAllocator();
            ::operator delete(mem);
            result.setError(ArgumentError("Invalid order range"));
            return result;
        }
        
        // Allocate backing pool from OS
        void* base = os_alloc(pool);
        if (!base) {
            buddy->~BuddyAllocator();
            ::operator delete(mem);
            result.setError(MemoryError("Failed to allocate OS memory pool"));
            return result;
        }
        
        // Allocate free-lists array
        BuddyBlock** free_lists = new (std::nothrow) BuddyBlock*[num_levels];
        if (!free_lists) {
            os_free(base, pool);
            buddy->~BuddyAllocator();
            ::operator delete(mem);
            result.setError(MemoryError("Failed to allocate free-lists array"));
            return result;
        }
        
        // Initialize free-lists to nullptr
        for (uint32_t i = 0; i < num_levels; ++i) {
            free_lists[i] = nullptr;
        }
        
        // Populate buddy structure
        buddy->base_ = base;
        buddy->pool_size_ = pool;
        buddy->min_order_ = min_order;
        buddy->max_order_ = max_order;
        buddy->num_levels_ = num_levels;
        buddy->free_lists_ = free_lists;
        buddy->base_align_ = base_align;
        buddy->user_offset_ = user_offset;
        
        // Base class initialization
        buddy->default_alignment_ = base_align;
        buddy->mem_type_ = static_cast<uint8_t>(DYNAMIC);
        buddy->owns_memory_ = static_cast<uint8_t>(true);
        buddy->size_ = 0;
        buddy->alloc_ = pool;
        
        // Calculate total_alloc: pool + free_lists + BuddyAllocator struct
        size_t total = pool;
        size_t lists_bytes = num_levels * sizeof(BuddyBlock*);
        
        // Overflow guards
        if (total > SIZE_MAX - lists_bytes) {
            os_free(base, pool);
            delete[] free_lists;
            buddy->~BuddyAllocator();
            ::operator delete(mem);
            result.setError(CapacityOverflowError("Total allocation size overflow"));
            return result;
        }
        total += lists_bytes;
        
        if (total > SIZE_MAX - sizeof(BuddyAllocator)) {
            os_free(base, pool);
            delete[] free_lists;
            buddy->~BuddyAllocator();
            ::operator delete(mem);
            result.setError(CapacityOverflowError("Total allocation size overflow"));
            return result;
        }
        total += sizeof(BuddyAllocator);
        
        buddy->total_alloc_ = total;
        
        // Seed top free list with one large block
        BuddyBlock* initial_block = static_cast<BuddyBlock*>(base);
        initial_block->next = nullptr;
        
        uint32_t top_level = buddy->order_to_level(max_order);
        buddy->free_lists_[top_level] = initial_block;
        
        // Create UniquePtr and return
        UniquePtr<BuddyAllocator, BuddyDeleter> ptr(buddy, BuddyDeleter{});
        result.setValue(cslt::move(ptr));
        return result;
    }
// -------------------------------------------------------------------------------- 

    Expected<void*> BuddyAllocator::alloc(size_t bytes, bool zeroed) {
        Expected<void*> result;
        
        if (bytes == 0) {
            result.setError(ArgumentError("Allocation size must be non-zero"));
            return result;
        }
        
        // Total = user payload + header (guard overflow)
        if (bytes > SIZE_MAX - sizeof(BuddyHeader)) {
            result.setError(CapacityOverflowError("Size too large"));
            return result;
        }
        size_t total = bytes + sizeof(BuddyHeader);
        
        // Ensure at least min block size
        size_t min_block = (size_t)1 << min_order_;
        if (total < min_block) {
            total = min_block;
        } else {
            total = next_pow2(total);
            if (total == 0) {
                result.setError(CapacityOverflowError("Size overflow during rounding"));
                return result;
            }
        }
        
        // Request cannot exceed pool
        if (total > pool_size_) {
            result.setError(MemoryError("Request exceeds pool capacity"));
            return result;
        }
        
        // Compute order for total (defensive clamp)
        uint32_t order = ilog2(total);
        if (order < min_order_) order = min_order_;
        if (order > max_order_) {
            result.setError(MemoryError("Request too large"));
            return result;
        }
        
        // Find a free block
        uint32_t desired_level = order_to_level(order);
        int32_t lvl = find_nonempty_level(desired_level);
        if (lvl < 0) {
            result.setError(MemoryError("No free blocks available"));
            return result;
        }
        
        // Pop a block from level 'lvl'
        BuddyBlock* block = free_lists_[lvl];
        free_lists_[lvl] = block->next;
        block->next = nullptr;
        
        uint32_t current_order = level_to_order(static_cast<uint32_t>(lvl));
        size_t current_size = (size_t)1 << current_order;
        
        // Split down until we reach the desired order
        while (current_order > order) {
            current_order--;
            current_size >>= 1;
            
            BuddyBlock* split_block = 
                reinterpret_cast<BuddyBlock*>(reinterpret_cast<uint8_t*>(block) + current_size);
            split_block->next = nullptr;
            
            uint32_t split_level = order_to_level(current_order);
            freelist_push(&free_lists_[split_level], split_block);
        }
        
        // Final block is size 2^order
        uint8_t* block_bytes = reinterpret_cast<uint8_t*>(block);
        
        // Write header
        BuddyHeader* hdr = reinterpret_cast<BuddyHeader*>(block_bytes);
        hdr->order = order;
        hdr->block_offset = static_cast<size_t>(block_bytes - static_cast<uint8_t*>(base_));
        
        size_t block_size = (size_t)1 << order;
        
        // Defensive: ensure accounting cannot overflow
        if (block_size > SIZE_MAX - size_) {
            result.setError(CapacityOverflowError("Accounting overflow"));
            return result;
        }
        size_ += block_size;  // Base class accounting
        
        // User pointer starts after header
        uint8_t* user_ptr = block_bytes + sizeof(BuddyHeader);
        
        // Zero-initialize if requested
        if (zeroed) {
            memset(user_ptr, 0, block_size - sizeof(BuddyHeader));
        }
        
        result.setValue(static_cast<void*>(user_ptr));
        return result;
    }
// -------------------------------------------------------------------------------- 

    Expected<void*> BuddyAllocator::alloc_aligned(size_t bytes, size_t alignment, bool zeroed) {
        Expected<void*> result;
        
        if (bytes == 0) {
            result.setError(ArgumentError("Allocation size must be non-zero"));
            return result;
        }
        
        // Normalize alignment: 0 -> natural alignment
        if (alignment == 0) {
            alignment = alignof(max_align_t);
        }
        
        // Require power-of-two alignment; round up if needed
        if ((alignment & (alignment - 1)) != 0) {
            alignment = next_pow2(alignment);
            if (alignment == 0) {
                result.setError(AlignmentError("Alignment too large"));
                return result;
            }
        }
        
        // Validate alignment
        if (alignment == 0) {
            result.setError(AlignmentError("Invalid alignment"));
            return result;
        }
        
        // Guard: total = size + header + (align-1)
        if (bytes > SIZE_MAX - sizeof(BuddyHeader) - (alignment - 1)) {
            result.setError(CapacityOverflowError("Size too large with alignment"));
            return result;
        }
        
        size_t total = bytes + sizeof(BuddyHeader) + (alignment - 1);
        
        // Ensure at least min block size
        size_t min_block = (size_t)1 << min_order_;
        if (total < min_block) {
            total = min_block;
        } else {
            total = next_pow2(total);
            if (total == 0) {
                result.setError(CapacityOverflowError("Size overflow during rounding"));
                return result;
            }
        }
        
        if (total > pool_size_) {
            result.setError(MemoryError("Request exceeds pool capacity"));
            return result;
        }
        
        uint32_t order = ilog2(total);
        if (order < min_order_) order = min_order_;
        if (order > max_order_) {
            result.setError(MemoryError("Request too large"));
            return result;
        }
        
        // Find a free block
        uint32_t desired_level = order_to_level(order);
        int32_t lvl = find_nonempty_level(desired_level);
        if (lvl < 0) {
            result.setError(MemoryError("No free blocks available"));
            return result;
        }
        
        // Take a block from level 'lvl'
        BuddyBlock* block = free_lists_[lvl];
        free_lists_[lvl] = block->next;
        block->next = nullptr;
        
        uint32_t current_order = level_to_order(static_cast<uint32_t>(lvl));
        size_t current_size = (size_t)1 << current_order;
        
        // Split down until we reach the desired order
        while (current_order > order) {
            current_order--;
            current_size >>= 1;
            
            BuddyBlock* split_block = 
                reinterpret_cast<BuddyBlock*>(reinterpret_cast<uint8_t*>(block) + current_size);
            split_block->next = nullptr;
            
            uint32_t split_level = order_to_level(current_order);
            freelist_push(&free_lists_[split_level], split_block);
        }
        
        size_t block_size = (size_t)1 << order;
        uint8_t* block_bytes = reinterpret_cast<uint8_t*>(block);
        
        // Find an aligned user pointer inside this block
        uintptr_t block_addr = reinterpret_cast<uintptr_t>(block_bytes);
        
        // min_user = block_addr + sizeof(header)
        if (static_cast<uintptr_t>(sizeof(BuddyHeader)) > (UINTPTR_MAX - block_addr)) {
            // Should never happen; treat as overflow and undo allocation
            uint32_t lvl_final = order_to_level(order);
            freelist_push(&free_lists_[lvl_final], block);
            result.setError(CapacityOverflowError("Pointer arithmetic overflow"));
            return result;
        }
        uintptr_t min_user = block_addr + static_cast<uintptr_t>(sizeof(BuddyHeader));
        
        // aligned_user = align_up(min_user, align)
        uintptr_t aligned_user = (min_user + static_cast<uintptr_t>(alignment - 1)) & 
                                 ~static_cast<uintptr_t>(alignment - 1);
        
        // Ensure aligned_user + size <= block_addr + block_size
        uintptr_t block_end = block_addr + static_cast<uintptr_t>(block_size);
        if (static_cast<uintptr_t>(bytes) > (UINTPTR_MAX - aligned_user) || 
            (aligned_user + static_cast<uintptr_t>(bytes)) > block_end) {
            // Defensive: return block to free list
            uint32_t lvl_final = order_to_level(order);
            freelist_push(&free_lists_[lvl_final], block);
            result.setError(MemoryError("Insufficient space for alignment"));
            return result;
        }
        
        uint8_t* user_ptr = reinterpret_cast<uint8_t*>(aligned_user);
        
        // Header lives immediately before user_ptr
        BuddyHeader* hdr = reinterpret_cast<BuddyHeader*>(user_ptr - sizeof(BuddyHeader));
        hdr->order = order;
        hdr->block_offset = static_cast<size_t>(block_bytes - static_cast<uint8_t*>(base_));
        
        size_ += block_size;  // Base class accounting
        
        if (zeroed) {
            // Zero the usable payload region from user_ptr to end-of-block
            size_t payload_bytes = static_cast<size_t>((block_bytes + block_size) - user_ptr);
            memset(user_ptr, 0, payload_bytes);
        }
        
        result.setValue(static_cast<void*>(user_ptr));
        return result;
    }
// -------------------------------------------------------------------------------- 

    Expected<void*> BuddyAllocator::realloc(void* ptr, size_t old_bytes, size_t new_bytes, bool zeroed) {
        Expected<void*> result;
        
        // realloc(NULL, n) => alloc(n)
        if (!ptr) {
            if (new_bytes == 0) {
                result.setError(ArgumentError("Cannot realloc NULL to zero size"));
                return result;
            }
            return alloc(new_bytes, zeroed);
        }
        
        // realloc(ptr, 0) => free(ptr), return NULL
        if (new_bytes == 0) {
            return_element(ptr, old_bytes);
            result.setValue(nullptr);
            return result;
        }
        
        // Caller must provide meaningful old_bytes when ptr != NULL
        if (old_bytes == 0) {
            result.setError(ArgumentError("old_bytes must be non-zero when ptr is non-NULL"));
            return result;
        }
        
        // Determine usable capacity behind old_ptr (based on header order)
        BuddyHeader* hdr = reinterpret_cast<BuddyHeader*>(
            static_cast<uint8_t*>(ptr) - sizeof(BuddyHeader));
        
        uint32_t order = hdr->order;
        
        // Defensive: order must be within allocator range
        if (order < min_order_ || order > max_order_) {
            result.setError(ArgumentError("Invalid block header"));
            return result;
        }
        
        size_t block_size = (size_t)1 << order;
        if (block_size < sizeof(BuddyHeader)) {
            result.setError(CapacityOverflowError("Invalid block size"));
            return result;
        }
        
        size_t usable_old = block_size - sizeof(BuddyHeader);
        
        // If it fits, keep same pointer (no shrink; optional zero tail)
        if (new_bytes <= usable_old) {
            if (zeroed && new_bytes > old_bytes) {
                // Clamp old_bytes to usable_old to avoid writing past block
                size_t logical_old = (old_bytes < usable_old) ? old_bytes : usable_old;
                size_t extra = new_bytes - logical_old;
                memset(static_cast<uint8_t*>(ptr) + logical_old, 0, extra);
            }
            result.setValue(ptr);
            return result;
        }
        
        // Need a bigger block
        Expected<void*> new_expect = alloc(new_bytes, zeroed);
        if (!new_expect.hasValue()) {
            // old_ptr remains valid
            return new_expect;
        }
        
        void* new_ptr = new_expect.value();
        
        // Copy min(old_bytes, usable_old) bytes
        size_t copy_bytes = (old_bytes < usable_old) ? old_bytes : usable_old;
        memcpy(new_ptr, ptr, copy_bytes);
        
        // Return old block
        return_element(ptr, old_bytes);
        
        result.setValue(new_ptr);
        return result;
    }
// -------------------------------------------------------------------------------- 

    Expected<void*> BuddyAllocator::realloc_aligned(void* ptr, size_t old_bytes, size_t new_bytes,
                                                     size_t alignment, bool zeroed) {
        Expected<void*> result;
        
        // realloc(NULL, 0) => success with NULL (no-op)
        if (!ptr) {
            if (new_bytes == 0) {
                result.setValue(nullptr);
                return result;
            }
            return alloc_aligned(new_bytes, alignment, zeroed);
        }
        
        // realloc(p, 0) => free(p), success with NULL
        if (new_bytes == 0) {
            return_element(ptr, old_bytes);
            result.setValue(nullptr);
            return result;
        }
        
        // Caller claims old_bytes==0 but passed a pointer
        if (old_bytes == 0) {
            result.setError(ArgumentError("old_bytes must be non-zero when ptr is non-NULL"));
            return result;
        }
        
        // Normalize requested alignment: 0 -> natural; non-pow2 -> round up
        size_t eff_align = (alignment != 0) ? alignment : alignof(max_align_t);
        if ((eff_align & (eff_align - 1)) != 0) {
            eff_align = next_pow2(eff_align);
            if (eff_align == 0) {
                result.setError(AlignmentError("Alignment too large"));
                return result;
            }
        }
        
        // Introspect old block header
        BuddyHeader* hdr = reinterpret_cast<BuddyHeader*>(
            static_cast<uint8_t*>(ptr) - sizeof(BuddyHeader));
        uint32_t order = hdr->order;
        
        // Defensive: shifting by >= word bits is UB; sanity-check order
        if (order > max_order_) {
            result.setError(ArgumentError("Invalid block header"));
            return result;
        }
        
        size_t block_size = (size_t)1 << order;
        
        // Usable space behind old_ptr
        if (block_size < sizeof(BuddyHeader)) {
            result.setError(ArgumentError("Invalid block size"));
            return result;
        }
        size_t usable_old = block_size - sizeof(BuddyHeader);
        
        // If it fits and pointer already satisfies requested alignment, reuse
        if (new_bytes <= usable_old &&
            ((reinterpret_cast<uintptr_t>(ptr) & (eff_align - 1)) == 0))
        {
            if (zeroed && new_bytes > old_bytes) {
                memset(static_cast<uint8_t*>(ptr) + old_bytes, 0, new_bytes - old_bytes);
            }
            result.setValue(ptr);
            return result;
        }
        
        // Need a new block with (possibly stricter) alignment
        Expected<void*> new_expect = alloc_aligned(new_bytes, eff_align, zeroed);
        if (!new_expect.hasValue()) {
            // Old pointer remains valid; propagate error
            return new_expect;
        }
        
        void* new_ptr = new_expect.value();
        
        // Copy min(logical old size, old usable capacity)
        size_t copy_bytes = (old_bytes < usable_old) ? old_bytes : usable_old;
        memcpy(new_ptr, ptr, copy_bytes);
        
        return_element(ptr, old_bytes);
        
        result.setValue(new_ptr);
        return result;
    }
// -------------------------------------------------------------------------------- 

    void BuddyAllocator::return_element(void* ptr, size_t bytes, size_t alignment) {
        // bytes and alignment parameters ignored (interface compatibility)
        (void)bytes;
        (void)alignment;
        
        if (!ptr) {
            // Like free(NULL): no-op, success
            return;
        }
        
        uint8_t* user = static_cast<uint8_t*>(ptr);
        
        // Header is immediately before the user pointer
        BuddyHeader* hdr = reinterpret_cast<BuddyHeader*>(user - sizeof(BuddyHeader));
        
        uint32_t order = hdr->order;
        if (order < min_order_ || order > max_order_) {
            // Invalid header - silent failure
            return;
        }
        
        size_t block_size = (size_t)1 << order;
        
        // Block start is base + block_offset
        uint8_t* base = static_cast<uint8_t*>(base_);
        size_t off = hdr->block_offset;
        if (off + block_size > pool_size_) {
            // Invalid offset - silent failure
            return;
        }
        
        BuddyBlock* block = reinterpret_cast<BuddyBlock*>(base + off);
        
        if (size_ >= block_size) {
            size_ -= block_size;  // Base class accounting
        } else {
            size_ = 0;
        }
        
        // Coalesce with buddy blocks
        size_t cur_off = off;
        uint32_t cur_order = order;
        
        while (cur_order < max_order_) {
            size_t buddy_off = cur_off ^ ((size_t)1 << cur_order);
            uint8_t* buddy_addr = base + buddy_off;
            
            uint32_t lvl = order_to_level(cur_order);
            
            BuddyBlock* buddy_in_list = freelist_find(free_lists_[lvl], 
                                                       static_cast<void*>(buddy_addr));
            
            if (!buddy_in_list) {
                // Buddy not free, stop coalescing
                break;
            }
            
            // Remove buddy from free list
            freelist_remove(&free_lists_[lvl], buddy_in_list);
            
            // New merged block starts at the lower address
            if (buddy_off < cur_off) {
                cur_off = buddy_off;
                block = reinterpret_cast<BuddyBlock*>(base + cur_off);
            }
            
            cur_order++;
        }
        
        // Insert final (possibly coalesced) block into appropriate free list
        uint32_t final_level = order_to_level(cur_order);
        freelist_push(&free_lists_[final_level], block);
        
        return;
    }
// -------------------------------------------------------------------------------- 

    bool BuddyAllocator::reset(bool trim) {
        // trim parameter ignored for buddy allocators (no resizing)
        (void)trim;
        
        if (!base_ || pool_size_ == 0 || num_levels_ == 0 || max_order_ < min_order_) {
            return false;
        }
        
        // Clear all free lists
        for (uint32_t i = 0; i < num_levels_; ++i) {
            free_lists_[i] = nullptr;
        }
        
        // Single big free block spanning the whole pool
        BuddyBlock* initial_block = static_cast<BuddyBlock*>(base_);
        initial_block->next = nullptr;
        
        uint32_t top_level = order_to_level(max_order_);
        free_lists_[top_level] = initial_block;
        
        // No bytes "in use" from the pool any more
        size_ = 0;  // Base class accounting
        
        return true;
    }
// -------------------------------------------------------------------------------- 

    bool BuddyAllocator::is_ptr(void* ptr) const {
        if (!ptr) {
            return false;
        }
        if (!ptr_in_pool_(ptr)) return false;
        
        const uint8_t* p = static_cast<const uint8_t*>(ptr);
        const uint8_t* pool_start = static_cast<const uint8_t*>(base_);
        const uint8_t* pool_end = pool_start + pool_size_;
        
        // Step 0: ptr must lie inside the buddy pool's user range
        if (p < pool_start + sizeof(BuddyHeader) || p >= pool_end) {
            return false;
        }
        
        // Now it's safe to look at the header
        const BuddyHeader* hdr = reinterpret_cast<const BuddyHeader*>(p - sizeof(BuddyHeader));
        
        // Step 2: order must be valid
        if (hdr->order < min_order_ || hdr->order > max_order_) {
            return false;
        }
        
        size_t block_size = (size_t)1 << hdr->order;
        
        // Step 3: block_offset must be within pool
        if (hdr->block_offset + block_size > pool_size_) {
            return false;
        }
        
        // Step 4: the block must be aligned correctly
        if (hdr->block_offset & (block_size - 1)) {
            return false;
        }
        
        // Step 5: user pointer must lie inside the block
        const uint8_t* block_start = pool_start + hdr->block_offset;
        const uint8_t* block_end = block_start + block_size;
        
        if (p < block_start + sizeof(BuddyHeader) || p >= block_end) {
            return false;
        }
        
        // Everything checks out
        return true;
    }
// -------------------------------------------------------------------------------- 

    bool BuddyAllocator::is_ptr_sized(void* ptr, size_t bytes) const {
        if (!is_ptr(ptr)) {
            return false;
        }
        
        const BuddyHeader* hdr = reinterpret_cast<const BuddyHeader*>(
            static_cast<const uint8_t*>(ptr) - sizeof(BuddyHeader));
        
        size_t block_size = (size_t)1 << hdr->order;
        size_t usable = block_size - sizeof(BuddyHeader);
        
        if (bytes > usable) {
            return false;
        }
        
        return true;
    }
// -------------------------------------------------------------------------------- 

    bool BuddyAllocator::stats(char* buffer, size_t buffer_size) const {
        size_t offset = 0;
        
        if (!buffer || buffer_size == 0) {
            return false;
        }
        
        if (!base_) {
            if (!_buf_appendf(buffer, buffer_size, &offset, "%s", "Buddy: NULL\n")) {
                return false;
            }
            return true;
        }
        
        // Header
        if (!_buf_appendf(buffer, buffer_size, &offset, "%s", "Buddy Statistics:\n")) {
            return false;
        }
        
        // Basic capacity/usage numbers
        size_t const pool = pool_size_;
        size_t const used = size_;
        size_t const remaining = (pool > used) ? (pool - used) : 0;
        size_t const total_overhead = total_alloc_;
        
        size_t const min_block = (size_t)1 << min_order_;
        size_t const max_block = (size_t)1 << max_order_;
        size_t const largest = largest_block();
        
        if (!_buf_appendf(buffer, buffer_size, &offset,
                          "  Pool size: %zu bytes\n", pool)) {
            return false;
        }
        
        if (!_buf_appendf(buffer, buffer_size, &offset,
                          "  Min block size: %zu bytes\n", min_block)) {
            return false;
        }
        
        if (!_buf_appendf(buffer, buffer_size, &offset,
                          "  Max block size: %zu bytes\n", max_block)) {
            return false;
        }
        
        if (!_buf_appendf(buffer, buffer_size, &offset,
                          "  Used: %zu bytes\n", used)) {
            return false;
        }
        
        if (!_buf_appendf(buffer, buffer_size, &offset,
                          "  Remaining: %zu bytes\n", remaining)) {
            return false;
        }
        
        if (!_buf_appendf(buffer, buffer_size, &offset,
                          "  Total (with overhead): %zu bytes\n", total_overhead)) {
            return false;
        }
        
        if (!_buf_appendf(buffer, buffer_size, &offset,
                          "  Largest free block: %zu bytes\n", largest)) {
            return false;
        }
        
        // Utilization with divide-by-zero guard
        if (pool == 0) {
            if (!_buf_appendf(buffer, buffer_size, &offset,
                              "%s", "  Utilization: N/A (pool size is 0)\n")) {
                return false;
            }
        } else {
            double const util = (100.0 * static_cast<double>(used)) / static_cast<double>(pool);
            if (!_buf_appendf(buffer, buffer_size, &offset,
                              "  Utilization: %.1f%%\n", util)) {
                return false;
            }
        }
        
        // Per-level free list stats
        if (!_buf_appendf(buffer, buffer_size, &offset,
                          "%s", "  Free lists by level:\n")) {
            return false;
        }
        
        size_t total_free_bytes_from_lists = 0;
        
        for (uint32_t level = 0; level < num_levels_; ++level) {
            uint32_t const order = min_order_ + level;
            size_t const block_size = (size_t)1 << order;
            
            size_t count = 0;
            for (BuddyBlock* blk = free_lists_[level]; blk != nullptr; blk = blk->next) {
                count++;
            }
            
            size_t level_free_bytes = count * block_size;
            total_free_bytes_from_lists += level_free_bytes;
            
            if (!_buf_appendf(buffer, buffer_size, &offset,
                              "    Level %u (order %u, block %zu bytes): "
                              "%zu blocks, %zu bytes free\n",
                              level, order, block_size,
                              count, level_free_bytes)) {
                return false;
            }
        }
        
        // Optional cross-check of free bytes vs remaining
        if (!_buf_appendf(buffer, buffer_size, &offset,
                          "  Free bytes (sum of free lists): %zu bytes\n",
                          total_free_bytes_from_lists)) {
            return false;
        }
        
        return true;
    }
// -------------------------------------------------------------------------------- 

    size_t BuddyAllocator::remaining() const noexcept {
        if (pool_size_ > size_) {
            return pool_size_ - size_;
        }
        return 0;
    }
// -------------------------------------------------------------------------------- 

    size_t BuddyAllocator::largest_block() const noexcept {
        // Scan from highest order (largest blocks) down to smallest
        for (int32_t lvl = static_cast<int32_t>(num_levels_) - 1; lvl >= 0; --lvl) {
            if (free_lists_[lvl] != nullptr) {
                // This level has at least one free block
                uint32_t order = min_order_ + static_cast<uint32_t>(lvl);
                return (size_t)1 << order;
            }
        }
        
        return 0;
    }
// ================================================================================ 
// ================================================================================ 
} /* cslt namespace */
// ================================================================================
// ================================================================================
// eof
