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
#ifdef _WIN32
    #include <malloc.h>  // for _aligned_malloc/_aligned_free
#else
    #include <stdlib.h>  // for posix_memalign/free
#endif
// ================================================================================ 
// ================================================================================ 

namespace cslt {

    // Helper function for formatted appending
    static bool _buf_appendf(char *buffer,
                            size_t buffer_size,
                            size_t *p_offset,
                            const char *fmt, ...) {
        if ((buffer == NULL) || (p_offset == NULL) || (fmt == NULL)) {
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

    void HeapAllocator::return_element(void *ptr, size_t bytes, size_t alignment) {
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
        
        if ((buffer == NULL) || (buffer_size == 0U)) {
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
}
// ================================================================================
// ================================================================================
// eof
