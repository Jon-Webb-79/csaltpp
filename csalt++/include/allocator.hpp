// ================================================================================
// ================================================================================
// - File:    allocator.hpp
// - Purpose: Describe the file purpose here
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

#include "error.hpp"

#include <iostream>
#include <cstddef>
// ================================================================================ 
// ================================================================================ 

namespace cslt {
    class Allocator {
    protected:
        size_t default_alignment_;
    public:
        explicit Allocator(size_t alignment = alignof(max_align_t))  noexcept
            : default_alignment_(alignment) {}
// -------------------------------------------------------------------------------- 

        virtual ~Allocator() noexcept = default;
// -------------------------------------------------------------------------------- 

        size_t default_alignment() const noexcept {return default_alignment_; }
// -------------------------------------------------------------------------------- 

        virtual Expected<void*> alloc(size_t bytes,
                                      bool zeroed = false) = 0;
// -------------------------------------------------------------------------------- 

        virtual Expected<void*> alloc_aligned(size_t bytes,
                                              size_t alignment,
                                              bool zeroed = false) = 0;
// -------------------------------------------------------------------------------- 

        virtual Expected<void*> realloc(void* ptr,
                                        size_t old_bytes,
                                        size_t new_bytes,
                                        bool zeroed = false) = 0;
// -------------------------------------------------------------------------------- 

        virtual Expected<void*> realloc_aligned(void *ptr,
                                                size_t old_bytes,
                                                size_t new_bytes,
                                                size_t alignment,
                                                bool zeroed = false) = 0;
// -------------------------------------------------------------------------------- 

        // Returns element within allocator if appropriate, should be NO-OP if not applicable
        virtual void return_element(void *ptr, size_t bytes, size_t alignment) = 0;
    };
// ================================================================================ 
// ================================================================================ 

    class HeapAllocator : public Allocator {
    public:
        using Allocator::Allocator; // Inherit constructor

        ~HeapAllocator() noexcept override  = default;
// -------------------------------------------------------------------------------- 

        Expected<void*> alloc(size_t bytes,
                              bool zeroed = false) override;
// -------------------------------------------------------------------------------- 

        Expected<void*> alloc_aligned(size_t bytes,
                                      size_t alignment,
                                      bool zeroed = false) override;
// -------------------------------------------------------------------------------- 

        Expected<void*> realloc(void* ptr,
                                size_t old_bytes,
                                size_t new_bytes,
                                bool zeroed = false) override;
// -------------------------------------------------------------------------------- 

        Expected<void*> realloc_aligned(void* ptr,
                                        size_t old_bytes,
                                        size_t new_bytes,
                                        size_t alignment,
                                        bool zeroed = false) override;
// --------------------------------------------------------------------------------

        void return_element(void *ptr, size_t bytes, size_t alignment) override;
    };
}
// ================================================================================ 
// ================================================================================ 
#endif /* file_name_HPP */
// ================================================================================
// ================================================================================
// eof
