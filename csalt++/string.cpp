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

#include "string.hpp"
#include <cstring>
#include <new>
// ================================================================================
// ================================================================================

namespace cslt {

    // ============================================================================
    // String Constructor
    // ============================================================================

    String::String(const char* cstr, size_t capacity_bytes, Allocator& allocator)
        : str_(nullptr), len_(0), alloc_(0), allocator_(&allocator) {
        
        size_t const src_len = std::strlen(cstr);
        
        // Determine actual capacity
        if (capacity_bytes == 0u) {
            capacity_bytes = src_len;  // Default: fit entire string
        }
        
        size_t const buf_bytes = capacity_bytes + 1u;  // Always reserve null terminator
        size_t const copy_len = (src_len < capacity_bytes) ? src_len : capacity_bytes;
        
        // Allocate buffer
        auto buf_result = allocator.alloc(buf_bytes, false);
        if (!buf_result.hasValue()) {
            // Leave str_ as nullptr to indicate allocation failure
            return;
        }
        
        str_ = static_cast<char*>(buf_result.value());
        len_ = copy_len;
        alloc_ = buf_bytes;
        
        // Copy data
        if (copy_len > 0u) {
            std::memcpy(str_, cstr, copy_len);
        }
        str_[copy_len] = '\0';  // Always null-terminate
    }
// --------------------------------------------------------------------------------

    // ============================================================================
    // String Destructor
    // ============================================================================

    String::~String() noexcept {
        if (str_ && allocator_) {
            allocator_->return_element(str_, alloc_, allocator_->default_alignment());
            str_ = nullptr;
        }
    }
// --------------------------------------------------------------------------------

    // ============================================================================
    // String::init - Static Factory Function
    // ============================================================================

    Expected<String*> String::init(const char* cstr, 
                                    size_t capacity_bytes,
                                    Allocator& allocator) noexcept {
        Expected<String*> result;
        
        if (!cstr) {
            result.setError(ArgumentError("NULL pointer passed to String::init"));
            return result;
        }
        
        // Allocate String object
        auto obj_result = allocator.alloc(sizeof(String), true);
        if (!obj_result.hasValue()) {
            result.setError(obj_result.error());
            return result;
        }
        
        // Placement new to construct String
        String* s = new (obj_result.value()) String(cstr, capacity_bytes, allocator);
        
        // Check if buffer allocation succeeded in constructor
        if (!s->str_) {
            // Buffer allocation failed, clean up String object
            s->~String();
            allocator.return_element(obj_result.value(), sizeof(String), 
                                    allocator.default_alignment());
            result.setError(MemoryError("Failed to allocate string buffer"));
            return result;
        }
        
        result.setValue(s);
        return result;
    }
// --------------------------------------------------------------------------------

    // ============================================================================
    // StringDeleter Implementation
    // ============================================================================

    void StringDeleter::operator()(String* s) const noexcept {
        if (!s) return;
        
        // Save allocator pointer before destruction
        Allocator* allocator = s->allocator_;
        
        // Call destructor to free buffer
        s->~String();
        
        // Free the String object itself
        if (allocator) {
            allocator->return_element(static_cast<void*>(s), 
                                     sizeof(String), 
                                     allocator->default_alignment());
        }
    }
// ================================================================================ 
// ================================================================================ 
} // namespace cslt
// ================================================================================ 
// ================================================================================ 
// eof
// ================================================================================
// ================================================================================
// eof
