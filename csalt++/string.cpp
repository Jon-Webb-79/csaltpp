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

#if defined(__AVX512BW__) && defined(__AVX512VL__)
  #include <immintrin.h>
  #include "simd_avx512_char.inl"

#elif defined(__AVX2__)
  #include <immintrin.h>
  #include "simd_avx2_char.inl"

#elif defined(__AVX__)
  #include <immintrin.h>
  #include "simd_avx_char.inl"

#elif defined(__SSE4_1__)
  #include <immintrin.h>
  #include "simd_sse41_char.inl"

#elif defined(__SSE3__)
  #include <immintrin.h>
  #include "simd_sse3_char.inl"

#elif defined(__SSE2__)
  #include <immintrin.h>
  #include "simd_sse2_char.inl"

#elif defined(__ARM_FEATURE_SVE2)
  #include <arm_sve.h>
  #include "simd_sve2_char.inl"

#elif defined(__ARM_FEATURE_SVE)
  #include <arm_sve.h>
  #include "simd_sve_char.inl"

#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
  #include <arm_neon.h>
  #include "simd_neon_char.inl"

#else
  #include "simd_scalar_char.inl"
#endif
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

    bool String::concat(const char* str) noexcept {
        if (!str_ || !str || !allocator_) return false;

        size_t const len2 = std::strlen(str);
        
        if (len2 == 0u) return true;  // Nothing to append

        // Overflow guard: len_ + len2 + 1
        if (len2 > (SIZE_MAX - 1u - len_)) return false;

        size_t const needed = len_ + len2 + 1u;  // Total bytes including null

        // Check for self-aliasing
        const char* const s_begin = str_;
        const char* const s_end = str_ + (len_ + 1u);
        bool const overlaps = (str >= s_begin && str < s_end);

        const char* src = str;
        char* temp = nullptr;

        // If we must grow and src aliases our buffer, copy src aside first
        if (needed > alloc_ && overlaps) {
            auto temp_result = allocator_->alloc(len2, false);
            if (!temp_result.hasValue()) return false;

            temp = static_cast<char*>(temp_result.value());
            std::memcpy(temp, str, len2);
            src = temp;
        }

        // Ensure capacity
        if (needed > alloc_) {
            // Try reallocate if available
            auto realloc_result = allocator_->realloc(
                str_, alloc_, needed, false
            );
            
            if (realloc_result.hasValue()) {
                str_ = static_cast<char*>(realloc_result.value());
                alloc_ = needed;
            } else {
                // Fallback: allocate new buffer
                auto new_result = allocator_->alloc(needed, false);
                if (!new_result.hasValue()) {
                    if (temp) {
                        allocator_->return_element(temp, len2, 
                                                  allocator_->default_alignment());
                    }
                    return false;
                }

                char* newbuf = static_cast<char*>(new_result.value());
                
                // Copy old content
                std::memcpy(newbuf, str_, len_ + 1u);
                
                // Return old buffer
                allocator_->return_element(str_, alloc_, 
                                          allocator_->default_alignment());
                
                str_ = newbuf;
                alloc_ = needed;
            }
        }

        // Append safely (memmove handles overlap)
        std::memmove(str_ + len_, src, len2);
        str_[len_ + len2] = '\0';
        len_ = len_ + len2;

        if (temp) {
            allocator_->return_element(temp, len2, allocator_->default_alignment());
        }
        
        return true;
    }
// -------------------------------------------------------------------------------- 

    bool String::concat(const String& str) noexcept {
        if (!str.str_) return false;
        return concat(str.str_);
    }
// -------------------------------------------------------------------------------- 

    int8_t String::compare(const char* str) const noexcept {
        // Error sentinel value
        constexpr int8_t COMPARE_ERROR = -128;
        
        if (!str_ || !str) {
            return COMPARE_ERROR;
        }

        // Compare up to len_ characters, but stop early if str ends
        for (size_t i = 0u; i < len_; ++i) {
            uint8_t const a = static_cast<uint8_t>(static_cast<unsigned char>(str_[i]));
            uint8_t const b = static_cast<uint8_t>(static_cast<unsigned char>(str[i]));

            // If C string ended before this String did
            if (b == 0u) {
                // If this String has '\0' here too, they match through len_ (treat as equal)
                return (a == 0u) ? static_cast<int8_t>(0) : static_cast<int8_t>(1);
            }

            if (a < b) { return static_cast<int8_t>(-1); }
            if (a > b) { return static_cast<int8_t>(1); }
        }

        // All len_ characters matched. If str has more chars, this is shorter => less
        return (str[len_] == '\0') ? static_cast<int8_t>(0) : static_cast<int8_t>(-1);
    }
// -------------------------------------------------------------------------------- 

    int8_t String::compare(const String& other) const noexcept {
        // Error sentinel value
        constexpr int8_t COMPARE_ERROR = -128;
        
        if (!str_ || !other.str_) {
            return COMPARE_ERROR;
        }

        size_t const n = (len_ < other.len_) ? len_ : other.len_;

        // Use SIMD to find first difference
        size_t const k = simd_first_diff_u8(
            reinterpret_cast<const uint8_t*>(str_),
            reinterpret_cast<const uint8_t*>(other.str_),
            n
        );

        // If difference found within common length
        if (k < n) {
            uint8_t const x = static_cast<uint8_t>(static_cast<unsigned char>(str_[k]));
            uint8_t const y = static_cast<uint8_t>(static_cast<unsigned char>(other.str_[k]));
            return (x < y) ? static_cast<int8_t>(-1) : static_cast<int8_t>(1);
        }

        // All common characters matched - compare lengths
        if (len_ < other.len_) return static_cast<int8_t>(-1);
        if (len_ > other.len_) return static_cast<int8_t>(1);
        return static_cast<int8_t>(0);
    }
// -------------------------------------------------------------------------------- 

    void String::reset() noexcept {
        if (!str_) {
            return;
        }
        
        len_ = 0u;
        str_[0] = '\0';
    }
// -------------------------------------------------------------------------------- 

    Expected<String*> String::copy() const noexcept {
        Expected<String*> result;
        
        if (!str_) {
            result.setError(ArgumentError("Cannot copy String with null buffer"));
            return result;
        }
        
        if (!allocator_) {
            result.setError(ArgumentError("No allocator available"));
            return result;
        }
        
        // String::init returns Expected<String*>
        // We need to extract the value or error and put it in our result
        auto init_result = String::init(str_, len_, *allocator_);
        
        if (init_result.hasValue()) {
            result.setValue(init_result.value());
        } else {
            result.setError(init_result.error());
        }
        
        return result;
    }
// --------------------------------------------------------------------------------

    Expected<String*> String::copy(Allocator& allocator) const noexcept {
        Expected<String*> result;
        
        if (!str_) {
            result.setError(ArgumentError("Cannot copy String with null buffer"));
            return result;
        }
        
        // String::init returns Expected<String*>
        // We need to extract the value or error and put it in our result
        auto init_result = String::init(str_, len_, allocator);
        
        if (init_result.hasValue()) {
            result.setValue(init_result.value());
        } else {
            result.setError(init_result.error());
        }
        
        return result;
    } 
// -------------------------------------------------------------------------------- 

    bool String::is_ptr(const void* ptr) const noexcept {
        if (!str_ || !ptr) {
            return false;
        }
        
        uintptr_t const begin = reinterpret_cast<uintptr_t>(static_cast<const void*>(str_));
        uintptr_t const end = begin + alloc_;  // one-past-end
        uintptr_t const addr = reinterpret_cast<uintptr_t>(ptr);
        
        // Overflow-safe containment check
        if (addr < begin) {
            return false;
        }
        if (addr >= end) {
            return false;
        }
        
        return true;
    }
// --------------------------------------------------------------------------------

    bool String::is_ptr(const void* ptr, size_t bytes) const noexcept {
        if (!str_ || !ptr || bytes == 0u) {
            return false;
        }
        
        uintptr_t const begin = reinterpret_cast<uintptr_t>(static_cast<const void*>(str_));
        uintptr_t const end = begin + alloc_;  // one-past-end
        uintptr_t const p = reinterpret_cast<uintptr_t>(ptr);
        
        // ptr must be within [begin, end)
        if (p < begin || p >= end) {
            return false;
        }
        
        // Overflow-safe: require bytes <= end - p (equivalent to p + bytes <= end)
        if (bytes > static_cast<size_t>(end - p)) {
            return false;
        }
        
        return true;
    }
// -------------------------------------------------------------------------------- 

    size_t String::find(const String& needle,
                        const void* begin,
                        const void* end,
                        direction_t dir) const noexcept {
        if (!str_ || !needle.str_) {
            return SIZE_MAX;
        }
        
        const uint8_t* const hs_base = reinterpret_cast<const uint8_t*>(static_cast<const void*>(str_));
        const uint8_t* const hs_used_end = hs_base + len_;  // exclude terminator
        
        // Defaults if begin/end omitted (nullptr)
        const uint8_t* search_begin = (begin == nullptr) 
            ? hs_base 
            : reinterpret_cast<const uint8_t*>(begin);
        
        const uint8_t* search_end = (end == nullptr) 
            ? hs_used_end 
            : reinterpret_cast<const uint8_t*>(end);
        
        // Validate begin and end are within buffer using existing is_ptr()
        if (!is_ptr(search_begin) || !is_ptr(search_end)) {
            return SIZE_MAX;
        }
        
        // Check monotonic ordering
        if (search_begin > search_end) {
            return SIZE_MAX;
        }
        
        // Clamp to used region (typical string-search semantics)
        if (search_begin > hs_used_end) {
            return SIZE_MAX;
        }
        if (search_end > hs_used_end) {
            search_end = hs_used_end;
        }
        
        size_t const region_len = static_cast<size_t>(search_end - search_begin);
        size_t const nlen = needle.len_;
        
        // Empty needle: define as found at start of region (offset 0)
        if (nlen == 0u) {
            return 0u;
        }
        
        if (nlen > region_len) {
            return SIZE_MAX;
        }
       
        size_t offset_from_search_start = simd_find_substr_u8(
            search_begin, region_len,
            reinterpret_cast<const uint8_t*>(static_cast<const void*>(needle.str_)), nlen,
            dir
        );
        
        // Convert to offset from string start
        if (offset_from_search_start == SIZE_MAX) {
            return SIZE_MAX;
        }
        
        return (search_begin - hs_base) + offset_from_search_start;
    }
// --------------------------------------------------------------------------------

    size_t String::find(const char* needle,
                        const void* begin,
                        const void* end,
                        direction_t dir) const noexcept {
        if (!str_ || !needle) {
            return SIZE_MAX;
        }
        
        const uint8_t* const hs_base = reinterpret_cast<const uint8_t*>(static_cast<const void*>(str_));
        const uint8_t* const hs_used_end = hs_base + len_;
        
        // Defaults if begin/end omitted
        const uint8_t* search_begin = (begin == nullptr) 
            ? hs_base 
            : reinterpret_cast<const uint8_t*>(begin);
        
        const uint8_t* search_end = (end == nullptr) 
            ? hs_used_end 
            : reinterpret_cast<const uint8_t*>(end);
        
        // Validate begin and end are within buffer using existing is_ptr()
        if (!is_ptr(search_begin) || !is_ptr(search_end)) {
            return SIZE_MAX;
        }
        
        // Check monotonic ordering
        if (search_begin > search_end) {
            return SIZE_MAX;
        }
        
        // Clamp to used region
        if (search_begin > hs_used_end) {
            return SIZE_MAX;
        }
        if (search_end > hs_used_end) {
            search_end = hs_used_end;
        }
        
        size_t const region_len = static_cast<size_t>(search_end - search_begin);
        size_t const needle_len = std::strlen(needle);
        
        // Empty needle
        if (needle_len == 0u) {
            return 0u;
        }
        
        if (needle_len > region_len) {
            return SIZE_MAX;
        }
        
        // Delegate to SIMD/scalar implementation
        size_t offset_from_search_start = simd_find_substr_u8(
            search_begin, region_len,
            reinterpret_cast<const uint8_t*>(static_cast<const void*>(needle)), needle_len,
            dir
        );
        
        // Convert to offset from string start
        if (offset_from_search_start == SIZE_MAX) {
            return SIZE_MAX;
        }
        
        return (search_begin - hs_base) + offset_from_search_start;
    }
// -------------------------------------------------------------------------------- 

    size_t String::words(const String& word,
                         const void*   begin,
                         const void*   end) const noexcept {
        if (!str_ || !word.str_) return 0u;
        if (word.len_ == 0u)     return 0u;

        const uint8_t* const base = reinterpret_cast<const uint8_t*>(
            static_cast<const void*>(str_));

        const uint8_t* cur = (begin != nullptr)
            ? reinterpret_cast<const uint8_t*>(begin)
            : base;

        const uint8_t* const win_end = (end != nullptr)
            ? reinterpret_cast<const uint8_t*>(end)
            : nullptr;

        size_t count = 0u;
        for (;;) {
            size_t pos = find(word, cur, win_end, FORWARD);
            if (pos == SIZE_MAX) break;
            ++count;
            /* Advance past this match (non-overlapping) */
            cur = base + pos + word.len_;
            if ((win_end != nullptr) && (cur >= win_end)) break;
        }
        return count;
    }
// --------------------------------------------------------------------------------

    size_t String::words(const char* word,
                         const void* begin,
                         const void* end) const noexcept {
        if (!str_ || !word) return 0u;
        const size_t wlen = std::strlen(word);
        if (wlen == 0u) return 0u;

        const uint8_t* const base = reinterpret_cast<const uint8_t*>(
            static_cast<const void*>(str_));

        const uint8_t* cur = (begin != nullptr)
            ? reinterpret_cast<const uint8_t*>(begin)
            : base;

        const uint8_t* const win_end = (end != nullptr)
            ? reinterpret_cast<const uint8_t*>(end)
            : nullptr;

        size_t count = 0u;
        for (;;) {
            size_t pos = find(word, cur, win_end, FORWARD);
            if (pos == SIZE_MAX) break;
            ++count;
            cur = base + pos + wlen;
            if ((win_end != nullptr) && (cur >= win_end)) break;
        }
        return count;
    }
// -------------------------------------------------------------------------------- 

    size_t String::tokens(const String& delim,
                          const void*   begin,
                          const void*   end) const noexcept {
        if (!str_ || !delim.str_) return SIZE_MAX;

        const uint8_t* const base     = reinterpret_cast<const uint8_t*>(
                                            static_cast<const void*>(str_));
        const uint8_t* const used_end = base + len_;

        const uint8_t* search_begin = (begin == nullptr)
            ? base
            : reinterpret_cast<const uint8_t*>(begin);

        const uint8_t* search_end = (end == nullptr)
            ? used_end
            : reinterpret_cast<const uint8_t*>(end);

        // Validate pointers lie within the allocation
        if (!is_ptr(search_begin) || !is_ptr(search_end)) return SIZE_MAX;

        // Monotonic ordering required
        if (search_begin > search_end) return SIZE_MAX;

        // Clamp to used region
        if (search_begin > used_end) return SIZE_MAX;
        if (search_end   > used_end) search_end = used_end;

        // Empty window
        if (search_begin >= search_end) return 0u;

        size_t const n    = static_cast<size_t>(search_end - search_begin);
        size_t const dlen = delim.len_;

        // No delimiters — entire window is one token
        if (dlen == 0u) return 1u;

        return simd_token_count_u8(search_begin, n, delim.str_, dlen);
    }
// --------------------------------------------------------------------------------

    size_t String::tokens(const char* delim,
                          const void* begin,
                          const void* end) const noexcept {
        if (!str_ || !delim) return SIZE_MAX;

        const uint8_t* const base     = reinterpret_cast<const uint8_t*>(
                                            static_cast<const void*>(str_));
        const uint8_t* const used_end = base + len_;

        const uint8_t* search_begin = (begin == nullptr)
            ? base
            : reinterpret_cast<const uint8_t*>(begin);

        const uint8_t* search_end = (end == nullptr)
            ? used_end
            : reinterpret_cast<const uint8_t*>(end);

        // Validate pointers lie within the allocation
        if (!is_ptr(search_begin) || !is_ptr(search_end)) return SIZE_MAX;

        // Monotonic ordering required
        if (search_begin > search_end) return SIZE_MAX;

        // Clamp to used region
        if (search_begin > used_end) return SIZE_MAX;
        if (search_end   > used_end) search_end = used_end;

        // Empty window
        if (search_begin >= search_end) return 0u;

        size_t const n    = static_cast<size_t>(search_end - search_begin);
        size_t const dlen = std::strlen(delim);

        // No delimiters — entire window is one token
        if (dlen == 0u) return 1u;

        return simd_token_count_u8(search_begin, n, delim, dlen);
    }
// ================================================================================
// ================================================================================
// eof
// ================================================================================ 
// ================================================================================ 


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
