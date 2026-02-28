
#ifndef CSALT_SIMD_SSE3_UINT8_INL
#define CSALT_SIMD_SSE3_UINT8_INL

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <tmmintrin.h>   /* SSSE3 */
// ================================================================================ 
// ================================================================================ 

// ================================================================================
// Internal: reverse bytes within a 128-bit register using pshufb
// ================================================================================

static inline __m128i _ssse3_reverse_bytes(__m128i v) {
    /* Shuffle mask: byte 15 goes to position 0, 14 to 1, ..., 0 to 15 */
    const __m128i mask = _mm_set_epi8(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );
    return _mm_shuffle_epi8(v, mask);
}

// ================================================================================
// Public interface
// ================================================================================

static void simd_reverse_uint8(uint8_t* data, size_t len, size_t data_size) {
    if (data == NULL || len < 2u || data_size == 0u) return;

    if (data_size <= 16u && (16u % data_size == 0u)) {
        size_t lo = 0u;
        size_t hi = len - 1u;
        size_t elems_per_reg = 16u / data_size;

        while (lo + elems_per_reg <= hi + 1u) {
            uint8_t* lo_ptr = data + lo * data_size;
            uint8_t* hi_ptr = data + hi * data_size;

            if ((hi - lo + 1u) >= 2u * elems_per_reg) {
                __m128i vlo = _mm_loadu_si128((__m128i*)lo_ptr);
                __m128i vhi = _mm_loadu_si128((__m128i*)hi_ptr);

                vlo = _ssse3_reverse_bytes(vlo);
                vhi = _ssse3_reverse_bytes(vhi);

                _mm_storeu_si128((__m128i*)hi_ptr, vlo);
                _mm_storeu_si128((__m128i*)lo_ptr, vhi);

                lo += elems_per_reg;
                hi -= elems_per_reg;
                continue;
            }

            /* Scalar remainder */
            uint8_t tmp[16];
            memcpy(tmp,     lo_ptr, data_size);
            memcpy(lo_ptr,  hi_ptr, data_size);
            memcpy(hi_ptr,  tmp,    data_size);
            lo++;
            hi--;
        }
        return;
    }

    /* Scalar fallback */
    size_t lo = 0u;
    size_t hi = len - 1u;
    uint8_t tmp[256];

    if (data_size <= sizeof(tmp)) {
        while (lo < hi) {
            uint8_t* lo_ptr = data + lo * data_size;
            uint8_t* hi_ptr = data + hi * data_size;
            memcpy(tmp,    lo_ptr, data_size);
            memcpy(lo_ptr, hi_ptr, data_size);
            memcpy(hi_ptr, tmp,    data_size);
            lo++; hi--;
        }
    } else {
        while (lo < hi) {
            uint8_t* lo_ptr = data + lo * data_size;
            uint8_t* hi_ptr = data + hi * data_size;
            for (size_t b = 0u; b < data_size; b++) {
                uint8_t byte = lo_ptr[b];
                lo_ptr[b]    = hi_ptr[b];
                hi_ptr[b]    = byte;
            }
            lo++; hi--;
        }
    }
}
// -------------------------------------------------------------------------------- 

// ================================================================================
// Internal: broadcast needle into a 128-bit register using pshufb
// ================================================================================

static inline __m128i _ssse3_broadcast(const uint8_t* needle, size_t data_size) {
    /* Load needle into low bytes, then shuffle to repeat across all 16 */
    uint8_t buf[16] = {0};
    for (size_t i = 0; i < 16; i++) {
        buf[i] = needle[i % data_size];
    }
    return _mm_loadu_si128((__m128i*)buf);
}

// ================================================================================
// Internal: first element match from movemask result
// ================================================================================

static inline size_t _ssse3_first_match(int mask,
                                         size_t data_size,
                                         size_t elem_base,
                                         size_t end) {
    int    elem_mask     = (data_size == 16) ? -1 : (1 << (int)data_size) - 1;
    size_t elems_per_reg = 16u / data_size;

    for (size_t e = 0; e < elems_per_reg; e++) {
        int    shift = (int)(e * data_size);
        int    got   = (mask >> shift) & elem_mask;
        size_t idx   = elem_base + e;
        if (idx >= end) break;
        if (got == elem_mask) return idx;
    }
    return SIZE_MAX;
}

// ================================================================================
// Public interface
// ================================================================================

static size_t simd_contains_uint8(const uint8_t* data,
                                   size_t         start,
                                   size_t         end,
                                   size_t         data_size,
                                   const uint8_t* needle) {
    if (data_size <= 16u && (16u % data_size == 0u)) {
        size_t  elems_per_reg = 16u / data_size;
        __m128i vneedle       = _ssse3_broadcast(needle, data_size);
        size_t  i             = start;

        while (i + elems_per_reg <= end) {
            __m128i chunk = _mm_loadu_si128(
                (__m128i*)(data + i * data_size)
            );
            int mask = _mm_movemask_epi8(_mm_cmpeq_epi8(chunk, vneedle));
            if (mask != 0) {
                size_t found = _ssse3_first_match(mask, data_size, i, end);
                if (found != SIZE_MAX) return found;
            }
            i += elems_per_reg;
        }

        /* Scalar remainder */
        for (; i < end; i++) {
            if (memcmp(data + i * data_size, needle, data_size) == 0) {
                return i;
            }
        }
        return SIZE_MAX;
    }

    /* Scalar fallback */
    for (size_t i = start; i < end; i++) {
        if (memcmp(data + i * data_size, needle, data_size) == 0) {
            return i;
        }
    }
    return SIZE_MAX;
}
// ================================================================================ 
// ================================================================================ 
#endif /* CSALT_SIMD_SSE3_UINT8_INL */

