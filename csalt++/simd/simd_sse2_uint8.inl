
#ifndef CSALT_SIMD_SSE2_UINT8_INL
#define CSALT_SIMD_SSE2_UINT8_INL

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <emmintrin.h>   /* SSE2 */
// ================================================================================ 
// ================================================================================ 

static inline __m128i _sse2_reverse_bytes(__m128i v) {
    /* Step 1: reverse the 8 x uint16_t words within the register.
       _MM_SHUFFLE(0,1,2,3) reverses 4 x uint32_t lanes; we then fix
       the byte order within each lane in step 2. */
    v = _mm_shufflelo_epi16(v, _MM_SHUFFLE(0, 1, 2, 3));
    v = _mm_shufflehi_epi16(v, _MM_SHUFFLE(0, 1, 2, 3));
    v = _mm_shuffle_epi32(v, _MM_SHUFFLE(1, 0, 3, 2));

    /* Step 2: byte-swap each 16-bit word.
       ((v >> 8) & 0x00FF00FF00FF00FF) | ((v << 8) & 0xFF00FF00FF00FF00) */
    __m128i lo  = _mm_srli_epi16(v, 8);
    __m128i hi  = _mm_slli_epi16(v, 8);
    return _mm_or_si128(lo, hi);
}

// ================================================================================
// Public interface
// ================================================================================

static void simd_reverse_uint8(uint8_t* data, size_t len, size_t data_size) {
    if (data == NULL || len < 2u || data_size == 0u) return;

    /* SIMD fast path: only for element sizes that evenly divide 16 */
    if (data_size <= 16u && (16u % data_size == 0u)) {
        size_t lo = 0u;
        size_t hi = len - 1u;

        while (lo < hi) {
            uint8_t* lo_ptr = data + lo * data_size;
            uint8_t* hi_ptr = data + hi * data_size;

            /* How many elements fit in one 16-byte register? */
            size_t elems_per_reg = 16u / data_size;

            /* Only use SIMD when there is a full register's worth to swap
               on both sides; otherwise fall to scalar for the remainder. */
            if ((hi - lo + 1u) >= 2u * elems_per_reg) {
                __m128i vlo = _mm_loadu_si128((__m128i*)lo_ptr);
                __m128i vhi = _mm_loadu_si128((__m128i*)hi_ptr);

                vlo = _sse2_reverse_bytes(vlo);
                vhi = _sse2_reverse_bytes(vhi);

                _mm_storeu_si128((__m128i*)hi_ptr, vlo);
                _mm_storeu_si128((__m128i*)lo_ptr, vhi);

                lo += elems_per_reg;
                hi -= elems_per_reg;
                continue;
            }

            /* Scalar swap for the remaining elements */
            uint8_t tmp[16];
            memcpy(tmp,     lo_ptr, data_size);
            memcpy(lo_ptr,  hi_ptr, data_size);
            memcpy(hi_ptr,  tmp,    data_size);
            lo++;
            hi--;
        }
        return;
    }

    /* Scalar fallback for non-power-of-two or large element sizes */
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
// Internal: broadcast the first data_size bytes of needle into a 128-bit register.
// Repeats the element pattern across all 16 bytes.
// ================================================================================

static inline __m128i _sse2_broadcast(const uint8_t* needle, size_t data_size) {
    uint8_t buf[16] = {0};
    for (size_t i = 0; i < 16; i++) {
        buf[i] = needle[i % data_size];
    }
    return _mm_loadu_si128((__m128i*)buf);
}

// ================================================================================
// Internal: given a movemask result and element size, return the index of the
// first element whose data_size bytes all match, or SIZE_MAX if none.
// elem_base is the index of the first element in this 16-byte chunk.
// ================================================================================

static inline size_t _sse2_first_match(int mask,
                                        size_t data_size,
                                        size_t elem_base,
                                        size_t end) {
    /* Build the expected per-element mask: data_size consecutive set bits */
    int elem_mask = (data_size == 16) ? -1 : (1 << (int)data_size) - 1;
    size_t elems_per_reg = 16u / data_size;

    for (size_t e = 0; e < elems_per_reg; e++) {
        int shift    = (int)(e * data_size);
        int got      = (mask >> shift) & elem_mask;
        size_t idx   = elem_base + e;
        if (idx >= end) break;
        if (got == elem_mask) return idx;
    }
    return SIZE_MAX;
}
// -------------------------------------------------------------------------------- 

static size_t simd_contains_uint8(const uint8_t* data,
                                   size_t         start,
                                   size_t         end,
                                   size_t         data_size,
                                   const uint8_t* needle) {
    if (data_size <= 16u && (16u % data_size == 0u)) {
        size_t elems_per_reg = 16u / data_size;
        __m128i vneedle      = _sse2_broadcast(needle, data_size);

        size_t i = start;

        /* Align to element-reg boundary then process full registers */
        while (i + elems_per_reg <= end) {
            __m128i chunk = _mm_loadu_si128(
                (__m128i*)(data + i * data_size)
            );
            int mask = _mm_movemask_epi8(_mm_cmpeq_epi8(chunk, vneedle));
            if (mask != 0) {
                size_t found = _sse2_first_match(mask, data_size, i, end);
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
#endif /* CSALT_SIMD_SSE2_UINT8_INL */

