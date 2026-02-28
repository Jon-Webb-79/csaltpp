
#ifndef CSALT_SIMD_SSE41_UINT8_INL
#define CSALT_SIMD_SSE41_UINT8_INL

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <smmintrin.h>
// ================================================================================ 
// ================================================================================ 

// ================================================================================
// Internal: reverse bytes within a 128-bit register using SSE2 only
//
// Strategy: treat the 16 bytes as 8 x uint16_t, shuffle the 16-bit words
// into reverse order, then byte-swap each 16-bit word.
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
// Public interface
// ================================================================================

static size_t simd_contains_uint8(const uint8_t* data,
                                   size_t         start,
                                   size_t         end,
                                   size_t         data_size,
                                   const uint8_t* needle) {
    size_t i = start;

    if (data_size == 1u) {
        /* Broadcast single byte across all 16 lanes */
        __m128i vneedle = _mm_set1_epi8((char)*needle);
        while (i + 16u <= end) {
            __m128i chunk = _mm_loadu_si128((__m128i*)(data + i));
            int     mask  = _mm_movemask_epi8(_mm_cmpeq_epi8(chunk, vneedle));
            if (mask != 0) {
                /* __builtin_ctz gives position of lowest set bit */
                return i + (size_t)__builtin_ctz((unsigned)mask);
            }
            i += 16u;
        }
    } else if (data_size == 2u) {
        __m128i vneedle = _mm_set1_epi16(*(const int16_t*)needle);
        while (i + 8u <= end) {
            __m128i chunk = _mm_loadu_si128((__m128i*)(data + i * 2u));
            /* cmpeq_epi16 sets each 16-bit lane to 0xFFFF on match */
            __m128i cmp   = _mm_cmpeq_epi16(chunk, vneedle);
            int     mask  = _mm_movemask_epi8(cmp);
            if (mask != 0) {
                /* Two bits set per matching element; find first pair */
                for (size_t e = 0; e < 8u && i + e < end; e++) {
                    if (((mask >> (int)(e * 2u)) & 0x3) == 0x3) return i + e;
                }
            }
            i += 8u;
        }
    } else if (data_size == 4u) {
        __m128i vneedle = _mm_set1_epi32(*(const int32_t*)needle);
        while (i + 4u <= end) {
            __m128i chunk = _mm_loadu_si128((__m128i*)(data + i * 4u));
            __m128i cmp   = _mm_cmpeq_epi32(chunk, vneedle);
            int     mask  = _mm_movemask_epi8(cmp);
            if (mask != 0) {
                /* Four bits per element */
                for (size_t e = 0; e < 4u && i + e < end; e++) {
                    if (((mask >> (int)(e * 4u)) & 0xF) == 0xF) return i + e;
                }
            }
            i += 4u;
        }
    } else if (data_size == 8u) {
        /* _mm_cmpeq_epi64 is SSE4.1 */
        __m128i vneedle = _mm_set1_epi64x(*(const int64_t*)needle);
        while (i + 2u <= end) {
            __m128i chunk = _mm_loadu_si128((__m128i*)(data + i * 8u));
            __m128i cmp   = _mm_cmpeq_epi64(chunk, vneedle);
            int     mask  = _mm_movemask_epi8(cmp);
            if (mask != 0) {
                if ((mask & 0x00FF) == 0x00FF && i     < end) return i;
                if ((mask & 0xFF00) == 0xFF00 && i + 1 < end) return i + 1u;
            }
            i += 2u;
        }
    }

    /* Scalar remainder for all sizes */
    for (; i < end; i++) {
        if (memcmp(data + i * data_size, needle, data_size) == 0) {
            return i;
        }
    }
    return SIZE_MAX;
}
// ================================================================================ 
// ================================================================================ 
#endif /* CSALT_SIMD_SSE41_UINT8_INL */

