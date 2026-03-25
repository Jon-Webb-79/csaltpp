
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

static inline __m128i _sse41_reverse_bytes(__m128i v) {
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

    if (data_size < 16u && (16u % data_size == 0u)) {
        size_t lo = 0u;
        size_t hi = len - 1u;
        size_t elems_per_reg = 16u / data_size;

        while (lo < hi) {
            uint8_t* lo_ptr = data + lo * data_size;
            uint8_t* hi_ptr = data + hi * data_size;

            if ((hi - lo + 1u) >= 2u * elems_per_reg) {
                __m128i vlo = _mm_loadu_si128((__m128i*)lo_ptr);
                __m128i vhi = _mm_loadu_si128((__m128i*)hi_ptr);

                vlo = _sse41_reverse_bytes(vlo);
                vhi = _sse41_reverse_bytes(vhi);

                _mm_storeu_si128((__m128i*)hi_ptr, vlo);
                _mm_storeu_si128((__m128i*)lo_ptr, vhi);

                lo += elems_per_reg;
                hi -= elems_per_reg;
                continue;
            }

            uint8_t tmp[16];
            memcpy(tmp,    lo_ptr, data_size);
            memcpy(lo_ptr, hi_ptr, data_size);
            memcpy(hi_ptr, tmp,    data_size);
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
// -------------------------------------------------------------------------------- 

static inline uint8_t _sse41_hmin_u8(__m128i v) {
    __m128i lo8  = _mm_and_si128(v, _mm_set1_epi16(0x00FF));
    __m128i hi8  = _mm_srli_epi16(v, 8);
    __m128i min8 = _mm_min_epu8(lo8, hi8);
    __m128i pos  = _mm_minpos_epu16(min8);
    return (uint8_t)_mm_cvtsi128_si32(pos);
}

static inline uint8_t _sse41_hmax_u8(__m128i v) {
    __m128i inv = _mm_xor_si128(v, _mm_set1_epi8((char)0xFF));
    return (uint8_t)(0xFF ^ _sse41_hmin_u8(inv));
}

static size_t simd_min_uint8(const uint8_t* data,
                              size_t         len,
                              size_t         data_size,
                              int          (*cmp)(const void*, const void*)) {
    if (data_size == 1u) {
        __m128i vmin = _mm_set1_epi8((char)0xFF);
        size_t  i    = 0u;
        while (i + 16u <= len) {
            vmin = _mm_min_epu8(vmin, _mm_loadu_si128((__m128i*)(data + i)));
            i += 16u;
        }
        for (; i < len; i++) {
            __m128i v = _mm_set1_epi8((char)data[i]);
            vmin = _mm_min_epu8(vmin, v);
        }
        uint8_t min_val = _sse41_hmin_u8(vmin);
        for (size_t j = 0u; j < len; j++)
            if (data[j] == min_val) return j;
    }
    size_t best = 0u;
    for (size_t i = 1u; i < len; i++)
        if (cmp(data + i * data_size, data + best * data_size) < 0) best = i;
    return best;
}
// -------------------------------------------------------------------------------- 

static size_t simd_max_uint8(const uint8_t* data,
                              size_t         len,
                              size_t         data_size,
                              int          (*cmp)(const void*, const void*)) {
    if (data_size == 1u) {
        __m128i vmax = _mm_setzero_si128();
        size_t  i    = 0u;
        while (i + 16u <= len) {
            vmax = _mm_max_epu8(vmax, _mm_loadu_si128((__m128i*)(data + i)));
            i += 16u;
        }
        for (; i < len; i++) {
            __m128i v = _mm_set1_epi8((char)data[i]);
            vmax = _mm_max_epu8(vmax, v);
        }
        uint8_t max_val = _sse41_hmax_u8(vmax);
        for (size_t j = 0u; j < len; j++)
            if (data[j] == max_val) return j;
    }
    size_t best = 0u;
    for (size_t i = 1u; i < len; i++)
        if (cmp(data + i * data_size, data + best * data_size) > 0) best = i;
    return best;
}
// -------------------------------------------------------------------------------- 

static void simd_sum_uint8(const uint8_t* data,
                            size_t         len,
                            size_t         data_size,
                            void*          accum,
                            void         (*add)(void* accum, const void* element)) {
    size_t i = 0u;
    if (data_size == 1u) {
        while (i + 16u <= len) {
            __m128i chunk = _mm_loadu_si128((__m128i*)(data + i));
            uint8_t v;
            v=(uint8_t)_mm_extract_epi8(chunk, 0);add(accum,&v);
            v=(uint8_t)_mm_extract_epi8(chunk, 1);add(accum,&v);
            v=(uint8_t)_mm_extract_epi8(chunk, 2);add(accum,&v);
            v=(uint8_t)_mm_extract_epi8(chunk, 3);add(accum,&v);
            v=(uint8_t)_mm_extract_epi8(chunk, 4);add(accum,&v);
            v=(uint8_t)_mm_extract_epi8(chunk, 5);add(accum,&v);
            v=(uint8_t)_mm_extract_epi8(chunk, 6);add(accum,&v);
            v=(uint8_t)_mm_extract_epi8(chunk, 7);add(accum,&v);
            v=(uint8_t)_mm_extract_epi8(chunk, 8);add(accum,&v);
            v=(uint8_t)_mm_extract_epi8(chunk, 9);add(accum,&v);
            v=(uint8_t)_mm_extract_epi8(chunk,10);add(accum,&v);
            v=(uint8_t)_mm_extract_epi8(chunk,11);add(accum,&v);
            v=(uint8_t)_mm_extract_epi8(chunk,12);add(accum,&v);
            v=(uint8_t)_mm_extract_epi8(chunk,13);add(accum,&v);
            v=(uint8_t)_mm_extract_epi8(chunk,14);add(accum,&v);
            v=(uint8_t)_mm_extract_epi8(chunk,15);add(accum,&v);
            i += 16u;
        }
    } else if (data_size == 2u) {
        while (i + 8u <= len) {
            __m128i chunk = _mm_loadu_si128((__m128i*)(data + i * 2u));
            uint16_t v;
            v=(uint16_t)_mm_extract_epi16(chunk,0);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(chunk,1);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(chunk,2);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(chunk,3);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(chunk,4);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(chunk,5);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(chunk,6);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(chunk,7);add(accum,&v);
            i += 8u;
        }
    } else if (data_size == 4u) {
        while (i + 4u <= len) {
            __m128i chunk = _mm_loadu_si128((__m128i*)(data + i * 4u));
            uint32_t v;
            v=(uint32_t)_mm_extract_epi32(chunk,0);add(accum,&v);
            v=(uint32_t)_mm_extract_epi32(chunk,1);add(accum,&v);
            v=(uint32_t)_mm_extract_epi32(chunk,2);add(accum,&v);
            v=(uint32_t)_mm_extract_epi32(chunk,3);add(accum,&v);
            i += 4u;
        }
    } else if (data_size == 8u) {
        while (i + 2u <= len) {
            __m128i chunk = _mm_loadu_si128((__m128i*)(data + i * 8u));
            uint64_t v;
            v=(uint64_t)_mm_extract_epi64(chunk,0);add(accum,&v);
            v=(uint64_t)_mm_extract_epi64(chunk,1);add(accum,&v);
            i += 2u;
        }
    }
    for (; i < len; i++) add(accum, data + i * data_size);
}
// ================================================================================ 
// ================================================================================ 
#endif /* CSALT_SIMD_SSE41_UINT8_INL */

