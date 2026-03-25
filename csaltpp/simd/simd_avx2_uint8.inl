
#ifndef CSALT_SIMD_AVX2_UINT8_INL
#define CSALT_SIMD_AVX2_UINT8_INL

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <immintrin.h>   /* AVX2 */
// ================================================================================ 
// ================================================================================ 

// ================================================================================
// Internal: reverse bytes within a 256-bit register
//
// vpshufb works per 128-bit lane independently, so:
//   1. Reverse bytes within each lane using vpshufb.
//   2. Swap the two lanes using vperm2i128.
// ================================================================================

static inline __m256i _avx2_reverse_bytes(__m256i v) {
    /* Per-lane byte-reversal mask (same mask applied to both 128-bit lanes) */
    const __m256i mask = _mm256_set_epi8(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );

    /* Reverse bytes within each lane */
    v = _mm256_shuffle_epi8(v, mask);

    /* Swap the two 128-bit lanes: 0x01 means take lane 1 as low, lane 0 as high */
    return _mm256_permute2x128_si256(v, v, 0x01);
}

// ================================================================================
// Public interface
// ================================================================================

static void simd_reverse_uint8(uint8_t* data, size_t len, size_t data_size) {
    if (data == NULL || len < 2u || data_size == 0u) return;

    if (data_size < 32u && (32u % data_size == 0u)) {
        size_t lo = 0u;
        size_t hi = len - 1u;
        size_t elems_per_reg = 32u / data_size;

        while (lo < hi) {
            uint8_t* lo_ptr = data + lo * data_size;
            uint8_t* hi_ptr = data + hi * data_size;

            if ((hi - lo + 1u) >= 2u * elems_per_reg) {
                __m256i vlo = _mm256_loadu_si256((__m256i*)lo_ptr);
                __m256i vhi = _mm256_loadu_si256((__m256i*)hi_ptr);

                vlo = _avx2_reverse_bytes(vlo);
                vhi = _avx2_reverse_bytes(vhi);

                _mm256_storeu_si256((__m256i*)hi_ptr, vlo);
                _mm256_storeu_si256((__m256i*)lo_ptr, vhi);

                lo += elems_per_reg;
                hi -= elems_per_reg;
                continue;
            }

            /* Scalar remainder */
            uint8_t tmp[32];
            memcpy(tmp,    lo_ptr, data_size);
            memcpy(lo_ptr, hi_ptr, data_size);
            memcpy(hi_ptr, tmp,    data_size);
            lo++;
            hi--;
        }

        _mm256_zeroupper();
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
    _mm256_zeroupper();
}
// -------------------------------------------------------------------------------- 

static size_t simd_contains_uint8(const uint8_t* data,
                                   size_t         start,
                                   size_t         end,
                                   size_t         data_size,
                                   const uint8_t* needle) {
    size_t i = start;

    if (data_size == 1u) {
        __m256i vn = _mm256_set1_epi8((char)*needle);
        while (i + 32u <= end) {
            __m256i chunk = _mm256_loadu_si256((__m256i*)(data + i));
            int     mask  = _mm256_movemask_epi8(_mm256_cmpeq_epi8(chunk, vn));
            if (mask != 0) {
                _mm256_zeroupper();
                return i + (size_t)__builtin_ctz((unsigned)mask);
            }
            i += 32u;
        }
    } else if (data_size == 2u) {
        __m256i vn = _mm256_set1_epi16(*(const int16_t*)needle);
        while (i + 16u <= end) {
            __m256i chunk = _mm256_loadu_si256((__m256i*)(data + i * 2u));
            __m256i cmp   = _mm256_cmpeq_epi16(chunk, vn);
            int     mask  = _mm256_movemask_epi8(cmp);
            if (mask != 0) {
                for (size_t e = 0; e < 16u && i + e < end; e++) {
                    if (((mask >> (int)(e * 2u)) & 0x3) == 0x3) {
                        _mm256_zeroupper();
                        return i + e;
                    }
                }
            }
            i += 16u;
        }
    } else if (data_size == 4u) {
        __m256i vn = _mm256_set1_epi32(*(const int32_t*)needle);
        while (i + 8u <= end) {
            __m256i chunk = _mm256_loadu_si256((__m256i*)(data + i * 4u));
            __m256i cmp   = _mm256_cmpeq_epi32(chunk, vn);
            int     mask  = _mm256_movemask_epi8(cmp);
            if (mask != 0) {
                for (size_t e = 0; e < 8u && i + e < end; e++) {
                    if (((mask >> (int)(e * 4u)) & 0xF) == 0xF) {
                        _mm256_zeroupper();
                        return i + e;
                    }
                }
            }
            i += 8u;
        }
    } else if (data_size == 8u) {
        __m256i vn = _mm256_set1_epi64x(*(const int64_t*)needle);
        while (i + 4u <= end) {
            __m256i chunk = _mm256_loadu_si256((__m256i*)(data + i * 8u));
            __m256i cmp   = _mm256_cmpeq_epi64(chunk, vn);
            int     mask  = _mm256_movemask_epi8(cmp);
            if (mask != 0) {
                for (size_t e = 0; e < 4u && i + e < end; e++) {
                    if (((mask >> (int)(e * 8u)) & 0xFF) == 0xFF) {
                        _mm256_zeroupper();
                        return i + e;
                    }
                }
            }
            i += 4u;
        }
    }

    _mm256_zeroupper();

    /* Scalar remainder for all sizes */
    for (; i < end; i++) {
        if (memcmp(data + i * data_size, needle, data_size) == 0) {
            return i;
        }
    }
    return SIZE_MAX;
}
// -------------------------------------------------------------------------------- 

static inline uint8_t _avx2_hmin_u8(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i m  = _mm_min_epu8(lo, hi);
    m = _mm_min_epu8(m, _mm_srli_si128(m, 8));
    m = _mm_min_epu8(m, _mm_srli_si128(m, 4));
    m = _mm_min_epu8(m, _mm_srli_si128(m, 2));
    m = _mm_min_epu8(m, _mm_srli_si128(m, 1));
    return (uint8_t)_mm_cvtsi128_si32(m);
}

static inline uint8_t _avx2_hmax_u8(__m256i v) {
    __m128i lo = _mm256_castsi256_si128(v);
    __m128i hi = _mm256_extracti128_si256(v, 1);
    __m128i m  = _mm_max_epu8(lo, hi);
    m = _mm_max_epu8(m, _mm_srli_si128(m, 8));
    m = _mm_max_epu8(m, _mm_srli_si128(m, 4));
    m = _mm_max_epu8(m, _mm_srli_si128(m, 2));
    m = _mm_max_epu8(m, _mm_srli_si128(m, 1));
    return (uint8_t)_mm_cvtsi128_si32(m);
}

static size_t simd_min_uint8(const uint8_t* data,
                              size_t         len,
                              size_t         data_size,
                              int          (*cmp)(const void*, const void*)) {
    if (data_size == 1u) {
        __m256i vmin = _mm256_set1_epi8((char)0xFF);
        size_t  i    = 0u;
        while (i + 32u <= len) {
            vmin = _mm256_min_epu8(vmin,
                       _mm256_loadu_si256((__m256i*)(data + i)));
            i += 32u;
        }
        if (i + 16u <= len) {
            __m128i tail = _mm_loadu_si128((__m128i*)(data + i));
            __m128i lo   = _mm256_castsi256_si128(vmin);
            lo   = _mm_min_epu8(lo, tail);
            vmin = _mm256_inserti128_si256(
                       _mm256_castsi128_si256(lo),
                       _mm256_extracti128_si256(vmin, 1), 1);
            i += 16u;
        }
        for (; i < len; i++) {
            __m256i v = _mm256_set1_epi8((char)data[i]);
            vmin = _mm256_min_epu8(vmin, v);
        }
        uint8_t min_val = _avx2_hmin_u8(vmin);
        _mm256_zeroupper();
        for (size_t j = 0u; j < len; j++)
            if (data[j] == min_val) return j;
    }
    _mm256_zeroupper();
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
        __m256i vmax = _mm256_setzero_si256();
        size_t  i    = 0u;
        while (i + 32u <= len) {
            vmax = _mm256_max_epu8(vmax,
                       _mm256_loadu_si256((__m256i*)(data + i)));
            i += 32u;
        }
        if (i + 16u <= len) {
            __m128i tail = _mm_loadu_si128((__m128i*)(data + i));
            __m128i lo   = _mm256_castsi256_si128(vmax);
            lo   = _mm_max_epu8(lo, tail);
            vmax = _mm256_inserti128_si256(
                       _mm256_castsi128_si256(lo),
                       _mm256_extracti128_si256(vmax, 1), 1);
            i += 16u;
        }
        for (; i < len; i++) {
            __m256i v = _mm256_set1_epi8((char)data[i]);
            vmax = _mm256_max_epu8(vmax, v);
        }
        uint8_t max_val = _avx2_hmax_u8(vmax);
        _mm256_zeroupper();
        for (size_t j = 0u; j < len; j++)
            if (data[j] == max_val) return j;
    }
    _mm256_zeroupper();
    size_t best = 0u;
    for (size_t i = 1u; i < len; i++)
        if (cmp(data + i * data_size, data + best * data_size) > 0) best = i;
    return best;
}
// -------------------------------------------------------------------------------- 

static inline void _avx2_drain_u8(__m128i lane, void* accum,
                                   void (*add)(void*, const void*)) {
    uint8_t v;
    v=(uint8_t)_mm_extract_epi8(lane, 0);add(accum,&v);
    v=(uint8_t)_mm_extract_epi8(lane, 1);add(accum,&v);
    v=(uint8_t)_mm_extract_epi8(lane, 2);add(accum,&v);
    v=(uint8_t)_mm_extract_epi8(lane, 3);add(accum,&v);
    v=(uint8_t)_mm_extract_epi8(lane, 4);add(accum,&v);
    v=(uint8_t)_mm_extract_epi8(lane, 5);add(accum,&v);
    v=(uint8_t)_mm_extract_epi8(lane, 6);add(accum,&v);
    v=(uint8_t)_mm_extract_epi8(lane, 7);add(accum,&v);
    v=(uint8_t)_mm_extract_epi8(lane, 8);add(accum,&v);
    v=(uint8_t)_mm_extract_epi8(lane, 9);add(accum,&v);
    v=(uint8_t)_mm_extract_epi8(lane,10);add(accum,&v);
    v=(uint8_t)_mm_extract_epi8(lane,11);add(accum,&v);
    v=(uint8_t)_mm_extract_epi8(lane,12);add(accum,&v);
    v=(uint8_t)_mm_extract_epi8(lane,13);add(accum,&v);
    v=(uint8_t)_mm_extract_epi8(lane,14);add(accum,&v);
    v=(uint8_t)_mm_extract_epi8(lane,15);add(accum,&v);
}

static void simd_sum_uint8(const uint8_t* data,
                            size_t         len,
                            size_t         data_size,
                            void*          accum,
                            void         (*add)(void* accum, const void* element)) {
    size_t i = 0u;
    if (data_size == 1u) {
        while (i + 32u <= len) {
            __m256i chunk = _mm256_loadu_si256((__m256i*)(data + i));
            _avx2_drain_u8(_mm256_castsi256_si128(chunk),         accum, add);
            _avx2_drain_u8(_mm256_extracti128_si256(chunk, 1),    accum, add);
            i += 32u;
        }
    } else if (data_size == 2u) {
        while (i + 16u <= len) {
            __m256i chunk = _mm256_loadu_si256((__m256i*)(data + i * 2u));
            __m128i lo    = _mm256_castsi256_si128(chunk);
            __m128i hi    = _mm256_extracti128_si256(chunk, 1);
            uint16_t v;
            v=(uint16_t)_mm_extract_epi16(lo,0);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(lo,1);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(lo,2);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(lo,3);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(lo,4);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(lo,5);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(lo,6);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(lo,7);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(hi,0);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(hi,1);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(hi,2);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(hi,3);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(hi,4);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(hi,5);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(hi,6);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(hi,7);add(accum,&v);
            i += 16u;
        }
    } else if (data_size == 4u) {
        while (i + 8u <= len) {
            __m256i chunk = _mm256_loadu_si256((__m256i*)(data + i * 4u));
            __m128i lo    = _mm256_castsi256_si128(chunk);
            __m128i hi    = _mm256_extracti128_si256(chunk, 1);
            uint32_t v;
            v=(uint32_t)_mm_extract_epi32(lo,0);add(accum,&v);
            v=(uint32_t)_mm_extract_epi32(lo,1);add(accum,&v);
            v=(uint32_t)_mm_extract_epi32(lo,2);add(accum,&v);
            v=(uint32_t)_mm_extract_epi32(lo,3);add(accum,&v);
            v=(uint32_t)_mm_extract_epi32(hi,0);add(accum,&v);
            v=(uint32_t)_mm_extract_epi32(hi,1);add(accum,&v);
            v=(uint32_t)_mm_extract_epi32(hi,2);add(accum,&v);
            v=(uint32_t)_mm_extract_epi32(hi,3);add(accum,&v);
            i += 8u;
        }
    } else if (data_size == 8u) {
        while (i + 4u <= len) {
            __m256i chunk = _mm256_loadu_si256((__m256i*)(data + i * 8u));
            __m128i lo    = _mm256_castsi256_si128(chunk);
            __m128i hi    = _mm256_extracti128_si256(chunk, 1);
            uint64_t v;
            v=(uint64_t)_mm_extract_epi64(lo,0);add(accum,&v);
            v=(uint64_t)_mm_extract_epi64(lo,1);add(accum,&v);
            v=(uint64_t)_mm_extract_epi64(hi,0);add(accum,&v);
            v=(uint64_t)_mm_extract_epi64(hi,1);add(accum,&v);
            i += 4u;
        }
    }
    _mm256_zeroupper();
    for (; i < len; i++) add(accum, data + i * data_size);
}
// ================================================================================ 
// ================================================================================ 
#endif /* CSALT_SIMD_AVX2_UINT8_INL */

