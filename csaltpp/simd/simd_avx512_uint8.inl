
#ifndef CSALT_SIMD_AVX512_UINT8_INL
#define CSALT_SIMD_AVX512_UINT8_INL

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <immintrin.h>   /* AVX-512 */

// ================================================================================ 
// ================================================================================ 

static inline __m512i _avx512_reverse_bytes(__m512i v) {
#ifdef __AVX512VBMI__
    /* VBMI provides vpermb: a full 512-bit byte permutation in one instruction */
    const __m512i idx = _mm512_set_epi8(
         0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
    );
    return _mm512_permutexvar_epi8(idx, v);
#else
    /* AVX-512BW only: reverse each 128-bit lane then rotate the four lanes */
    const __m128i lane_mask = _mm_set_epi8(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );

    __m128i l0 = _mm512_extracti32x4_epi32(v, 0);
    __m128i l1 = _mm512_extracti32x4_epi32(v, 1);
    __m128i l2 = _mm512_extracti32x4_epi32(v, 2);
    __m128i l3 = _mm512_extracti32x4_epi32(v, 3);

    l0 = _mm_shuffle_epi8(l0, lane_mask);
    l1 = _mm_shuffle_epi8(l1, lane_mask);
    l2 = _mm_shuffle_epi8(l2, lane_mask);
    l3 = _mm_shuffle_epi8(l3, lane_mask);

    /* Reassemble: old lane 3 -> pos 0, 2 -> 1, 1 -> 2, 0 -> 3 */
    __m512i result = _mm512_castsi128_si512(l3);
    result = _mm512_inserti32x4(result, l2, 1);
    result = _mm512_inserti32x4(result, l1, 2);
    result = _mm512_inserti32x4(result, l0, 3);
    return result;
#endif
}

// ================================================================================
// Public interface
// ================================================================================

static void simd_reverse_uint8(uint8_t* data, size_t len, size_t data_size) {
    if (data == NULL || len < 2u || data_size == 0u) return;

    if (data_size < 64u && (64u % data_size == 0u)) {
        size_t lo = 0u;
        size_t hi = len - 1u;
        size_t elems_per_reg = 64u / data_size;

        while (lo < hi) {
            uint8_t* lo_ptr = data + lo * data_size;
            uint8_t* hi_ptr = data + hi * data_size;

            if ((hi - lo + 1u) >= 2u * elems_per_reg) {
                __m512i vlo = _mm512_loadu_si512((__m512i*)lo_ptr);
                __m512i vhi = _mm512_loadu_si512((__m512i*)hi_ptr);

                vlo = _avx512_reverse_bytes(vlo);
                vhi = _avx512_reverse_bytes(vhi);

                _mm512_storeu_si512((__m512i*)hi_ptr, vlo);
                _mm512_storeu_si512((__m512i*)lo_ptr, vhi);

                lo += elems_per_reg;
                hi -= elems_per_reg;
                continue;
            }

            /* Scalar remainder */
            uint8_t tmp[64];
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

static size_t simd_contains_uint8(const uint8_t* data,
                                   size_t         start,
                                   size_t         end,
                                   size_t         data_size,
                                   const uint8_t* needle) {
    size_t i = start;

    if (data_size == 1u) {
        __m512i vn = _mm512_set1_epi8((char)*needle);
        while (i + 64u <= end) {
            __m512i  chunk = _mm512_loadu_si512((__m512i*)(data + i));
            __mmask64 mask = _mm512_cmpeq_epi8_mask(chunk, vn);
            if (mask != 0) {
                return i + (size_t)__builtin_ctzll((unsigned long long)mask);
            }
            i += 64u;
        }
    } else if (data_size == 2u) {
        __m512i vn = _mm512_set1_epi16(*(const int16_t*)needle);
        while (i + 32u <= end) {
            __m512i  chunk = _mm512_loadu_si512((__m512i*)(data + i * 2u));
            __mmask32 mask = _mm512_cmpeq_epi16_mask(chunk, vn);
            if (mask != 0) {
                return i + (size_t)__builtin_ctz((unsigned)mask);
            }
            i += 32u;
        }
    } else if (data_size == 4u) {
        __m512i vn = _mm512_set1_epi32(*(const int32_t*)needle);
        while (i + 16u <= end) {
            __m512i  chunk = _mm512_loadu_si512((__m512i*)(data + i * 4u));
            __mmask16 mask = _mm512_cmpeq_epi32_mask(chunk, vn);
            if (mask != 0) {
                return i + (size_t)__builtin_ctz((unsigned)mask);
            }
            i += 16u;
        }
    } else if (data_size == 8u) {
        __m512i vn = _mm512_set1_epi64(*(const int64_t*)needle);
        while (i + 8u <= end) {
            __m512i chunk = _mm512_loadu_si512((__m512i*)(data + i * 8u));
            __mmask8 mask = _mm512_cmpeq_epi64_mask(chunk, vn);
            if (mask != 0) {
                return i + (size_t)__builtin_ctz((unsigned)mask);
            }
            i += 8u;
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

static inline uint8_t _avx512_hmin_u8(__m512i v) {
#ifdef __AVX512BW__
    return (uint8_t)_mm512_reduce_min_epu8(v);
#else
    __m128i l0 = _mm512_extracti32x4_epi32(v, 0);
    __m128i l1 = _mm512_extracti32x4_epi32(v, 1);
    __m128i l2 = _mm512_extracti32x4_epi32(v, 2);
    __m128i l3 = _mm512_extracti32x4_epi32(v, 3);
    __m128i m  = _mm_min_epu8(_mm_min_epu8(l0, l1), _mm_min_epu8(l2, l3));
    m = _mm_min_epu8(m, _mm_srli_si128(m, 8));
    m = _mm_min_epu8(m, _mm_srli_si128(m, 4));
    m = _mm_min_epu8(m, _mm_srli_si128(m, 2));
    m = _mm_min_epu8(m, _mm_srli_si128(m, 1));
    return (uint8_t)_mm_cvtsi128_si32(m);
#endif
}

static inline uint8_t _avx512_hmax_u8(__m512i v) {
#ifdef __AVX512BW__
    return (uint8_t)_mm512_reduce_max_epu8(v);
#else
    __m128i l0 = _mm512_extracti32x4_epi32(v, 0);
    __m128i l1 = _mm512_extracti32x4_epi32(v, 1);
    __m128i l2 = _mm512_extracti32x4_epi32(v, 2);
    __m128i l3 = _mm512_extracti32x4_epi32(v, 3);
    __m128i m  = _mm_max_epu8(_mm_max_epu8(l0, l1), _mm_max_epu8(l2, l3));
    m = _mm_max_epu8(m, _mm_srli_si128(m, 8));
    m = _mm_max_epu8(m, _mm_srli_si128(m, 4));
    m = _mm_max_epu8(m, _mm_srli_si128(m, 2));
    m = _mm_max_epu8(m, _mm_srli_si128(m, 1));
    return (uint8_t)_mm_cvtsi128_si32(m);
#endif
}

static size_t simd_min_uint8(const uint8_t* data,
                              size_t         len,
                              size_t         data_size,
                              int          (*cmp)(const void*, const void*)) {
    if (data_size == 1u) {
        __m512i vmin = _mm512_set1_epi8((char)0xFF);
        size_t  i    = 0u;
        while (i + 64u <= len) {
            vmin = _mm512_min_epu8(vmin,
                       _mm512_loadu_si512((__m512i*)(data + i)));
            i += 64u;
        }
        if (i + 32u <= len) {
            __mmask64 mask = ((__mmask64)1 << 32) - 1;
            __m512i t = _mm512_castsi256_si512(
                            _mm256_loadu_si256((__m256i*)(data + i)));
            vmin = _mm512_mask_min_epu8(vmin, mask, vmin, t);
            i += 32u;
        }
        if (i + 16u <= len) {
            __mmask64 mask = ((__mmask64)1 << 16) - 1;
            __m512i t = _mm512_castsi128_si512(
                            _mm_loadu_si128((__m128i*)(data + i)));
            vmin = _mm512_mask_min_epu8(vmin, mask, vmin, t);
            i += 16u;
        }
        for (; i < len; i++) {
            __mmask64 mask = (__mmask64)1;
            __m512i v = _mm512_set1_epi8((char)data[i]);
            vmin = _mm512_mask_min_epu8(vmin, mask, vmin, v);
        }
        uint8_t min_val = _avx512_hmin_u8(vmin);
        for (size_t j = 0u; j < len; j++)
            if (data[j] == min_val) return j;
    }
    size_t best = 0u;
    for (size_t i = 1u; i < len; i++)
        if (cmp(data + i * data_size, data + best * data_size) < 0) best = i;
    return best;
}

static size_t simd_max_uint8(const uint8_t* data,
                              size_t         len,
                              size_t         data_size,
                              int          (*cmp)(const void*, const void*)) {
    if (data_size == 1u) {
        __m512i vmax = _mm512_setzero_si512();
        size_t  i    = 0u;
        while (i + 64u <= len) {
            vmax = _mm512_max_epu8(vmax,
                       _mm512_loadu_si512((__m512i*)(data + i)));
            i += 64u;
        }
        if (i + 32u <= len) {
            __mmask64 mask = ((__mmask64)1 << 32) - 1;
            __m512i t = _mm512_castsi256_si512(
                            _mm256_loadu_si256((__m256i*)(data + i)));
            vmax = _mm512_mask_max_epu8(vmax, mask, vmax, t);
            i += 32u;
        }
        if (i + 16u <= len) {
            __mmask64 mask = ((__mmask64)1 << 16) - 1;
            __m512i t = _mm512_castsi128_si512(
                            _mm_loadu_si128((__m128i*)(data + i)));
            vmax = _mm512_mask_max_epu8(vmax, mask, vmax, t);
            i += 16u;
        }
        for (; i < len; i++) {
            __mmask64 mask = (__mmask64)1;
            __m512i v = _mm512_set1_epi8((char)data[i]);
            vmax = _mm512_mask_max_epu8(vmax, mask, vmax, v);
        }
        uint8_t max_val = _avx512_hmax_u8(vmax);
        for (size_t j = 0u; j < len; j++)
            if (data[j] == max_val) return j;
    }
    size_t best = 0u;
    for (size_t i = 1u; i < len; i++)
        if (cmp(data + i * data_size, data + best * data_size) > 0) best = i;
    return best;
}
// -------------------------------------------------------------------------------- 

static inline void _avx512_drain_u8(__m128i lane, void* accum,
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
        while (i + 64u <= len) {
            __m512i chunk = _mm512_loadu_si512((__m512i*)(data + i));
            _avx512_drain_u8(_mm512_extracti32x4_epi32(chunk, 0), accum, add);
            _avx512_drain_u8(_mm512_extracti32x4_epi32(chunk, 1), accum, add);
            _avx512_drain_u8(_mm512_extracti32x4_epi32(chunk, 2), accum, add);
            _avx512_drain_u8(_mm512_extracti32x4_epi32(chunk, 3), accum, add);
            i += 64u;
        }
    } else if (data_size == 2u) {
        while (i + 32u <= len) {
            __m512i chunk = _mm512_loadu_si512((__m512i*)(data + i * 2u));
            __m128i l0=_mm512_extracti32x4_epi32(chunk,0);
            __m128i l1=_mm512_extracti32x4_epi32(chunk,1);
            __m128i l2=_mm512_extracti32x4_epi32(chunk,2);
            __m128i l3=_mm512_extracti32x4_epi32(chunk,3);
            uint16_t v;
            v=(uint16_t)_mm_extract_epi16(l0,0);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(l0,1);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(l0,2);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(l0,3);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(l0,4);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(l0,5);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(l0,6);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(l0,7);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(l1,0);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(l1,1);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(l1,2);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(l1,3);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(l1,4);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(l1,5);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(l1,6);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(l1,7);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(l2,0);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(l2,1);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(l2,2);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(l2,3);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(l2,4);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(l2,5);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(l2,6);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(l2,7);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(l3,0);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(l3,1);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(l3,2);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(l3,3);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(l3,4);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(l3,5);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(l3,6);add(accum,&v);
            v=(uint16_t)_mm_extract_epi16(l3,7);add(accum,&v);
            i += 32u;
        }
    } else if (data_size == 4u) {
        while (i + 16u <= len) {
            __m512i chunk = _mm512_loadu_si512((__m512i*)(data + i * 4u));
            __m128i l0=_mm512_extracti32x4_epi32(chunk,0);
            __m128i l1=_mm512_extracti32x4_epi32(chunk,1);
            __m128i l2=_mm512_extracti32x4_epi32(chunk,2);
            __m128i l3=_mm512_extracti32x4_epi32(chunk,3);
            uint32_t v;
            v=(uint32_t)_mm_extract_epi32(l0,0);add(accum,&v);
            v=(uint32_t)_mm_extract_epi32(l0,1);add(accum,&v);
            v=(uint32_t)_mm_extract_epi32(l0,2);add(accum,&v);
            v=(uint32_t)_mm_extract_epi32(l0,3);add(accum,&v);
            v=(uint32_t)_mm_extract_epi32(l1,0);add(accum,&v);
            v=(uint32_t)_mm_extract_epi32(l1,1);add(accum,&v);
            v=(uint32_t)_mm_extract_epi32(l1,2);add(accum,&v);
            v=(uint32_t)_mm_extract_epi32(l1,3);add(accum,&v);
            v=(uint32_t)_mm_extract_epi32(l2,0);add(accum,&v);
            v=(uint32_t)_mm_extract_epi32(l2,1);add(accum,&v);
            v=(uint32_t)_mm_extract_epi32(l2,2);add(accum,&v);
            v=(uint32_t)_mm_extract_epi32(l2,3);add(accum,&v);
            v=(uint32_t)_mm_extract_epi32(l3,0);add(accum,&v);
            v=(uint32_t)_mm_extract_epi32(l3,1);add(accum,&v);
            v=(uint32_t)_mm_extract_epi32(l3,2);add(accum,&v);
            v=(uint32_t)_mm_extract_epi32(l3,3);add(accum,&v);
            i += 16u;
        }
    } else if (data_size == 8u) {
        while (i + 8u <= len) {
            __m512i chunk = _mm512_loadu_si512((__m512i*)(data + i * 8u));
            __m128i l0=_mm512_extracti32x4_epi32(chunk,0);
            __m128i l1=_mm512_extracti32x4_epi32(chunk,1);
            __m128i l2=_mm512_extracti32x4_epi32(chunk,2);
            __m128i l3=_mm512_extracti32x4_epi32(chunk,3);
            uint64_t v;
            v=(uint64_t)_mm_extract_epi64(l0,0);add(accum,&v);
            v=(uint64_t)_mm_extract_epi64(l0,1);add(accum,&v);
            v=(uint64_t)_mm_extract_epi64(l1,0);add(accum,&v);
            v=(uint64_t)_mm_extract_epi64(l1,1);add(accum,&v);
            v=(uint64_t)_mm_extract_epi64(l2,0);add(accum,&v);
            v=(uint64_t)_mm_extract_epi64(l2,1);add(accum,&v);
            v=(uint64_t)_mm_extract_epi64(l3,0);add(accum,&v);
            v=(uint64_t)_mm_extract_epi64(l3,1);add(accum,&v);
            i += 8u;
        }
    }
    for (; i < len; i++) add(accum, data + i * data_size);
}
// ================================================================================ 
// ================================================================================ 
#endif /* CSALT_SIMD_AVX512_UINT8_INL */

