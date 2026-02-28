
#ifndef CSALT_SIMD_AVX_UINT8_INL
#define CSALT_SIMD_AVX_UINT8_INL

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <immintrin.h>   /* AVX */
// ================================================================================ 
// ================================================================================ 

static inline __m256i _avx_reverse_bytes(__m256i v) {
    const __m128i byte_mask = _mm_set_epi8(
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
    );

    /* Extract and reverse each 128-bit lane independently */
    __m128i lo128 = _mm256_extractf128_si256(v, 0);
    __m128i hi128 = _mm256_extractf128_si256(v, 1);

    lo128 = _mm_shuffle_epi8(lo128, byte_mask);
    hi128 = _mm_shuffle_epi8(hi128, byte_mask);

    /* Reassemble with lanes swapped: old-hi becomes new-lo, old-lo becomes new-hi */
    __m256i result = _mm256_castsi128_si256(hi128);
    result = _mm256_insertf128_si256(result, lo128, 1);
    return result;
}

// ================================================================================
// Public interface
// ================================================================================

static void simd_reverse_uint8(uint8_t* data, size_t len, size_t data_size) {
    if (data == NULL || len < 2u || data_size == 0u) return;

    if (data_size <= 32u && (32u % data_size == 0u)) {
        size_t lo = 0u;
        size_t hi = len - 1u;
        size_t elems_per_reg = 32u / data_size;

        while (lo < hi) {
            uint8_t* lo_ptr = data + lo * data_size;
            uint8_t* hi_ptr = data + hi * data_size;

            if ((hi - lo + 1u) >= 2u * elems_per_reg) {
                __m256i vlo = _mm256_loadu_si256((__m256i*)lo_ptr);
                __m256i vhi = _mm256_loadu_si256((__m256i*)hi_ptr);

                vlo = _avx_reverse_bytes(vlo);
                vhi = _avx_reverse_bytes(vhi);

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

// ================================================================================
// Internal: broadcast needle pattern into a 128-bit register
// ================================================================================

static inline __m128i _avx_broadcast128(const uint8_t* needle, size_t data_size) {
    uint8_t buf[16] = {0};
    for (size_t i = 0; i < 16; i++) {
        buf[i] = needle[i % data_size];
    }
    return _mm_loadu_si128((__m128i*)buf);
}

// ================================================================================
// Internal: find first matching element within a 32-byte chunk
// mask is a 32-bit value combining two 16-bit movemask results.
// ================================================================================

static inline size_t _avx_first_match32(int    mask,
                                         size_t data_size,
                                         size_t elem_base,
                                         size_t end) {
    int    elem_mask     = (data_size >= 16) ? -1 : (1 << (int)data_size) - 1;
    size_t elems_per_reg = 32u / data_size;

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
        size_t  elems_per_reg = 32u / data_size;
        __m128i vn128         = _avx_broadcast128(needle, data_size);
        size_t  i             = start;

        while (i + elems_per_reg <= end) {
            const uint8_t* ptr = data + i * data_size;

            /* Load as two 128-bit halves — AVX has no 256-bit integer cmpeq */
            __m128i lo   = _mm_loadu_si128((__m128i*)ptr);
            __m128i hi   = _mm_loadu_si128((__m128i*)(ptr + 16));
            int     mlo  = _mm_movemask_epi8(_mm_cmpeq_epi8(lo, vn128));
            int     mhi  = _mm_movemask_epi8(_mm_cmpeq_epi8(hi, vn128));
            int     mask = mlo | (mhi << 16);

            if (mask != 0) {
                size_t found = _avx_first_match32(mask, data_size, i, end);
                if (found != SIZE_MAX) {
                    _mm256_zeroupper();
                    return found;
                }
            }
            i += elems_per_reg;
        }

        /* Scalar remainder */
        _mm256_zeroupper();
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
#endif /* CSALT_SIMD_AVX_UINT8_INL */

