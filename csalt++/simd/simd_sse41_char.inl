/* simd_sse41_char.inl
   SSE4.1 path uses SSE2 integer ops for this routine.
   Requires: <smmintrin.h> and <emmintrin.h>
*/
#ifndef CSALT_SIMD_SSE41_CHAR_INL
#define CSALT_SIMD_SSE41_CHAR_INL

#include <stddef.h>
#include <stdint.h>
#include <smmintrin.h>
#include <emmintrin.h>
// ================================================================================ 
// ================================================================================ 

/* Count trailing zeros in a 16-bit nonzero value. Result 0..15. */
static inline unsigned simd_ctz16_(uint16_t x)
{
    unsigned idx = 0u;
    while ((x & (uint16_t)1u) == (uint16_t)0u) {
        x = (uint16_t)(x >> 1u);
        ++idx;
    }
    return idx;
}

static inline size_t simd_first_diff_u8(const uint8_t* a,
                                        const uint8_t* b,
                                        size_t n)
{
    if (n == 0u) { return 0u; }
    if ((a == NULL) || (b == NULL)) { return 0u; }

    size_t i = 0u;

    while ((i + 16u) <= n) {
        const __m128i va = _mm_loadu_si128((const __m128i*)(const void*)(a + i));
        const __m128i vb = _mm_loadu_si128((const __m128i*)(const void*)(b + i));

        const __m128i veq = _mm_cmpeq_epi8(va, vb);
        const uint16_t m  = (uint16_t)(unsigned)_mm_movemask_epi8(veq);

        if (m != (uint16_t)0xFFFFu) {
            const uint16_t mism = (uint16_t)((uint16_t)0xFFFFu ^ m);
            const unsigned bit  = simd_ctz16_(mism);
            return i + (size_t)bit;
        }

        i += 16u;
    }

    for (; i < n; ++i) {
        if (a[i] != b[i]) { return i; }
    }
    return n;
}
// -------------------------------------------------------------------------------- 

#ifndef SIZE_MAX
  #define SIZE_MAX ((size_t)-1)
#endif

#ifndef ITER_DIR_H
#define ITER_DIR_H
    typedef enum {
        FORWARD = 0,
        REVERSE = 1
    }direction_t;
#endif /* ITER_DIR_H*/

static inline unsigned csalt_ctz16(uint16_t m)
{
    unsigned n = 0u;
    while ((m & 1u) == 0u) { m = (uint16_t)(m >> 1u); ++n; }
    return n;
}

static inline unsigned csalt_clz16(uint16_t m)
{
    unsigned n = 0u;
    uint16_t bit = (uint16_t)0x8000u;
    while ((m & bit) == 0u) { bit = (uint16_t)(bit >> 1u); ++n; }
    return n;
}

static inline unsigned csalt_first_bit16(uint16_t m) { return csalt_ctz16(m); }
static inline unsigned csalt_last_bit16 (uint16_t m) { return 15u - csalt_clz16(m); }

static inline size_t simd_find_substr_u8(const uint8_t* hay,
                                         size_t hay_len,
                                         const uint8_t* needle,
                                         size_t needle_len,
                                         direction_t dir)
{
    if ((hay == NULL) || (needle == NULL)) { return SIZE_MAX; }
    if (needle_len == 0u) { return 0u; }
    if (needle_len > hay_len) { return SIZE_MAX; }

    const uint8_t first = needle[0];
    const __m128i vfirst = _mm_set1_epi8((char)first);

    const size_t last_start = hay_len - needle_len;

    /* For very small haystacks, a vector load would overread.
       Handle entirely in scalar to guarantee no OOB reads. */
    if (hay_len < 16u) {
        if (dir == FORWARD) {
            for (size_t i = 0u; i <= last_start; ++i) {
                if (hay[i] != first) continue;
                if (needle_len == 1u) return i;
                if (memcmp(hay + i + 1u, needle + 1u, needle_len - 1u) == 0) {
                    return i;
                }
            }
            return SIZE_MAX;
        } else { /* REVERSE */
            for (size_t i = last_start + 1u; i-- > 0u; ) {
                if (hay[i] != first) continue;
                if (needle_len == 1u) return i;
                if (memcmp(hay + i + 1u, needle + 1u, needle_len - 1u) == 0) {
                    return i;
                }
                if (i == 0u) break;
            }
            return SIZE_MAX;
        }
    }

    /* hay_len >= 16 from here on */

    if (dir == FORWARD) {
        size_t i = 0u;

        /* Only load at positions i where i+16 <= hay_len */
        const size_t vec_end = hay_len - 16u;

        while (i <= last_start && i <= vec_end) {
            __m128i v  = _mm_loadu_si128((const __m128i*)(const void*)(hay + i));
            __m128i eq = _mm_cmpeq_epi8(v, vfirst);
            uint16_t mask = (uint16_t)_mm_movemask_epi8(eq);

            while (mask != 0u) {
                unsigned bit = csalt_first_bit16(mask);
                size_t pos = i + (size_t)bit;

                if (pos <= last_start) {
                    if (needle_len == 1u) { return pos; }
                    if (memcmp(hay + pos + 1u, needle + 1u, needle_len - 1u) == 0) {
                        return pos;
                    }
                }

                mask = (uint16_t)(mask & (uint16_t)(mask - 1u));
            }

            i += 16u;
        }

        /* Scalar tail for remaining start positions */
        for (; i <= last_start; ++i) {
            if (hay[i] != first) continue;
            if (needle_len == 1u) return i;
            if (memcmp(hay + i + 1u, needle + 1u, needle_len - 1u) == 0) {
                return i;
            }
        }

        return SIZE_MAX;
    }

    /* REVERSE */
    {
        size_t i = last_start;

        for (;;) {
            /* Choose a block that contains i but also stays safe for a 16B load */
            size_t block_start = (i >= 15u) ? (i - 15u) : 0u;

            /* Ensure block_start+16 <= hay_len (safe load) */
            if (block_start + 16u > hay_len) {
                block_start = hay_len - 16u; /* safe because hay_len >= 16 */
            }

            __m128i v  = _mm_loadu_si128((const __m128i*)(const void*)(hay + block_start));
            __m128i eq = _mm_cmpeq_epi8(v, vfirst);
            uint16_t mask = (uint16_t)_mm_movemask_epi8(eq);

            /* Only keep candidate positions <= max_pos within this block */
            size_t max_pos = (i < last_start) ? i : last_start;

            if (max_pos < block_start + 15u) {
                unsigned keep = (unsigned)(max_pos - block_start + 1u); /* 1..16 */
                if (keep < 16u) {
                    mask = (uint16_t)(mask & (uint16_t)((1u << keep) - 1u));
                }
            }

            while (mask != 0u) {
                unsigned bit = csalt_last_bit16(mask);
                size_t pos = block_start + (size_t)bit;

                if (needle_len == 1u) { return pos; }
                if (memcmp(hay + pos + 1u, needle + 1u, needle_len - 1u) == 0) {
                    return pos;
                }

                mask = (uint16_t)(mask & (uint16_t)~(uint16_t)(1u << bit));
            }

            if (block_start == 0u) { break; }
            i = block_start - 1u;
        }
    }

    return SIZE_MAX;
}
// -------------------------------------------------------------------------------- 

size_t simd_token_count_u8(const uint8_t* s, size_t n,
                           const char* delim, size_t dlen) {
    if ((s == NULL) || (delim == NULL)) return SIZE_MAX;
    if (n == 0u) return 0u;
    if (dlen == 0u) return 1u;

    uint8_t lut[256];
    memset(lut, 0, sizeof(lut));
    for (size_t j = 0; j < dlen; ++j) {
        lut[(unsigned char)delim[j]] = 1u;
    }

    size_t i = 0u;
    size_t count = 0u;
    uint32_t prev_is_delim = 1u;

    for (; i + 16u <= n; i += 16u) {
        __m128i v = _mm_loadu_si128((const __m128i*)(const void*)(s + i));
        __m128i m = _mm_setzero_si128();

        for (size_t j = 0; j < dlen; ++j) {
            __m128i dj = _mm_set1_epi8((char)delim[j]);
            m = _mm_or_si128(m, _mm_cmpeq_epi8(v, dj));
        }

        uint32_t dm  = (uint32_t)_mm_movemask_epi8(m) & 0xFFFFu;
        uint32_t non = (~dm) & 0xFFFFu;

        uint32_t starts = non & ((dm << 1) | (prev_is_delim & 1u));
        count += (size_t)__builtin_popcount(starts);

        prev_is_delim = (dm >> 15) & 1u;
    }

    bool in_token = (prev_is_delim == 0u);
    for (; i < n; ++i) {
        const bool is_delim = (lut[s[i]] != 0u);
        if (!is_delim) {
            if (!in_token) { ++count; in_token = true; }
        } else {
            in_token = false;
        }
    }

    return count;
}
// -------------------------------------------------------------------------------- 

static inline __m128i ascii_range_mask_sse41(__m128i x, __m128i lo, __m128i hi)
{
    const __m128i one = _mm_set1_epi8(1);
    __m128i ge_lo = _mm_cmpgt_epi8(x, _mm_sub_epi8(lo, one));
    __m128i le_hi = _mm_cmpgt_epi8(_mm_add_epi8(hi, one), x);
    return _mm_and_si128(ge_lo, le_hi);
}

void simd_ascii_upper_u8(uint8_t* p, size_t n)
{
    if ((p == NULL) || (n == 0u)) return;

    size_t i = 0u;

    const __m128i lo  = _mm_set1_epi8('a');
    const __m128i hi  = _mm_set1_epi8('z');
    const __m128i sub = _mm_set1_epi8(0x20);

    for (; i + 16u <= n; i += 16u) {
        __m128i v = _mm_loadu_si128((const __m128i*)(const void*)(p + i));

        __m128i mask  = ascii_range_mask_sse41(v, lo, hi);
        __m128i upper = _mm_sub_epi8(v, sub);

        /* mask selects upper where 0xFF, else v */
        __m128i out = _mm_blendv_epi8(v, upper, mask);

        _mm_storeu_si128((__m128i*)(void*)(p + i), out);
    }

    for (; i < n; ++i) {
        uint8_t c = p[i];
        if ((c >= (uint8_t)'a') && (c <= (uint8_t)'z')) p[i] = (uint8_t)(c - 0x20u);
    }
}

void simd_ascii_lower_u8(uint8_t* p, size_t n)
{
    if ((p == NULL) || (n == 0u)) return;

    size_t i = 0u;

    const __m128i lo  = _mm_set1_epi8('A');
    const __m128i hi  = _mm_set1_epi8('Z');
    const __m128i add = _mm_set1_epi8(0x20);

    for (; i + 16u <= n; i += 16u) {
        __m128i v = _mm_loadu_si128((const __m128i*)(const void*)(p + i));

        __m128i mask  = ascii_range_mask_sse41(v, lo, hi);
        __m128i lower = _mm_add_epi8(v, add);

        __m128i out = _mm_blendv_epi8(v, lower, mask);

        _mm_storeu_si128((__m128i*)(void*)(p + i), out);
    }

    for (; i < n; ++i) {
        uint8_t c = p[i];
        if ((c >= (uint8_t)'A') && (c <= (uint8_t)'Z')) p[i] = (uint8_t)(c + 0x20u);
    }
}
// ================================================================================ 
// ================================================================================ 
#endif /* CSALT_SIMD_SSE41_CHAR_INL */

