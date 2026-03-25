/* simd_avx2_char.inl
   AVX2 helpers for last-occurrence search in unsigned bytes.
   Requires: <immintrin.h>, compiled with -mavx2
*/
#ifndef CSALT_SIMD_AVX2_CHAR_INL
#define CSALT_SIMD_AVX2_CHAR_INL

#if !defined(__AVX512F__) || !defined(__AVX512BW__)
  #error "simd_avx512_char.inl requires __AVX512F__ and __AVX512BW__"
#endif

#include <stddef.h>
#include <stdint.h>
#include <immintrin.h>
// ================================================================================ 
// ================================================================================ 

/* Return index of least-significant 1-bit in a 16-bit value.
   Precondition: x != 0. Result: 0..15 */
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
        __m128i va = _mm_loadu_si128((const __m128i*)(const void*)(a + i));
        __m128i vb = _mm_loadu_si128((const __m128i*)(const void*)(b + i));

        __m128i eq = _mm_cmpeq_epi8(va, vb);
        uint16_t m = (uint16_t)(unsigned)_mm_movemask_epi8(eq);

        if (m != (uint16_t)0xFFFFu) {
            uint16_t mism = (uint16_t)((uint16_t)0xFFFFu ^ m);
            unsigned bit  = simd_ctz16_(mism);
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

/* ------------------------- portable bit scan helpers ------------------------- */
/* Preconditions: input mask m != 0 */
static inline unsigned csalt_ctz32_portable_(uint32_t m)
{
    unsigned n = 0u;
    while ((m & 1u) == 0u) {
        m >>= 1u;
        ++n;
    }
    return n; /* 0..31 */
}

static inline unsigned csalt_clz32_portable_(uint32_t m)
{
    unsigned n = 0u;
    uint32_t bit = 0x80000000u;
    while ((m & bit) == 0u) {
        bit >>= 1u;
        ++n;
    }
    return n; /* 0..31 */
}

static inline unsigned simd_first_bit_(uint32_t m)
{
    /* m != 0 required */
    return csalt_ctz32_portable_(m);
}

static inline unsigned simd_last_bit_(uint32_t m)
{
    /* last = 31 - clz(m); m != 0 required */
    return 31u - csalt_clz32_portable_(m);
}

/* ------------------------------ AVX2 forward -------------------------------- */
static inline size_t simd_find_substr_u8_forward_(const uint8_t* hay,
                                                  size_t hay_len,
                                                  const uint8_t* needle,
                                                  size_t needle_len)
{
    if ((hay == NULL) || (needle == NULL)) { return SIZE_MAX; }
    if (needle_len == 0u) { return 0u; }
    if (needle_len > hay_len) { return SIZE_MAX; }

    uint8_t const first = needle[0];
    __m256i const vfirst = _mm256_set1_epi8((char)first);

    size_t const last_start = hay_len - needle_len;
    size_t i = 0u;

    while (i <= last_start) {
        /* Safe to load 32 bytes only if i+32 <= hay_len */
        if ((i + 32u) <= hay_len) {
            __m256i v  = _mm256_loadu_si256((const __m256i*)(const void*)(hay + i));
            __m256i eq = _mm256_cmpeq_epi8(v, vfirst);
            uint32_t mask = (uint32_t)_mm256_movemask_epi8(eq);

            while (mask != 0u) {
                unsigned const bit = simd_first_bit_(mask); /* portable */
                size_t const pos = i + (size_t)bit;

                if (pos <= last_start) {
                    if (needle_len == 1u) { return pos; }
                    if (memcmp(hay + pos + 1u, needle + 1u, needle_len - 1u) == 0) {
                        return pos;
                    }
                }

                mask &= (mask - 1u); /* clear lowest set bit */
            }

            i += 32u;
        } else {
            break;
        }
    }

    /* Scalar tail over remaining possible starts */
    for (; i <= last_start; ++i) {
        if (hay[i] != first) { continue; }
        if (needle_len == 1u) { return i; }
        if (memcmp(hay + i + 1u, needle + 1u, needle_len - 1u) == 0) {
            return i;
        }
    }

    return SIZE_MAX;
}

/* ------------------------------ AVX2 reverse -------------------------------- */
static inline size_t simd_find_substr_u8_reverse_(const uint8_t* hay,
                                                  size_t hay_len,
                                                  const uint8_t* needle,
                                                  size_t needle_len)
{
    if ((hay == NULL) || (needle == NULL)) { return SIZE_MAX; }
    if (needle_len == 0u) { return 0u; }
    if (needle_len > hay_len) { return SIZE_MAX; }

    uint8_t const first = needle[0];
    __m256i const vfirst = _mm256_set1_epi8((char)first);

    size_t const last_start = hay_len - needle_len;
    size_t i = last_start;

    while (true) {
        size_t block_start = (i >= 31u) ? (i - 31u) : 0u;

        __m256i v  = _mm256_loadu_si256((const __m256i*)(const void*)(hay + block_start));
        __m256i eq = _mm256_cmpeq_epi8(v, vfirst);
        uint32_t mask = (uint32_t)_mm256_movemask_epi8(eq);

        /* Only candidates <= max_pos are valid within this block */
        {
            size_t const block_end = block_start + 31u;
            size_t const max_pos   = (i < last_start) ? i : last_start;

            if (max_pos < block_end) {
                unsigned const keep_bits =
                    (unsigned)(max_pos - block_start + 1u); /* 1..32 */

                if (keep_bits < 32u) {
                    mask &= ((1u << keep_bits) - 1u);
                }
            }

            while (mask != 0u) {
                unsigned const bit = simd_last_bit_(mask); /* portable */
                size_t const pos = block_start + (size_t)bit;

                if (needle_len == 1u) { return pos; }
                if (memcmp(hay + pos + 1u, needle + 1u, needle_len - 1u) == 0) {
                    return pos;
                }

                mask &= ~(1u << bit); /* clear that bit */
            }
        }

        if (block_start == 0u) { break; }
        i = block_start - 1u;
    }

    return SIZE_MAX;
}

static inline size_t simd_find_substr_u8(const uint8_t* hay,
                                         size_t hay_len,
                                         const uint8_t* needle,
                                         size_t needle_len,
                                         direction_t dir)
{
    return (dir == REVERSE)
        ? simd_find_substr_u8_reverse_(hay, hay_len, needle, needle_len)
        : simd_find_substr_u8_forward_(hay, hay_len, needle, needle_len);
}
// -------------------------------------------------------------------------------- 

static inline size_t simd_token_count_u8(const uint8_t* p,
                                         size_t n,
                                         const char* delim,
                                         size_t dlen) {
    if ((p == NULL) || (delim == NULL)) return SIZE_MAX;

    if (n == 0u) return 0u;
    if (dlen == 0u) return 1u; /* no delimiters => whole window is one token */

    /* LUT for scalar tail and for optional scalar checks */
    uint8_t lut[256];
    memset(lut, 0, sizeof(lut));
    for (size_t j = 0; j < dlen; ++j) {
        lut[(unsigned char)delim[j]] = 1u;
    }

    size_t i = 0u;
    size_t count = 0u;

    /* Treat start-of-window as delimiter boundary */
    uint32_t prev_is_delim = 1u;

    for (; i + 32u <= n; i += 32u) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(const void*)(p + i));

        /* Build delimiter mask for this 32B block:
           dm bit k = 1 if v[k] is a delimiter */
        __m256i m = _mm256_setzero_si256();
        for (size_t j = 0; j < dlen; ++j) {
            __m256i dj = _mm256_set1_epi8((char)delim[j]);
            m = _mm256_or_si256(m, _mm256_cmpeq_epi8(v, dj));
        }

        uint32_t dm = (uint32_t)_mm256_movemask_epi8(m); /* 32-bit mask */
        uint32_t non = ~dm;

        /* Token start where current is non-delim AND previous is delim.
           Previous for lane 0 comes from prev_is_delim. */
        uint32_t prev_delim_mask = (dm << 1) | (prev_is_delim & 1u);
        uint32_t starts = non & prev_delim_mask;

        count += (size_t)__builtin_popcount(starts);

        /* Carry last lane delimiter state */
        prev_is_delim = (dm >> 31) & 1u;
    }

    /* Scalar tail */
    bool in_token = (prev_is_delim == 0u);

    for (; i < n; ++i) {
        const bool is_delim = (lut[p[i]] != 0u);
        if (!is_delim) {
            if (!in_token) {
                ++count;
                in_token = true;
            }
        } else {
            in_token = false;
        }
    }

    return count;
}
// -------------------------------------------------------------------------------- 

static inline __m256i _blendv_epi8(__m256i a, __m256i b, __m256i mask)
{
    return _mm256_or_si256(_mm256_andnot_si256(mask, a),
                           _mm256_and_si256(mask, b));
}

/* mask for (x >= lo) && (x <= hi) for ASCII bytes */
static inline __m256i _ascii_range_mask(__m256i x, __m256i lo, __m256i hi)
{
    const __m256i one = _mm256_set1_epi8(1);

    /* x > lo-1  &&  hi+1 > x  (signed compares ok for ASCII range) */
    __m256i ge_lo = _mm256_cmpgt_epi8(x, _mm256_sub_epi8(lo, one));
    __m256i le_hi = _mm256_cmpgt_epi8(_mm256_add_epi8(hi, one), x);
    return _mm256_and_si256(ge_lo, le_hi);
}

void simd_ascii_upper_u8(uint8_t* p, size_t n)
{
    if ((p == NULL) || (n == 0u)) return;

    size_t i = 0u;
    const __m256i lo   = _mm256_set1_epi8('a');
    const __m256i hi   = _mm256_set1_epi8('z');
    const __m256i sub  = _mm256_set1_epi8(0x20);

    for (; i + 32u <= n; i += 32u) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(const void*)(p + i));
        __m256i mask  = _ascii_range_mask(v, lo, hi);
        __m256i upper = _mm256_sub_epi8(v, sub);
        __m256i out   = _blendv_epi8(v, upper, mask);
        _mm256_storeu_si256((__m256i*)(void*)(p + i), out);
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
    const __m256i lo   = _mm256_set1_epi8('A');
    const __m256i hi   = _mm256_set1_epi8('Z');
    const __m256i add  = _mm256_set1_epi8(0x20);

    for (; i + 32u <= n; i += 32u) {
        __m256i v = _mm256_loadu_si256((const __m256i*)(const void*)(p + i));
        __m256i mask  = _ascii_range_mask(v, lo, hi);
        __m256i lower = _mm256_add_epi8(v, add);
        __m256i out   = _blendv_epi8(v, lower, mask);
        _mm256_storeu_si256((__m256i*)(void*)(p + i), out);
    }

    for (; i < n; ++i) {
        uint8_t c = p[i];
        if ((c >= (uint8_t)'A') && (c <= (uint8_t)'Z')) p[i] = (uint8_t)(c + 0x20u);
    }
}
// ================================================================================ 
// ================================================================================ 
#endif /* CSALT_SIMD_AVX2_CHAR_INL */

