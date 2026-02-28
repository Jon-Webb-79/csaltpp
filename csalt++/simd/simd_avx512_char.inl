/* simd_avx512_char.inl
   AVX-512BW helpers for last-occurrence search in unsigned bytes.
   Requires: <immintrin.h>, compiled with -mavx512bw
*/
#ifndef CSALT_SIMD_AVX512_CHAR_INL
#define CSALT_SIMD_AVX512_CHAR_INL

#include <stddef.h>
#include <stdint.h>
#include <immintrin.h>
#include <string.h>
// ================================================================================ 
// ================================================================================ 

static inline unsigned simd_ctz64_(uint64_t x) {
    /* Precondition: x != 0 */
    unsigned idx = 0u;
    while ((x & 1ull) == 0ull) {
        x >>= 1u;
        ++idx;
    }
    return idx;
}

static inline size_t simd_first_diff_u8(const uint8_t* a,
                                        const uint8_t* b,
                                        size_t n) {
    if ((a == NULL) || (b == NULL) || (n == 0u)) {
        return 0u;
    }

    size_t i = 0u;

    while ((i + 64u) <= n) {
        __m512i va = _mm512_loadu_si512((const void*)(a + i));
        __m512i vb = _mm512_loadu_si512((const void*)(b + i));

        __mmask64 eqmask = _mm512_cmpeq_epi8_mask(va, vb);

        if ((uint64_t)eqmask != UINT64_MAX) {
            uint64_t mism = (uint64_t)(~(uint64_t)eqmask); /* 1 where different */
            unsigned bit  = simd_ctz64_(mism);
            return i + (size_t)bit;
        }

        i += 64u;
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
static inline unsigned csalt_ctz64_portable_(uint64_t m)
{
    unsigned n = 0u;
    while ((m & 1ull) == 0ull) {
        m >>= 1ull;
        ++n;
    }
    return n; /* 0..63 */
}

static inline unsigned csalt_clz64_portable_(uint64_t m)
{
    unsigned n = 0u;
    uint64_t bit = 0x8000000000000000ull;
    while ((m & bit) == 0ull) {
        bit >>= 1ull;
        ++n;
    }
    return n; /* 0..63 */
}

static inline unsigned simd_first_bit64_(uint64_t m)
{
    return csalt_ctz64_portable_(m);
}

static inline unsigned simd_last_bit64_(uint64_t m)
{
    return 63u - csalt_clz64_portable_(m);
}

/* ---------------------------- AVX-512 forward -------------------------------- */
static inline size_t simd512_find_substr_u8_forward_(const uint8_t* hay,
                                                     size_t hay_len,
                                                     const uint8_t* needle,
                                                     size_t needle_len)
{
    if ((hay == NULL) || (needle == NULL)) { return SIZE_MAX; }
    if (needle_len == 0u) { return 0u; }
    if (needle_len > hay_len) { return SIZE_MAX; }

    uint8_t const first = needle[0];
    __m512i const vfirst = _mm512_set1_epi8((char)first);

    size_t const last_start = hay_len - needle_len;
    size_t i = 0u;

    while (i <= last_start) {
        /* Safe to load 64 bytes only if i+64 <= hay_len */
        if ((i + 64u) <= hay_len) {
            __m512i v = _mm512_loadu_si512((const void*)(hay + i));
            __mmask64 eqmask = _mm512_cmpeq_epi8_mask(v, vfirst);
            uint64_t mask = (uint64_t)eqmask;

            while (mask != 0ull) {
                unsigned const bit = simd_first_bit64_(mask);
                size_t const pos = i + (size_t)bit;

                if (pos <= last_start) {
                    if (needle_len == 1u) { return pos; }
                    if (memcmp(hay + pos + 1u, needle + 1u, needle_len - 1u) == 0) {
                        return pos;
                    }
                }

                mask &= (mask - 1ull); /* clear lowest set bit */
            }

            i += 64u;
        } else {
            break;
        }
    }

    /* Scalar tail */
    for (; i <= last_start; ++i) {
        if (hay[i] != first) { continue; }
        if (needle_len == 1u) { return i; }
        if (memcmp(hay + i + 1u, needle + 1u, needle_len - 1u) == 0) {
            return i;
        }
    }

    return SIZE_MAX;
}

/* ---------------------------- AVX-512 reverse -------------------------------- */
static inline size_t simd512_find_substr_u8_reverse_(const uint8_t* hay,
                                                     size_t hay_len,
                                                     const uint8_t* needle,
                                                     size_t needle_len)
{
    if ((hay == NULL) || (needle == NULL)) { return SIZE_MAX; }
    if (needle_len == 0u) { return 0u; }
    if (needle_len > hay_len) { return SIZE_MAX; }

    uint8_t const first = needle[0];
    __m512i const vfirst = _mm512_set1_epi8((char)first);

    size_t const last_start = hay_len - needle_len;
    size_t i = last_start;

    while (true) {
        /* Choose a 64-byte block that ends at i (or earlier) */
        size_t block_start = (i >= 63u) ? (i - 63u) : 0u;

        __m512i v = _mm512_loadu_si512((const void*)(hay + block_start));
        __mmask64 eqmask = _mm512_cmpeq_epi8_mask(v, vfirst);
        uint64_t mask = (uint64_t)eqmask;

        {
            size_t const block_end = block_start + 63u;
            size_t const max_pos   = (i < last_start) ? i : last_start;

            /* Clear bits above max_pos (relative to block_start). */
            if (max_pos < block_end) {
                unsigned const keep_bits =
                    (unsigned)(max_pos - block_start + 1u); /* 1..64 */
                if (keep_bits < 64u) {
                    mask &= ((1ull << keep_bits) - 1ull);
                }
            }

            while (mask != 0ull) {
                unsigned const bit = simd_last_bit64_(mask);
                size_t const pos = block_start + (size_t)bit;

                if (needle_len == 1u) { return pos; }
                if (memcmp(hay + pos + 1u, needle + 1u, needle_len - 1u) == 0) {
                    return pos;
                }

                mask &= ~(1ull << bit); /* clear that bit */
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
        ? simd512_find_substr_u8_reverse_(hay, hay_len, needle, needle_len)
        : simd512_find_substr_u8_forward_(hay, hay_len, needle, needle_len);
}
// -------------------------------------------------------------------------------- 

static inline size_t simd_token_count_u8(const uint8_t* p,
                                         size_t n,
                                         const char* delim,
                                         size_t dlen) {
    if ((p == NULL) || (delim == NULL)) return SIZE_MAX;
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

    for (; i + 64u <= n; i += 64u) {
        __m512i v = _mm512_loadu_si512((const void*)(p + i));

        /* dm bit k = 1 if v[k] is delimiter */
        __mmask64 dm = 0;
        for (size_t j = 0; j < dlen; ++j) {
            __m512i dj = _mm512_set1_epi8((char)delim[j]);
            dm |= _mm512_cmpeq_epi8_mask(v, dj);
        }

        /* starts where current is non-delim and previous is delim */
        __mmask64 non = ~dm;

        /* Shift dm left by 1 bit for “previous lane delim”, and inject carry-in */
        uint64_t dm_u = (uint64_t)dm;
        uint64_t prev_delim_u = (dm_u << 1) | (uint64_t)(prev_is_delim & 1u);

        uint64_t starts_u = ((uint64_t)non) & prev_delim_u;

        count += (size_t)__builtin_popcountll(starts_u);
        prev_is_delim = (uint32_t)((dm_u >> 63) & 1u);
    }

    bool in_token = (prev_is_delim == 0u);
    for (; i < n; ++i) {
        const bool is_delim = (lut[p[i]] != 0u);
        if (!is_delim) {
            if (!in_token) { ++count; in_token = true; }
        } else {
            in_token = false;
        }
    }

    return count;
}
// -------------------------------------------------------------------------------- 

static inline __mmask64 ascii_range_mask_512(__m512i x, __m512i lo, __m512i hi)
{
    /* Unsigned compare is ideal here. These intrinsics require AVX-512BW. */
    __mmask64 ge_lo = _mm512_cmpge_epu8_mask(x, lo);
    __mmask64 le_hi = _mm512_cmple_epu8_mask(x, hi);
    return (ge_lo & le_hi);
}

void simd_ascii_upper_u8(uint8_t* p, size_t n)
{
    if ((p == NULL) || (n == 0u)) return;

    size_t i = 0u;
    const __m512i lo  = _mm512_set1_epi8('a');
    const __m512i hi  = _mm512_set1_epi8('z');
    const __m512i sub = _mm512_set1_epi8(0x20);

    for (; i + 64u <= n; i += 64u) {
        __m512i v = _mm512_loadu_si512((const void*)(p + i));

        __mmask64 m = ascii_range_mask_512(v, lo, hi);

        /* Only subtract on masked lanes */
        __m512i up = _mm512_sub_epi8(v, sub);
        __m512i out = _mm512_mask_mov_epi8(v, m, up);

        _mm512_storeu_si512((void*)(p + i), out);
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
    const __m512i lo  = _mm512_set1_epi8('A');
    const __m512i hi  = _mm512_set1_epi8('Z');
    const __m512i add = _mm512_set1_epi8(0x20);

    for (; i + 64u <= n; i += 64u) {
        __m512i v = _mm512_loadu_si512((const void*)(p + i));

        __mmask64 m = ascii_range_mask_512(v, lo, hi);

        __m512i low = _mm512_add_epi8(v, add);
        __m512i out = _mm512_mask_mov_epi8(v, m, low);

        _mm512_storeu_si512((void*)(p + i), out);
    }

    for (; i < n; ++i) {
        uint8_t c = p[i];
        if ((c >= (uint8_t)'A') && (c <= (uint8_t)'Z')) p[i] = (uint8_t)(c + 0x20u);
    }
}
// ================================================================================ 
// ================================================================================ 
#endif /* CSALT_SIMD_AVX512_CHAR_INL */

