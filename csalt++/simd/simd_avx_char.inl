/* simd_avx_char.inl
   AVX (no AVX2): 256-bit integer compares are unavailable, so we use SSE2 16B steps.
   Requires: <immintrin.h> (or <emmintrin.h>)
*/
#ifndef CSALT_SIMD_AVX_CHAR_INL
#define CSALT_SIMD_AVX_CHAR_INL

#include <stddef.h>
#include <stdint.h>
#include <emmintrin.h>
#include <string.h>
// ================================================================================ 
// ================================================================================ 

/* Return index of least-significant 1-bit in a 16-bit value.
   Precondition: x != 0. Result: 0..15 */
static inline unsigned simd_ctz16_(uint16_t x) {
    unsigned idx = 0u;
    while ((x & (uint16_t)1u) == (uint16_t)0u) {
        x = (uint16_t)(x >> 1u);
        ++idx;
    }
    return idx;
}

static inline size_t simd_first_diff_u8(const uint8_t* a,
                                        const uint8_t* b,
                                        size_t n) {
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

/* direction_t must be visible before this include.
   If you already define it elsewhere, remove this block. */

#ifndef ITER_DIR_H
#define ITER_DIR_H
    typedef enum {
        FORWARD = 0,
        REVERSE = 1
    }direction_t;
#endif /* ITER_DIR_H*/

/* ---------- portable bit scan (no compiler builtins) ---------- */
/* Precondition: m != 0 */
static inline unsigned csalt_ctz32(uint32_t m)
{
    unsigned n = 0u;
    while ((m & 1u) == 0u) { m >>= 1u; ++n; }
    return n; /* 0..31 */
}

/* Precondition: m != 0 */
static inline unsigned csalt_clz32(uint32_t m)
{
    unsigned n = 0u;
    uint32_t bit = 0x80000000u;
    while ((m & bit) == 0u) { bit >>= 1u; ++n; }
    return n; /* 0..31 */
}

static inline unsigned csalt_first_bit32(uint32_t m) { return csalt_ctz32(m); }
static inline unsigned csalt_last_bit32 (uint32_t m) { return 31u - csalt_clz32(m); }

/* Build a 32-bit mask of matches for byte == first over hay[i..i+31],
   using two SSE2 loads and movemasks. */
static inline uint32_t csalt_match32_first(const uint8_t* p, __m128i vfirst)
{
    __m128i v0 = _mm_loadu_si128((const __m128i*)(const void*)(p));
    __m128i v1 = _mm_loadu_si128((const __m128i*)(const void*)(p + 16u));

    __m128i e0 = _mm_cmpeq_epi8(v0, vfirst);
    __m128i e1 = _mm_cmpeq_epi8(v1, vfirst);

    uint32_t m0 = (uint32_t)(uint16_t)_mm_movemask_epi8(e0);
    uint32_t m1 = (uint32_t)(uint16_t)_mm_movemask_epi8(e1);

    return (m0 | (m1 << 16));
}

/* ---------- unified public function (must be named exactly this) ---------- */
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

    /* ================= FORWARD ================= */
    if (dir == FORWARD) {
        size_t i = 0u;

        while (i <= last_start) {
            if ((i + 32u) <= hay_len) {
                uint32_t mask = csalt_match32_first(hay + i, vfirst);

                while (mask != 0u) {
                    unsigned bit = csalt_first_bit32(mask);
                    size_t pos = i + (size_t)bit;

                    if (pos <= last_start) {
                        if (needle_len == 1u) { return pos; }
                        if (memcmp(hay + pos + 1u,
                                   needle + 1u,
                                   needle_len - 1u) == 0)
                        {
                            return pos;
                        }
                    }

                    mask &= (mask - 1u);
                }

                i += 32u;
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

    /* ================= REVERSE ================= */
    {
        size_t i = last_start;

        while (true) {
            size_t block_start = (i >= 31u) ? (i - 31u) : 0u;

            uint32_t mask = csalt_match32_first(hay + block_start, vfirst);

            /* Keep only candidates within [block_start, max_pos] */
            {
                size_t block_end = block_start + 31u;
                size_t max_pos   = (i < last_start) ? i : last_start;

                if (max_pos < block_end) {
                    unsigned keep = (unsigned)(max_pos - block_start + 1u); /* 1..32 */
                    if (keep < 32u) {
                        mask &= ((1u << keep) - 1u);
                    }
                }

                while (mask != 0u) {
                    unsigned bit = csalt_last_bit32(mask);
                    size_t pos = block_start + (size_t)bit;

                    if (needle_len == 1u) { return pos; }
                    if (memcmp(hay + pos + 1u,
                               needle + 1u,
                               needle_len - 1u) == 0)
                    {
                        return pos;
                    }

                    mask &= ~(1u << bit);
                }
            }

            if (block_start == 0u) { break; }
            i = block_start - 1u;
        }
    }

    return SIZE_MAX;
}
// ================================================================================ 
// ================================================================================ 

static inline int csalt_highbit_u32(uint32_t m) { return 31 - __builtin_clz(m); }

static inline size_t simd_last_index_u8_avx_fallback_sse2(const unsigned char* s, size_t n, unsigned char c) {
    size_t i = 0, last = SIZE_MAX;
    const __m128i needle = _mm_set1_epi8((char)c);

    while (i + 16 <= n) {
        __m128i v  = _mm_loadu_si128((const __m128i*)(s + i));
        __m128i eq = _mm_cmpeq_epi8(v, needle);
        uint32_t mask = (uint32_t)_mm_movemask_epi8(eq);
        if (mask) last = i + (size_t)csalt_highbit_u32(mask);
        i += 16;
    }
    for (size_t j = n; j-- > i; ) if (s[j] == c) return j;
    return last;
}

static inline size_t simd_first_substr_index_avx(const unsigned char* s, size_t n,
                                                 const unsigned char* pat, size_t m) {
    if (m == 0) return 0;
    if (m == 1) {
        const void* p = memchr(s, pat[0], n);
        return p ? (size_t)((const unsigned char*)p - s) : SIZE_MAX;
    }
    size_t i = 0;
    const __m128i needle0 = _mm_set1_epi8((char)pat[0]);

    while (i + 16 <= n) {
        __m128i v  = _mm_loadu_si128((const __m128i*)(s + i));
        __m128i eq = _mm_cmpeq_epi8(v, needle0);
        uint32_t mask = (uint32_t)_mm_movemask_epi8(eq);
        while (mask) {
            int pos = __builtin_ctz(mask);
            size_t cand = i + (size_t)pos;
            if (cand + m <= n && memcmp(s + cand, pat, m) == 0) return cand;
            mask &= (mask - 1);
        }
        i += 16;
    }
    for (; i + m <= n; ++i)
        if (s[i] == pat[0] && memcmp(s + i, pat, m) == 0) return i;
    return SIZE_MAX;
}
static inline int csalt_hi_bit32(uint32_t m) { return 31 - __builtin_clz(m); }
static inline size_t simd_last_substr_index_avx(const unsigned char* s, size_t n,
                                                const unsigned char* pat, size_t m) {
    if (m == 0) return n;
    if (m == 1) { for (size_t i = n; i-- > 0; ) if (s[i] == pat[0]) return i; return SIZE_MAX; }

    size_t i = 0, last = SIZE_MAX;
    const __m128i needle0 = _mm_set1_epi8((char)pat[0]);

    while (i + 16 <= n) {
        __m128i v  = _mm_loadu_si128((const __m128i*)(s + i));
        __m128i eq = _mm_cmpeq_epi8(v, needle0);
        uint32_t mask = (uint32_t)_mm_movemask_epi8(eq);

        while (mask) {
            int pos = csalt_hi_bit32(mask);             /* highest set bit in this chunk */
            size_t cand = i + (size_t)pos;
            if (cand + m <= n && memcmp(s + cand, pat, m) == 0) { last = cand; break; }
            mask &= (1u << pos) - 1;                    /* drop bits >= pos; keep lower */
        }
        i += 16;
    }

    /* tail: [i, n-m] descending — return first found; else prior 'last' */
    for (size_t j = (n >= m ? n - m : 0); j + 1 > i; --j)
        if (s[j] == pat[0] && memcmp(s + j, pat, m) == 0) return j;

    return last;
}

static inline int csalt_hi_bit32_avx(uint32_t m) { return 31 - __builtin_clz(m); }

static inline size_t simd_last_substr_index_avx(const unsigned char* s, size_t n,
                                                const unsigned char* pat, size_t m) {
    if (m == 0) return n;
    if (m == 1) { for (size_t i=n; i-- > 0; ) if (s[i]==pat[0]) return i; return SIZE_MAX; }
    if (n < m) return SIZE_MAX;

    size_t i = 0, last = SIZE_MAX;
    const __m128i b0 = _mm_set1_epi8((char)pat[0]);

    while (i + 16 <= n) {
        __m128i v  = _mm_loadu_si128((const __m128i*)(s + i));
        __m128i eq = _mm_cmpeq_epi8(v, b0);
        uint32_t mask = (uint32_t)_mm_movemask_epi8(eq);
        while (mask) {
            int pos = csalt_hi_bit32_avx(mask);
            size_t cand = i + (size_t)pos;
            if (cand + m <= n && memcmp(s + cand, pat, m) == 0) { last = cand; break; }
            mask &= (1u << pos) - 1u;
        }
        i += 16;
    }

    for (size_t j = (n - m); j + 1 > i; --j)
        if (s[j] == pat[0] && memcmp(s + j, pat, m) == 0) return j;

    return last;
}
static inline size_t simd_token_count_avx(const char* s, size_t n,
                                          const char* delim, size_t dlen)
{
    size_t i = 0, count = 0;
    uint32_t prev_is_delim = 1;

    for (; i + 16 <= n; i += 16) {
        __m128i v = _mm_loadu_si128((const __m128i*)(s + i));
        __m128i m = _mm_setzero_si128();
        for (size_t j = 0; j < dlen; ++j) {
            __m128i dj = _mm_set1_epi8((char)delim[j]);
            m = _mm_or_si128(m, _mm_cmpeq_epi8(v, dj));
        }
        uint32_t dm = (uint32_t)_mm_movemask_epi8(m);
        uint32_t non = ~dm & 0xFFFFu;
        uint32_t starts = non & ((dm << 1) | (prev_is_delim & 1));
        count += (size_t)__builtin_popcount(starts);
        prev_is_delim = (dm >> 15) & 1u;
    }

    uint8_t lut[256]; memset(lut, 0, sizeof(lut));
    for (const unsigned char* p = (const unsigned char*)delim; *p; ++p) lut[*p] = 1;
    bool in_token = !prev_is_delim;
    for (; i < n; ++i) {
        const bool is_delim = lut[(unsigned char)s[i]] != 0;
        if (!is_delim) { if (!in_token) { ++count; in_token = true; } }
        else in_token = false;
    }
    return count;
}

#endif /* CSALT_SIMD_AVX_CHAR_INL */

