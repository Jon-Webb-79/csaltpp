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
// ================================================================================ 
// ================================================================================ 
static inline int csalt_highbit_u32(uint32_t m) { return 31 - __builtin_clz(m); }

static inline size_t simd_last_index_u8_sse41(const unsigned char* s, size_t n, unsigned char c) {
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

static inline size_t simd_first_substr_index_sse41(const unsigned char* s, size_t n,
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
static inline int csalt_hi_bit32_sse41(uint32_t m) { return 31 - __builtin_clz(m); }

static inline size_t simd_last_substr_index_sse41(const unsigned char* s, size_t n,
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
            int pos = csalt_hi_bit32_sse41(mask);
            size_t cand = i + (size_t)pos;
            if (cand + m <= n && memcmp(s + cand, pat, m) == 0) { last = cand; break; }
            mask &= (1u << pos) - 1;
        }
        i += 16;
    }
    for (size_t j = (n >= m ? n - m : 0); j + 1 > i; --j)
        if (s[j] == pat[0] && memcmp(s + j, pat, m) == 0) return j;

    return last;
}
static inline size_t simd_token_count_sse41(const char* s, size_t n,
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

#endif /* CSALT_SIMD_SSE41_CHAR_INL */

