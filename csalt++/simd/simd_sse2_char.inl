/* simd_sse2_char.inl
   SSE2 helpers for last-occurrence search in bytes.
   Requires: <emmintrin.h>
*/
#ifndef CSALT_SIMD_SSE2_CHAR_INL
#define CSALT_SIMD_SSE2_CHAR_INL

#if !defined(__SSE2__)
  #error "simd_sse2_char.inl requires __SSE2__"
#endif

#include <stddef.h>
#include <stdint.h>
#include <emmintrin.h>
// ================================================================================ 
// ================================================================================ 

/* Return index of least-significant 1-bit in a 16-bit value.
   Precondition: x != 0.
   Result range: 0..15
*/
static inline unsigned simd_ctz16_(uint16_t x) {
    /* Portable, MISRA-friendly: bounded loop */
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
            uint16_t mism = (uint16_t)((uint16_t)0xFFFFu ^ m); /* 1 where different */
            unsigned bit  = simd_ctz16_(mism);                 /* 0..15 */
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

/* ---------- portable bit scan (no compiler builtins) ---------- */
/* Precondition: m != 0 */
static inline unsigned csalt_ctz16(uint16_t m)
{
    unsigned n = 0u;
    while ((m & 1u) == 0u) { m = (uint16_t)(m >> 1u); ++n; }
    return n; /* 0..15 */
}

/* Precondition: m != 0 */
static inline unsigned csalt_clz16(uint16_t m)
{
    unsigned n = 0u;
    uint16_t bit = (uint16_t)0x8000u;
    while ((m & bit) == 0u) { bit = (uint16_t)(bit >> 1u); ++n; }
    return n; /* 0..15 */
}

static inline unsigned csalt_first_bit16(uint16_t m) { return csalt_ctz16(m); }
static inline unsigned csalt_last_bit16 (uint16_t m) { return 15u - csalt_clz16(m); }

/* ---------- unified public function ---------- */
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
            if ((i + 16u) <= hay_len) {
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
            } else {
                break;
            }
        }

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
            size_t block_start = (i >= 15u) ? (i - 15u) : 0u;

            __m128i v  = _mm_loadu_si128((const __m128i*)(const void*)(hay + block_start));
            __m128i eq = _mm_cmpeq_epi8(v, vfirst);
            uint16_t mask = (uint16_t)_mm_movemask_epi8(eq);

            /* Keep only candidates within [block_start, max_pos] */
            {
                size_t block_end = block_start + 15u;
                size_t max_pos   = (i < last_start) ? i : last_start;

                if (max_pos < block_end) {
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

static inline __m128i blendv_epi8_sse2(__m128i a, __m128i b, __m128i mask)
{
    /* mask: 0xFF -> choose b, 0x00 -> choose a */
    return _mm_or_si128(_mm_andnot_si128(mask, a),
                        _mm_and_si128(mask, b));
}

static inline __m128i ascii_range_mask_sse2(__m128i x, __m128i lo, __m128i hi)
{
    const __m128i one = _mm_set1_epi8(1);

    /* x > lo-1 */
    __m128i ge_lo = _mm_cmpgt_epi8(x, _mm_sub_epi8(lo, one));
    /* hi+1 > x  <=> x < hi+1 */
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

        __m128i mask  = ascii_range_mask_sse2(v, lo, hi);
        __m128i upper = _mm_sub_epi8(v, sub);
        __m128i out   = blendv_epi8_sse2(v, upper, mask);

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

        __m128i mask  = ascii_range_mask_sse2(v, lo, hi);
        __m128i lower = _mm_add_epi8(v, add);
        __m128i out   = blendv_epi8_sse2(v, lower, mask);

        _mm_storeu_si128((__m128i*)(void*)(p + i), out);
    }

    for (; i < n; ++i) {
        uint8_t c = p[i];
        if ((c >= (uint8_t)'A') && (c <= (uint8_t)'Z')) p[i] = (uint8_t)(c + 0x20u);
    }
}
// ================================================================================ 
// ================================================================================ 


static inline int csalt_highbit_u32(uint32_t m) { return 31 - __builtin_clz(m); }

static inline size_t simd_last_index_u8_sse2(const unsigned char* s, size_t n, unsigned char c) {
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

static inline size_t simd_first_substr_index_sse2(const unsigned char* s, size_t n,
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
            mask &= mask - 1;
        }
        i += 16;
    }
    for (; i + m <= n; ++i) {
        if (s[i] == pat[0] && memcmp(s + i, pat, m) == 0) return i;
    }
    return SIZE_MAX;
}
static inline size_t simd_last_substr_index_sse2(const unsigned char* s, size_t n,
                                                   const unsigned char* pat, size_t m) {
      if (m == 0) return n;
      if (m == 1) {
          for (size_t i = n; i-- > 0; ) if (s[i] == pat[0]) return i;
          return SIZE_MAX;
      }
      size_t i = 0, last = SIZE_MAX;
      const __m128i needle0 = _mm_set1_epi8((char)pat[0]);

      while (i + 16 <= n) {
          __m128i v  = _mm_loadu_si128((const __m128i*)(s + i));
          __m128i eq = _mm_cmpeq_epi8(v, needle0);
          uint32_t mask = (uint32_t)_mm_movemask_epi8(eq);
          while (mask) {
              int pos = hi_bit32(mask);
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

static inline int csalt_hi_bit32_sse2(uint32_t m) { return 31 - __builtin_clz(m); }

static inline size_t simd_last_substr_index_sse2(const unsigned char* s, size_t n,
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
            int pos = csalt_hi_bit32_sse2(mask);
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
#endif /* CSALT_SIMD_SSE2_CHAR_INL */

