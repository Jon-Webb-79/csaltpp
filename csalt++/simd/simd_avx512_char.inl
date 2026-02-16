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
// ================================================================================ 
// ================================================================================ 

static inline int csalt_highbit_u64(uint64_t m) { return 63 - __builtin_clzll(m); }

/* Returns index of last occurrence of c in s[0..n), or SIZE_MAX if not found. */
static inline size_t simd_last_index_u8_avx512bw(const unsigned char* s, size_t n, unsigned char c) {
    size_t i = 0, last = SIZE_MAX;
    const __m512i needle = _mm512_set1_epi8((char)c);

    while (i + 64 <= n) {
        __m512i v = _mm512_loadu_si512((const void*)(s + i));
        __mmask64 k = _mm512_cmpeq_epi8_mask(v, needle);
        if (k) last = i + (size_t)csalt_highbit_u64((uint64_t)k);
        i += 64;
    }
    /* tail */
    for (size_t j = n; j-- > i; ) {
        if (s[j] == c) return j;
    }
    return last;
}

static inline size_t simd_first_substr_index_avx512bw(const unsigned char* s, size_t n,
                                                      const unsigned char* pat, size_t m) {
    if (m == 0) return 0;
    if (m == 1) {
        const void* p = memchr(s, pat[0], n);
        return p ? (size_t)((const unsigned char*)p - s) : SIZE_MAX;
    }
    size_t i = 0;
    const __m512i needle0 = _mm512_set1_epi8((char)pat[0]);

    while (i + 64 <= n) {
        __m512i v = _mm512_loadu_si512((const void*)(s + i));
        __mmask64 k = _mm512_cmpeq_epi8_mask(v, needle0);
        while (k) {
            unsigned long pos = __builtin_ctzll((uint64_t)k);
            size_t cand = i + (size_t)pos;
            if (cand + m <= n && memcmp(s + cand, pat, m) == 0) return cand;
            k &= (k - 1);  /* clear lowest set bit */
        }
        i += 64;
    }
    for (; i + m <= n; ++i)
        if (s[i] == pat[0] && memcmp(s + i, pat, m) == 0) return i;
    return SIZE_MAX;
}
static inline size_t simd_last_substr_index_avx512bw(const unsigned char* s, size_t n,
                                                       const unsigned char* pat, size_t m) {
      if (m == 0) return n;
      if (m == 1) {
          /* reuse a simple reverse char scan */
          for (size_t i = n; i-- > 0; ) if (s[i] == pat[0]) return i;
          return SIZE_MAX;
      }
      size_t i = 0, last = SIZE_MAX;
      const __m512i needle0 = _mm512_set1_epi8((char)pat[0]);

      while (i + 64 <= n) {
          __m512i v = _mm512_loadu_si512((const void*)(s + i));
          __mmask64 k = _mm512_cmpeq_epi8_mask(v, needle0);
          while (k) {
              int pos = 63 - __builtin_clzll((uint64_t)k);      /* highest set bit */
              size_t cand = i + (size_t)pos;
              if (cand + m <= n && memcmp(s + cand, pat, m) == 0) { last = cand; break; }
              k &= (~0ULL >> (64 - pos));                       /* drop bits >= pos */
          }
          i += 64;
      }
      /* tail range [i, n-m] descending – return first found, else earlier chunk 'last' */
      for (size_t j = (n >= m ? n - m : 0); j + 1 > i; --j)
          if (s[j] == pat[0] && memcmp(s + j, pat, m) == 0) return j;
      return last;
}

static inline size_t simd_last_substr_index_avx512bw(const unsigned char* s, size_t n,
                                                     const unsigned char* pat, size_t m) {
    if (m == 0) return n;
    if (m == 1) { for (size_t i=n; i-- > 0; ) if (s[i]==pat[0]) return i; return SIZE_MAX; }
    if (n < m) return SIZE_MAX;

    size_t i = 0, last = SIZE_MAX;
    const __m512i b0 = _mm512_set1_epi8((char)pat[0]);

    while (i + 64 <= n) {
        __m512i v = _mm512_loadu_si512((const void*)(s + i));
        __mmask64 k = _mm512_cmpeq_epi8_mask(v, b0);
        while (k) {
            int pos = 63 - __builtin_clzll((unsigned long long)k); /* highest set bit */
            size_t cand = i + (size_t)pos;
            if (cand + m <= n && memcmp(s + cand, pat, m) == 0) { last = cand; break; }
            k &= ((1ULL << pos) - 1ULL); /* keep lower bits only */
        }
        i += 64;
    }

    for (size_t j = (n - m); j + 1 > i; --j)
        if (s[j] == pat[0] && memcmp(s + j, pat, m) == 0) return j;

    return last;
}
static inline size_t simd_token_count_avx512bw(const char* s, size_t n,
                                               const char* delim, size_t dlen)
{
    size_t i = 0, count = 0;
    /* treat virtual byte before s[0] as delimiter */
    uint64_t prev_is_delim = 1;

    for (; i + 64 <= n; i += 64) {
        __m512i v = _mm512_loadu_si512((const void*)(s + i));
        __mmask64 dm = 0;
        for (size_t j = 0; j < dlen; ++j) {
            __m512i dj = _mm512_set1_epi8((char)delim[j]);
            dm |= _mm512_cmpeq_epi8_mask(v, dj);
        }
        __mmask64 non = ~dm;
        /* starts = non & (dm << 1 | prev) */
        __mmask64 starts = non & ((dm << 1) | (prev_is_delim & 1));
        count += (size_t)_mm_popcnt_u64(starts);
        prev_is_delim = (dm >> 63) & 1u;
    }

    /* scalar tail with LUT */
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


#endif /* CSALT_SIMD_AVX512_CHAR_INL */

