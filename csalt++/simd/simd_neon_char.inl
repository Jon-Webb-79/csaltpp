/* simd_neon_char.inl
   NEON helpers for last-occurrence search in bytes.
   Requires: <arm_neon.h>
*/
#ifndef CSALT_SIMD_NEON_CHAR_INL
#define CSALT_SIMD_NEON_CHAR_INL

#include <stddef.h>
#include <stdint.h>
#include <arm_neon.h>
// ================================================================================ 
// ================================================================================ 

static inline size_t simd_first_diff_u8(const uint8_t* a,
                                        const uint8_t* b,
                                        size_t n) {
    if (n == 0u) { return 0u; }
    if ((a == NULL) || (b == NULL)) { return 0u; }

    size_t i = 0u;

    while ((i + 16u) <= n) {
        /* Load 16 bytes from each */
        const uint8x16_t va = vld1q_u8(a + i);
        const uint8x16_t vb = vld1q_u8(b + i);

        /* Byte-wise equality: lanes are 0xFF if equal, 0x00 if not */
        const uint8x16_t veq = vceqq_u8(va, vb);

        /* Check if ALL lanes are 0xFF without needing AArch64-only reductions.
           Reinterpret as two 64-bit lanes and check both == UINT64_MAX. */
        const uint64x2_t eq64 = vreinterpretq_u64_u8(veq);

        if ((vgetq_lane_u64(eq64, 0) != UINT64_MAX) ||
            (vgetq_lane_u64(eq64, 1) != UINT64_MAX))
        {
            /* There is at least one mismatch in this 16-byte block.
               Find the first mismatch safely with a tiny scalar scan. */
            for (size_t j = 0u; j < 16u; ++j) {
                if (a[i + j] != b[i + j]) {
                    return i + j;
                }
            }

            /* Defensive: should not happen if the check above detects mismatch */
            return i;
        }

        i += 16u;
    }

    /* Scalar tail (0..15 bytes) */
    for (; i < n; ++i) {
        if (a[i] != b[i]) { return i; }
    }

    return n;
}
// ================================================================================ 
// ================================================================================ 

/* Build a 16-bit movemask from MSBs of each byte */
static inline uint32_t csalt_neon_movemask_u8(uint8x16_t v) {
    /* MSB -> bit0 per lane */
    uint8x16_t msb = vshrq_n_u8(v, 7);
    /* Multiply by powers of two and horizontally add to a 16-bit mask */
    static const uint8_t kbits_data[16] = {
        1,2,4,8,16,32,64,128, 1,2,4,8,16,32,64,128
    };
    uint8x16_t kbits = vld1q_u8(kbits_data);
    uint8x16_t masked = vandq_u8(msb, kbits);

    uint16x8_t sum16 = vpaddlq_u8(masked);   /* 16 -> 8 */
    uint32x4_t sum32 = vpaddlq_u16(sum16);   /* 8  -> 4 */
    uint64x2_t sum64 = vpaddlq_u32(sum32);   /* 4  -> 2 */
    uint64_t lo = vgetq_lane_u64(sum64, 0);
    uint64_t hi = vgetq_lane_u64(sum64, 1);
    return (uint32_t)(lo | (hi << 8));       /* 16-bit mask in low bits */
}

static inline int csalt_highbit_u32(uint32_t m) { return 31 - __builtin_clz(m); }

static inline size_t simd_last_index_u8_neon(const unsigned char* s, size_t n, unsigned char c) {
    size_t i = 0, last = SIZE_MAX;
    const uint8x16_t needle = vdupq_n_u8(c);

    while (i + 16 <= n) {
        uint8x16_t v  = vld1q_u8(s + i);
        uint8x16_t eq = vceqq_u8(v, needle);
        uint32_t mask = csalt_neon_movemask_u8(eq);
        if (mask) last = i + (size_t)csalt_highbit_u32(mask);
        i += 16;
    }
    for (size_t j = n; j-- > i; ) if (s[j] == c) return j;
    return last;
}

static inline uint32_t neon_movemask_u8(uint8x16_t v) {
    // MSB of each lane -> 16-bit mask
    uint8x16_t msb = vshrq_n_u8(v, 7);
    static const uint8_t bits_arr[16] = {1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128};
    uint8x16_t bits = vld1q_u8(bits_arr);
    uint8x16_t andv = vandq_u8(msb, bits);
    uint16x8_t s16  = vpaddlq_u8(andv);
    uint32x4_t s32  = vpaddlq_u16(s16);
    uint64x2_t s64  = vpaddlq_u32(s32);
    uint64_t lo = vgetq_lane_u64(s64, 0);
    uint64_t hi = vgetq_lane_u64(s64, 1);
    return (uint32_t)(lo | (hi << 8));   // 16-bit mask
}

static inline size_t simd_first_substr_index_neon(const unsigned char* s, size_t n,
                                                  const unsigned char* pat, size_t m) {
    if (m == 0) return 0;
    if (m == 1) {
        const void* p = memchr(s, pat[0], n);
        return p ? (size_t)((const unsigned char*)p - s) : SIZE_MAX;
    }

    size_t i = 0;
    const uint8x16_t needle0 = vdupq_n_u8(pat[0]);

    while (i + 16 <= n) {
        uint8x16_t v  = vld1q_u8(s + i);
        uint8x16_t eq = vceqq_u8(v, needle0);
        uint32_t mask = neon_movemask_u8(eq);

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
static inline size_t simd_last_substr_index_neon(const unsigned char* s, size_t n,
                                                   const unsigned char* pat, size_t m) {
      if (m == 0) return n;
      if (m == 1) {
          for (size_t i = n; i-- > 0; ) if (s[i] == pat[0]) return i;
          return SIZE_MAX;
      }
      size_t i = 0, last = SIZE_MAX;
      const uint8x16_t needle0 = vdupq_n_u8(pat[0]);

      while (i + 16 <= n) {
          uint8x16_t v  = vld1q_u8(s + i);
          uint8x16_t eq = vceqq_u8(v, needle0);
          uint32_t mask = neon_movemask_u8(eq);
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

static inline uint32_t csalt_neon_movemask_u8(uint8x16_t v) {
    /* Convert top bits to a 16-bit mask. */
    static const uint8_t shift_tbl[16] = {1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128};
    uint8x16_t msb  = vshrq_n_u8(v, 7);
    uint8x16_t bits = vld1q_u8(shift_tbl);
    uint8x16_t andv = vandq_u8(msb, bits);
    uint16x8_t s16  = vpaddlq_u8(andv);
    uint32x4_t s32  = vpaddlq_u16(s16);
    uint64x2_t s64  = vpaddlq_u32(s32);
    uint64_t lo = vgetq_lane_u64(s64, 0);
    uint64_t hi = vgetq_lane_u64(s64, 1);
    return (uint32_t)(lo | (hi << 8));
}

static inline int csalt_hi_bit32_neon(uint32_t m) { return 31 - __builtin_clz(m); }

static inline size_t simd_last_substr_index_neon(const unsigned char* s, size_t n,
                                                 const unsigned char* pat, size_t m) {
    if (m == 0) return n;
    if (m == 1) { for (size_t i=n; i-- > 0; ) if (s[i]==pat[0]) return i; return SIZE_MAX; }
    if (n < m) return SIZE_MAX;

    size_t i = 0, last = SIZE_MAX;
    const uint8x16_t b0 = vdupq_n_u8(pat[0]);

    while (i + 16 <= n) {
        uint8x16_t v  = vld1q_u8(s + i);
        uint8x16_t eq = vceqq_u8(v, b0);
        uint32_t mask = csalt_neon_movemask_u8(eq);
        while (mask) {
            int pos = csalt_hi_bit32_neon(mask);
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
static inline size_t simd_token_count_neon(const char* s, size_t n,
                                           const char* delim, size_t dlen)
{
    size_t i = 0, count = 0;
    uint8_t prev_last = 0xFF; /* virtual delimiter before start */

    for (; i + 16 <= n; i += 16) {
        uint8x16_t v = vld1q_u8((const uint8_t*)(s + i));
        uint8x16_t dm = vceqq_u8(v, vdupq_n_u8((uint8_t)delim[0]));
        for (size_t j = 1; j < dlen; ++j) {
            dm = vorrq_u8(dm, vceqq_u8(v, vdupq_n_u8((uint8_t)delim[j])));
        }
        uint8x16_t non = vmvnq_u8(dm);
        uint8x16_t prev_fill = vdupq_n_u8(prev_last);
        /* previous-delim per position: [prev_last, dm[0], dm[1], ... dm[14]] */
        uint8x16_t prev = vextq_u8(prev_fill, dm, 15);
        uint8x16_t starts = vandq_u8(non, prev);

        /* convert 0xFF -> 1, 0x00 -> 0, then horizontal sum */
        uint8x16_t ones = vshrq_n_u8(starts, 7);
        uint16x8_t  s16  = vpaddlq_u8(ones);
        uint32x4_t  s32  = vpaddlq_u16(s16);
        uint64x2_t  s64  = vpaddlq_u32(s32);
        count += (size_t)(vgetq_lane_u64(s64, 0) + vgetq_lane_u64(s64, 1));

        /* carry last-lane delimiter flag (0xFF or 0x00) */
        prev_last = (uint8_t)(vgetq_lane_u8(dm, 15) ? 0xFF : 0x00);
    }

    /* scalar tail */
    uint8_t lut[256]; memset(lut, 0, sizeof(lut));
    for (const unsigned char* p = (const unsigned char*)delim; *p; ++p) lut[*p] = 1;
    bool in_token = (prev_last == 0x00);
    for (; i < n; ++i) {
        const bool is_delim = lut[(unsigned char)s[i]] != 0;
        if (!is_delim) { if (!in_token) { ++count; in_token = true; } }
        else in_token = false;
    }
    return count;
}


#endif /* CSALT_SIMD_NEON_CHAR_INL */

