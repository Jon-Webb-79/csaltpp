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
    const uint8x16_t vfirst = vdupq_n_u8(first);

    const size_t last_start = hay_len - needle_len;

    /* ================= FORWARD ================= */
    if (dir == FORWARD) {
        size_t i = 0u;

        while (i <= last_start) {
            if ((i + 16u) <= hay_len) {
                uint8x16_t v  = vld1q_u8(hay + i);
                uint8x16_t eq = vceqq_u8(v, vfirst);

                uint8_t lanes[16];
                vst1q_u8(lanes, eq);

                for (unsigned lane = 0u; lane < 16u; ++lane) {
                    if (lanes[lane] == 0u) { continue; }

                    size_t pos = i + (size_t)lane;
                    if (pos > last_start) { break; }

                    if (needle_len == 1u) { return pos; }
                    if (memcmp(hay + pos + 1u, needle + 1u, needle_len - 1u) == 0) {
                        return pos;
                    }
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

            uint8x16_t v  = vld1q_u8(hay + block_start);
            uint8x16_t eq = vceqq_u8(v, vfirst);

            uint8_t lanes[16];
            vst1q_u8(lanes, eq);

            size_t const max_pos   = (i < last_start) ? i : last_start;
            unsigned const max_lane = (unsigned)(max_pos - block_start); /* 0..15 */

            for (unsigned lane = max_lane + 1u; lane-- > 0u; ) {
                if (lanes[lane] == 0u) {
                    if (lane == 0u) { break; }
                    continue;
                }

                size_t pos = block_start + (size_t)lane;

                if (needle_len == 1u) { return pos; }
                if (memcmp(hay + pos + 1u, needle + 1u, needle_len - 1u) == 0) {
                    return pos;
                }

                if (lane == 0u) { break; }
            }

            if (block_start == 0u) { break; }
            i = block_start - 1u;
        }
    }

    return SIZE_MAX;
}
// -------------------------------------------------------------------------------- 

size_t simd_token_count_u8(const uint8_t* s, size_t n,
                           const char* delim, size_t dlen)
{
    if ((s == NULL) || (delim == NULL)) return SIZE_MAX;
    if (n == 0u) return 0u;
    if (dlen == 0u) return 1u; /* no delimiters => whole window is one token */

    /* LUT for scalar tail */
    uint8_t lut[256];
    memset(lut, 0, sizeof(lut));
    for (size_t j = 0; j < dlen; ++j) {
        lut[(unsigned char)delim[j]] = 1u;
    }

    size_t i = 0u;
    size_t count = 0u;
    uint32_t prev_is_delim = 1u;

    /* Constants for NEON movemask (8-lane halves) */
    const uint8x16_t bitweights =
        (uint8x16_t){ 1u,2u,4u,8u,16u,32u,64u,128u,  1u,2u,4u,8u,16u,32u,64u,128u };

    for (; i + 16u <= n; i += 16u) {
        uint8x16_t v = vld1q_u8(s + i);

        /* Build per-lane delimiter mask as 0xFF where delim, else 0x00 */
        uint8x16_t m = vdupq_n_u8(0u);
        for (size_t j = 0; j < dlen; ++j) {
            uint8x16_t dj = vdupq_n_u8((uint8_t)(unsigned char)delim[j]);
            m = vorrq_u8(m, vceqq_u8(v, dj));
        }

        /* movemask16: compress top-bit of each lane to a 16-bit mask
           Here we use bitweights + horizontal pairwise sums to produce:
             low 8 bits in lane0 of sum64[0], high 8 bits in lane0 of sum64[1]. */
        uint8x16_t weighted = vandq_u8(m, bitweights);
        uint16x8_t sum16 = vpaddlq_u8(weighted);
        uint32x4_t sum32 = vpaddlq_u16(sum16);
        uint64x2_t sum64 = vpaddlq_u32(sum32);

        uint64_t lo = vgetq_lane_u64(sum64, 0);
        uint64_t hi = vgetq_lane_u64(sum64, 1);

        uint32_t dm = (uint32_t)((lo & 0xFFu) | ((hi & 0xFFu) << 8)); /* 16-bit used */

        uint32_t non = (~dm) & 0xFFFFu;
        uint32_t starts = non & ((dm << 1) | (prev_is_delim & 1u));

        count += (size_t)__builtin_popcount(starts);
        prev_is_delim = (dm >> 15) & 1u;
    }

    /* Scalar tail */
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
#endif /* CSALT_SIMD_NEON_CHAR_INL */

