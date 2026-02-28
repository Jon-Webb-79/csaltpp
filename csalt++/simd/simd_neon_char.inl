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
// -------------------------------------------------------------------------------- 

void simd_ascii_upper_u8(uint8_t* p, size_t n)
{
    if ((p == NULL) || (n == 0u)) return;

    size_t i = 0u;

    const uint8x16_t lo  = vdupq_n_u8((uint8_t)'a');
    const uint8x16_t hi  = vdupq_n_u8((uint8_t)'z');
    const uint8x16_t sub = vdupq_n_u8(0x20u);

    for (; i + 16u <= n; i += 16u) {
        uint8x16_t v = vld1q_u8(p + i);

        /* mask = (v >= 'a') & (v <= 'z') */
        uint8x16_t ge_lo = vcgeq_u8(v, lo);
        uint8x16_t le_hi = vcleq_u8(v, hi);
        uint8x16_t mask  = vandq_u8(ge_lo, le_hi);

        uint8x16_t upper = vsubq_u8(v, sub);

        /* select: mask ? upper : v */
        uint8x16_t out = vbslq_u8(mask, upper, v);

        vst1q_u8(p + i, out);
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

    const uint8x16_t lo  = vdupq_n_u8((uint8_t)'A');
    const uint8x16_t hi  = vdupq_n_u8((uint8_t)'Z');
    const uint8x16_t add = vdupq_n_u8(0x20u);

    for (; i + 16u <= n; i += 16u) {
        uint8x16_t v = vld1q_u8(p + i);

        uint8x16_t ge_lo = vcgeq_u8(v, lo);
        uint8x16_t le_hi = vcleq_u8(v, hi);
        uint8x16_t mask  = vandq_u8(ge_lo, le_hi);

        uint8x16_t lower = vaddq_u8(v, add);
        uint8x16_t out   = vbslq_u8(mask, lower, v);

        vst1q_u8(p + i, out);
    }

    for (; i < n; ++i) {
        uint8_t c = p[i];
        if ((c >= (uint8_t)'A') && (c <= (uint8_t)'Z')) p[i] = (uint8_t)(c + 0x20u);
    }
}
// ================================================================================ 
// ================================================================================ 
#endif /* CSALT_SIMD_NEON_CHAR_INL */

