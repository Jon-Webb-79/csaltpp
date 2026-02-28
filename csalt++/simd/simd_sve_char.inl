/* simd_sve_char.inl
   SVE helpers for last-occurrence search in bytes (VL-agnostic).
   Requires: <arm_sve.h>
*/
#ifndef CSALT_SIMD_SVE_CHAR_INL
#define CSALT_SIMD_SVE_CHAR_INL

#include <stddef.h>
#include <stdint.h>
#include <arm_sve.h>
#include <string.h>
// ================================================================================ 
// ================================================================================ 

static inline size_t simd_first_diff_u8(const uint8_t* a,
                                        const uint8_t* b,
                                        size_t n) {
    if (n == 0u) { return 0u; }
    if ((a == NULL) || (b == NULL)) { return 0u; }

    size_t i = 0u;

    while (i < n) {
        /* Active lanes for remaining bytes */
        const svbool_t p = svwhilelt_b8((uint64_t)i, (uint64_t)n);

        /* Load only active lanes */
        const svuint8_t va = svld1_u8(p, a + i);
        const svuint8_t vb = svld1_u8(p, b + i);

        /* eq lanes are true where equal (only under p) */
        const svbool_t eq = svcmpeq_u8(p, va, vb);

        /* If all active lanes are equal, continue */
        if (svptest_all(p, eq)) {
            /* advance by the number of active lanes this iteration */
            i += (size_t)svcntp_b8(svptrue_b8(), p);
            continue;
        }

        /* There is a mismatch inside active lanes.
           Build a predicate that is true from first mismatch onward, then count
           lanes BEFORE that mismatch. */
        const svbool_t mism = svnot_z(p, eq);         /* true where mismatch */
        const svbool_t after_first = svbrkb_z(p, mism); /* true from first mism onwards */
        const svbool_t before_first = svnot_z(p, after_first); /* true before first mism */

        /* Count bytes before first mismatch among active lanes */
        const size_t off = (size_t)svcntp_b8(svptrue_b8(), before_first);
        return i + off;
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

/* Find first true lane in predicate for up to 'lanes' bytes. Returns lanes if none. */
static inline size_t csalt_pred_first_true_u8(svbool_t pg, size_t lanes)
{
    for (size_t j = 0u; j < lanes; ++j) {
        /* svptest_any checks any active lane true under a mask; we need per-lane.
           Use a single-lane mask: svwhilelt_b8(j, j+1). */
        svbool_t one = svwhilelt_b8((uint64_t)j, (uint64_t)(j + 1u));
        if (svptest_any(svptrue_b8(), svand_b_z(svptrue_b8(), pg, one))) {
            return j;
        }
    }
    return lanes;
}

/* Find last true lane in predicate for up to 'lanes' bytes. Returns lanes if none. */
static inline size_t csalt_pred_last_true_u8(svbool_t pg, size_t lanes)
{
    for (size_t jj = 0u; jj < lanes; ++jj) {
        size_t j = (lanes - 1u) - jj;
        svbool_t one = svwhilelt_b8((uint64_t)j, (uint64_t)(j + 1u));
        if (svptest_any(svptrue_b8(), svand_b_z(svptrue_b8(), pg, one))) {
            return j;
        }
    }
    return lanes;
}

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
    const svuint8_t vfirst = svdup_u8(first);

    const size_t last_start = hay_len - needle_len;

    /* Vector length in bytes (runtime) */
    const size_t VL = svcntb();

    /* ================= FORWARD ================= */
    if (dir == FORWARD) {
        size_t i = 0u;

        while (i <= last_start) {
            if ((i + VL) <= hay_len) {
                svbool_t pg = svptrue_b8();
                svuint8_t v = svld1_u8(pg, hay + i);
                svbool_t eq = svcmpeq_u8(pg, v, vfirst);

                /* Scan matches within this vector */
                while (svptest_any(pg, eq)) {
                    size_t bit = csalt_pred_first_true_u8(eq, VL);
                    if (bit >= VL) { break; }

                    size_t pos = i + bit;
                    if (pos <= last_start) {
                        if (needle_len == 1u) { return pos; }
                        if (memcmp(hay + pos + 1u, needle + 1u, needle_len - 1u) == 0) {
                            return pos;
                        }
                    }

                    /* Clear this lane from eq */
                    svbool_t one = svwhilelt_b8((uint64_t)bit, (uint64_t)(bit + 1u));
                    eq = svand_b_z(pg, eq, svnot_b_z(pg, one));
                }

                i += VL;
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
            /* Choose a block that ends at i (inclusive) and is VL bytes long when possible. */
            size_t block_start = (i + 1u >= VL) ? (i + 1u - VL) : 0u;

            /* Load block_start..block_start+VL-1 safely (only if within hay_len) */
            if ((block_start + VL) <= hay_len) {
                svbool_t pg = svptrue_b8();
                svuint8_t v = svld1_u8(pg, hay + block_start);
                svbool_t eq = svcmpeq_u8(pg, v, vfirst);

                /* Mask out positions > max_pos within this block */
                {
                    size_t max_pos = (i < last_start) ? i : last_start;
                    size_t max_in_block = max_pos - block_start; /* 0..VL-1 */
                    if (max_in_block + 1u < VL) {
                        /* lanes [0, max_in_block] kept, [max_in_block+1, VL-1] cleared */
                        svbool_t keep = svwhilele_b8((uint64_t)0, (uint64_t)max_in_block);
                        eq = svand_b_z(pg, eq, keep);
                    }
                }

                while (svptest_any(pg, eq)) {
                    size_t bit = csalt_pred_last_true_u8(eq, VL);
                    if (bit >= VL) { break; }

                    size_t pos = block_start + bit;
                    if (needle_len == 1u) { return pos; }
                    if (memcmp(hay + pos + 1u, needle + 1u, needle_len - 1u) == 0) {
                        return pos;
                    }

                    svbool_t one = svwhilelt_b8((uint64_t)bit, (uint64_t)(bit + 1u));
                    eq = svand_b_z(pg, eq, svnot_b_z(pg, one));
                }
            } else {
                /* If we can't safely vector-load, break to scalar reverse tail */
                break;
            }

            if (block_start == 0u) { break; }
            i = block_start - 1u;
        }

        /* Scalar reverse tail over remaining starts */
        for (size_t pos = (i <= last_start ? i : last_start) + 1u; pos-- > 0u; ) {
            if (pos > last_start) { continue; }
            if (hay[pos] != first) { continue; }
            if (needle_len == 1u) { return pos; }
            if (memcmp(hay + pos + 1u, needle + 1u, needle_len - 1u) == 0) {
                return pos;
            }
            if (pos == 0u) { break; }
        }

        return SIZE_MAX;
    }
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

    size_t count = 0u;
    uint32_t prev_is_delim = 1u; /* start-of-window boundary */

    size_t i = 0u;
    while (i < n) {
        svbool_t pg = svwhilelt_b8(i, n);
        svuint8_t v = svld1_u8(pg, s + i);

        /* Build delimiter predicate dm: lane true where v == delim[j] */
        svbool_t dm = svdup_b8(false);
        for (size_t j = 0; j < dlen; ++j) {
            svuint8_t dj = svdup_n_u8((uint8_t)(unsigned char)delim[j]);
            dm = svorr_b_z(pg, dm, svcmpeq_u8(pg, v, dj));
        }

        /* non-delim predicate */
        svbool_t non = svnot_b_z(pg, dm);

        /* Token starts = non && previous is delim.
           We compute starts lane-by-lane using predicate iteration to avoid
           relying on optional mask-to-integer intrinsics. */
        svbool_t starts = svdup_b8(false);

        /* Iterate active lanes in increasing index order */
        svbool_t it = pg;
        bool prev = (prev_is_delim != 0u);

        while (svptest_any(svptrue_b8(), it)) {
            /* index of next active lane */
            uint64_t idx = svclastb_u64(svptrue_b8(), svcompact_u64(it, svindex_u64(0, 1)));

            /* Read lane flags using svpextract-like pattern:
               Make a predicate for that single lane. */
            svbool_t lane = svdup_b8(false);
            lane = svsetq_lane_b8(true, lane, (uint32_t)idx);

            bool is_non = svptest_any(svptrue_b8(), svand_b_z(pg, non, lane));
            bool is_del = svptest_any(svptrue_b8(), svand_b_z(pg, dm, lane));

            if (is_non && prev) {
                starts = svorr_b_z(pg, starts, lane);
            }

            /* update prev based on current lane */
            prev = is_del;

            /* clear this lane from iterator */
            it = svand_b_z(pg, it, svnot_b_z(pg, lane));
        }

        /* Count starts */
        count += (size_t)svcntp_b8(pg, starts);

        /* Carry last-lane delimiter state to next chunk:
           Determine last active lane in this chunk and query dm there. */
        {
            /* last active byte lane index is (i + active_count - 1) relative, but easier:
               build reverse index and find last true lane. */
            uint32_t active = (uint32_t)svcntp_b8(pg, pg);
            if (active != 0u) {
                uint32_t last_lane = active - 1u;
                svbool_t lastp = svdup_b8(false);
                lastp = svsetq_lane_b8(true, lastp, last_lane);
                prev_is_delim = svptest_any(svptrue_b8(), svand_b_z(pg, dm, lastp)) ? 1u : 0u;
            }
        }

        i += (size_t)svcntb();
    }

    return count;
}
// -------------------------------------------------------------------------------- 

void simd_ascii_upper_u8(uint8_t* p, size_t n)
{
    if ((p == NULL) || (n == 0u)) return;

    size_t i = 0u;

    const svuint8_t vlo  = svdup_n_u8((uint8_t)'a');
    const svuint8_t vhi  = svdup_n_u8((uint8_t)'z');
    const svuint8_t vsub = svdup_n_u8(0x20u);

    while (i < n) {
        svbool_t pg = svwhilelt_b8(i, n);

        svuint8_t v = svld1_u8(pg, p + i);

        /* mask = (v >= 'a') & (v <= 'z')  (unsigned compares) */
        svbool_t ge_lo = svcmpge_u8(pg, v, vlo);
        svbool_t le_hi = svcmple_u8(pg, v, vhi);
        svbool_t m     = svand_b_z(pg, ge_lo, le_hi);

        /* upper = v - 0x20 on masked lanes */
        svuint8_t upper = svsub_u8_z(pg, v, vsub);

        /* select: m ? upper : v */
        svuint8_t out = svsel_u8(m, upper, v);

        svst1_u8(pg, p + i, out);

        i += (size_t)svcntb();
    }
}

void simd_ascii_lower_u8(uint8_t* p, size_t n)
{
    if ((p == NULL) || (n == 0u)) return;

    size_t i = 0u;

    const svuint8_t vlo  = svdup_n_u8((uint8_t)'A');
    const svuint8_t vhi  = svdup_n_u8((uint8_t)'Z');
    const svuint8_t vadd = svdup_n_u8(0x20u);

    while (i < n) {
        svbool_t pg = svwhilelt_b8(i, n);

        svuint8_t v = svld1_u8(pg, p + i);

        svbool_t ge_lo = svcmpge_u8(pg, v, vlo);
        svbool_t le_hi = svcmple_u8(pg, v, vhi);
        svbool_t m     = svand_b_z(pg, ge_lo, le_hi);

        svuint8_t lower = svadd_u8_z(pg, v, vadd);
        svuint8_t out   = svsel_u8(m, lower, v);

        svst1_u8(pg, p + i, out);

        i += (size_t)svcntb();
    }
}
// ================================================================================ 
// ================================================================================ 
#endif /* CSALT_SIMD_SVE_CHAR_INL */

