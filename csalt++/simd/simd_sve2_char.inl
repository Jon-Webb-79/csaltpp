/* simd_sve2_char.inl
   SVE2 helpers for last-occurrence search in bytes (VL-agnostic).
   Requires: <arm_sve.h>
*/
#ifndef CSALT_SIMD_SVE2_CHAR_INL
#define CSALT_SIMD_SVE2_CHAR_INL

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
        const svbool_t p = svwhilelt_b8((uint64_t)i, (uint64_t)n);

        const svuint8_t va = svld1_u8(p, a + i);
        const svuint8_t vb = svld1_u8(p, b + i);

        const svbool_t eq = svcmpeq_u8(p, va, vb);

        if (svptest_all(p, eq)) {
            i += (size_t)svcntp_b8(svptrue_b8(), p);
            continue;
        }

        const svbool_t mism = svnot_z(p, eq);
        const svbool_t after_first  = svbrkb_z(p, mism);
        const svbool_t before_first = svnot_z(p, after_first);

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

static inline size_t csalt_pred_first_true_u8(svbool_t pg, size_t lanes)
{
    for (size_t j = 0u; j < lanes; ++j) {
        svbool_t one = svwhilelt_b8((uint64_t)j, (uint64_t)(j + 1u));
        if (svptest_any(svptrue_b8(), svand_b_z(svptrue_b8(), pg, one))) {
            return j;
        }
    }
    return lanes;
}

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
    const size_t VL = svcntb();

    if (dir == FORWARD) {
        size_t i = 0u;

        while (i <= last_start) {
            if ((i + VL) <= hay_len) {
                svbool_t pg = svptrue_b8();
                svuint8_t v = svld1_u8(pg, hay + i);
                svbool_t eq = svcmpeq_u8(pg, v, vfirst);

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

                    svbool_t one = svwhilelt_b8((uint64_t)bit, (uint64_t)(bit + 1u));
                    eq = svand_b_z(pg, eq, svnot_b_z(pg, one));
                }

                i += VL;
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

    /* REVERSE */
    {
        size_t i = last_start;

        while (true) {
            size_t block_start = (i + 1u >= VL) ? (i + 1u - VL) : 0u;

            if ((block_start + VL) <= hay_len) {
                svbool_t pg = svptrue_b8();
                svuint8_t v = svld1_u8(pg, hay + block_start);
                svbool_t eq = svcmpeq_u8(pg, v, vfirst);

                size_t max_pos = (i < last_start) ? i : last_start;
                size_t max_in_block = max_pos - block_start;
                if (max_in_block + 1u < VL) {
                    svbool_t keep = svwhilele_b8((uint64_t)0, (uint64_t)max_in_block);
                    eq = svand_b_z(pg, eq, keep);
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
                break;
            }

            if (block_start == 0u) { break; }
            i = block_start - 1u;
        }

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
    uint32_t prev_is_delim = 1u;

    size_t i = 0u;
    while (i < n) {
        svbool_t pg = svwhilelt_b8(i, n);
        svuint8_t v = svld1_u8(pg, s + i);

        svbool_t dm = svdup_b8(false);
        for (size_t j = 0; j < dlen; ++j) {
            svuint8_t dj = svdup_n_u8((uint8_t)(unsigned char)delim[j]);
            dm = svorr_b_z(pg, dm, svcmpeq_u8(pg, v, dj));
        }

        svbool_t non = svnot_b_z(pg, dm);

        /* same safe predicate-iteration approach */
        svbool_t starts = svdup_b8(false);
        svbool_t it = pg;
        bool prev = (prev_is_delim != 0u);

        while (svptest_any(svptrue_b8(), it)) {
            uint64_t idx = svclastb_u64(svptrue_b8(), svcompact_u64(it, svindex_u64(0, 1)));

            svbool_t lane = svdup_b8(false);
            lane = svsetq_lane_b8(true, lane, (uint32_t)idx);

            bool is_non = svptest_any(svptrue_b8(), svand_b_z(pg, non, lane));
            bool is_del = svptest_any(svptrue_b8(), svand_b_z(pg, dm, lane));

            if (is_non && prev) {
                starts = svorr_b_z(pg, starts, lane);
            }
            prev = is_del;
            it = svand_b_z(pg, it, svnot_b_z(pg, lane));
        }

        count += (size_t)svcntp_b8(pg, starts);

        /* carry */
        uint32_t active = (uint32_t)svcntp_b8(pg, pg);
        if (active != 0u) {
            uint32_t last_lane = active - 1u;
            svbool_t lastp = svdup_b8(false);
            lastp = svsetq_lane_b8(true, lastp, last_lane);
            prev_is_delim = svptest_any(svptrue_b8(), svand_b_z(pg, dm, lastp)) ? 1u : 0u;
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

        svbool_t ge_lo = svcmpge_u8(pg, v, vlo);
        svbool_t le_hi = svcmple_u8(pg, v, vhi);
        svbool_t m     = svand_b_z(pg, ge_lo, le_hi);

        svuint8_t upper = svsub_u8_z(pg, v, vsub);
        svuint8_t out   = svsel_u8(m, upper, v);

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
static inline size_t simd_last_index_u8_sve2(const unsigned char* s, size_t n, unsigned char c) {
    size_t i = 0, last = SIZE_MAX;

    while (i < n) {
        svbool_t pg = svwhilelt_b8((uint64_t)i, (uint64_t)n);
        svuint8_t v = svld1_u8(pg, (const uint8_t*)(s + i));
        svbool_t  m = svcmpeq_n_u8(pg, v, c);

        if (svptest_any(svptrue_b8(), m)) {
            svuint8_t idx0 = svindex_u8(0, 1);
            svuint8_t idxc = svcompact_u8(m, idx0);
            uint8_t rel = svlasta_u8(svptrue_b8(), idxc);
            last = i + (size_t)rel;
        }

        i += svcntb();
    }
    return last;
}

static inline size_t simd_first_substr_index_sve2(const unsigned char* s, size_t n,
                                                  const unsigned char* pat, size_t m) {
    if (m == 0) return 0;
    if (m == 1) {
        const void* p = memchr(s, pat[0], n);
        return p ? (size_t)((const unsigned char*)p - s) : SIZE_MAX;
    }

    size_t i = 0;
    while (i < n) {
        svbool_t pg = svwhilelt_b8((uint64_t)i, (uint64_t)n);
        svuint8_t v = svld1_u8(pg, (const uint8_t*)(s + i));
        svbool_t  msk = svcmpeq_n_u8(pg, v, pat[0]);

        if (svptest_any(svptrue_b8(), msk)) {
            uint64_t vl = svcntb();
            uint64_t end = i + vl;
            if (end > n) end = n;
            for (size_t j = i; j < end; ++j) {
                if (s[j] == pat[0]) {
                    if (j + m <= n && memcmp(s + j, pat, m) == 0) return j;
                }
            }
        }
        i += svcntb();
    }
    return SIZE_MAX;
}
static inline size_t simd_last_substr_index_sve2(const unsigned char* s, size_t n,
                                                 const unsigned char* pat, size_t m) {
    if (m == 0) return n;
    if (m == 1) { for (size_t i = n; i-- > 0; ) if (s[i] == pat[0]) return i; return SIZE_MAX; }

    size_t i = 0, last = SIZE_MAX;
    while (i < n) {
        svbool_t pg = svwhilelt_b8((uint64_t)i, (uint64_t)n);
        svuint8_t v = svld1_u8(pg, (const uint8_t*)(s + i));
        svbool_t  msk = svcmpeq_n_u8(pg, v, pat[0]);

        if (svptest_any(svptrue_b8(), msk)) {
            uint64_t vl = svcntb();
            uint64_t end = i + vl; if (end > n) end = n;
            for (size_t j = end; j-- > i; ) {
                if (s[j] == pat[0]) {
                    if (j + m <= n && memcmp(s + j, pat, m) == 0) { last = j; break; }
                }
            }
        }
        i += svcntb();
    }
    return last;
}

static inline size_t simd_last_substr_index_sve2(const unsigned char* s, size_t n,
                                                 const unsigned char* pat, size_t m) {
    if (m == 0) return n;
    if (m == 1) { for (size_t i=n; i-- > 0; ) if (s[i]==pat[0]) return i; return SIZE_MAX; }
    if (n < m) return SIZE_MAX;

    size_t i = 0, last = SIZE_MAX;
    while (i < n) {
        svbool_t pg = svwhilelt_b8((uint64_t)i, (uint64_t)n);
        svuint8_t v = svld1_u8(pg, (const uint8_t*)(s + i));
        svbool_t  msk = svcmpeq_n_u8(pg, v, pat[0]);

        if (svptest_any(svptrue_b8(), msk)) {
            uint64_t vl = svcntb();
            uint64_t end = i + vl; if (end > n) end = n;
            for (size_t j = end; j-- > i; ) {
                if (s[j] == pat[0]) {
                    if (j + m <= n && memcmp(s + j, pat, m) == 0) { last = j; break; }
                }
            }
        }
        i += svcntb();
    }
    return last;
}

#endif /* CSALT_SIMD_SVE2_CHAR_INL */

