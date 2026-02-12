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
// ================================================================================ 
// ================================================================================ 
static inline size_t simd_last_index_u8_sve(const unsigned char* s, size_t n, unsigned char c) {
    size_t i = 0, last = SIZE_MAX;

    while (i < n) {
        svbool_t pg = svwhilelt_b8((uint64_t)i, (uint64_t)n);
        svuint8_t v = svld1_u8(pg, (const uint8_t*)(s + i));
        svbool_t  m = svcmpeq_n_u8(pg, v, c);

        if (svptest_any(svptrue_b8(), m)) {
            /* Build lane indices [0..VL-1] then compact under match predicate. */
            svuint8_t idx0 = svindex_u8(0, 1);
            svuint8_t idxc = svcompact_u8(m, idx0);
            /* Number of matches in this vector */
            uint64_t count = svcntp_b8(svptrue_b8(), m);
            /* Last packed index (relative to this chunk) */
            uint8_t rel = svlasta_u8(svptrue_b8(), idxc);
            /* If multiple matches exist, rel will hold the highest matched lane index */
            (void)count; /* rel already represents last match after compact */
            last = i + (size_t)rel;
        }

        i += svcntb();
    }
    return last;
}

static inline size_t simd_first_substr_index_sve(const unsigned char* s, size_t n,
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
            /* Simple, robust: scan this chunk scalarly to get the FIRST lane. */
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
static inline size_t simd_last_substr_index_sve(const unsigned char* s, size_t n,
                                                const unsigned char* pat, size_t m) {
    if (m == 0) return n;
    if (m == 1) { for (size_t i = n; i-- > 0; ) if (s[i] == pat[0]) return i; return SIZE_MAX; }

    size_t i = 0, last = SIZE_MAX;
    while (i < n) {
        svbool_t pg = svwhilelt_b8((uint64_t)i, (uint64_t)n);
        svuint8_t v = svld1_u8(pg, (const uint8_t*)(s + i));
        svbool_t  msk = svcmpeq_n_u8(pg, v, pat[0]);

        if (svptest_any(svptrue_b8(), msk)) {
            /* Descend within the chunk scalarly to get the LAST lane and verify. */
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

static inline size_t simd_last_substr_index_sve(const unsigned char* s, size_t n,
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
            /* Scan this chunk descending to get the rightmost candidate. */
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

static inline size_t simd_token_count_sve(const char* s, size_t n,
                                          const char* delim, size_t dlen)
{
    size_t i = 0, count = 0;
    /* byte mask 0xFF (delim) / 0x00 (non) for "previous" before start */
    uint8_t prev_last = 0xFF;

    while (i < n) {
        svbool_t pg = svwhilelt_b8(i, n);
        svuint8_t v = svld1_u8(pg, (const uint8_t*)(s + i));

        /* build byte mask 0xFF/0x00 for delimiters */
        svuint8_t dm = svdup_u8(0);
        for (size_t j = 0; j < dlen; ++j) {
            svbool_t eq = svcmpeq_u8(pg, v, svdup_u8((uint8_t)delim[j]));
            dm = svorr_u8_x(pg, dm, svdup_u8_z(eq, 0xFF));
        }
        svuint8_t non = svnot_u8_z(pg, dm);

        /* prev per position: [prev_last, dm[0], dm[1], ...] */
        svuint8_t prev_fill = svdup_u8(prev_last);
        svuint8_t prev = svext_u8(prev_fill, dm, 1);

        svuint8_t starts = svand_u8_x(pg, non, prev);

        /* Count bytes != 0 via population count of 1-bit per byte.
           Shift 0xFF->1, 0x00->0, then sum lanes. */
        svuint8_t ones = svlsr_n_u8_x(pg, starts, 7);
        /* widen and accumulate */
        svuint16_t s16 = svuaddlb_u8(ones, svdup_u8(0));   /* pairwise add low */
        svuint16_t s16h= svuaddlt_u8(ones, svdup_u8(0));   /* pairwise add high */
        uint64_t block = (uint64_t)svaddv_u16(pg, s16) + (uint64_t)svaddv_u16(pg, s16h);
        count += (size_t)block;

        /* carry last-lane flag: extract the last active lane byte of dm */
        uint64_t active = (uint64_t)svcntp_b8(svptrue_b8(), pg); /* active lanes */
        uint8_t last = svlastb_u8(pg, dm); /* last active element */
        prev_last = (last != 0) ? 0xFF : 0x00;

        i += svcntb();
    }
    return count;
}


#endif /* CSALT_SIMD_SVE_CHAR_INL */

