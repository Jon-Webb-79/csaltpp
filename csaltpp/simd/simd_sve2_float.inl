#ifndef SIMD_SVE2_FLOAT_INL
#define SIMD_SVE2_FLOAT_INL
// ================================================================================ 
// ================================================================================ 

/**
 * @file simd_sve2_float.inl
 * @brief ARM SVE2 implementation of tolerance-aware float search.
 *
 * SVE2 adds no instructions that improve on the SVE float comparison path
 * for single-needle search — svmatch operates on integer elements and is
 * designed for multi-needle matching, not floating-point tolerance.  This
 * file therefore uses the identical SVE algorithm and exists as a separate
 * dispatch target so that future SVE2-specific optimisations (e.g.,
 * svmatch on quantised data, or SVE2 gather/scatter patterns) can be
 * introduced here without disturbing the rest of the dispatch chain.
 *
 * See contains_sve_float.inl for a full description of the algorithm.
 *
 * Requires: -march=armv9-a or -march=armv8-a+sve2
 */

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <arm_sve.h>

// ================================================================================
// Public interface
// ================================================================================

static size_t simd_contains_float(const float* data,
                                   size_t       start,
                                   size_t       end,
                                   float        needle,
                                   float        abs_tol,
                                   float        rel_tol) {
    size_t vl = svcntw();
    size_t i  = start;

    if (abs_tol == 0.0f && rel_tol == 0.0f) {
        union { float f; uint32_t u; } nu; nu.f = needle;
        svuint32_t vn = svdup_n_u32(nu.u);

        while (i + vl <= end) {
            svbool_t    pg     = svptrue_b32();
            svfloat32_t fchunk = svld1_f32(pg, data + i);
            svuint32_t  chunk  = svreinterpret_u32_f32(fchunk);
            svbool_t    cmp    = svcmpeq_u32(pg, chunk, vn);
            if (svptest_any(pg, cmp)) {
                return i + (size_t)svcntp_b32(pg, svbrkb_z(pg, cmp));
            }
            i += vl;
        }
    } else {
        svfloat32_t vn          = svdup_n_f32(needle);
        svfloat32_t vabs_tol    = svdup_n_f32(abs_tol);
        svfloat32_t vrel_tol    = svdup_n_f32(rel_tol);
        svfloat32_t vabs_needle = svdup_n_f32(fabsf(needle));

        while (i + vl <= end) {
            svbool_t    pg         = svptrue_b32();
            svfloat32_t chunk      = svld1_f32(pg, data + i);
            svfloat32_t diff       = svabs_f32_z(pg, svsub_f32_z(pg, chunk, vn));
            svfloat32_t abs_chunk  = svabs_f32_z(pg, chunk);
            svfloat32_t rel_base   = svmax_f32_z(pg, abs_chunk, vabs_needle);
            svfloat32_t rel_thresh = svmul_f32_z(pg, vrel_tol, rel_base);
            svfloat32_t threshold  = svmax_f32_z(pg, vabs_tol, rel_thresh);
            svbool_t    cmp        = svcmple_f32(pg, diff, threshold);
            if (svptest_any(pg, cmp)) {
                return i + (size_t)svcntp_b32(pg, svbrkb_z(pg, cmp));
            }
            i += vl;
        }
    }

    /* Scalar remainder */
    if (abs_tol == 0.0f && rel_tol == 0.0f) {
        union { float f; uint32_t u; } n; n.f = needle;
        for (; i < end; i++) {
            union { float f; uint32_t u; } v; v.f = data[i];
            if (v.u == n.u) return i;
        }
    } else {
        for (; i < end; i++) {
            float diff      = fabsf(data[i] - needle);
            float threshold = fmaxf(abs_tol,
                                    rel_tol * fmaxf(fabsf(data[i]), fabsf(needle)));
            if (diff <= threshold) return i;
        }
    }
    return SIZE_MAX;
}
// -------------------------------------------------------------------------------- 

static size_t simd_binary_search_float(const float* data,
                                        size_t       len,
                                        float        needle,
                                        float        abs_tol,
                                        float        rel_tol,
                                        int        (*cmp)(const void*, const void*)) {
    if (len == 0u) return SIZE_MAX;

    /* ── Bisection (exact comparator, sort contract unchanged) ── */
    size_t lo = 0u, hi = len - 1u, mid = 0u;
    while (lo < hi) {
        mid = lo + (hi - lo) / 2u;
        int c = cmp(data + mid, &needle);
        if      (c < 0) lo = mid + 1u;
        else if (c > 0) { if (mid == 0u) { lo = hi + 1u; break; } hi = mid; }
        else            { lo = hi = mid; break; }
    }
    if (lo > hi) return SIZE_MAX;   /* empty range after underflow guard */
    mid = lo;

    /* ── Tolerance fan-out ── */
    size_t vl        = svcntw();
    size_t best      = SIZE_MAX;
    float  abs_needle = fabsf(needle);
    svfloat32_t vn       = svdup_n_f32(needle);
    svfloat32_t vabs_tol = svdup_n_f32(abs_tol);
    svfloat32_t vrel_tol = svdup_n_f32(rel_tol);
    svfloat32_t vabs_ndl = svdup_n_f32(abs_needle);

    /* Check landing */
    {
        float diff = fabsf(data[mid] - needle);
        float thr  = fmaxf(abs_tol, rel_tol * fmaxf(fabsf(data[mid]), abs_needle));
        if (diff <= thr) best = mid;
    }

    /* SVE left walk */
    size_t L = mid;
    while (L >= vl) {
        size_t      base  = L - vl;
        svbool_t    pg    = svptrue_b32();
        svfloat32_t chunk = svld1_f32(pg, data + base);
        svfloat32_t diff  = svabs_f32_z(pg, svsub_f32_z(pg, chunk, vn));
        svfloat32_t abs_c = svabs_f32_z(pg, chunk);
        svfloat32_t thr   = svmax_f32_z(pg, vabs_tol,
                                svmul_f32_z(pg, vrel_tol,
                                    svmax_f32_z(pg, abs_c, vabs_ndl)));
        svbool_t    match = svcmple_f32(pg, diff, thr);
        svbool_t    nomatch = svnot_b_z(pg, match);
        if (!svptest_any(pg, nomatch)) {
            /* All lanes match */
            best = base; L = base; continue;
        }
        /* Find first non-matching lane from the low end */
        size_t n_match = (size_t)svcntp_b32(pg, svbrkb_z(pg, nomatch));
        for (size_t e = 0; e < n_match && base + e < L; e++)
            best = base + e;
        L = base + n_match;
        break;
    }
    /* Scalar left mop-up */
    while (L > 0u) {
        float diff = fabsf(data[L - 1u] - needle);
        float thr  = fmaxf(abs_tol, rel_tol * fmaxf(fabsf(data[L - 1u]), abs_needle));
        if (diff <= thr) { best = --L; } else { break; }
    }

    /* SVE right walk */
    if (best == SIZE_MAX) {
        size_t R = mid + 1u;
        while (R + vl <= len) {
            svbool_t    pg    = svptrue_b32();
            svfloat32_t chunk = svld1_f32(pg, data + R);
            svfloat32_t diff  = svabs_f32_z(pg, svsub_f32_z(pg, chunk, vn));
            svfloat32_t abs_c = svabs_f32_z(pg, chunk);
            svfloat32_t thr   = svmax_f32_z(pg, vabs_tol,
                                    svmul_f32_z(pg, vrel_tol,
                                        svmax_f32_z(pg, abs_c, vabs_ndl)));
            svbool_t    match = svcmple_f32(pg, diff, thr);
            if (!svptest_any(pg, match)) break;
            /* Record only the first match in this block */
            if (best == SIZE_MAX) {
                size_t first = (size_t)svcntp_b32(pg, svbrkb_z(pg, match));
                if (R + first < len) best = R + first;
            }
            /* Stop when a non-matching lane is found */
            svbool_t nomatch = svnot_b_z(pg, match);
            if (svptest_any(pg, nomatch)) break;
            R += vl;
        }
        /* Scalar right mop-up */
        while (R < len && best == SIZE_MAX) {
            float diff = fabsf(data[R] - needle);
            float thr  = fmaxf(abs_tol, rel_tol * fmaxf(fabsf(data[R]), abs_needle));
            if (diff <= thr) { best = R++; } else { break; }
        }
    }
    return best;
}
// ================================================================================ 
// ================================================================================ 
#endif /* SIMD_SVE2_FLOAT_INL */ 
// ================================================================================ 
// ================================================================================ 
// eof
