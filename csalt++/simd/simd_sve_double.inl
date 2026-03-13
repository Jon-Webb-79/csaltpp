#ifndef SIMD_SVE_DOUBLE_INL
#define SIMD_SVE_DOUBLE_INL

/**
 * @file simd_sve_double.inl
 * @brief ARM SVE implementation of tolerance-aware double search.
 *
 * The vector length is queried at runtime via svcntd() (number of 64-bit lanes).
 * 
 * Exact mode: svreinterpret_u64_f64 + svcmpeq_u64 — bitwise equality.
 * Tolerance mode: svabs_f64_z, svsub_f64_z, svmax_f64_z, svmul_f64_z, svcmple_f64.
 *
 * Requires: -march=armv8-a+sve or -march=armv9-a
 */

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <string.h>
#include <arm_sve.h>

static inline bool _biteq_double(double a, double b) {
    uint64_t ua, ub;
    memcpy(&ua, &a, sizeof ua);
    memcpy(&ub, &b, sizeof ub);
    return ua == ub;
}


static size_t simd_contains_double(const double* data,
                                    size_t        start,
                                    size_t        end,
                                    double        needle,
                                    double        abs_tol,
                                    double        rel_tol) {
    size_t vl = svcntd();
    size_t i  = start;

    if (abs_tol == 0.0 && rel_tol == 0.0) {
        union { double d; uint64_t u; } nu; nu.d = needle;
        svuint64_t vn = svdup_n_u64(nu.u);
        while (i + vl <= end) {
            svbool_t    pg    = svptrue_b64();
            svfloat64_t fchunk = svld1_f64(pg, data + i);
            svuint64_t  chunk  = svreinterpret_u64_f64(fchunk);
            svbool_t    cmp    = svcmpeq_u64(pg, chunk, vn);
            if (svptest_any(pg, cmp))
                return i + (size_t)svcntp_b64(pg, svbrkb_z(pg, cmp));
            i += vl;
        }
    } else {
        svfloat64_t vn          = svdup_n_f64(needle);
        svfloat64_t vabs_tol    = svdup_n_f64(abs_tol);
        svfloat64_t vrel_tol    = svdup_n_f64(rel_tol);
        svfloat64_t vabs_needle = svdup_n_f64(fabs(needle));
        while (i + vl <= end) {
            svbool_t    pg         = svptrue_b64();
            svfloat64_t chunk      = svld1_f64(pg, data + i);
            svfloat64_t diff       = svabs_f64_z(pg, svsub_f64_z(pg, chunk, vn));
            svfloat64_t abs_chunk  = svabs_f64_z(pg, chunk);
            svfloat64_t rel_base   = svmax_f64_z(pg, abs_chunk, vabs_needle);
            svfloat64_t rel_thresh = svmul_f64_z(pg, vrel_tol, rel_base);
            svfloat64_t threshold  = svmax_f64_z(pg, vabs_tol, rel_thresh);
            svbool_t    cmp        = svcmple_f64(pg, diff, threshold);
            if (svptest_any(pg, cmp))
                return i + (size_t)svcntp_b64(pg, svbrkb_z(pg, cmp));
            i += vl;
        }
    }

    /* Scalar remainder */
    if (abs_tol == 0.0 && rel_tol == 0.0) {
        union { double d; uint64_t u; } n; n.d = needle;
        for (; i < end; i++) {
            union { double d; uint64_t u; } v; v.d = data[i];
            if (v.u == n.u) return i;
        }
    } else {
        for (; i < end; i++) {
            double diff      = fabs(data[i] - needle);
            double threshold = fmax(abs_tol,
                                    rel_tol * fmax(fabs(data[i]), fabs(needle)));
            if (diff <= threshold) return i;
        }
    }
    return SIZE_MAX;
}

// ================================================================================
// simd_binary_search_double
// ================================================================================

static size_t simd_binary_search_double(const double* data,
                                         size_t        len,
                                         double        needle,
                                         double        abs_tol,
                                         double        rel_tol,
                                         int         (*cmp)(const void*,
                                                            const void*)) {
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
    if (lo > hi) return SIZE_MAX;
    mid = lo;

    size_t vl        = svcntd();
    size_t best      = SIZE_MAX;
    double abs_needle = fabs(needle);
    svfloat64_t vn       = svdup_n_f64(needle);
    svfloat64_t vabs_tol = svdup_n_f64(abs_tol);
    svfloat64_t vrel_tol = svdup_n_f64(rel_tol);
    svfloat64_t vabs_ndl = svdup_n_f64(abs_needle);

    /* Exact-mode bitwise helper (valid for the whole function scope). */

    if (abs_tol == 0.0 && rel_tol == 0.0) {
        if (_biteq_double(data[mid], needle)) best = mid;
    } else {
        double diff = fabs(data[mid] - needle);
        double thr  = fmax(abs_tol, rel_tol * fmax(fabs(data[mid]), abs_needle));
        if (diff <= thr) best = mid;
    }

    size_t L = mid;
    while (L >= vl) {
        size_t      base  = L - vl;
        svbool_t    pg    = svptrue_b64();
        svfloat64_t chunk = svld1_f64(pg, data + base);
        svfloat64_t diff  = svabs_f64_z(pg, svsub_f64_z(pg, chunk, vn));
        svfloat64_t abs_c = svabs_f64_z(pg, chunk);
        svfloat64_t thr   = svmax_f64_z(pg, vabs_tol,
                                svmul_f64_z(pg, vrel_tol,
                                    svmax_f64_z(pg, abs_c, vabs_ndl)));
        svbool_t    match   = svcmple_f64(pg, diff, thr);
        svbool_t    nomatch = svnot_b_z(pg, match);
        if (!svptest_any(pg, nomatch)) {
            best = base; L = base; continue;
        }
        size_t n_match = (size_t)svcntp_b64(pg, svbrkb_z(pg, nomatch));
        for (size_t e = 0; e < n_match && base + e < L; e++)
            best = base + e;
        L = base + n_match;
        break;
    }
    if (abs_tol == 0.0 && rel_tol == 0.0) {
        while (L > 0u && _biteq_double(data[L - 1u], needle)) { best = --L; }
    } else {
        while (L > 0u) {
            double diff = fabs(data[L - 1u] - needle);
            double thr  = fmax(abs_tol, rel_tol * fmax(fabs(data[L - 1u]), abs_needle));
            if (diff <= thr) { best = --L; } else { break; }
        }
    }

    if (best == SIZE_MAX) {
        size_t R = mid + 1u;
        while (R + vl <= len) {
            svbool_t    pg    = svptrue_b64();
            svfloat64_t chunk = svld1_f64(pg, data + R);
            svfloat64_t diff  = svabs_f64_z(pg, svsub_f64_z(pg, chunk, vn));
            svfloat64_t abs_c = svabs_f64_z(pg, chunk);
            svfloat64_t thr   = svmax_f64_z(pg, vabs_tol,
                                    svmul_f64_z(pg, vrel_tol,
                                        svmax_f64_z(pg, abs_c, vabs_ndl)));
            svbool_t match = svcmple_f64(pg, diff, thr);
            if (!svptest_any(pg, match)) break;
            if (best == SIZE_MAX) {
                size_t first = (size_t)svcntp_b64(pg, svbrkb_z(pg, match));
                if (R + first < len) best = R + first;
            }
            if (svptest_any(pg, svnot_b_z(pg, match))) break;
            R += vl;
        }
        if (abs_tol == 0.0 && rel_tol == 0.0) {
            while (R < len && best == SIZE_MAX && _biteq_double(data[R], needle)) { best = R++; }
        } else {
            while (R < len && best == SIZE_MAX) {
                double diff = fabs(data[R] - needle);
                double thr  = fmax(abs_tol, rel_tol * fmax(fabs(data[R]), abs_needle));
                if (diff <= thr) { best = R++; } else { break; }
            }
        }
    }
    return best;
}

#endif /* SIMD_SVE_DOUBLE_INL */
