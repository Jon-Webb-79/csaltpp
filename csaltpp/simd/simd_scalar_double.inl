#ifndef SIMD_SCALAR_DOUBLE_INL
#define SIMD_SCALAR_DOUBLE_INL

/**
 * @file simd_scalar_double.inl
 * @brief Scalar fallback for tolerance-aware double search.
 *
 * Exact mode (abs_tol == 0.0 && rel_tol == 0.0) uses a union reinterpret of
 * both values as uint64_t for bitwise equality: NaN != NaN, -0.0 != +0.0.
 * Tolerance mode applies:
 *   fabs(element - needle) <= fmax(abs_tol, rel_tol * fmax(|element|, |needle|))
 */

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <string.h>

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
    if (abs_tol == 0.0 && rel_tol == 0.0) {
        union { double d; uint64_t u; } n; n.d = needle;
        for (size_t i = start; i < end; i++) {
            union { double d; uint64_t u; } v; v.d = data[i];
            if (v.u == n.u) return i;
        }
        return SIZE_MAX;
    }
    for (size_t i = start; i < end; i++) {
        double diff      = fabs(data[i] - needle);
        double threshold = fmax(abs_tol,
                                rel_tol * fmax(fabs(data[i]), fabs(needle)));
        if (diff <= threshold) return i;
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

    size_t best = SIZE_MAX;

    if (abs_tol == 0.0 && rel_tol == 0.0) {
        /* Exact mode: bitwise equality avoids NaN from (inf - inf). */
        if (_biteq_double(data[mid], needle)) best = mid;
        size_t L = mid;
        while (L > 0u && _biteq_double(data[L - 1u], needle)) { L--; best = L; }
        if (best == SIZE_MAX) {
            size_t R = mid + 1u;
            while (R < len && _biteq_double(data[R], needle)) { best = (best == SIZE_MAX) ? R : best; R++; }
        }
    } else {
        /* Tolerance mode. */
        #define _IN_TOL_D(idx) \
            (fabs(data[(idx)] - needle) <= \
                 fmax(abs_tol, rel_tol * fmax(fabs(data[(idx)]), fabs(needle))))
        if (_IN_TOL_D(mid)) best = mid;
        size_t L = mid;
        while (L > 0u && _IN_TOL_D(L - 1u)) { L--; best = L; }
        if (best == SIZE_MAX) {
            size_t R = mid + 1u;
            while (R < len && _IN_TOL_D(R)) { best = (best == SIZE_MAX) ? R : best; R++; }
        }
        #undef _IN_TOL_D
    }
    return best;
}

#endif /* SIMD_SCALAR_DOUBLE_INL */
