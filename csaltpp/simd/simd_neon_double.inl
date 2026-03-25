#ifndef SIMD_NEON_DOUBLE_INL
#define SIMD_NEON_DOUBLE_INL

/**
 * @file simd_neon_double.inl
 * @brief ARM NEON implementation of tolerance-aware double search.
 *
 * Processes 2 doubles per iteration using 128-bit NEON Q registers.
 * AArch64 only — double-precision NEON requires the 64-bit ARM ISA.
 *
 * Exact mode: vceqq_u64 on vreinterpretq_u64_f64 for bitwise equality.
 *   NaN != NaN, -0.0 != +0.0.
 * Tolerance mode: vabsq_f64, vsubq_f64, vcleq_f64, vmaxq_f64, vmulq_f64.
 *
 * Requires: AArch64 (implicit double-precision NEON)
 */

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <string.h>
#include <arm_neon.h>

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
    size_t i = start;

    if (abs_tol == 0.0 && rel_tol == 0.0) {
        uint64_t nbits;
        memcpy(&nbits, &needle, 8);
        uint64x2_t vn = vdupq_n_u64(nbits);
        while (i + 2u <= end) {
            uint64x2_t chunk = vld1q_u64((const uint64_t*)(data + i));
            uint64x2_t cmp   = vceqq_u64(chunk, vn);
            uint64_t lo = vgetq_lane_u64(cmp, 0);
            uint64_t hi = vgetq_lane_u64(cmp, 1);
            if (lo && i     < end) return i;
            if (hi && i + 1 < end) return i + 1;
            i += 2u;
        }
    } else {
        float64x2_t vn          = vdupq_n_f64(needle);
        float64x2_t vabs_tol    = vdupq_n_f64(abs_tol);
        float64x2_t vrel_tol    = vdupq_n_f64(rel_tol);
        float64x2_t vabs_needle = vdupq_n_f64(fabs(needle));
        while (i + 2u <= end) {
            float64x2_t chunk     = vld1q_f64(data + i);
            float64x2_t diff      = vabsq_f64(vsubq_f64(chunk, vn));
            float64x2_t abs_chunk = vabsq_f64(chunk);
            float64x2_t rel_base  = vmaxq_f64(abs_chunk, vabs_needle);
            float64x2_t rel_thr   = vmulq_f64(vrel_tol, rel_base);
            float64x2_t threshold = vmaxq_f64(vabs_tol, rel_thr);
            uint64x2_t  cmp       = vcleq_f64(diff, threshold);
            uint64_t lo = vgetq_lane_u64(cmp, 0);
            uint64_t hi = vgetq_lane_u64(cmp, 1);
            if (lo && i     < end) return i;
            if (hi && i + 1 < end) return i + 1;
            i += 2u;
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

    size_t best      = SIZE_MAX;
    double abs_needle = fabs(needle);
    float64x2_t vn        = vdupq_n_f64(needle);
    float64x2_t vabs_tol  = vdupq_n_f64(abs_tol);
    float64x2_t vrel_tol  = vdupq_n_f64(rel_tol);
    float64x2_t vabs_ndl  = vdupq_n_f64(abs_needle);
    if (abs_tol == 0.0 && rel_tol == 0.0) {
        /* Exact mode: bitwise equality avoids NaN from (inf - inf). */
        if (_biteq_double(data[mid], needle)) best = mid;
    } else {

    {
        double diff = fabs(data[mid] - needle);
        double thr  = fmax(abs_tol, rel_tol * fmax(fabs(data[mid]), abs_needle));
        if (diff <= thr) best = mid;
    }
    }

    size_t L = mid;
    while (L >= 2u) {
        size_t      base  = L - 2u;
        float64x2_t chunk = vld1q_f64(data + base);
        float64x2_t diff  = vabsq_f64(vsubq_f64(chunk, vn));
        float64x2_t abs_c = vabsq_f64(chunk);
        float64x2_t thr   = vmaxq_f64(vabs_tol,
                                vmulq_f64(vrel_tol, vmaxq_f64(abs_c, vabs_ndl)));
        uint64x2_t  cmp   = vcleq_f64(diff, thr);
        uint64_t    lo    = vgetq_lane_u64(cmp, 0);
        uint64_t    hi    = vgetq_lane_u64(cmp, 1);
        /* All match? */
        if (lo && hi) { best = base; L = base; continue; }
        /* Scan high→low: lane 1 (hi) is index base+1, closer to mid. */
        if (hi) { best = base + 1u; }  /* lane 1 matches */
        if (!hi) { L = base + 1u; goto left_done_neon_d; } /* lane 1 misses → stop above it */
        /* hi matched; lo (lane 0) misses → stop at base+1 */
        L = base + 1u; goto left_done_neon_d;
        L = base;
    }
    left_done_neon_d:
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
        while (R + 2u <= len) {
            float64x2_t chunk = vld1q_f64(data + R);
            float64x2_t diff  = vabsq_f64(vsubq_f64(chunk, vn));
            float64x2_t abs_c = vabsq_f64(chunk);
            float64x2_t thr   = vmaxq_f64(vabs_tol,
                                    vmulq_f64(vrel_tol, vmaxq_f64(abs_c, vabs_ndl)));
            uint64x2_t  cmp   = vcleq_f64(diff, thr);
            uint64_t    lo    = vgetq_lane_u64(cmp, 0);
            uint64_t    hi    = vgetq_lane_u64(cmp, 1);
            if (!lo) break;
            if (best == SIZE_MAX) best = R;
            if (!hi) goto right_done_neon_d;
            R += 2u;
        }
        right_done_neon_d:
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

#endif /* SIMD_NEON_DOUBLE_INL */
