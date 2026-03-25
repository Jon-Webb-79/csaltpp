#ifndef SIMD_AVX512_DOUBLE_INL
#define SIMD_AVX512_DOUBLE_INL

/**
 * @file simd_avx512_double.inl
 * @brief AVX-512 implementation of tolerance-aware double search.
 *
 * Processes 8 doubles per iteration using 512-bit ZMM registers.
 *
 * Exact mode: _mm512_cmpeq_epi64_mask returns an 8-bit mask directly.
 * Tolerance mode: _mm512_cmp_pd_mask(_CMP_LE_OQ) returns an 8-bit mask.
 *
 * Requires: -mavx512f -mavx512bw
 */

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>

static inline bool _biteq_double(double a, double b) {
    uint64_t ua, ub;
    memcpy(&ua, &a, sizeof ua);
    memcpy(&ub, &b, sizeof ub);
    return ua == ub;
}


static inline __m512d _avx512_fabs_pd(__m512d v) {
    __m512d sign_mask = _mm512_set1_pd(-0.0);
    return _mm512_andnot_pd(sign_mask, v);
}

static inline __mmask8 _avx512_exact_mask_d(__m512d chunk, __m512d vn) {
    __m512i ci = _mm512_castpd_si512(chunk);
    __m512i ni = _mm512_castpd_si512(vn);
    return _mm512_cmpeq_epi64_mask(ci, ni);
}

static inline __mmask8 _avx512_tol_mask_d(__m512d chunk, __m512d vn,
                                            __m512d vabs_tol, __m512d vrel_tol,
                                            __m512d vabs_needle) {
    __m512d diff       = _avx512_fabs_pd(_mm512_sub_pd(chunk, vn));
    __m512d abs_chunk  = _avx512_fabs_pd(chunk);
    __m512d rel_base   = _mm512_max_pd(abs_chunk, vabs_needle);
    __m512d rel_thresh = _mm512_mul_pd(vrel_tol, rel_base);
    __m512d threshold  = _mm512_max_pd(vabs_tol, rel_thresh);
    return _mm512_cmp_pd_mask(diff, threshold, _CMP_LE_OQ);
}

static size_t simd_contains_double(const double* data,
                                    size_t        start,
                                    size_t        end,
                                    double        needle,
                                    double        abs_tol,
                                    double        rel_tol) {
    size_t  i  = start;
    __m512d vn = _mm512_set1_pd(needle);

    if (abs_tol == 0.0 && rel_tol == 0.0) {
        while (i + 8u <= end) {
            __m512d  chunk = _mm512_loadu_pd(data + i);
            __mmask8 mask  = _avx512_exact_mask_d(chunk, vn);
            if (mask != 0) return i + (size_t)__builtin_ctz((unsigned)mask);
            i += 8u;
        }
    } else {
        __m512d vabs_tol    = _mm512_set1_pd(abs_tol);
        __m512d vrel_tol    = _mm512_set1_pd(rel_tol);
        __m512d vabs_needle = _mm512_set1_pd(fabs(needle));
        while (i + 8u <= end) {
            __m512d  chunk = _mm512_loadu_pd(data + i);
            __mmask8 mask  = _avx512_tol_mask_d(chunk, vn, vabs_tol, vrel_tol,
                                                 vabs_needle);
            if (mask != 0) return i + (size_t)__builtin_ctz((unsigned)mask);
            i += 8u;
        }
    }

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

    size_t  best       = SIZE_MAX;
    double  abs_needle = fabs(needle);
    __m512d vn         = _mm512_set1_pd(needle);
    __m512d vabs_tol   = _mm512_set1_pd(abs_tol);
    __m512d vrel_tol   = _mm512_set1_pd(rel_tol);
    __m512d vabs_ndl   = _mm512_set1_pd(abs_needle);
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
    while (L >= 8u) {
        size_t   base  = L - 8u;
        __m512d  chunk = _mm512_loadu_pd(data + base);
        __mmask8 mask  = _avx512_tol_mask_d(chunk, vn, vabs_tol, vrel_tol, vabs_ndl);
        if (mask == 0xFFu) { best = base; L = base; continue; }
        /* Scan high→low so a miss stops the walk without skipping higher matches. */
        for (int _e = 7; _e >= 0; _e--) {
            if ((mask >> (unsigned)_e) & 1u) { best = base + (size_t)_e; }
            else                             { L = base + (size_t)_e + 1u; goto left_done_avx512_d; }
        }
        L = base;
    }
    left_done_avx512_d:
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
        while (R + 8u <= len) {
            __m512d  chunk = _mm512_loadu_pd(data + R);
            __mmask8 mask  = _avx512_tol_mask_d(chunk, vn, vabs_tol, vrel_tol, vabs_ndl);
            if (mask == 0) break;
            for (size_t e = 0; e < 8u && R + e < len; e++) {
                if ((mask >> (unsigned)e) & 1u) { if (best == SIZE_MAX) best = R + e; }
                else                            { goto right_done_avx512_d; }
            }
            R += 8u;
        }
        right_done_avx512_d:
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

#endif /* SIMD_AVX512_DOUBLE_INL */
