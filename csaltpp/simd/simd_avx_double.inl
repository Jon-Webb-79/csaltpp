#ifndef SIMD_AVX_DOUBLE_INL
#define SIMD_AVX_DOUBLE_INL

/**
 * @file simd_avx_double.inl
 * @brief AVX implementation of tolerance-aware double search.
 *
 * Processes 4 doubles per iteration using 256-bit YMM registers.
 *
 * Exact mode: AVX has no 256-bit integer compare, so the register is split
 * into two 128-bit halves and each half compared with _mm_cmpeq_epi32
 * using the SSE2 emulated 64-bit compare approach.
 *
 * Tolerance mode: _mm256_cmp_pd(_CMP_LE_OQ).
 *
 * Requires: -mavx
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


static inline __m256d _avx_fabs_pd(__m256d v) {
    __m256d sign_mask = _mm256_set1_pd(-0.0);
    return _mm256_andnot_pd(sign_mask, v);
}

static inline int _avx_exact_mask_d(__m256d chunk, __m256d vn) {
    /*
     * Split into 128-bit halves, emulate 64-bit cmpeq with 32-bit pairs.
     * Returns a 4-bit mask (one bit per double lane).
     */
    __m128i lo_c  = _mm_castpd_si128(_mm256_castpd256_pd128(chunk));
    __m128i hi_c  = _mm_castpd_si128(_mm256_extractf128_pd(chunk, 1));
    __m128i lo_n  = _mm_castpd_si128(_mm256_castpd256_pd128(vn));
    __m128i hi_n  = _mm_castpd_si128(_mm256_extractf128_pd(vn, 1));

    /* Emulate 64-bit cmpeq using 32-bit cmpeq + AND of adjacent halves */
    __m128i clo32 = _mm_cmpeq_epi32(lo_c, lo_n);
    __m128i chi32 = _mm_cmpeq_epi32(hi_c, hi_n);
    __m128i slo   = _mm_shuffle_epi32(clo32, _MM_SHUFFLE(2, 3, 0, 1));
    __m128i shi   = _mm_shuffle_epi32(chi32, _MM_SHUFFLE(2, 3, 0, 1));
    __m128i clo64 = _mm_and_si128(clo32, slo);
    __m128i chi64 = _mm_and_si128(chi32, shi);

    int mlo = _mm_movemask_pd(_mm_castsi128_pd(clo64));   /* 2-bit */
    int mhi = _mm_movemask_pd(_mm_castsi128_pd(chi64));   /* 2-bit */
    return mlo | (mhi << 2);                               /* 4-bit */
}

static inline int _avx_tol_mask_d(__m256d chunk, __m256d vn,
                                   __m256d vabs_tol, __m256d vrel_tol,
                                   __m256d vabs_needle) {
    __m256d diff       = _avx_fabs_pd(_mm256_sub_pd(chunk, vn));
    __m256d abs_chunk  = _avx_fabs_pd(chunk);
    __m256d rel_base   = _mm256_max_pd(abs_chunk, vabs_needle);
    __m256d rel_thresh = _mm256_mul_pd(vrel_tol, rel_base);
    __m256d threshold  = _mm256_max_pd(vabs_tol, rel_thresh);
    return _mm256_movemask_pd(_mm256_cmp_pd(diff, threshold, _CMP_LE_OQ));
}

static size_t simd_contains_double(const double* data,
                                    size_t        start,
                                    size_t        end,
                                    double        needle,
                                    double        abs_tol,
                                    double        rel_tol) {
    size_t  i  = start;
    __m256d vn = _mm256_set1_pd(needle);

    if (abs_tol == 0.0 && rel_tol == 0.0) {
        while (i + 4u <= end) {
            __m256d chunk = _mm256_loadu_pd(data + i);
            int     mask  = _avx_exact_mask_d(chunk, vn);
            if (mask != 0) {
                for (size_t e = 0; e < 4u && i + e < end; e++)
                    if ((mask >> (int)e) & 1) { _mm256_zeroupper(); return i + e; }
            }
            i += 4u;
        }
    } else {
        __m256d vabs_tol    = _mm256_set1_pd(abs_tol);
        __m256d vrel_tol    = _mm256_set1_pd(rel_tol);
        __m256d vabs_needle = _mm256_set1_pd(fabs(needle));
        while (i + 4u <= end) {
            __m256d chunk = _mm256_loadu_pd(data + i);
            int     mask  = _avx_tol_mask_d(chunk, vn, vabs_tol, vrel_tol,
                                             vabs_needle);
            if (mask != 0) {
                for (size_t e = 0; e < 4u && i + e < end; e++)
                    if ((mask >> (int)e) & 1) { _mm256_zeroupper(); return i + e; }
            }
            i += 4u;
        }
    }
    _mm256_zeroupper();

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

    size_t  best       = SIZE_MAX;
    double  abs_needle = fabs(needle);
    __m256d vn         = _mm256_set1_pd(needle);
    __m256d vabs_tol   = _mm256_set1_pd(abs_tol);
    __m256d vrel_tol   = _mm256_set1_pd(rel_tol);
    __m256d vabs_ndl   = _mm256_set1_pd(abs_needle);
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
    while (L >= 4u) {
        size_t  base  = L - 4u;
        __m256d chunk = _mm256_loadu_pd(data + base);
        int     mask  = _avx_tol_mask_d(chunk, vn, vabs_tol, vrel_tol, vabs_ndl);
        if (mask == 0xF) { best = base; L = base; continue; }
        /* Scan high→low so a miss stops the walk without skipping higher matches. */
        for (int _e = (int)4u - 1; _e >= 0; _e--) {
            if ((mask >> _e) & 1) { best = base + (size_t)_e; }
            else                  { L = base + (size_t)_e + 1u; goto left_done_avx_d; }
        }
        L = base;
    }
    left_done_avx_d:
    _mm256_zeroupper();
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
        while (R + 4u <= len) {
            __m256d chunk = _mm256_loadu_pd(data + R);
            int     mask  = _avx_tol_mask_d(chunk, vn, vabs_tol, vrel_tol, vabs_ndl);
            if (mask == 0) break;
            for (size_t e = 0; e < 4u && R + e < len; e++) {
                if ((mask >> (int)e) & 1) { if (best == SIZE_MAX) best = R + e; }
                else                      { goto right_done_avx_d; }
            }
            R += 4u;
        }
        right_done_avx_d:
        _mm256_zeroupper();
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

#endif /* SIMD_AVX_DOUBLE_INL */
