#ifndef SIMD_AVX2_FLOAT_INL
#define SIMD_AVX2_FLOAT_INL
// ================================================================================ 
// ================================================================================ 

/**
 * @file simd_avx2_float.inl
 * @brief AVX2 implementation of tolerance-aware float search.
 *
 * Processes 8 floats per iteration using 256-bit YMM registers.
 *
 * Exact mode (both tolerances == 0.0f):
 *   AVX2 provides _mm256_cmpeq_epi32 for native 256-bit integer comparison.
 *   The float register is reinterpreted with _mm256_castps_si256 so the
 *   comparison is bitwise: NaN != NaN, -0.0f (0x80000000) != +0.0f.
 *   _mm256_movemask_ps on the cast result gives an 8-bit lane mask directly.
 *
 * Tolerance mode:
 *   Uses _mm256_cmp_ps(_CMP_LE_OQ) on the absolute difference against the
 *   per-lane threshold.  _mm256_andnot_ps clears the sign bit for fabsf.
 *
 * Requires: -mavx2
 */

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <immintrin.h>   /* AVX2 */

// ================================================================================
// Internal helpers
// ================================================================================

static inline __m256 _avx2_fabs_ps(__m256 v) {
    __m256 sign_mask = _mm256_set1_ps(-0.0f);
    return _mm256_andnot_ps(sign_mask, v);
}

/* Exact bitwise match using AVX2 native 256-bit integer compare. */
static inline int _avx2_exact_mask(__m256 chunk, __m256 vn) {
    __m256i ci  = _mm256_castps_si256(chunk);
    __m256i ni  = _mm256_castps_si256(vn);
    __m256i cmp = _mm256_cmpeq_epi32(ci, ni);
    return _mm256_movemask_ps(_mm256_castsi256_ps(cmp));
}

static inline int _avx2_tol_mask(__m256 chunk, __m256 vn,
                                  __m256 vabs_tol, __m256 vrel_tol,
                                  __m256 vabs_needle) {
    __m256 diff       = _avx2_fabs_ps(_mm256_sub_ps(chunk, vn));
    __m256 abs_chunk  = _avx2_fabs_ps(chunk);
    __m256 rel_base   = _mm256_max_ps(abs_chunk, vabs_needle);
    __m256 rel_thresh = _mm256_mul_ps(vrel_tol, rel_base);
    __m256 threshold  = _mm256_max_ps(vabs_tol, rel_thresh);
    return _mm256_movemask_ps(_mm256_cmp_ps(diff, threshold, _CMP_LE_OQ));
}

// ================================================================================
// Public interface
// ================================================================================

static size_t simd_contains_float(const float* data,
                                   size_t       start,
                                   size_t       end,
                                   float        needle,
                                   float        abs_tol,
                                   float        rel_tol) {
    size_t i  = start;
    __m256 vn = _mm256_set1_ps(needle);

    if (abs_tol == 0.0f && rel_tol == 0.0f) {
        while (i + 8u <= end) {
            __m256 chunk = _mm256_loadu_ps(data + i);
            int    mask  = _avx2_exact_mask(chunk, vn);
            if (mask != 0) {
                /* __builtin_ctz gives index of lowest set bit — one bit per lane */
                _mm256_zeroupper();
                return i + (size_t)__builtin_ctz((unsigned)mask);
            }
            i += 8u;
        }
    } else {
        __m256 vabs_tol    = _mm256_set1_ps(abs_tol);
        __m256 vrel_tol    = _mm256_set1_ps(rel_tol);
        __m256 vabs_needle = _mm256_set1_ps(fabsf(needle));

        while (i + 8u <= end) {
            __m256 chunk = _mm256_loadu_ps(data + i);
            int    mask  = _avx2_tol_mask(chunk, vn, vabs_tol, vrel_tol,
                                          vabs_needle);
            if (mask != 0) {
                _mm256_zeroupper();
                return i + (size_t)__builtin_ctz((unsigned)mask);
            }
            i += 8u;
        }
    }

    _mm256_zeroupper();

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
    size_t best      = SIZE_MAX;
    float  abs_needle = fabsf(needle);
    __m256 vn        = _mm256_set1_ps(needle);
    __m256 vabs_tol  = _mm256_set1_ps(abs_tol);
    __m256 vrel_tol  = _mm256_set1_ps(rel_tol);
    __m256 vabs_ndl  = _mm256_set1_ps(abs_needle);

    /* Check landing */
    {
        float diff = fabsf(data[mid] - needle);
        float thr  = fmaxf(abs_tol, rel_tol * fmaxf(fabsf(data[mid]), abs_needle));
        if (diff <= thr) best = mid;
    }

    /* SIMD left walk — 8 floats per step */
    size_t L = mid;
    while (L >= 8u) {
        size_t base  = L - 8u;
        __m256 chunk = _mm256_loadu_ps(data + base);
        int    mask  = _avx2_tol_mask(chunk, vn, vabs_tol, vrel_tol, vabs_ndl);
        if (mask == 0xFF) { best = base; L = base; continue; }
        for (size_t e = 0; e < 8u && base + e < L; e++) {
            if ((mask >> (int)e) & 1) { best = base + e; }
            else                      { L = base + e; goto left_done_avx2; }
        }
        L = base;
    }
    left_done_avx2:
    _mm256_zeroupper();
    while (L > 0u) {
        float diff = fabsf(data[L - 1u] - needle);
        float thr  = fmaxf(abs_tol, rel_tol * fmaxf(fabsf(data[L - 1u]), abs_needle));
        if (diff <= thr) { best = --L; } else { break; }
    }

    /* SIMD right walk */
    if (best == SIZE_MAX) {
        size_t R = mid + 1u;
        while (R + 8u <= len) {
            __m256 chunk = _mm256_loadu_ps(data + R);
            int    mask  = _avx2_tol_mask(chunk, vn, vabs_tol, vrel_tol, vabs_ndl);
            if (mask == 0) break;
            for (size_t e = 0; e < 8u && R + e < len; e++) {
                if ((mask >> (int)e) & 1) { if (best == SIZE_MAX) best = R + e; }
                else                      { goto right_done_avx2; }
            }
            R += 8u;
        }
        right_done_avx2:
        _mm256_zeroupper();
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
#endif /* SIMD_AVX2_FLOAT_INL */ 
// ================================================================================ 
// ================================================================================ 
// eof
