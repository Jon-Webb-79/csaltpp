#ifndef SIMD_AVX_FLOAT_INL
#define SIMD_AVX_FLOAT_INL
// ================================================================================ 
// ================================================================================ 

/**
 * @file simd_avx_float.inl
 * @brief AVX implementation of tolerance-aware float search.
 *
 * Processes 8 floats per iteration using 256-bit YMM registers.
 *
 * Exact mode (both tolerances == 0.0f):
 *   Unlike the integer SIMD paths, AVX does have native 256-bit float
 *   comparison via _mm256_cmp_ps with predicate _CMP_EQ_OQ (ordered,
 *   quiet). However, IEEE 754 defines 0.0f == -0.0f under _CMP_EQ_OQ,
 *   which would break the bitwise-exact contract. To preserve NaN != NaN
 *   and -0.0f != +0.0f semantics, the register is cast to integer and
 *   compared with two 128-bit _mm_cmpeq_epi32 calls (AVX has no 256-bit
 *   integer comparison — those require AVX2).
 *
 * Tolerance mode:
 *   Uses _mm256_cmp_ps(_CMP_LE_OQ) on the absolute difference against the
 *   per-lane threshold.  _mm256_andnot_ps clears the sign bit for fabsf.
 *
 * Requires: -mavx
 */

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <immintrin.h>   /* AVX */

// ================================================================================
// Internal helpers
// ================================================================================

/* Clear sign bit in all 8 lanes: implements fabsf lane-wise. */
static inline __m256 _avx_fabs_ps(__m256 v) {
    /* andnot(sign_mask, v) = v & ~sign_mask — clears bit 31 of each lane */
    __m256 sign_mask = _mm256_set1_ps(-0.0f);   /* 0x80000000 in each lane */
    return _mm256_andnot_ps(sign_mask, v);
}

/*
 * Exact bitwise match using two 128-bit integer compares (AVX has no
 * 256-bit _mm256_cmpeq_epi32 — that requires AVX2).
 * Returns an 8-bit mask, one bit per float lane.
 */
static inline int _avx_exact_mask(__m256 chunk, __m256 vn) {
    __m128i lo_c = _mm_castps_si128(_mm256_castps256_ps128(chunk));
    __m128i hi_c = _mm_castps_si128(_mm256_extractf128_ps(chunk, 1));
    __m128i lo_n = _mm_castps_si128(_mm256_castps256_ps128(vn));
    __m128i hi_n = _mm_castps_si128(_mm256_extractf128_ps(vn, 1));

    __m128i cmp_lo = _mm_cmpeq_epi32(lo_c, lo_n);
    __m128i cmp_hi = _mm_cmpeq_epi32(hi_c, hi_n);

    int mask_lo = _mm_movemask_ps(_mm_castsi128_ps(cmp_lo));  /* bits 0-3 */
    int mask_hi = _mm_movemask_ps(_mm_castsi128_ps(cmp_hi));  /* bits 0-3 */
    return mask_lo | (mask_hi << 4);                           /* 8-bit combined */
}

/*
 * Tolerance match: |chunk - needle| <= max(abs_tol, rel_tol * max(|chunk|, |needle|)).
 * Returns an 8-bit mask via _mm256_movemask_ps.
 */
static inline int _avx_tol_mask(__m256 chunk, __m256 vn,
                                 __m256 vabs_tol, __m256 vrel_tol,
                                 __m256 vabs_needle) {
    __m256 diff       = _avx_fabs_ps(_mm256_sub_ps(chunk, vn));
    __m256 abs_chunk  = _avx_fabs_ps(chunk);
    __m256 rel_base   = _mm256_max_ps(abs_chunk, vabs_needle);
    __m256 rel_thresh = _mm256_mul_ps(vrel_tol, rel_base);
    __m256 threshold  = _mm256_max_ps(vabs_tol, rel_thresh);
    /* _CMP_LE_OQ: ordered, quiet less-than-or-equal */
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
            int    mask  = _avx_exact_mask(chunk, vn);
            if (mask != 0) {
                for (size_t e = 0; e < 8u && i + e < end; e++) {
                    if ((mask >> (int)e) & 1) {
                        _mm256_zeroupper();
                        return i + e;
                    }
                }
            }
            i += 8u;
        }
    } else {
        __m256 vabs_tol    = _mm256_set1_ps(abs_tol);
        __m256 vrel_tol    = _mm256_set1_ps(rel_tol);
        __m256 vabs_needle = _mm256_set1_ps(fabsf(needle));

        while (i + 8u <= end) {
            __m256 chunk = _mm256_loadu_ps(data + i);
            int    mask  = _avx_tol_mask(chunk, vn, vabs_tol, vrel_tol,
                                         vabs_needle);
            if (mask != 0) {
                for (size_t e = 0; e < 8u && i + e < end; e++) {
                    if ((mask >> (int)e) & 1) {
                        _mm256_zeroupper();
                        return i + e;
                    }
                }
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
        int    mask  = _avx_tol_mask(chunk, vn, vabs_tol, vrel_tol, vabs_ndl);
        if (mask == 0xFF) { best = base; L = base; continue; }
        for (size_t e = 0; e < 8u && base + e < L; e++) {
            if ((mask >> (int)e) & 1) { best = base + e; }
            else                      { L = base + e; goto left_done_avx; }
        }
        L = base;
    }
    left_done_avx:
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
            int    mask  = _avx_tol_mask(chunk, vn, vabs_tol, vrel_tol, vabs_ndl);
            if (mask == 0) break;
            for (size_t e = 0; e < 8u && R + e < len; e++) {
                if ((mask >> (int)e) & 1) { if (best == SIZE_MAX) best = R + e; }
                else                      { goto right_done_avx; }
            }
            R += 8u;
        }
        right_done_avx:
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
#endif /* SIMD_AVX_FLOAT_INL */ 
// ================================================================================ 
// ================================================================================ 
// eof
