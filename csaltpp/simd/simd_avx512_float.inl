#ifndef SIMD_AVX512_FLOAT_INL
#define SIMD_AVX512_FLOAT_INL
// ================================================================================ 
// ================================================================================ 

/**
 * @file simd_avx512_float.inl
 * @brief AVX-512 implementation of tolerance-aware float search.
 *
 * Processes 16 floats per iteration using 512-bit ZMM registers.
 *
 * Exact mode (both tolerances == 0.0f):
 *   _mm512_cmpeq_epi32_mask reinterprets the ZMM register as 16 x int32
 *   and returns a 16-bit mask directly — no movemask needed.  Semantics
 *   are bitwise: NaN != NaN, -0.0f != +0.0f.
 *
 * Tolerance mode:
 *   _mm512_cmp_ps_mask(_CMP_LE_OQ) returns a 16-bit mask directly.
 *   _mm512_andnot_ps clears the sign bit for fabsf lane-wise.
 *   __builtin_ctz on the 16-bit mask gives the first matching lane.
 *
 * Requires: -mavx512f -mavx512bw
 */

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <immintrin.h>   /* AVX-512 */

// ================================================================================
// Internal helpers
// ================================================================================

static inline __m512 _avx512_fabs_ps(__m512 v) {
    __m512 sign_mask = _mm512_set1_ps(-0.0f);
    return _mm512_andnot_ps(sign_mask, v);
}

/* Exact bitwise match: cast to int32 lanes, compare, return 16-bit mask. */
static inline __mmask16 _avx512_exact_mask(__m512 chunk, __m512 vn) {
    __m512i ci = _mm512_castps_si512(chunk);
    __m512i ni = _mm512_castps_si512(vn);
    return _mm512_cmpeq_epi32_mask(ci, ni);
}

static inline __mmask16 _avx512_tol_mask(__m512 chunk, __m512 vn,
                                          __m512 vabs_tol, __m512 vrel_tol,
                                          __m512 vabs_needle) {
    __m512 diff       = _avx512_fabs_ps(_mm512_sub_ps(chunk, vn));
    __m512 abs_chunk  = _avx512_fabs_ps(chunk);
    __m512 rel_base   = _mm512_max_ps(abs_chunk, vabs_needle);
    __m512 rel_thresh = _mm512_mul_ps(vrel_tol, rel_base);
    __m512 threshold  = _mm512_max_ps(vabs_tol, rel_thresh);
    return _mm512_cmp_ps_mask(diff, threshold, _CMP_LE_OQ);
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
    __m512 vn = _mm512_set1_ps(needle);

    if (abs_tol == 0.0f && rel_tol == 0.0f) {
        while (i + 16u <= end) {
            __m512    chunk = _mm512_loadu_ps(data + i);
            __mmask16 mask  = _avx512_exact_mask(chunk, vn);
            if (mask != 0) {
                return i + (size_t)__builtin_ctz((unsigned)mask);
            }
            i += 16u;
        }
    } else {
        __m512 vabs_tol    = _mm512_set1_ps(abs_tol);
        __m512 vrel_tol    = _mm512_set1_ps(rel_tol);
        __m512 vabs_needle = _mm512_set1_ps(fabsf(needle));

        while (i + 16u <= end) {
            __m512    chunk = _mm512_loadu_ps(data + i);
            __mmask16 mask  = _avx512_tol_mask(chunk, vn, vabs_tol, vrel_tol,
                                               vabs_needle);
            if (mask != 0) {
                return i + (size_t)__builtin_ctz((unsigned)mask);
            }
            i += 16u;
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
    size_t best      = SIZE_MAX;
    float  abs_needle = fabsf(needle);
    __m512 vn        = _mm512_set1_ps(needle);
    __m512 vabs_tol  = _mm512_set1_ps(abs_tol);
    __m512 vrel_tol  = _mm512_set1_ps(rel_tol);
    __m512 vabs_ndl  = _mm512_set1_ps(abs_needle);

    /* Check landing */
    {
        float diff = fabsf(data[mid] - needle);
        float thr  = fmaxf(abs_tol, rel_tol * fmaxf(fabsf(data[mid]), abs_needle));
        if (diff <= thr) best = mid;
    }

    /* SIMD left walk — 16 floats per step */
    size_t L = mid;
    while (L >= 16u) {
        size_t    base  = L - 16u;
        __m512    chunk = _mm512_loadu_ps(data + base);
        __mmask16 mask  = _avx512_tol_mask(chunk, vn, vabs_tol, vrel_tol, vabs_ndl);
        if (mask == 0xFFFFu) { best = base; L = base; continue; }
        for (size_t e = 0; e < 16u && base + e < L; e++) {
            if ((mask >> (unsigned)e) & 1u) { best = base + e; }
            else                            { L = base + e; goto left_done_avx512; }
        }
        L = base;
    }
    left_done_avx512:
    while (L > 0u) {
        float diff = fabsf(data[L - 1u] - needle);
        float thr  = fmaxf(abs_tol, rel_tol * fmaxf(fabsf(data[L - 1u]), abs_needle));
        if (diff <= thr) { best = --L; } else { break; }
    }

    /* SIMD right walk */
    if (best == SIZE_MAX) {
        size_t R = mid + 1u;
        while (R + 16u <= len) {
            __m512    chunk = _mm512_loadu_ps(data + R);
            __mmask16 mask  = _avx512_tol_mask(chunk, vn, vabs_tol, vrel_tol, vabs_ndl);
            if (mask == 0) break;
            for (size_t e = 0; e < 16u && R + e < len; e++) {
                if ((mask >> (unsigned)e) & 1u) { if (best == SIZE_MAX) best = R + e; }
                else                            { goto right_done_avx512; }
            }
            R += 16u;
        }
        right_done_avx512:
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
#endif /* SIMD_AVX512_FLOAT_INL */ 
// ================================================================================ 
// ================================================================================ 
// eof
