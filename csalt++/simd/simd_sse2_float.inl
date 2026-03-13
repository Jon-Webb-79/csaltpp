#ifndef SIMD_SSE2_FLOAT_INL
#define SIMD_SSE2_FLOAT_INL
// ================================================================================ 
// ================================================================================ 

/**
 * @file simd_sse2_float.inl
 * @brief SSE2 implementation of tolerance-aware float search.
 *
 * Processes 4 floats per iteration using 128-bit XMM registers.
 *
 * Exact mode (both tolerances == 0.0f):
 *   Reinterprets the float register as integer with _mm_castps_si128 and
 *   uses _mm_cmpeq_epi32 for bitwise equality — identical semantics to the
 *   uint32 path in contains_sse2_uint8.inl.  NaN != NaN, -0.0f != +0.0f.
 *
 * Tolerance mode (either tolerance > 0.0f):
 *   Computes |element - needle| <= max(abs_tol, rel_tol * max(|element|, |needle|))
 *   per lane using _mm_cmpps/subps/andps/maxps.  The sign bit is cleared
 *   with _mm_andps against a mask of 0x7FFFFFFF to implement fabsf.
 *
 * Requires: -msse2 (implicit on x86-64)
 */

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <emmintrin.h>   /* SSE2 */

// ================================================================================
// Internal helpers
// ================================================================================

/* Absolute value mask: clears the sign bit of each 32-bit lane. */
static inline __m128 _sse2_fabs_ps(__m128 v) {
    __m128i mask = _mm_set1_epi32(0x7FFFFFFF);
    return _mm_and_ps(v, _mm_castsi128_ps(mask));
}

/*
 * For exact mode: return a 4-bit mask (one bit per lane) where bit k is set
 * if lane k matches needle exactly (bitwise).
 */
static inline int _sse2_exact_mask(__m128 chunk, __m128 vn) {
    __m128i ci = _mm_castps_si128(chunk);
    __m128i ni = _mm_castps_si128(vn);
    __m128i cmp = _mm_cmpeq_epi32(ci, ni);
    /* movemask_ps gives one bit per 32-bit lane based on sign bit */
    return _mm_movemask_ps(_mm_castsi128_ps(cmp));
}

/*
 * For tolerance mode: return a 4-bit mask where bit k is set if lane k
 * satisfies |chunk[k] - needle| <= max(abs_tol, rel_tol * max(|chunk[k]|, |needle|)).
 */
static inline int _sse2_tol_mask(__m128 chunk, __m128 vn,
                                  __m128 vabs_tol, __m128 vrel_tol,
                                  __m128 vabs_needle) {
    __m128 diff      = _sse2_fabs_ps(_mm_sub_ps(chunk, vn));
    __m128 abs_chunk = _sse2_fabs_ps(chunk);
    __m128 rel_base  = _mm_max_ps(abs_chunk, vabs_needle);
    __m128 rel_thresh = _mm_mul_ps(vrel_tol, rel_base);
    __m128 threshold  = _mm_max_ps(vabs_tol, rel_thresh);
    /* diff <= threshold: _mm_cmple_ps sets all bits in lane if true */
    return _mm_movemask_ps(_mm_cmple_ps(diff, threshold));
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
    size_t  i  = start;
    __m128  vn = _mm_set1_ps(needle);

    if (abs_tol == 0.0f && rel_tol == 0.0f) {
        /* Exact bitwise match — 4 floats per iteration */
        while (i + 4u <= end) {
            __m128 chunk = _mm_loadu_ps(data + i);
            int    mask  = _sse2_exact_mask(chunk, vn);
            if (mask != 0) {
                /* Find the first set bit (each lane is one bit in movemask_ps) */
                for (size_t e = 0; e < 4u && i + e < end; e++) {
                    if ((mask >> (int)e) & 1) return i + e;
                }
            }
            i += 4u;
        }
    } else {
        __m128 vabs_tol    = _mm_set1_ps(abs_tol);
        __m128 vrel_tol    = _mm_set1_ps(rel_tol);
        __m128 vabs_needle = _mm_set1_ps(fabsf(needle));

        while (i + 4u <= end) {
            __m128 chunk = _mm_loadu_ps(data + i);
            int    mask  = _sse2_tol_mask(chunk, vn, vabs_tol, vrel_tol,
                                          vabs_needle);
            if (mask != 0) {
                for (size_t e = 0; e < 4u && i + e < end; e++) {
                    if ((mask >> (int)e) & 1) return i + e;
                }
            }
            i += 4u;
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
    size_t best = SIZE_MAX;
    float  abs_needle = fabsf(needle);
    __m128 vn         = _mm_set1_ps(needle);
    __m128 vabs_tol   = _mm_set1_ps(abs_tol);
    __m128 vrel_tol   = _mm_set1_ps(rel_tol);
    __m128 vabs_ndl   = _mm_set1_ps(abs_needle);

    /* Check landing */
    {
        float diff = fabsf(data[mid] - needle);
        float thr  = fmaxf(abs_tol, rel_tol * fmaxf(fabsf(data[mid]), abs_needle));
        if (diff <= thr) best = mid;
    }

    /* SIMD left walk — 4 floats per step */
    size_t L = mid;
    while (L >= 4u) {
        size_t base  = L - 4u;
        __m128 chunk = _mm_loadu_ps(data + base);
        int    mask  = _sse2_tol_mask(chunk, vn, vabs_tol, vrel_tol, vabs_ndl);
        /* All 4 match — the entire block is within tolerance */
        if (mask == 0xF) { best = base; L = base; continue; }
        /* Partial match — find the rightmost matching lane from low end */
        for (size_t e = 0; e < 4u && base + e < L; e++) {
            if ((mask >> (int)e) & 1) { best = base + e; }
            else                      { L = base + e; goto left_done_sse2; }
        }
        L = base;
    }
    left_done_sse2:
    /* Scalar left mop-up */
    while (L > 0u) {
        float diff = fabsf(data[L - 1u] - needle);
        float thr  = fmaxf(abs_tol, rel_tol * fmaxf(fabsf(data[L - 1u]), abs_needle));
        if (diff <= thr) { best = --L; } else { break; }
    }

    /* SIMD right walk — only if landing didn't match */
    if (best == SIZE_MAX) {
        size_t R = mid + 1u;
        while (R + 4u <= len) {
            __m128 chunk = _mm_loadu_ps(data + R);
            int    mask  = _sse2_tol_mask(chunk, vn, vabs_tol, vrel_tol, vabs_ndl);
            if (mask == 0) break;   /* none match — tolerance region ended */
            for (size_t e = 0; e < 4u && R + e < len; e++) {
                if ((mask >> (int)e) & 1) { if (best == SIZE_MAX) best = R + e; }
                else                      { goto right_done_sse2; }
            }
            R += 4u;
        }
        right_done_sse2:
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
#endif /* SIMD_SSE2_FLOAT_INL */ 
// ================================================================================ 
// ================================================================================ 
// eof
