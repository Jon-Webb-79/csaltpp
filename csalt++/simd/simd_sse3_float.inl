#ifndef SIMD_SSE3_FLOAT_INL
#define SIMD_SSE3_FLOAT_INL
// ================================================================================ 
// ================================================================================ 

/**
 * @file simd_sse3_float.inl
 * @brief SSE3 implementation of tolerance-aware float search.
 *
 * SSE3 adds no floating-point comparison instructions beyond SSE2. The
 * meaningful addition for this operation is _mm_lddqu_si128, an unaligned
 * load hint that may perform better than _mm_loadu_si128 on some Intel
 * microarchitectures when the data crosses a cache-line boundary.  The
 * comparison and mask logic is otherwise identical to the SSE2 path.
 *
 * Requires: -msse3
 */

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <pmmintrin.h>   /* SSE3 */

// ================================================================================
// Internal helpers (identical to SSE2 path)
// ================================================================================

static inline __m128 _sse3_fabs_ps(__m128 v) {
    __m128i mask = _mm_set1_epi32(0x7FFFFFFF);
    return _mm_and_ps(v, _mm_castsi128_ps(mask));
}

static inline int _sse3_exact_mask(__m128 chunk, __m128 vn) {
    __m128i ci  = _mm_castps_si128(chunk);
    __m128i ni  = _mm_castps_si128(vn);
    __m128i cmp = _mm_cmpeq_epi32(ci, ni);
    return _mm_movemask_ps(_mm_castsi128_ps(cmp));
}

static inline int _sse3_tol_mask(__m128 chunk, __m128 vn,
                                  __m128 vabs_tol, __m128 vrel_tol,
                                  __m128 vabs_needle) {
    __m128 diff       = _sse3_fabs_ps(_mm_sub_ps(chunk, vn));
    __m128 abs_chunk  = _sse3_fabs_ps(chunk);
    __m128 rel_base   = _mm_max_ps(abs_chunk, vabs_needle);
    __m128 rel_thresh = _mm_mul_ps(vrel_tol, rel_base);
    __m128 threshold  = _mm_max_ps(vabs_tol, rel_thresh);
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
    size_t i  = start;
    __m128 vn = _mm_set1_ps(needle);

    if (abs_tol == 0.0f && rel_tol == 0.0f) {
        while (i + 4u <= end) {
            /*
             * _mm_lddqu_si128: unaligned load with a hint that the data may
             * cross a cache-line boundary. Cast back to __m128 for comparison.
             */
            __m128 chunk = _mm_castsi128_ps(
                _mm_lddqu_si128((__m128i*)(data + i)));
            int mask = _sse3_exact_mask(chunk, vn);
            if (mask != 0) {
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
            __m128 chunk = _mm_castsi128_ps(
                _mm_lddqu_si128((__m128i*)(data + i)));
            int mask = _sse3_tol_mask(chunk, vn, vabs_tol, vrel_tol,
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
        __m128 chunk = _mm_castsi128_ps(_mm_lddqu_si128((__m128i*)(data + base)));
        int    mask  = _sse3_tol_mask(chunk, vn, vabs_tol, vrel_tol, vabs_ndl);
        /* All 4 match — the entire block is within tolerance */
        if (mask == 0xF) { best = base; L = base; continue; }
        /* Partial match — find the rightmost matching lane from low end */
        for (size_t e = 0; e < 4u && base + e < L; e++) {
            if ((mask >> (int)e) & 1) { best = base + e; }
            else                      { L = base + e; goto left_done_sse3; }
        }
        L = base;
    }
    left_done_sse3:
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
            int    mask  = _sse3_tol_mask(chunk, vn, vabs_tol, vrel_tol, vabs_ndl);
            if (mask == 0) break;   /* none match — tolerance region ended */
            for (size_t e = 0; e < 4u && R + e < len; e++) {
                if ((mask >> (int)e) & 1) { if (best == SIZE_MAX) best = R + e; }
                else                      { goto right_done_sse3; }
            }
            R += 4u;
        }
        right_done_sse3:
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
#endif /* SIMD_SSE3_FLOAT_INL */ 
// ================================================================================ 
// ================================================================================ 
// eof
