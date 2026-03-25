#ifndef SIMD_NEON_FLOAT_INL
#define SIMD_NEON_FLOAT_INL
// ================================================================================ 
// ================================================================================ 

/**
 * @file simd_neon_float.inl
 * @brief ARM NEON implementation of tolerance-aware float search.
 *
 * Processes 4 floats per iteration using 128-bit NEON Q registers.
 *
 * Exact mode (both tolerances == 0.0f):
 *   vceqq_u32 on the reinterpreted float register provides bitwise
 *   equality: NaN != NaN, -0.0f (0x80000000) != +0.0f.  vmaxvq_u32
 *   reduces the 4-lane comparison result to a scalar to check for any
 *   hit before scanning lanes with vgetq_lane_u32.
 *
 * Tolerance mode:
 *   vabsq_f32 computes |element|, vsubq_f32 + vabsq_f32 computes the
 *   absolute difference, vmaxq_f32 + vmulq_f32 builds the per-lane
 *   threshold, and vcleq_f32 (compare less-than-or-equal) produces the
 *   match predicate.  vmaxvq_u32 on the cast result detects any hit.
 *
 * Requires: -mfpu=neon (ARMv7) or implicit on AArch64
 */

#include <stdint.h>
#include <stddef.h>
#include <math.h>
#include <arm_neon.h>

// ================================================================================
// Public interface
// ================================================================================

static size_t simd_contains_float(const float* data,
                                   size_t       start,
                                   size_t       end,
                                   float        needle,
                                   float        abs_tol,
                                   float        rel_tol) {
    size_t i = start;

    if (abs_tol == 0.0f && rel_tol == 0.0f) {
        /*
         * Exact bitwise match. Reinterpret as uint32 so that
         * -0.0f and NaN are handled correctly.
         */
        uint32x4_t vn = vdupq_n_u32(*(const uint32_t*)(const void*)&needle);

        while (i + 4u <= end) {
            uint32x4_t chunk = vld1q_u32((const uint32_t*)(data + i));
            uint32x4_t cmp   = vceqq_u32(chunk, vn);
            if (vmaxvq_u32(cmp) != 0u) {
                uint32_t tmp[4];
                vst1q_u32(tmp, cmp);
                for (size_t e = 0; e < 4u && i + e < end; e++) {
                    if (tmp[e] != 0u) return i + e;
                }
            }
            i += 4u;
        }
    } else {
        float32x4_t vn          = vdupq_n_f32(needle);
        float32x4_t vabs_tol    = vdupq_n_f32(abs_tol);
        float32x4_t vrel_tol    = vdupq_n_f32(rel_tol);
        float32x4_t vabs_needle = vdupq_n_f32(fabsf(needle));

        while (i + 4u <= end) {
            float32x4_t chunk     = vld1q_f32(data + i);
            float32x4_t diff      = vabsq_f32(vsubq_f32(chunk, vn));
            float32x4_t abs_chunk = vabsq_f32(chunk);
            float32x4_t rel_base  = vmaxq_f32(abs_chunk, vabs_needle);
            float32x4_t rel_thresh = vmulq_f32(vrel_tol, rel_base);
            float32x4_t threshold  = vmaxq_f32(vabs_tol, rel_thresh);
            /* vcleq_f32: returns 0xFFFFFFFF per lane where diff <= threshold */
            uint32x4_t  cmp        = vcleq_f32(diff, threshold);
            if (vmaxvq_u32(cmp) != 0u) {
                uint32_t tmp[4];
                vst1q_u32(tmp, cmp);
                for (size_t e = 0; e < 4u && i + e < end; e++) {
                    if (tmp[e] != 0u) return i + e;
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
    size_t best      = SIZE_MAX;
    float  abs_needle = fabsf(needle);
    float32x4_t vn        = vdupq_n_f32(needle);
    float32x4_t vabs_tol  = vdupq_n_f32(abs_tol);
    float32x4_t vrel_tol  = vdupq_n_f32(rel_tol);
    float32x4_t vabs_ndl  = vdupq_n_f32(abs_needle);

    /* Check landing */
    {
        float diff = fabsf(data[mid] - needle);
        float thr  = fmaxf(abs_tol, rel_tol * fmaxf(fabsf(data[mid]), abs_needle));
        if (diff <= thr) best = mid;
    }

    /* SIMD left walk — 4 floats per step */
    size_t L = mid;
    while (L >= 4u) {
        size_t      base  = L - 4u;
        float32x4_t chunk = vld1q_f32(data + base);
        float32x4_t diff  = vabsq_f32(vsubq_f32(chunk, vn));
        float32x4_t abs_c = vabsq_f32(chunk);
        float32x4_t thr   = vmaxq_f32(vabs_tol, vmulq_f32(vrel_tol,
                                vmaxq_f32(abs_c, vabs_ndl)));
        uint32x4_t  cmp   = vcleq_f32(diff, thr);
        uint32_t    tmp[4];
        vst1q_u32(tmp, cmp);
        if (tmp[0] && tmp[1] && tmp[2] && tmp[3]) {
            best = base; L = base; continue;
        }
        for (size_t e = 0; e < 4u && base + e < L; e++) {
            if (tmp[e]) { best = base + e; }
            else        { L = base + e; goto left_done_neon; }
        }
        L = base;
    }
    left_done_neon:
    while (L > 0u) {
        float diff = fabsf(data[L - 1u] - needle);
        float thr  = fmaxf(abs_tol, rel_tol * fmaxf(fabsf(data[L - 1u]), abs_needle));
        if (diff <= thr) { best = --L; } else { break; }
    }

    /* SIMD right walk */
    if (best == SIZE_MAX) {
        size_t R = mid + 1u;
        while (R + 4u <= len) {
            float32x4_t chunk = vld1q_f32(data + R);
            float32x4_t diff  = vabsq_f32(vsubq_f32(chunk, vn));
            float32x4_t abs_c = vabsq_f32(chunk);
            float32x4_t thr   = vmaxq_f32(vabs_tol, vmulq_f32(vrel_tol,
                                    vmaxq_f32(abs_c, vabs_ndl)));
            uint32x4_t  cmp   = vcleq_f32(diff, thr);
            uint32_t    tmp[4];
            vst1q_u32(tmp, cmp);
            if (!tmp[0]) break;
            for (size_t e = 0; e < 4u && R + e < len; e++) {
                if (tmp[e]) { if (best == SIZE_MAX) best = R + e; }
                else        { goto right_done_neon; }
            }
            R += 4u;
        }
        right_done_neon:
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
#endif /* SIMD_NEON_FLOAT_INL */ 
// ================================================================================ 
// ================================================================================ 
// eof
