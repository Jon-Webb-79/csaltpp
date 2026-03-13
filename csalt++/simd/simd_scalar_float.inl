#ifndef SIMD_SCALAR_FLOAT_INL
#define SIMD_SCALAR_FLOAT_INL
// ================================================================================ 
// ================================================================================ 

/**
 * @file simd_scalar_float.inl
 * @brief Scalar fallback implementation of tolerance-aware float search.
 *
 * Performs a linear scan over elements in [start, end). When both abs_tol
 * and rel_tol are 0.0f the comparison reduces to bitwise equality via a
 * union reinterpretation, matching the exact bit-pattern semantics of the
 * integer SIMD paths (NaN != NaN, -0.0f != +0.0f). When either tolerance
 * is non-zero the combined test
 *
 *     fabsf(element - needle) <= fmaxf(abs_tol, rel_tol * fmaxf(|element|, |needle|))
 *
 * is applied.  Passing abs_tol = 0.0f selects pure relative mode; passing
 * rel_tol = 0.0f selects pure absolute mode.
 *
 * This is the reference implementation against which all SIMD paths are
 * verified and the target compiled when no SIMD flag is set.
 */

#include <stdint.h>
#include <stddef.h>
#include <math.h>

// ================================================================================
// Scalar contains
//
// Returns the index of the first element in [start, end) that matches
// needle within the supplied tolerances, or SIZE_MAX if no match is found.
// ================================================================================

static size_t simd_contains_float(const float* data,
                                   size_t       start,
                                   size_t       end,
                                   float        needle,
                                   float        abs_tol,
                                   float        rel_tol) {
    if (abs_tol == 0.0f && rel_tol == 0.0f) {
        /*
         * Exact bit-pattern match. Use a union to reinterpret both values as
         * uint32_t so that the comparison is bitwise: NaN != NaN,
         * -0.0f (0x80000000) != +0.0f (0x00000000).
         */
        union { float f; uint32_t u; } n;
        n.f = needle;
        for (size_t i = start; i < end; i++) {
            union { float f; uint32_t u; } v;
            v.f = data[i];
            if (v.u == n.u) return i;
        }
        return SIZE_MAX;
    }

    for (size_t i = start; i < end; i++) {
        float diff      = fabsf(data[i] - needle);
        float threshold = fmaxf(abs_tol,
                                rel_tol * fmaxf(fabsf(data[i]), fabsf(needle)));
        if (diff <= threshold) return i;
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

    /* ── Tolerance fan-out ──────────────────────────────────────────────────
     * The bisection left lo == hi at the landing position closest to needle.
     * Because the array is sorted, all elements within tolerance form a
     * contiguous run.  We check the landing position, then walk left and
     * right, stopping as soon as an element falls outside tolerance.  The
     * lowest matching index is returned.
     *
     * When both tolerances are 0.0f the fan-out degenerates to an exact
     * match check plus the original leftward scan for first occurrence.
     */
    size_t best = SIZE_MAX;

    /* Inline tolerance test */
#define _IN_TOL(idx) \
    (fabsf(data[(idx)] - needle) <= \
         fmaxf(abs_tol, rel_tol * fmaxf(fabsf(data[(idx)]), fabsf(needle))))

    /* Check landing position */
    if (_IN_TOL(mid)) best = mid;

    /* Walk left */
    size_t L = mid;
    while (L > 0u && _IN_TOL(L - 1u)) { L--; best = L; }

    /* Walk right — only improves result if landing itself didn't match */
    if (best == SIZE_MAX) {
        size_t R = mid + 1u;
        while (R < len && _IN_TOL(R)) { best = (best == SIZE_MAX) ? R : best; R++; }
    }

#undef _IN_TOL
    return best;
}
// ================================================================================ 
// ================================================================================ 
#endif /* SIMD_SCALAR_FLOAT_INL */
// ================================================================================ 
// ================================================================================ 
// eof
