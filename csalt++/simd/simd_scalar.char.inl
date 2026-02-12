/* simd_avx2_char.inl
   AVX2 helpers for last-occurrence search in unsigned bytes.
   Requires: <immintrin.h>, compiled with -mavx2
*/
#ifndef CSALT_SIMD_SCALAR_CHAR_INL
#define CSALT_SIMD_SCALAR_CHAR_INL

#include <stddef.h>
#include <stdint.h>
// ================================================================================ 
// ================================================================================ 

/* Returns:
 *   n  if a[0..n-1] == b[0..n-1]
 *   i  (0..n-1) index of first mismatch otherwise
 *
 * Safety:
 *   - Never reads beyond n bytes.
 * Notes:
 *   - If a or b is NULL and n > 0, returns 0 (defensive).
 *     Prefer guarding against NULL at the caller for stricter semantics.
 */
static inline size_t csalt_first_diff_u8(const uint8_t* a,
                                        const uint8_t* b,
                                        size_t n) {
    if ((n == 0u)) {
        return 0u;
    }
    if ((a == NULL) || (b == NULL)) {
        return 0u;
    }

    for (size_t i = 0u; i < n; ++i) {
        if (a[i] != b[i]) {
            return i;
        }
    }
    return n;
}
// ================================================================================ 
// ================================================================================ 
#endif /* CSALT_SIMD_AVX2_CHAR_INL */
// ================================================================================ 
// ================================================================================ 
// eof
