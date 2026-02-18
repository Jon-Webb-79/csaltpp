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
// -------------------------------------------------------------------------------- 

#ifndef SIZE_MAX
  #define SIZE_MAX ((size_t)-1)
#endif

typedef enum {
    FORWARD = 0,
    REVERSE = 1
} direction_t;

/* scalar helper: forward search */
static inline size_t simd_find_substr_u8_forward_(const uint8_t* hay,
                                                  size_t hay_len,
                                                  const uint8_t* needle,
                                                  size_t needle_len)
{
    if ((hay == NULL) || (needle == NULL)) { return SIZE_MAX; }
    if (needle_len == 0u) { return 0u; }
    if (needle_len > hay_len) { return SIZE_MAX; }

    uint8_t const first = needle[0];

    for (size_t i = 0u; i <= (hay_len - needle_len); ++i) {
        if (hay[i] != first) { continue; }
        if (needle_len == 1u) { return i; }
        if (memcmp(hay + i + 1u, needle + 1u, needle_len - 1u) == 0) {
            return i;
        }
    }
    return SIZE_MAX;
}

/* scalar helper: reverse search (returns last occurrence) */
static inline size_t simd_find_substr_u8_reverse_(const uint8_t* hay,
                                                  size_t hay_len,
                                                  const uint8_t* needle,
                                                  size_t needle_len)
{
    if ((hay == NULL) || (needle == NULL)) { return SIZE_MAX; }
    if (needle_len == 0u) { return 0u; }
    if (needle_len > hay_len) { return SIZE_MAX; }

    uint8_t const first = needle[0];

    /* last possible start index */
    for (size_t i = (hay_len - needle_len) + 1u; i-- > 0u; ) {
        if (hay[i] != first) { continue; }
        if (needle_len == 1u) { return i; }
        if (memcmp(hay + i + 1u, needle + 1u, needle_len - 1u) == 0) {
            return i;
        }
        if (i == 0u) { break; } /* avoid wrap in i-- > 0 loop */
    }
    return SIZE_MAX;
}

static inline size_t simd_find_substr_u8(const uint8_t* hay,
                                         size_t hay_len,
                                         const uint8_t* needle,
                                         size_t needle_len,
                                         direction_t dir)
{
    if (dir == REVERSE) {
        return simd_find_substr_u8_reverse_(hay, hay_len, needle, needle_len);
    }
    return simd_find_substr_u8_forward_(hay, hay_len, needle, needle_len);
}
// -------------------------------------------------------------------------------- 

size_t simd_token_count_u8(const uint8_t* s, size_t n,
                           const char* delim, size_t dlen) {
    if ((s == NULL) || (delim == NULL)) return SIZE_MAX;
    if (n == 0u) return 0u;
    if (dlen == 0u) return 1u;

    uint8_t lut[256];
    memset(lut, 0, sizeof(lut));
    for (size_t j = 0; j < dlen; ++j) {
        lut[(unsigned char)delim[j]] = 1u;
    }

    size_t count = 0u;
    bool in_token = false;

    for (size_t i = 0u; i < n; ++i) {
        bool is_delim = (lut[s[i]] != 0u);
        if (!is_delim) {
            if (!in_token) { ++count; in_token = true; }
        } else {
            in_token = false;
        }
    }
    return count;
}
// ================================================================================ 
// ================================================================================ 
#endif /* CSALT_SIMD_AVX2_CHAR_INL */
// ================================================================================ 
// ================================================================================ 
// eof
