
#ifndef CSALT_SIMD_SCALAR_UINT8_INL
#define CSALT_SIMD_SCALAR_UINT8_INL

#include <stdint.h>
#include <stddef.h>
#include <string.h>
// ================================================================================ 
// ================================================================================ 

static void simd_reverse_uint8(uint8_t* data, size_t len, size_t data_size) {
    if (data == NULL || len < 2u || data_size == 0u) return;

    size_t lo = 0u;
    size_t hi = len - 1u;

    /* Stack buffer for the swap. Sized to cover all built-in types and
       reasonably sized user structs without heap allocation. Falls back
       to a byte-by-byte swap for anything larger. */
    uint8_t tmp[256];

    if (data_size <= sizeof(tmp)) {
        while (lo < hi) {
            uint8_t* lo_ptr = data + lo * data_size;
            uint8_t* hi_ptr = data + hi * data_size;

            memcpy(tmp,     lo_ptr,  data_size);
            memcpy(lo_ptr,  hi_ptr,  data_size);
            memcpy(hi_ptr,  tmp,     data_size);

            lo++;
            hi--;
        }
    } else {
        /* data_size > 256: byte-by-byte swap to avoid VLAs */
        while (lo < hi) {
            uint8_t* lo_ptr = data + lo * data_size;
            uint8_t* hi_ptr = data + hi * data_size;

            for (size_t b = 0u; b < data_size; b++) {
                uint8_t byte   = lo_ptr[b];
                lo_ptr[b]      = hi_ptr[b];
                hi_ptr[b]      = byte;
            }

            lo++;
            hi--;
        }
    }
}
// -------------------------------------------------------------------------------- 

static size_t simd_contains_uint8(const uint8_t* data,
                                   size_t         start,
                                   size_t         end,
                                   size_t         data_size,
                                   const uint8_t* needle) {
    for (size_t i = start; i < end; i++) {
        if (memcmp(data + i * data_size, needle, data_size) == 0) {
            return i;
        }
    }
    return SIZE_MAX;
}
// ================================================================================ 
// ================================================================================ 
#endif /* CSALT_SIMD_SCALAR_UINT8_INL */

