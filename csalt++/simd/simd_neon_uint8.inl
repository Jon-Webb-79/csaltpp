
#ifndef CSALT_SIMD_NEON_UINT8_INL
#define CSALT_SIMD_NEON_UINT8_INL

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <arm_sve.h>
// ================================================================================ 
// ================================================================================ 

static void simd_reverse_uint8(uint8_t* data, size_t len, size_t data_size) {
    if (data == NULL || len < 2u || data_size == 0u) return;

    if (data_size <= 16u && (16u % data_size == 0u)) {
        /* Query the hardware vector length in bytes at runtime */
        size_t vl          = svcntb();
        size_t elems_per_vec = vl / data_size;

        size_t lo = 0u;
        size_t hi = len - 1u;

        while (lo < hi && (hi - lo + 1u) >= 2u * elems_per_vec) {
            uint8_t* lo_ptr = data + lo * data_size;
            uint8_t* hi_ptr = data + hi * data_size;

            svbool_t pg = svptrue_b8();

            svuint8_t vlo = svld1_u8(pg, lo_ptr);
            svuint8_t vhi = svld1_u8(pg, hi_ptr);

            /* svrev_u8 reverses all active bytes within the vector */
            vlo = svrev_u8(vlo);
            vhi = svrev_u8(vhi);

            svst1_u8(pg, hi_ptr, vlo);
            svst1_u8(pg, lo_ptr, vhi);

            lo += elems_per_vec;
            hi -= elems_per_vec;
        }

        /* Scalar remainder */
        uint8_t tmp[256];
        while (lo < hi) {
            uint8_t* lo_ptr = data + lo * data_size;
            uint8_t* hi_ptr = data + hi * data_size;
            memcpy(tmp,    lo_ptr, data_size);
            memcpy(lo_ptr, hi_ptr, data_size);
            memcpy(hi_ptr, tmp,    data_size);
            lo++;
            hi--;
        }
        return;
    }

    /* Scalar fallback for non-aligned element sizes */
    size_t lo = 0u;
    size_t hi = len - 1u;
    uint8_t tmp[256];

    if (data_size <= sizeof(tmp)) {
        while (lo < hi) {
            uint8_t* lo_ptr = data + lo * data_size;
            uint8_t* hi_ptr = data + hi * data_size;
            memcpy(tmp,    lo_ptr, data_size);
            memcpy(lo_ptr, hi_ptr, data_size);
            memcpy(hi_ptr, tmp,    data_size);
            lo++; hi--;
        }
    } else {
        while (lo < hi) {
            uint8_t* lo_ptr = data + lo * data_size;
            uint8_t* hi_ptr = data + hi * data_size;
            for (size_t b = 0u; b < data_size; b++) {
                uint8_t byte = lo_ptr[b];
                lo_ptr[b]    = hi_ptr[b];
                hi_ptr[b]    = byte;
            }
            lo++; hi--;
        }
    }
}
// -------------------------------------------------------------------------------- 

static size_t simd_contains_uint8(const uint8_t* data,
                                   size_t         start,
                                   size_t         end,
                                   size_t         data_size,
                                   const uint8_t* needle) {
    size_t i = start;

    if (data_size == 1u) {
        uint8x16_t vn = vdupq_n_u8(*needle);
        while (i + 16u <= end) {
            uint8x16_t chunk = vld1q_u8(data + i);
            uint8x16_t cmp   = vceqq_u8(chunk, vn);
            /* vmaxvq_u8: horizontal max — non-zero means at least one match */
            if (vmaxvq_u8(cmp) != 0) {
                /* Find the exact lane */
                uint8_t tmp[16];
                vst1q_u8(tmp, cmp);
                for (size_t e = 0; e < 16u && i + e < end; e++) {
                    if (tmp[e] != 0) return i + e;
                }
            }
            i += 16u;
        }
    } else if (data_size == 2u) {
        uint16x8_t vn = vdupq_n_u16(*(const uint16_t*)needle);
        while (i + 8u <= end) {
            uint16x8_t chunk = vld1q_u16((const uint16_t*)(data + i * 2u));
            uint16x8_t cmp   = vceqq_u16(chunk, vn);
            if (vmaxvq_u16(cmp) != 0) {
                uint16_t tmp[8];
                vst1q_u16(tmp, cmp);
                for (size_t e = 0; e < 8u && i + e < end; e++) {
                    if (tmp[e] != 0) return i + e;
                }
            }
            i += 8u;
        }
    } else if (data_size == 4u) {
        uint32x4_t vn = vdupq_n_u32(*(const uint32_t*)needle);
        while (i + 4u <= end) {
            uint32x4_t chunk = vld1q_u32((const uint32_t*)(data + i * 4u));
            uint32x4_t cmp   = vceqq_u32(chunk, vn);
            if (vmaxvq_u32(cmp) != 0) {
                uint32_t tmp[4];
                vst1q_u32(tmp, cmp);
                for (size_t e = 0; e < 4u && i + e < end; e++) {
                    if (tmp[e] != 0) return i + e;
                }
            }
            i += 4u;
        }
    } else if (data_size == 8u) {
        uint64x2_t vn = vdupq_n_u64(*(const uint64_t*)needle);
        while (i + 2u <= end) {
            uint64x2_t chunk = vld1q_u64((const uint64_t*)(data + i * 8u));
            uint64x2_t cmp   = vceqq_u64(chunk, vn);
            if (vmaxvq_u64(cmp) != 0) {
                if (vgetq_lane_u64(cmp, 0) != 0 && i     < end) return i;
                if (vgetq_lane_u64(cmp, 1) != 0 && i + 1 < end) return i + 1u;
            }
            i += 2u;
        }
    }

    /* Scalar remainder for all sizes */
    for (; i < end; i++) {
        if (memcmp(data + i * data_size, needle, data_size) == 0) {
            return i;
        }
    }
    return SIZE_MAX;
}
// ================================================================================ 
// ================================================================================ 
#endif /* CSALT_SIMD_NEON_UINT8_INL */

