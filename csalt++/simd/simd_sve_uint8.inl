
#ifndef CSALT_SIMD_SVE_UINT8_INL
#define CSALT_SIMD_SVE_UINT8_INL

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
        size_t vl = svcntb();
        while (i + vl <= end) {
            svbool_t   pg  = svptrue_b8();
            svuint8_t  vn  = svdup_n_u8(*needle);
            svuint8_t  chunk = svld1_u8(pg, data + i);
            svbool_t   cmp   = svcmpeq_u8(pg, chunk, vn);
            if (svptest_any(pg, cmp)) {
                /* svfirst_b returns the lane index of first active predicate */
                return i + (size_t)svcntp_b8(svptrue_b8(),
                                              svbrkb_z(pg, cmp));
            }
            i += vl;
        }
    } else if (data_size == 2u) {
        size_t vl = svcnth();
        while (i + vl <= end) {
            svbool_t    pg    = svptrue_b16();
            svuint16_t  vn    = svdup_n_u16(*(const uint16_t*)needle);
            svuint16_t  chunk = svld1_u16(pg, (const uint16_t*)(data + i * 2u));
            svbool_t    cmp   = svcmpeq_u16(pg, chunk, vn);
            if (svptest_any(pg, cmp)) {
                return i + (size_t)svcntp_b16(svptrue_b16(),
                                               svbrkb_z(pg, cmp));
            }
            i += vl;
        }
    } else if (data_size == 4u) {
        size_t vl = svcntw();
        while (i + vl <= end) {
            svbool_t    pg    = svptrue_b32();
            svuint32_t  vn    = svdup_n_u32(*(const uint32_t*)needle);
            svuint32_t  chunk = svld1_u32(pg, (const uint32_t*)(data + i * 4u));
            svbool_t    cmp   = svcmpeq_u32(pg, chunk, vn);
            if (svptest_any(pg, cmp)) {
                return i + (size_t)svcntp_b32(svptrue_b32(),
                                               svbrkb_z(pg, cmp));
            }
            i += vl;
        }
    } else if (data_size == 8u) {
        size_t vl = svcntd();
        while (i + vl <= end) {
            svbool_t    pg    = svptrue_b64();
            svuint64_t  vn    = svdup_n_u64(*(const uint64_t*)needle);
            svuint64_t  chunk = svld1_u64(pg, (const uint64_t*)(data + i * 8u));
            svbool_t    cmp   = svcmpeq_u64(pg, chunk, vn);
            if (svptest_any(pg, cmp)) {
                return i + (size_t)svcntp_b64(svptrue_b64(),
                                               svbrkb_z(pg, cmp));
            }
            i += vl;
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
#endif /* CSALT_SIMD_SVE_UINT8_INL */

