
#ifndef CSALT_SIMD_SVE2_UINT8_INL
#define CSALT_SIMD_SVE2_UINT8_INL

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <arm_sve.h>
// ================================================================================ 
// ================================================================================ 

static void simd_reverse_uint8(uint8_t* data, size_t len, size_t data_size) {
    if (data == NULL || len < 2u || data_size == 0u) return;

    if (data_size < 16u && (16u % data_size == 0u)) {
        size_t vl            = svcntb();
        size_t elems_per_vec = vl / data_size;

        size_t lo = 0u;
        size_t hi = len - 1u;

        while (lo < hi && (hi - lo + 1u) >= 2u * elems_per_vec) {
            uint8_t* lo_ptr = data + lo * data_size;
            uint8_t* hi_ptr = data + hi * data_size;

            svbool_t   pg  = svptrue_b8();
            svuint8_t  vlo = svld1_u8(pg, lo_ptr);
            svuint8_t  vhi = svld1_u8(pg, hi_ptr);

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

    /* Scalar fallback */
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
            svbool_t  pg    = svptrue_b8();
            svuint8_t vn    = svdup_n_u8(*needle);
            svuint8_t chunk = svld1_u8(pg, data + i);
            svbool_t  cmp   = svcmpeq_u8(pg, chunk, vn);
            if (svptest_any(pg, cmp)) {
                return i + (size_t)svcntp_b8(svptrue_b8(),
                                              svbrkb_z(pg, cmp));
            }
            i += vl;
        }
    } else if (data_size == 2u) {
        size_t vl = svcnth();
        while (i + vl <= end) {
            svbool_t   pg    = svptrue_b16();
            svuint16_t vn    = svdup_n_u16(*(const uint16_t*)needle);
            svuint16_t chunk = svld1_u16(pg, (const uint16_t*)(data + i * 2u));
            svbool_t   cmp   = svcmpeq_u16(pg, chunk, vn);
            if (svptest_any(pg, cmp)) {
                return i + (size_t)svcntp_b16(svptrue_b16(),
                                               svbrkb_z(pg, cmp));
            }
            i += vl;
        }
    } else if (data_size == 4u) {
        size_t vl = svcntw();
        while (i + vl <= end) {
            svbool_t   pg    = svptrue_b32();
            svuint32_t vn    = svdup_n_u32(*(const uint32_t*)needle);
            svuint32_t chunk = svld1_u32(pg, (const uint32_t*)(data + i * 4u));
            svbool_t   cmp   = svcmpeq_u32(pg, chunk, vn);
            if (svptest_any(pg, cmp)) {
                return i + (size_t)svcntp_b32(svptrue_b32(),
                                               svbrkb_z(pg, cmp));
            }
            i += vl;
        }
    } else if (data_size == 8u) {
        size_t vl = svcntd();
        while (i + vl <= end) {
            svbool_t   pg    = svptrue_b64();
            svuint64_t vn    = svdup_n_u64(*(const uint64_t*)needle);
            svuint64_t chunk = svld1_u64(pg, (const uint64_t*)(data + i * 8u));
            svbool_t   cmp   = svcmpeq_u64(pg, chunk, vn);
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
// -------------------------------------------------------------------------------- 

static inline size_t _sve2_first_u8(const uint8_t* data, size_t len, uint8_t val) {
    size_t vl = svcntb(); svbool_t pg = svptrue_b8();
    svuint8_t needle = svdup_n_u8(val);
    size_t i = 0u;
    while (i + vl <= len) {
        svbool_t match = svmatch_u8(pg, svld1_u8(pg, data+i), needle);
        if (svptest_any(pg, match))
            return i + (size_t)svcntp_b8(svptrue_b8(), svbrkb_z(pg, match));
        i += vl;
    }
    for (; i < len; i++) if (data[i] == val) return i;
    return SIZE_MAX;
}

static inline size_t _sve2_first_u16(const uint16_t* data, size_t len, uint16_t val) {
    size_t vl = svcnth(); svbool_t pg = svptrue_b16();
    svuint16_t needle = svdup_n_u16(val);
    size_t i = 0u;
    while (i + vl <= len) {
        svbool_t match = svmatch_u16(pg, svld1_u16(pg, data+i), needle);
        if (svptest_any(pg, match))
            return i + (size_t)svcntp_b16(svptrue_b16(), svbrkb_z(pg, match));
        i += vl;
    }
    for (; i < len; i++) if (data[i] == val) return i;
    return SIZE_MAX;
}

static size_t simd_min_uint8(const uint8_t* data,
                              size_t         len,
                              size_t         data_size,
                              int          (*cmp)(const void*, const void*)) {
    if (data_size == 1u) {
        size_t vl = svcntb(); svbool_t pg = svptrue_b8();
        svuint8_t vmin = svdup_n_u8(0xFF); size_t i = 0u;
        while (i + vl <= len) {
            vmin = svmin_u8_z(pg, vmin, svld1_u8(pg, data+i)); i += vl;
        }
        if (i < len) {
            svbool_t tpg = svwhilelt_b8_u64((uint64_t)i, (uint64_t)len);
            vmin = svmin_u8_m(pg, vmin,
                              svsel_u8(tpg, svld1_u8(tpg,data+i), svdup_n_u8(0xFF)));
        }
        return _sve2_first_u8(data, len, svminv_u8(pg, vmin));
    } else if (data_size == 2u) {
        size_t vl = svcnth(); svbool_t pg = svptrue_b16();
        svuint16_t vmin = svdup_n_u16(0xFFFF);
        const uint16_t* p = (const uint16_t*)data; size_t i = 0u;
        while (i + vl <= len) {
            vmin = svmin_u16_z(pg, vmin, svld1_u16(pg, p+i)); i += vl;
        }
        if (i < len) {
            svbool_t tpg = svwhilelt_b16_u64((uint64_t)i, (uint64_t)len);
            vmin = svmin_u16_m(pg, vmin,
                               svsel_u16(tpg, svld1_u16(tpg,p+i), svdup_n_u16(0xFFFF)));
        }
        size_t first = _sve2_first_u16(p, len, svminv_u16(pg, vmin));
        size_t best = first;
        for (size_t k = first+1u; k < len; k++)
            if (cmp(data+k*2u, data+best*2u) < 0) best = k;
        return best;
    } else if (data_size == 4u) {
        size_t vl = svcntw(); svbool_t pg = svptrue_b32();
        svuint32_t vmin = svdup_n_u32(0xFFFFFFFFu);
        const uint32_t* p = (const uint32_t*)data; size_t i = 0u;
        while (i + vl <= len) {
            vmin = svmin_u32_z(pg, vmin, svld1_u32(pg, p+i)); i += vl;
        }
        if (i < len) {
            svbool_t tpg = svwhilelt_b32_u64((uint64_t)i, (uint64_t)len);
            vmin = svmin_u32_m(pg, vmin,
                               svsel_u32(tpg, svld1_u32(tpg,p+i), svdup_n_u32(0xFFFFFFFFu)));
        }
        uint32_t mv = svminv_u32(pg, vmin);
        for (size_t j = 0u; j < len; j++) if (p[j] == mv) {
            size_t best = j;
            for (size_t k = j+1u; k < len; k++)
                if (cmp(data+k*4u, data+best*4u) < 0) best = k;
            return best;
        }
    } else if (data_size == 8u) {
        size_t vl = svcntd(); svbool_t pg = svptrue_b64();
        svuint64_t vmin = svdup_n_u64(UINT64_MAX);
        const uint64_t* p = (const uint64_t*)data; size_t i = 0u;
        while (i + vl <= len) {
            vmin = svmin_u64_z(pg, vmin, svld1_u64(pg, p+i)); i += vl;
        }
        if (i < len) {
            svbool_t tpg = svwhilelt_b64_u64((uint64_t)i, (uint64_t)len);
            vmin = svmin_u64_m(pg, vmin,
                               svsel_u64(tpg, svld1_u64(tpg,p+i), svdup_n_u64(UINT64_MAX)));
        }
        uint64_t mv = svminv_u64(pg, vmin);
        for (size_t j = 0u; j < len; j++) if (p[j] == mv) {
            size_t best = j;
            for (size_t k = j+1u; k < len; k++)
                if (cmp(data+k*8u, data+best*8u) < 0) best = k;
            return best;
        }
    }
    size_t best = 0u;
    for (size_t i = 1u; i < len; i++)
        if (cmp(data+i*data_size, data+best*data_size) < 0) best = i;
    return best;
}

static size_t simd_max_uint8(const uint8_t* data,
                              size_t         len,
                              size_t         data_size,
                              int          (*cmp)(const void*, const void*)) {
    if (data_size == 1u) {
        size_t vl = svcntb(); svbool_t pg = svptrue_b8();
        svuint8_t vmax = svdup_n_u8(0x00); size_t i = 0u;
        while (i + vl <= len) {
            vmax = svmax_u8_z(pg, vmax, svld1_u8(pg, data+i)); i += vl;
        }
        if (i < len) {
            svbool_t tpg = svwhilelt_b8_u64((uint64_t)i, (uint64_t)len);
            vmax = svmax_u8_m(pg, vmax,
                              svsel_u8(tpg, svld1_u8(tpg,data+i), svdup_n_u8(0x00)));
        }
        return _sve2_first_u8(data, len, svmaxv_u8(pg, vmax));
    } else if (data_size == 2u) {
        size_t vl = svcnth(); svbool_t pg = svptrue_b16();
        svuint16_t vmax = svdup_n_u16(0x0000);
        const uint16_t* p = (const uint16_t*)data; size_t i = 0u;
        while (i + vl <= len) {
            vmax = svmax_u16_z(pg, vmax, svld1_u16(pg, p+i)); i += vl;
        }
        if (i < len) {
            svbool_t tpg = svwhilelt_b16_u64((uint64_t)i, (uint64_t)len);
            vmax = svmax_u16_m(pg, vmax,
                               svsel_u16(tpg, svld1_u16(tpg,p+i), svdup_n_u16(0x0000)));
        }
        size_t first = _sve2_first_u16(p, len, svmaxv_u16(pg, vmax));
        size_t best = first;
        for (size_t k = first+1u; k < len; k++)
            if (cmp(data+k*2u, data+best*2u) > 0) best = k;
        return best;
    } else if (data_size == 4u) {
        size_t vl = svcntw(); svbool_t pg = svptrue_b32();
        svuint32_t vmax = svdup_n_u32(0x00000000u);
        const uint32_t* p = (const uint32_t*)data; size_t i = 0u;
        while (i + vl <= len) {
            vmax = svmax_u32_z(pg, vmax, svld1_u32(pg, p+i)); i += vl;
        }
        if (i < len) {
            svbool_t tpg = svwhilelt_b32_u64((uint64_t)i, (uint64_t)len);
            vmax = svmax_u32_m(pg, vmax,
                               svsel_u32(tpg, svld1_u32(tpg,p+i), svdup_n_u32(0x00000000u)));
        }
        uint32_t mv = svmaxv_u32(pg, vmax);
        for (size_t j = 0u; j < len; j++) if (p[j] == mv) {
            size_t best = j;
            for (size_t k = j+1u; k < len; k++)
                if (cmp(data+k*4u, data+best*4u) > 0) best = k;
            return best;
        }
    } else if (data_size == 8u) {
        size_t vl = svcntd(); svbool_t pg = svptrue_b64();
        svuint64_t vmax = svdup_n_u64(0u);
        const uint64_t* p = (const uint64_t*)data; size_t i = 0u;
        while (i + vl <= len) {
            vmax = svmax_u64_z(pg, vmax, svld1_u64(pg, p+i)); i += vl;
        }
        if (i < len) {
            svbool_t tpg = svwhilelt_b64_u64((uint64_t)i, (uint64_t)len);
            vmax = svmax_u64_m(pg, vmax,
                               svsel_u64(tpg, svld1_u64(tpg,p+i), svdup_n_u64(0u)));
        }
        uint64_t mv = svmaxv_u64(pg, vmax);
        for (size_t j = 0u; j < len; j++) if (p[j] == mv) {
            size_t best = j;
            for (size_t k = j+1u; k < len; k++)
                if (cmp(data+k*8u, data+best*8u) > 0) best = k;
            return best;
        }
    }
    size_t best = 0u;
    for (size_t i = 1u; i < len; i++)
        if (cmp(data+i*data_size, data+best*data_size) > 0) best = i;
    return best;
}
// -------------------------------------------------------------------------------- 

static void simd_sum_uint8(const uint8_t* data,
                            size_t         len,
                            size_t         data_size,
                            void*          accum,
                            void         (*add)(void* accum, const void* element)) {
    size_t i = 0u;
    if (data_size == 1u) {
        size_t vl = svcntb(); svbool_t pg = svptrue_b8();
        uint8_t buf[256];
        while (i + vl <= len) {
            svst1_u8(pg, buf, svld1_u8(pg, data + i));
            for (size_t e = 0u; e < vl; e++) add(accum, &buf[e]);
            i += vl;
        }
    } else if (data_size == 2u) {
        size_t vl = svcnth(); svbool_t pg = svptrue_b16();
        uint16_t buf[128];
        while (i + vl <= len) {
            svst1_u16(pg, buf, svld1_u16(pg, (const uint16_t*)(data + i * 2u)));
            for (size_t e = 0u; e < vl; e++) add(accum, &buf[e]);
            i += vl;
        }
    } else if (data_size == 4u) {
        size_t vl = svcntw(); svbool_t pg = svptrue_b32();
        uint32_t buf[64];
        while (i + vl <= len) {
            svst1_u32(pg, buf, svld1_u32(pg, (const uint32_t*)(data + i * 4u)));
            for (size_t e = 0u; e < vl; e++) add(accum, &buf[e]);
            i += vl;
        }
    } else if (data_size == 8u) {
        size_t vl = svcntd(); svbool_t pg = svptrue_b64();
        uint64_t buf[32];
        while (i + vl <= len) {
            svst1_u64(pg, buf, svld1_u64(pg, (const uint64_t*)(data + i * 8u)));
            for (size_t e = 0u; e < vl; e++) add(accum, &buf[e]);
            i += vl;
        }
    }
    for (; i < len; i++) add(accum, data + i * data_size);
}
// ================================================================================ 
// ================================================================================ 
#endif /* CSALT_SIMD_SVE2_UINT8_INL */

