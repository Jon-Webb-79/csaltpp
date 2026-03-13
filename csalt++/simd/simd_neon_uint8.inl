
#ifndef CSALT_SIMD_NEON_UINT8_INL
#define CSALT_SIMD_NEON_UINT8_INL

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <arm_sve.h>
// ================================================================================ 
// ================================================================================ 

static inline uint8x16_t _neon_reverse_bytes(uint8x16_t v) {
    /* vrev64q_u8: reverse bytes within each 64-bit half */
    v = vrev64q_u8(v);
    /* Swap the two 64-bit halves to complete the full 128-bit reversal */
    return vcombine_u8(vget_high_u8(v), vget_low_u8(v));
}

// ================================================================================
// Public interface
// ================================================================================

static void simd_reverse_uint8(uint8_t* data, size_t len, size_t data_size) {
    if (data == NULL || len < 2u || data_size == 0u) return;

    if (data_size < 16u && (16u % data_size == 0u)) {
        size_t lo = 0u;
        size_t hi = len - 1u;
        size_t elems_per_reg = 16u / data_size;

        while (lo < hi) {
            uint8_t* lo_ptr = data + lo * data_size;
            uint8_t* hi_ptr = data + hi * data_size;

            if ((hi - lo + 1u) >= 2u * elems_per_reg) {
                uint8x16_t vlo = vld1q_u8(lo_ptr);
                uint8x16_t vhi = vld1q_u8(hi_ptr);

                vlo = _neon_reverse_bytes(vlo);
                vhi = _neon_reverse_bytes(vhi);

                vst1q_u8(hi_ptr, vlo);
                vst1q_u8(lo_ptr, vhi);

                lo += elems_per_reg;
                hi -= elems_per_reg;
                continue;
            }

            /* Scalar remainder */
            uint8_t tmp[16];
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
// -------------------------------------------------------------------------------- 

#ifdef __aarch64__
#  define _neon_hmin_u8(v)  vminvq_u8(v)
#  define _neon_hmax_u8(v)  vmaxvq_u8(v)
#  define _neon_hmin_u16(v) vminvq_u16(v)
#  define _neon_hmax_u16(v) vmaxvq_u16(v)
#  define _neon_hmin_u32(v) vminvq_u32(v)
#  define _neon_hmax_u32(v) vmaxvq_u32(v)
#else
static inline uint8_t _neon_hmin_u8(uint8x16_t v) {
    uint8x8_t lo = vmin_u8(vget_low_u8(v), vget_high_u8(v));
    lo = vpmin_u8(lo, lo); lo = vpmin_u8(lo, lo); lo = vpmin_u8(lo, lo);
    return vget_lane_u8(lo, 0);
}
static inline uint8_t _neon_hmax_u8(uint8x16_t v) {
    uint8x8_t lo = vmax_u8(vget_low_u8(v), vget_high_u8(v));
    lo = vpmax_u8(lo, lo); lo = vpmax_u8(lo, lo); lo = vpmax_u8(lo, lo);
    return vget_lane_u8(lo, 0);
}
static inline uint16_t _neon_hmin_u16(uint16x8_t v) {
    uint16x4_t lo = vmin_u16(vget_low_u16(v), vget_high_u16(v));
    lo = vpmin_u16(lo, lo); lo = vpmin_u16(lo, lo);
    return vget_lane_u16(lo, 0);
}
static inline uint16_t _neon_hmax_u16(uint16x8_t v) {
    uint16x4_t lo = vmax_u16(vget_low_u16(v), vget_high_u16(v));
    lo = vpmax_u16(lo, lo); lo = vpmax_u16(lo, lo);
    return vget_lane_u16(lo, 0);
}
static inline uint32_t _neon_hmin_u32(uint32x4_t v) {
    uint32x2_t lo = vmin_u32(vget_low_u32(v), vget_high_u32(v));
    lo = vpmin_u32(lo, lo);
    return vget_lane_u32(lo, 0);
}
static inline uint32_t _neon_hmax_u32(uint32x4_t v) {
    uint32x2_t lo = vmax_u32(vget_low_u32(v), vget_high_u32(v));
    lo = vpmax_u32(lo, lo);
    return vget_lane_u32(lo, 0);
}
#endif

static size_t simd_min_uint8(const uint8_t* data,
                              size_t         len,
                              size_t         data_size,
                              int          (*cmp)(const void*, const void*)) {
    if (data_size == 1u) {
        uint8x16_t vmin = vdupq_n_u8(0xFF);
        size_t i = 0u;
        while (i + 16u <= len) { vmin = vminq_u8(vmin, vld1q_u8(data+i)); i+=16u; }
        for (; i < len; i++) vmin = vminq_u8(vmin, vdupq_n_u8(data[i]));
        uint8_t min_val = _neon_hmin_u8(vmin);
        for (size_t j = 0u; j < len; j++) if (data[j] == min_val) return j;
    } else if (data_size == 2u) {
        const uint16_t* p = (const uint16_t*)data;
        uint16x8_t vmin = vdupq_n_u16(0xFFFF);
        size_t i = 0u;
        while (i + 8u <= len) { vmin = vminq_u16(vmin, vld1q_u16(p+i)); i+=8u; }
        for (; i < len; i++) vmin = vminq_u16(vmin, vdupq_n_u16(p[i]));
        uint16_t mv = _neon_hmin_u16(vmin);
        for (size_t j = 0u; j < len; j++) if (p[j] == mv) {
            size_t best = j;
            for (size_t k = j+1u; k < len; k++)
                if (cmp(data+k*2u, data+best*2u) < 0) best = k;
            return best;
        }
    } else if (data_size == 4u) {
        const uint32_t* p = (const uint32_t*)data;
        uint32x4_t vmin = vdupq_n_u32(0xFFFFFFFFu);
        size_t i = 0u;
        while (i + 4u <= len) { vmin = vminq_u32(vmin, vld1q_u32(p+i)); i+=4u; }
        for (; i < len; i++) vmin = vminq_u32(vmin, vdupq_n_u32(p[i]));
        uint32_t mv = _neon_hmin_u32(vmin);
        for (size_t j = 0u; j < len; j++) if (p[j] == mv) {
            size_t best = j;
            for (size_t k = j+1u; k < len; k++)
                if (cmp(data+k*4u, data+best*4u) < 0) best = k;
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
        uint8x16_t vmax = vdupq_n_u8(0x00);
        size_t i = 0u;
        while (i + 16u <= len) { vmax = vmaxq_u8(vmax, vld1q_u8(data+i)); i+=16u; }
        for (; i < len; i++) vmax = vmaxq_u8(vmax, vdupq_n_u8(data[i]));
        uint8_t max_val = _neon_hmax_u8(vmax);
        for (size_t j = 0u; j < len; j++) if (data[j] == max_val) return j;
    } else if (data_size == 2u) {
        const uint16_t* p = (const uint16_t*)data;
        uint16x8_t vmax = vdupq_n_u16(0x0000);
        size_t i = 0u;
        while (i + 8u <= len) { vmax = vmaxq_u16(vmax, vld1q_u16(p+i)); i+=8u; }
        for (; i < len; i++) vmax = vmaxq_u16(vmax, vdupq_n_u16(p[i]));
        uint16_t mv = _neon_hmax_u16(vmax);
        for (size_t j = 0u; j < len; j++) if (p[j] == mv) {
            size_t best = j;
            for (size_t k = j+1u; k < len; k++)
                if (cmp(data+k*2u, data+best*2u) > 0) best = k;
            return best;
        }
    } else if (data_size == 4u) {
        const uint32_t* p = (const uint32_t*)data;
        uint32x4_t vmax = vdupq_n_u32(0x00000000u);
        size_t i = 0u;
        while (i + 4u <= len) { vmax = vmaxq_u32(vmax, vld1q_u32(p+i)); i+=4u; }
        for (; i < len; i++) vmax = vmaxq_u32(vmax, vdupq_n_u32(p[i]));
        uint32_t mv = _neon_hmax_u32(vmax);
        for (size_t j = 0u; j < len; j++) if (p[j] == mv) {
            size_t best = j;
            for (size_t k = j+1u; k < len; k++)
                if (cmp(data+k*4u, data+best*4u) > 0) best = k;
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
        while (i + 16u <= len) {
            uint8x16_t c = vld1q_u8(data + i);
            uint8_t v;
            v=vgetq_lane_u8(c, 0);add(accum,&v); v=vgetq_lane_u8(c, 1);add(accum,&v);
            v=vgetq_lane_u8(c, 2);add(accum,&v); v=vgetq_lane_u8(c, 3);add(accum,&v);
            v=vgetq_lane_u8(c, 4);add(accum,&v); v=vgetq_lane_u8(c, 5);add(accum,&v);
            v=vgetq_lane_u8(c, 6);add(accum,&v); v=vgetq_lane_u8(c, 7);add(accum,&v);
            v=vgetq_lane_u8(c, 8);add(accum,&v); v=vgetq_lane_u8(c, 9);add(accum,&v);
            v=vgetq_lane_u8(c,10);add(accum,&v); v=vgetq_lane_u8(c,11);add(accum,&v);
            v=vgetq_lane_u8(c,12);add(accum,&v); v=vgetq_lane_u8(c,13);add(accum,&v);
            v=vgetq_lane_u8(c,14);add(accum,&v); v=vgetq_lane_u8(c,15);add(accum,&v);
            i += 16u;
        }
    } else if (data_size == 2u) {
        while (i + 8u <= len) {
            uint16x8_t c = vld1q_u16((const uint16_t*)(data + i * 2u));
            uint16_t v;
            v=vgetq_lane_u16(c,0);add(accum,&v); v=vgetq_lane_u16(c,1);add(accum,&v);
            v=vgetq_lane_u16(c,2);add(accum,&v); v=vgetq_lane_u16(c,3);add(accum,&v);
            v=vgetq_lane_u16(c,4);add(accum,&v); v=vgetq_lane_u16(c,5);add(accum,&v);
            v=vgetq_lane_u16(c,6);add(accum,&v); v=vgetq_lane_u16(c,7);add(accum,&v);
            i += 8u;
        }
    } else if (data_size == 4u) {
        while (i + 4u <= len) {
            uint32x4_t c = vld1q_u32((const uint32_t*)(data + i * 4u));
            uint32_t v;
            v=vgetq_lane_u32(c,0);add(accum,&v); v=vgetq_lane_u32(c,1);add(accum,&v);
            v=vgetq_lane_u32(c,2);add(accum,&v); v=vgetq_lane_u32(c,3);add(accum,&v);
            i += 4u;
        }
    } else if (data_size == 8u) {
        while (i + 2u <= len) {
            uint64x2_t c = vld1q_u64((const uint64_t*)(data + i * 8u));
            uint64_t v;
            v=vgetq_lane_u64(c,0);add(accum,&v);
            v=vgetq_lane_u64(c,1);add(accum,&v);
            i += 2u;
        }
    }
    for (; i < len; i++) add(accum, data + i * data_size);
}
// ================================================================================ 
// ================================================================================ 
#endif /* CSALT_SIMD_NEON_UINT8_INL */

