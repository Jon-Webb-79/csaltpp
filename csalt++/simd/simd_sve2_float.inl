#pragma once

#if defined(__ARM_FEATURE_SVE2)

#include <arm_sve.h>

namespace slt {

inline void simd_add_f32_sve2(const float* a, const float* b, float* r, std::size_t n) {
    std::size_t i = 0;
    for (; i < n; i += svcntw()) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t va = svld1(pg, &a[i]);
        svfloat32_t vb = svld1(pg, &b[i]);
        svfloat32_t vr = svadd_f32_m(pg, va, vb);
        svst1(pg, &r[i], vr);
    }
}

inline void simd_sub_f32_sve2(const float* a, const float* b, float* r, std::size_t n) {
    std::size_t i = 0;
    for (; i < n; i += svcntw()) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t va = svld1(pg, &a[i]);
        svfloat32_t vb = svld1(pg, &b[i]);
        svfloat32_t vr = svsub_f32_m(pg, va, vb);
        svst1(pg, &r[i], vr);
    }
}

inline void simd_add_scalar_f32_sve2(const float* a, float s, float* r, std::size_t n) {
    svfloat32_t vs = svdup_f32(s);
    std::size_t i = 0;
    for (; i < n; i += svcntw()) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t va = svld1(pg, &a[i]);
        svfloat32_t vr = svadd_f32_m(pg, va, vs);
        svst1(pg, &r[i], vr);
    }
}

inline void simd_sub_scalar_f32_sve2(const float* a, float s, float* r, std::size_t n) {
    svfloat32_t vs = svdup_f32(s);
    std::size_t i = 0;
    for (; i < n; i += svcntw()) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t va = svld1(pg, &a[i]);
        svfloat32_t vr = svsub_f32_m(pg, va, vs);
        svst1(pg, &r[i], vr);
    }
}

inline void simd_mul_f32_sve2(const float* a, const float* b, float* r, std::size_t n) {
    std::size_t i = 0;
    for (; i < n; i += svcntw()) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t va = svld1(pg, &a[i]);
        svfloat32_t vb = svld1(pg, &b[i]);
        svfloat32_t vr = svmul_f32_m(pg, va, vb);
        svst1(pg, &r[i], vr);
    }
}

inline void simd_mul_scalar_f32_sve2(const float* a, float s, float* r, std::size_t n) {
    svfloat32_t vs = svdup_f32(s);
    std::size_t i = 0;
    for (; i < n; i += svcntw()) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t va = svld1(pg, &a[i]);
        svfloat32_t vr = svmul_f32_m(pg, va, vs);
        svst1(pg, &r[i], vr);
    }
}

inline void simd_div_scalar_f32_sve2(const float* a, float s, float* r, std::size_t n) {
    svfloat32_t vs = svdup_f32(s);
    std::size_t i = 0;
    for (; i < n; i += svcntw()) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t va = svld1(pg, &a[i]);
        svfloat32_t vr = svdiv_f32_m(pg, va, vs);
        svst1(pg, &r[i], vr);
    }
}

inline void simd_copy_f32_sve2(const float* src, float* dst, std::size_t n) {
    std::size_t i = 0;
    for (; i < n; i += svcntw()) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t v = svld1(pg, &src[i]);
        svst1(pg, &dst[i], v);
    }
}

inline float simd_magnitude_squared_f32_sve2(const float* data, std::size_t n) {
    svfloat32_t sum = svdup_f32(0.0f);
    std::size_t i = 0;
    for (; i < n; i += svcntw()) {
        svbool_t pg = svwhilelt_b32(i, n);
        svfloat32_t v = svld1(pg, &data[i]);
        sum = svmla_f32_m(pg, sum, v, v);
    }
    return svaddv_f32(svptrue_b32(), sum);
}

} // namespace slt

#endif // __ARM_FEATURE_SVE2

