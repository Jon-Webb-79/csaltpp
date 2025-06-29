#pragma once

#if defined(__ARM_FEATURE_SVE)

#include <arm_sve.h>

namespace slt {

inline void simd_add_f64_sve(const double* a, const double* b, double* r, std::size_t n) {
    std::size_t i = 0;
    for (; i < n; i += svcntd()) {
        svbool_t pg = svwhilelt_b64(i, n);
        svfloat64_t va = svld1(pg, &a[i]);
        svfloat64_t vb = svld1(pg, &b[i]);
        svfloat64_t vr = svadd_f64_m(pg, va, vb);
        svst1(pg, &r[i], vr);
    }
}

inline void simd_sub_f64_sve(const double* a, const double* b, double* r, std::size_t n) {
    std::size_t i = 0;
    for (; i < n; i += svcntd()) {
        svbool_t pg = svwhilelt_b64(i, n);
        svfloat64_t va = svld1(pg, &a[i]);
        svfloat64_t vb = svld1(pg, &b[i]);
        svfloat64_t vr = svsub_f64_m(pg, va, vb);
        svst1(pg, &r[i], vr);
    }
}

inline void simd_add_scalar_f64_sve(const double* a, double s, double* r, std::size_t n) {
    std::size_t i = 0;
    svfloat64_t vs = svdup_f64(s);
    for (; i < n; i += svcntd()) {
        svbool_t pg = svwhilelt_b64(i, n);
        svfloat64_t va = svld1(pg, &a[i]);
        svfloat64_t vr = svadd_f64_m(pg, va, vs);
        svst1(pg, &r[i], vr);
    }
}

inline void simd_sub_scalar_f64_sve(const double* a, double s, double* r, std::size_t n) {
    std::size_t i = 0;
    svfloat64_t vs = svdup_f64(s);
    for (; i < n; i += svcntd()) {
        svbool_t pg = svwhilelt_b64(i, n);
        svfloat64_t va = svld1(pg, &a[i]);
        svfloat64_t vr = svsub_f64_m(pg, va, vs);
        svst1(pg, &r[i], vr);
    }
}

inline void simd_mul_f64_sve(const double* a, const double* b, double* r, std::size_t n) {
    std::size_t i = 0;
    for (; i < n; i += svcntd()) {
        svbool_t pg = svwhilelt_b64(i, n);
        svfloat64_t va = svld1(pg, &a[i]);
        svfloat64_t vb = svld1(pg, &b[i]);
        svfloat64_t vr = svmul_f64_m(pg, va, vb);
        svst1(pg, &r[i], vr);
    }
}

inline void simd_mul_scalar_f64_sve(const double* a, double s, double* r, std::size_t n) {
    std::size_t i = 0;
    svfloat64_t vs = svdup_f64(s);
    for (; i < n; i += svcntd()) {
        svbool_t pg = svwhilelt_b64(i, n);
        svfloat64_t va = svld1(pg, &a[i]);
        svfloat64_t vr = svmul_f64_m(pg, va, vs);
        svst1(pg, &r[i], vr);
    }
}

inline void simd_div_scalar_f64_sve(const double* a, double s, double* r, std::size_t n) {
    std::size_t i = 0;
    svfloat64_t vs = svdup_f64(s);
    for (; i < n; i += svcntd()) {
        svbool_t pg = svwhilelt_b64(i, n);
        svfloat64_t va = svld1(pg, &a[i]);
        svfloat64_t vr = svdiv_f64_m(pg, va, vs);
        svst1(pg, &r[i], vr);
    }
}

inline void simd_copy_f64_sve(const double* src, double* dst, std::size_t n) {
    std::size_t i = 0;
    for (; i < n; i += svcntd()) {
        svbool_t pg = svwhilelt_b64(i, n);
        svfloat64_t v = svld1(pg, &src[i]);
        svst1(pg, &dst[i], v);
    }
}

inline double simd_magnitude_squared_f64_sve(const double* data, std::size_t n) {
    svfloat64_t vsum = svdup_f64(0.0);
    std::size_t i = 0;
    for (; i < n; i += svcntd()) {
        svbool_t pg = svwhilelt_b64(i, n);
        svfloat64_t v = svld1(pg, &data[i]);
        vsum = svmla_f64_m(pg, vsum, v, v);
    }
    return svaddv_f64(svptrue_b64(), vsum);
}

}

#endif // __ARM_FEATURE_SVE

