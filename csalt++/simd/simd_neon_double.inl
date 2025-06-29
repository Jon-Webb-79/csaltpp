#pragma once
#if defined(__ARM_NEON)
// safe to include <arm_neon.h> and use intrinsics
#include <arm_neon.h>


namespace slt {

inline void simd_add_f64_neon(const double* a, const double* b, double* r, std::size_t n) {
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vb = vld1q_f64(&b[i]);
        float64x2_t vr = vaddq_f64(va, vb);
        vst1q_f64(&r[i], vr);
    }
    for (; i < n; ++i) r[i] = a[i] + b[i];
}

inline void simd_sub_f64_neon(const double* a, const double* b, double* r, std::size_t n) {
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vb = vld1q_f64(&b[i]);
        float64x2_t vr = vsubq_f64(va, vb);
        vst1q_f64(&r[i], vr);
    }
    for (; i < n; ++i) r[i] = a[i] - b[i];
}

inline void simd_mul_f64_neon(const double* a, const double* b, double* r, std::size_t n) {
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vb = vld1q_f64(&b[i]);
        float64x2_t vr = vmulq_f64(va, vb);
        vst1q_f64(&r[i], vr);
    }
    for (; i < n; ++i) r[i] = a[i] * b[i];
}

inline void simd_div_f64_neon(const double* a, double scalar, double* r, std::size_t n) {
    float64x2_t vs = vdupq_n_f64(scalar);
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t va = vld1q_f64(&a[i]);
        float64x2_t vr = vdivq_f64(va, vs);  // Only in AArch64
        vst1q_f64(&r[i], vr);
    }
    for (; i < n; ++i) r[i] = a[i] / scalar;
}

inline void simd_copy_f64_neon(const double* src, double* dst, std::size_t n) {
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t v = vld1q_f64(&src[i]);
        vst1q_f64(&dst[i], v);
    }
    for (; i < n; ++i) dst[i] = src[i];
}

inline double simd_magnitude_squared_f64_neon(const double* data, std::size_t n) {
    float64x2_t vsum = vdupq_n_f64(0.0);
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        float64x2_t v = vld1q_f64(&data[i]);
        vsum = vfmaq_f64(vsum, v, v);
    }
    double buffer[2];
    vst1q_f64(buffer, vsum);
    double total = buffer[0] + buffer[1];
    for (; i < n; ++i) total += data[i] * data[i];
    return total;
}

} // namespace slt
#endif /* ARM_NEON */
