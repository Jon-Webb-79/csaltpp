#pragma once
#if defined(__ARM_NEON)
// safe to include <arm_neon.h> and use intrinsics
#include <arm_neon.h>


namespace slt {

inline void simd_add_f32_neon(const float* a, const float* b, float* r, std::size_t n) {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        float32x4_t vr = vaddq_f32(va, vb);
        vst1q_f32(&r[i], vr);
    }
    for (; i < n; ++i) r[i] = a[i] + b[i];
}

inline void simd_sub_f32_neon(const float* a, const float* b, float* r, std::size_t n) {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        float32x4_t vr = vsubq_f32(va, vb);
        vst1q_f32(&r[i], vr);
    }
    for (; i < n; ++i) r[i] = a[i] - b[i];
}

inline void simd_mul_f32_neon(const float* a, const float* b, float* r, std::size_t n) {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        float32x4_t vr = vmulq_f32(va, vb);
        vst1q_f32(&r[i], vr);
    }
    for (; i < n; ++i) r[i] = a[i] * b[i];
}

inline void simd_div_f32_neon(const float* a, float scalar, float* r, std::size_t n) {
    float32x4_t vs = vdupq_n_f32(scalar);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vr = vdivq_f32(va, vs); // Note: only available on AArch64
        vst1q_f32(&r[i], vr);
    }
    for (; i < n; ++i) r[i] = a[i] / scalar;
}

inline void simd_copy_f32_neon(const float* src, float* dst, std::size_t n) {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(&src[i]);
        vst1q_f32(&dst[i], v);
    }
    for (; i < n; ++i) dst[i] = src[i];
}

inline float simd_magnitude_squared_f32_neon(const float* data, std::size_t n) {
    float32x4_t vsum = vdupq_n_f32(0.0f);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(&data[i]);
        vsum = vmlaq_f32(vsum, v, v);
    }
    float buffer[4];
    vst1q_f32(buffer, vsum);
    float total = buffer[0] + buffer[1] + buffer[2] + buffer[3];
    for (; i < n; ++i) total += data[i] * data[i];
    return total;
}

} // namespace slt
#endif /* ARM_NEON */
