#pragma once

#include <immintrin.h>

namespace slt {

// ============================================================================
// AVX2 SIMD helpers for float (float = f32)
// ============================================================================

inline void simd_add_f32_avx2(const float* a, const float* b, float* r, std::size_t n) {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(&r[i], vr);
    }
    for (; i < n; ++i)
        r[i] = a[i] + b[i];
}

inline void simd_sub_f32_avx2(const float* a, const float* b, float* r, std::size_t n) {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vr = _mm256_sub_ps(va, vb);
        _mm256_storeu_ps(&r[i], vr);
    }
    for (; i < n; ++i)
        r[i] = a[i] - b[i];
}

inline void simd_add_scalar_f32_avx2(const float* a, float s, float* r, std::size_t n) {
    __m256 vs = _mm256_set1_ps(s);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vr = _mm256_add_ps(va, vs);
        _mm256_storeu_ps(&r[i], vr);
    }
    for (; i < n; ++i)
        r[i] = a[i] + s;
}

inline void simd_sub_scalar_f32_avx2(const float* a, float s, float* r, std::size_t n) {
    __m256 vs = _mm256_set1_ps(s);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vr = _mm256_sub_ps(va, vs);
        _mm256_storeu_ps(&r[i], vr);
    }
    for (; i < n; ++i)
        r[i] = a[i] - s;
}

inline void simd_mul_f32_avx2(const float* a, const float* b, float* r, std::size_t n) {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vr = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(&r[i], vr);
    }
    for (; i < n; ++i)
        r[i] = a[i] * b[i];
}

inline void simd_mul_scalar_f32_avx2(const float* a, float s, float* r, std::size_t n) {
    __m256 vs = _mm256_set1_ps(s);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vr = _mm256_mul_ps(va, vs);
        _mm256_storeu_ps(&r[i], vr);
    }
    for (; i < n; ++i)
        r[i] = a[i] * s;
}

inline void simd_div_scalar_f32_avx2(const float* a, float s, float* r, std::size_t n) {
    __m256 vs = _mm256_set1_ps(s);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vr = _mm256_div_ps(va, vs);
        _mm256_storeu_ps(&r[i], vr);
    }
    for (; i < n; ++i)
        r[i] = a[i] / s;
}

inline void simd_copy_f32_avx2(const float* src, float* dst, std::size_t n) {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(&src[i]);
        _mm256_storeu_ps(&dst[i], v);
    }
    for (; i < n; ++i)
        dst[i] = src[i];
}

inline float simd_magnitude_squared_f32_avx2(const float* data, std::size_t n) {
    __m256 vsum = _mm256_setzero_ps();
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 v = _mm256_loadu_ps(&data[i]);
        vsum = _mm256_add_ps(vsum, _mm256_mul_ps(v, v));
    }

    alignas(32) float buffer[8];
    _mm256_store_ps(buffer, vsum);

    float total = buffer[0] + buffer[1] + buffer[2] + buffer[3] +
                  buffer[4] + buffer[5] + buffer[6] + buffer[7];

    for (; i < n; ++i)
        total += data[i] * data[i];

    return total;
}

} // namespace slt

