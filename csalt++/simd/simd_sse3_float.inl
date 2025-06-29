// simd_sse3_float.inl
#pragma once

#include <pmmintrin.h>  // SSE3
#include <cstddef>

namespace slt {

inline void simd_add_f32_sse3(const float* a, const float* b, float* r, std::size_t n) {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        __m128 vr = _mm_add_ps(va, vb);
        _mm_storeu_ps(&r[i], vr);
    }
    for (; i < n; ++i)
        r[i] = a[i] + b[i];
}

inline void simd_sub_f32_sse3(const float* a, const float* b, float* r, std::size_t n) {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        __m128 vr = _mm_sub_ps(va, vb);
        _mm_storeu_ps(&r[i], vr);
    }
    for (; i < n; ++i)
        r[i] = a[i] - b[i];
}

inline void simd_add_scalar_f32_sse3(const float* a, float s, float* r, std::size_t n) {
    __m128 vs = _mm_set1_ps(s);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vr = _mm_add_ps(va, vs);
        _mm_storeu_ps(&r[i], vr);
    }
    for (; i < n; ++i)
        r[i] = a[i] + s;
}

inline void simd_sub_scalar_f32_sse3(const float* a, float s, float* r, std::size_t n) {
    __m128 vs = _mm_set1_ps(s);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vr = _mm_sub_ps(va, vs);
        _mm_storeu_ps(&r[i], vr);
    }
    for (; i < n; ++i)
        r[i] = a[i] - s;
}

inline void simd_mul_f32_sse3(const float* a, const float* b, float* r, std::size_t n) {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        __m128 vr = _mm_mul_ps(va, vb);
        _mm_storeu_ps(&r[i], vr);
    }
    for (; i < n; ++i)
        r[i] = a[i] * b[i];
}

inline void simd_mul_scalar_f32_sse3(const float* a, float s, float* r, std::size_t n) {
    __m128 vs = _mm_set1_ps(s);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vr = _mm_mul_ps(va, vs);
        _mm_storeu_ps(&r[i], vr);
    }
    for (; i < n; ++i)
        r[i] = a[i] * s;
}

inline void simd_div_scalar_f32_sse3(const float* a, float s, float* r, std::size_t n) {
    __m128 vs = _mm_set1_ps(s);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vr = _mm_div_ps(va, vs);
        _mm_storeu_ps(&r[i], vr);
    }
    for (; i < n; ++i)
        r[i] = a[i] / s;
}

inline void simd_copy_f32_sse3(const float* src, float* dst, std::size_t n) {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 v = _mm_loadu_ps(&src[i]);
        _mm_storeu_ps(&dst[i], v);
    }
    for (; i < n; ++i)
        dst[i] = src[i];
}

inline float simd_magnitude_squared_f32_sse3(const float* data, std::size_t n) {
    __m128 vsum = _mm_setzero_ps();
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 v = _mm_loadu_ps(&data[i]);
        vsum = _mm_add_ps(vsum, _mm_mul_ps(v, v));
    }

    alignas(16) float buffer[4];
    _mm_store_ps(buffer, vsum);

    float total = buffer[0] + buffer[1] + buffer[2] + buffer[3];

    for (; i < n; ++i)
        total += data[i] * data[i];

    return total;
}

} // namespace slt

