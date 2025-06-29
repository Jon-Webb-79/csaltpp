// simd_avx512_float.inl
#pragma once

#include <immintrin.h>
#include <cstddef>

namespace slt {

inline void simd_add_f32_avx512(const float* a, const float* b, float* r, std::size_t n) {
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        __m512 vr = _mm512_add_ps(va, vb);
        _mm512_storeu_ps(&r[i], vr);
    }
    for (; i < n; ++i) r[i] = a[i] + b[i];
}

inline void simd_sub_f32_avx512(const float* a, const float* b, float* r, std::size_t n) {
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        __m512 vr = _mm512_sub_ps(va, vb);
        _mm512_storeu_ps(&r[i], vr);
    }
    for (; i < n; ++i) r[i] = a[i] - b[i];
}

inline void simd_add_scalar_f32_avx512(const float* a, float s, float* r, std::size_t n) {
    __m512 vs = _mm512_set1_ps(s);
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vr = _mm512_add_ps(va, vs);
        _mm512_storeu_ps(&r[i], vr);
    }
    for (; i < n; ++i) r[i] = a[i] + s;
}

inline void simd_sub_scalar_f32_avx512(const float* a, float s, float* r, std::size_t n) {
    __m512 vs = _mm512_set1_ps(s);
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vr = _mm512_sub_ps(va, vs);
        _mm512_storeu_ps(&r[i], vr);
    }
    for (; i < n; ++i) r[i] = a[i] - s;
}

inline void simd_mul_f32_avx512(const float* a, const float* b, float* r, std::size_t n) {
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        __m512 vr = _mm512_mul_ps(va, vb);
        _mm512_storeu_ps(&r[i], vr);
    }
    for (; i < n; ++i) r[i] = a[i] * b[i];
}

inline void simd_mul_scalar_f32_avx512(const float* a, float s, float* r, std::size_t n) {
    __m512 vs = _mm512_set1_ps(s);
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vr = _mm512_mul_ps(va, vs);
        _mm512_storeu_ps(&r[i], vr);
    }
    for (; i < n; ++i) r[i] = a[i] * s;
}

inline void simd_div_scalar_f32_avx512(const float* a, float s, float* r, std::size_t n) {
    __m512 vs = _mm512_set1_ps(s);
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vr = _mm512_div_ps(va, vs);
        _mm512_storeu_ps(&r[i], vr);
    }
    for (; i < n; ++i) r[i] = a[i] / s;
}

inline void simd_copy_f32_avx512(const float* src, float* dst, std::size_t n) {
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(&src[i]);
        _mm512_storeu_ps(&dst[i], v);
    }
    for (; i < n; ++i) dst[i] = src[i];
}

inline float simd_magnitude_squared_f32_avx512(const float* data, std::size_t n) {
    __m512 vsum = _mm512_setzero_ps();
    std::size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 v = _mm512_loadu_ps(&data[i]);
        vsum = _mm512_fmadd_ps(v, v, vsum);
    }
    alignas(64) float buffer[16];
    _mm512_store_ps(buffer, vsum);
    float total = 0.0f;
    for (int j = 0; j < 16; ++j) total += buffer[j];
    for (; i < n; ++i) total += data[i] * data[i];
    return total;
}

} // namespace slt

