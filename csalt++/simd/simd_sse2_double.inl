#pragma once

#include <emmintrin.h>

namespace slt {

// ============================================================================
// SSE2 SIMD helpers for double (double = f64)
// ============================================================================

inline void simd_add_f64_sse2(const double* a, const double* b, double* r, std::size_t n) {
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        __m128d va = _mm_loadu_pd(&a[i]);
        __m128d vb = _mm_loadu_pd(&b[i]);
        __m128d vr = _mm_add_pd(va, vb);
        _mm_storeu_pd(&r[i], vr);
    }
    for (; i < n; ++i)
        r[i] = a[i] + b[i];
}

inline void simd_sub_f64_sse2(const double* a, const double* b, double* r, std::size_t n) {
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        __m128d va = _mm_loadu_pd(&a[i]);
        __m128d vb = _mm_loadu_pd(&b[i]);
        __m128d vr = _mm_sub_pd(va, vb);
        _mm_storeu_pd(&r[i], vr);
    }
    for (; i < n; ++i)
        r[i] = a[i] - b[i];
}

inline void simd_add_scalar_f64_sse2(const double* a, double s, double* r, std::size_t n) {
    __m128d vs = _mm_set1_pd(s);
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        __m128d va = _mm_loadu_pd(&a[i]);
        __m128d vr = _mm_add_pd(va, vs);
        _mm_storeu_pd(&r[i], vr);
    }
    for (; i < n; ++i)
        r[i] = a[i] + s;
}

inline void simd_sub_scalar_f64_sse2(const double* a, double s, double* r, std::size_t n) {
    __m128d vs = _mm_set1_pd(s);
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        __m128d va = _mm_loadu_pd(&a[i]);
        __m128d vr = _mm_sub_pd(va, vs);
        _mm_storeu_pd(&r[i], vr);
    }
    for (; i < n; ++i)
        r[i] = a[i] - s;
}

inline void simd_mul_f64_sse2(const double* a, const double* b, double* r, std::size_t n) {
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        __m128d va = _mm_loadu_pd(&a[i]);
        __m128d vb = _mm_loadu_pd(&b[i]);
        __m128d vr = _mm_mul_pd(va, vb);
        _mm_storeu_pd(&r[i], vr);
    }
    for (; i < n; ++i)
        r[i] = a[i] * b[i];
}

inline void simd_mul_scalar_f64_sse2(const double* a, double s, double* r, std::size_t n) {
    __m128d vs = _mm_set1_pd(s);
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        __m128d va = _mm_loadu_pd(&a[i]);
        __m128d vr = _mm_mul_pd(va, vs);
        _mm_storeu_pd(&r[i], vr);
    }
    for (; i < n; ++i)
        r[i] = a[i] * s;
}

inline void simd_div_scalar_f64_sse2(const double* a, double s, double* r, std::size_t n) {
    __m128d vs = _mm_set1_pd(s);
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        __m128d va = _mm_loadu_pd(&a[i]);
        __m128d vr = _mm_div_pd(va, vs);
        _mm_storeu_pd(&r[i], vr);
    }
    for (; i < n; ++i)
        r[i] = a[i] / s;
}

inline void simd_copy_f64_sse2(const double* src, double* dst, std::size_t n) {
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        __m128d v = _mm_loadu_pd(&src[i]);
        _mm_storeu_pd(&dst[i], v);
    }
    for (; i < n; ++i)
        dst[i] = src[i];
}

inline double simd_magnitude_squared_f64_sse2(const double* data, std::size_t n) {
    __m128d vsum = _mm_setzero_pd();
    std::size_t i = 0;
    for (; i + 2 <= n; i += 2) {
        __m128d v = _mm_loadu_pd(&data[i]);
        vsum = _mm_add_pd(vsum, _mm_mul_pd(v, v));
    }

    alignas(16) double buffer[2];
    _mm_store_pd(buffer, vsum);

    double total = buffer[0] + buffer[1];

    for (; i < n; ++i)
        total += data[i] * data[i];

    return total;
}

} // namespace slt

