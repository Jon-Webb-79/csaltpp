#pragma once

#include <immintrin.h>

namespace slt {

// ============================================================================
// AVX2 SIMD helpers for double (double = f64)
// ============================================================================

inline void simd_add_f64_avx2(const double* a, const double* b, double* r, std::size_t n) {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d vr = _mm256_add_pd(va, vb);
        _mm256_storeu_pd(&r[i], vr);
    }
    for (; i < n; ++i)
        r[i] = a[i] + b[i];
}

inline void simd_sub_f64_avx2(const double* a, const double* b, double* r, std::size_t n) {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d vr = _mm256_sub_pd(va, vb);
        _mm256_storeu_pd(&r[i], vr);
    }
    for (; i < n; ++i)
        r[i] = a[i] - b[i];
}

inline void simd_add_scalar_f64_avx2(const double* a, double s, double* r, std::size_t n) {
    __m256d vs = _mm256_set1_pd(s);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vr = _mm256_add_pd(va, vs);
        _mm256_storeu_pd(&r[i], vr);
    }
    for (; i < n; ++i)
        r[i] = a[i] + s;
}

inline void simd_sub_scalar_f64_avx2(const double* a, double s, double* r, std::size_t n) {
    __m256d vs = _mm256_set1_pd(s);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vr = _mm256_sub_pd(va, vs);
        _mm256_storeu_pd(&r[i], vr);
    }
    for (; i < n; ++i)
        r[i] = a[i] - s;
}

inline void simd_mul_f64_avx2(const double* a, const double* b, double* r, std::size_t n) {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vb = _mm256_loadu_pd(&b[i]);
        __m256d vr = _mm256_mul_pd(va, vb);
        _mm256_storeu_pd(&r[i], vr);
    }
    for (; i < n; ++i)
        r[i] = a[i] * b[i];
}

inline void simd_mul_scalar_f64_avx2(const double* a, double s, double* r, std::size_t n) {
    __m256d vs = _mm256_set1_pd(s);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vr = _mm256_mul_pd(va, vs);
        _mm256_storeu_pd(&r[i], vr);
    }
    for (; i < n; ++i)
        r[i] = a[i] * s;
}

inline void simd_div_scalar_f64_avx2(const double* a, double s, double* r, std::size_t n) {
    __m256d vs = _mm256_set1_pd(s);
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d va = _mm256_loadu_pd(&a[i]);
        __m256d vr = _mm256_div_pd(va, vs);
        _mm256_storeu_pd(&r[i], vr);
    }
    for (; i < n; ++i)
        r[i] = a[i] / s;
}

inline void simd_copy_f64_avx2(const double* src, double* dst, std::size_t n) {
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d v = _mm256_loadu_pd(&src[i]);
        _mm256_storeu_pd(&dst[i], v);
    }
    for (; i < n; ++i)
        dst[i] = src[i];
}

inline double simd_magnitude_squared_f64_avx2(const double* data, std::size_t n) {
    __m256d vsum = _mm256_setzero_pd();
    std::size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m256d v = _mm256_loadu_pd(&data[i]);
        vsum = _mm256_add_pd(vsum, _mm256_mul_pd(v, v));
    }

    alignas(32) double buffer[4];
    _mm256_store_pd(buffer, vsum);

    double total = buffer[0] + buffer[1] + buffer[2] + buffer[3];

    for (; i < n; ++i)
        total += data[i] * data[i];

    return total;
}

} // namespace slt

