// simd_avx512_double.inl
#pragma once

#include <immintrin.h>
#include <cstddef>

namespace slt {

inline void simd_add_f64_avx512(const double* a, const double* b, double* r, std::size_t n) {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vb = _mm512_loadu_pd(&b[i]);
        __m512d vr = _mm512_add_pd(va, vb);
        _mm512_storeu_pd(&r[i], vr);
    }
    for (; i < n; ++i) r[i] = a[i] + b[i];
}

inline void simd_sub_f64_avx512(const double* a, const double* b, double* r, std::size_t n) {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vb = _mm512_loadu_pd(&b[i]);
        __m512d vr = _mm512_sub_pd(va, vb);
        _mm512_storeu_pd(&r[i], vr);
    }
    for (; i < n; ++i) r[i] = a[i] - b[i];
}

inline void simd_add_scalar_f64_avx512(const double* a, double s, double* r, std::size_t n) {
    __m512d vs = _mm512_set1_pd(s);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vr = _mm512_add_pd(va, vs);
        _mm512_storeu_pd(&r[i], vr);
    }
    for (; i < n; ++i) r[i] = a[i] + s;
}

inline void simd_sub_scalar_f64_avx512(const double* a, double s, double* r, std::size_t n) {
    __m512d vs = _mm512_set1_pd(s);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vr = _mm512_sub_pd(va, vs);
        _mm512_storeu_pd(&r[i], vr);
    }
    for (; i < n; ++i) r[i] = a[i] - s;
}

inline void simd_mul_f64_avx512(const double* a, const double* b, double* r, std::size_t n) {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vb = _mm512_loadu_pd(&b[i]);
        __m512d vr = _mm512_mul_pd(va, vb);
        _mm512_storeu_pd(&r[i], vr);
    }
    for (; i < n; ++i) r[i] = a[i] * b[i];
}

inline void simd_mul_scalar_f64_avx512(const double* a, double s, double* r, std::size_t n) {
    __m512d vs = _mm512_set1_pd(s);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vr = _mm512_mul_pd(va, vs);
        _mm512_storeu_pd(&r[i], vr);
    }
    for (; i < n; ++i) r[i] = a[i] * s;
}

inline void simd_div_scalar_f64_avx512(const double* a, double s, double* r, std::size_t n) {
    __m512d vs = _mm512_set1_pd(s);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d va = _mm512_loadu_pd(&a[i]);
        __m512d vr = _mm512_div_pd(va, vs);
        _mm512_storeu_pd(&r[i], vr);
    }
    for (; i < n; ++i) r[i] = a[i] / s;
}

inline void simd_copy_f64_avx512(const double* src, double* dst, std::size_t n) {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d v = _mm512_loadu_pd(&src[i]);
        _mm512_storeu_pd(&dst[i], v);
    }
    for (; i < n; ++i) dst[i] = src[i];
}

inline double simd_magnitude_squared_f64_avx512(const double* data, std::size_t n) {
    __m512d vsum = _mm512_setzero_pd();
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m512d v = _mm512_loadu_pd(&data[i]);
        vsum = _mm512_fmadd_pd(v, v, vsum);
    }
    alignas(64) double buffer[8];
    _mm512_store_pd(buffer, vsum);
    double total = 0.0;
    for (int j = 0; j < 8; ++j) total += buffer[j];
    for (; i < n; ++i) total += data[i] * data[i];
    return total;
}

} // namespace slt

