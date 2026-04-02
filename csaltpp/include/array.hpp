// ================================================================================
// ================================================================================
// - File:    array.hpp
// - Purpose: Describe the file purpose here
//
// Source Metadata
// - Author:  Jonathan A. Webb
// - Date:    March 09, 2026
// - Version: 1.0
// - Copyright: Copyright 2022, Jon Webb Inc.
// ================================================================================
// ================================================================================
// Include modules here

#ifndef array_HPP
#define array_HPP

#include "error.hpp"
#include "allocator.hpp"

#include <cstddef>
#include <cstring>
#include <cstdint>
#include <new>
#include <type_traits>
#include <utility>

#if defined(__AVX512BW__)
#  include "simd_avx512_uint8.inl"
#elif defined(__AVX2__)
#  include "simd_avx2_uint8.inl"
#elif defined(__AVX__)
#  include "simd_avx_uint8.inl"
#elif defined(__SSE4_1__)
#  include "simd_sse41_uint8.inl"
#elif defined(__SSSE3__)
#  include "simd_sse3_uint8.inl"
#elif defined(__SSE2__)
#  include "simd_sse2_uint8.inl"
#elif defined(__ARM_FEATURE_SVE2)
#  include "simd_sve2_uint8.inl"
#elif defined(__ARM_FEATURE_SVE)
#  include "simd_sve_uint8.inl"
#elif defined(__ARM_NEON)
#  include "simd_neon_uint8.inl"
#else
#  include "simd_scalar_uint8.inl"
#endif
// ================================================================================ 
// ================================================================================ 

namespace cslt {

    #ifndef ITER_DIR_H
    enum class Direction {
        FORWARD = 0,  ///< Ascending / forward order
        REVERSE = 1   ///< Descending / reverse order
    };
    #endif

    /**
     * @class Array
     * @brief Allocator-backed generic dynamic array container
     *
     * @details Provides a resizable array container that uses custom allocators for
     * memory management. Both the Array struct and its internal data buffer are
     * allocated through the provided allocator.
     *
     * Key features:
     * - Custom allocator support (heap, arena, buddy, slab, etc.)
     * - Generic element type via template parameter
     * - Tiered growth strategy to avoid runaway allocation at large sizes
     * - Safe insertion and removal via push/pop family of methods
     * - Bounds-checked read via operator[](size_t) const returning Expected<T>
     * - Bounds-checked write via set(size_t, const T&) returning Expected<bool>
     * - Copy factory via copy(); move construction explicitly deleted
     * - RAII-based memory management through init factory function
     *
     * Growth tiers (in elements):
     * - 0            -> 1
     * - < 1024       -> 2x
     * - < 8192       -> 1.5x  (cap + cap/2)
     * - < 65536      -> 1.25x (cap + cap/4)
     * - >= 65536     -> cap + 256  (linear increment)
     *
     * Shift strategy:
     * - Trivially copyable T: std::memmove (compiler-vectorisable)
     * - Non-trivial T:        placement-new / explicit destructor loop
     *
     * Usage pattern:
     * - Must be initialised via the static factory function init()
     * - Cannot be constructed directly (private constructor)
     * - Automatically manages both the Array object and its buffer memory
     * - Cleaned up through ArrayDeleter
     *
     * @tparam T Element type stored in the array
     *
     * @code{.cpp}
     * cslt::HeapAllocator allocator;
     * auto result = cslt::Array<int>::init(8, allocator);
     *
     * if (result.hasValue()) {
     *     cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(result.value());
     *
     *     arr->push_back(1);
     *     arr->push_back(2);
     *     arr->push_back(3);
     *
     *     (*arr)[1] = 99;            // overwrite index 1
     *     auto r = (*arr)[1];        // read index 1 -> Expected<int>
     *     arr->pop_back();           // removes 3
     * }
     * @endcode
     *
     * @note This class cannot be instantiated directly. Use the init() factory
     *       function to create instances.
     */
    template <typename T>
    class Array {
    private:
        T*         data_;       ///< Pointer to the element buffer
        size_t     len_;        ///< Number of elements currently stored
        size_t     cap_;        ///< Total element capacity of the buffer
        Allocator* allocator_;  ///< Allocator used for all memory operations

        // ========================================================================
        // Private helpers
        // ========================================================================

        /**
         * @brief Compute the next capacity using a tiered growth strategy
         *
         * @param current Current capacity in elements
         * @return New capacity in elements
         *
         * @details Tiered to avoid runaway allocation at large sizes while
         *          maintaining fast ramp-up at small sizes:
         *          - 0 elements        -> 1
         *          - < 1024 elements   -> 2x
         *          - < 8192 elements   -> 1.5x  (cap + cap/2)
         *          - < 65536 elements  -> 1.25x (cap + cap/4)
         *          - >= 65536 elements -> linear increment of 256
         */
        static size_t _compute_new_cap(size_t current) noexcept {
            if (current == 0u)    return 1u;
            if (current < 1024u)  return current * 2u;
            if (current < 8192u)  return current + current / 2u;
            if (current < 65536u) return current + current / 4u;
            return current + 256u;
        }
// --------------------------------------------------------------------------------

        /**
         * @brief Reallocate the element buffer to new_cap elements
         *
         * @param new_cap Target capacity in elements
         * @return true on success; false if overflow guard triggers or
         *         the allocator reallocation fails. On failure the array
         *         is left completely unchanged.
         *
         * @details Guards against size_t overflow before computing byte counts,
         *          mirroring the overflow check in the C _grow_array helper.
         */
        bool _grow(size_t new_cap) noexcept {
            // Overflow guard: new_cap * sizeof(T) must not wrap
            if (new_cap > SIZE_MAX / sizeof(T)) return false;

            size_t const old_bytes = cap_    * sizeof(T);
            size_t const new_bytes = new_cap * sizeof(T);

            auto result = allocator_->realloc(static_cast<void*>(data_),
                                              old_bytes,
                                              new_bytes,
                                              false);
            if (!result.hasValue()) return false;

            data_ = static_cast<T*>(result.value());
            cap_  = new_cap;
            return true;
        }
// --------------------------------------------------------------------------------

        /**
         * @brief Ensure at least one free slot exists, growing if necessary
         *
         * @return true if there is room to write (either already, or after
         *         a successful grow); false if the grow failed.
         */
        bool _ensure_capacity() noexcept {
            if (len_ < cap_) return true;
            return _grow(_compute_new_cap(cap_));
        }
// --------------------------------------------------------------------------------

        /**
         * @brief Shift elements right by one slot over the range [index, len_)
         *        to open a gap at @p index
         *
         * @param index First position to move
         *
         * @details For trivially copyable T, delegates to std::memmove which
         *          the compiler can vectorise. For non-trivial T, uses a
         *          placement-new / explicit-destructor loop to preserve correct
         *          construction and destruction semantics.
         *
         * @pre  len_ < cap_  (caller must have ensured capacity first)
         * @pre  index <= len_
         */
        void _shift_right(size_t index) noexcept {
            if constexpr (std::is_trivially_copyable_v<T>) {
                std::memmove(static_cast<void*>(data_ + index + 1u),
                             static_cast<const void*>(data_ + index),
                             (len_ - index) * sizeof(T));
            } else {
                for (size_t i = len_; i > index; --i) {
                    new (static_cast<void*>(data_ + i)) T(data_[i - 1u]);
                    data_[i - 1u].~T();
                }
            }
        }
// --------------------------------------------------------------------------------

        /**
         * @brief Shift elements left by one slot over the range (index, len_)
         *        to close the gap at @p index
         *
         * @param index Position of the element that has already been destroyed
         *
         * @details For trivially copyable T, delegates to std::memmove.
         *          For non-trivial T, uses a placement-new / explicit-destructor
         *          loop.
         *
         * @pre  The element at @p index has already been destroyed by the caller
         * @pre  index < len_
         */
        void _shift_left(size_t index) noexcept {
            if constexpr (std::is_trivially_copyable_v<T>) {
                std::memmove(static_cast<void*>(data_ + index),
                             static_cast<const void*>(data_ + index + 1u),
                             (len_ - index - 1u) * sizeof(T));
            } else {
                for (size_t i = index; i < len_ - 1u; ++i) {
                    new (static_cast<void*>(data_ + i)) T(data_[i + 1u]);
                    data_[i + 1u].~T();
                }
            }
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Apply Direction to a raw comparator result
         *
         * @param cmp_result Raw result from the caller-supplied comparator
         * @param dir        FORWARD leaves the result unchanged; REVERSE negates it
         * @return Adjusted comparator result
         */
        template <typename Func>
        static int _apply_dir(int cmp_result, Direction dir) noexcept {
            return (dir == Direction::FORWARD) ? cmp_result : -cmp_result;
        }
// --------------------------------------------------------------------------------
 
        /**
         * @brief Swap two elements using move semantics (non-trivial T)
         */
        static void _swap_move(T* a, T* b) noexcept {
            T tmp(std::move(*a));
            a->~T();
            new (static_cast<void*>(a)) T(std::move(*b));
            b->~T();
            new (static_cast<void*>(b)) T(std::move(tmp));
        }
// --------------------------------------------------------------------------------
 
        /**
         * @brief Swap two elements using memcpy (trivially copyable T)
         */
        static void _swap_trivial(T* a, T* b) noexcept {
            T tmp;
            std::memcpy(&tmp, a,    sizeof(T));
            std::memcpy(a,    b,    sizeof(T));
            std::memcpy(b,    &tmp, sizeof(T));
        }
// --------------------------------------------------------------------------------
 
        /**
         * @brief Median-of-three pivot selection
         *
         * @return Pointer to the median element among a, b, and c
         */
        template <typename Func>
        static T* _median_of_three(T* a, T* b, T* c,
                                   Func cmp, Direction dir) noexcept {
            int ab = _apply_dir<Func>(cmp(*a, *b), dir);
            int bc = _apply_dir<Func>(cmp(*b, *c), dir);
            int ac = _apply_dir<Func>(cmp(*a, *c), dir);
 
            if (ab <= 0) {
                if (bc <= 0) return b;   // a <= b <= c
                if (ac <= 0) return c;   // a <= c < b
                return a;                // c < a <= b
            }
            if (ac <= 0) return a;       // b < a <= c
            if (bc <= 0) return c;       // b <= c < a
            return b;                    // c < b < a
        }
// --------------------------------------------------------------------------------
 
        /**
         * @brief Insertion sort for small partitions [lo, hi] inclusive
         */
        template <typename Func>
        void _insertion_sort(size_t lo, size_t hi,
                             Func cmp, Direction dir) noexcept {
            for (size_t i = lo + 1u; i <= hi; ++i) {
                size_t j = i;
                while (j > lo &&
                       _apply_dir<Func>(cmp(data_[j - 1u], data_[j]), dir) > 0) {
                    if constexpr (std::is_trivially_copyable_v<T>)
                        _swap_trivial(data_ + j - 1u, data_ + j);
                    else
                        _swap_move(data_ + j - 1u, data_ + j);
                    --j;
                }
            }
        }
// --------------------------------------------------------------------------------
 
        /**
         * @brief Partition data_[lo..hi] around a median-of-three pivot
         *
         * @return Final index of the pivot element
         */
        template <typename Func>
        size_t _partition(size_t lo, size_t hi,
                          Func cmp, Direction dir) noexcept {
            size_t mid = lo + (hi - lo) / 2u;
            T* pivot_ptr = _median_of_three(data_ + lo, data_ + mid, data_ + hi,
                                            cmp, dir);
 
            // Move pivot to end so it is out of the way during partitioning
            if (pivot_ptr != data_ + hi) {
                if constexpr (std::is_trivially_copyable_v<T>)
                    _swap_trivial(pivot_ptr, data_ + hi);
                else
                    _swap_move(pivot_ptr, data_ + hi);
            }
 
            size_t i = lo;
            for (size_t j = lo; j < hi; ++j) {
                if (_apply_dir<Func>(cmp(data_[j], data_[hi]), dir) < 0) {
                    if constexpr (std::is_trivially_copyable_v<T>)
                        _swap_trivial(data_ + i, data_ + j);
                    else
                        _swap_move(data_ + i, data_ + j);
                    ++i;
                }
            }
 
            // Place pivot in its final position
            if constexpr (std::is_trivially_copyable_v<T>)
                _swap_trivial(data_ + i, data_ + hi);
            else
                _swap_move(data_ + i, data_ + hi);
 
            return i;
        }
// --------------------------------------------------------------------------------
 
        /**
         * @brief Iterative quicksort with median-of-three pivot, insertion sort
         *        fallback for small partitions, and tail-call optimisation
         */
        template <typename Func>
        void _quicksort(size_t lo, size_t hi,
                        Func cmp, Direction dir) noexcept {
            constexpr size_t INSERTION_THRESHOLD = 10u;
 
            while (lo < hi) {
                if (hi - lo < INSERTION_THRESHOLD) {
                    _insertion_sort(lo, hi, cmp, dir);
                    break;
                }
 
                size_t pi = _partition(lo, hi, cmp, dir);
 
                // Recurse into the smaller partition, iterate into the larger
                // to keep worst-case stack depth at O(log n)
                if (pi > lo && pi - lo <= hi - pi) {
                    _quicksort(lo, pi - 1u, cmp, dir);
                    lo = pi + 1u;
                } else {
                    if (pi + 1u < hi)
                        _quicksort(pi + 1u, hi, cmp, dir);
                    if (pi == 0u) break;
                    hi = pi - 1u;
                }
            }
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Return the index of the minimum element
         *
         * @details For T == uint8_t delegates to the SIMD back-end selected at
         *          compile time.  For all other types performs a scalar linear
         *          scan using the caller-supplied comparator.
         */
        template <typename Func>
        size_t _min_index(Func cmp) const noexcept {
            if constexpr (std::is_same_v<T, uint8_t>) {
                return simd_min_uint8(data_, len_);
            } else {
                size_t best = 0u;
                for (size_t i = 1u; i < len_; ++i)
                    if (cmp(data_[i], data_[best]) < 0) best = i;
                return best;
            }
        }
// --------------------------------------------------------------------------------
 
        /**
         * @brief Return the index of the maximum element
         *
         * @details For T == uint8_t delegates to the SIMD back-end selected at
         *          compile time.  For all other types performs a scalar linear
         *          scan using the caller-supplied comparator.
         */
        template <typename Func>
        size_t _max_index(Func cmp) const noexcept {
            if constexpr (std::is_same_v<T, uint8_t>) {
                return simd_max_uint8(data_, len_);
            } else {
                size_t best = 0u;
                for (size_t i = 1u; i < len_; ++i)
                    if (cmp(data_[i], data_[best]) > 0) best = i;
                return best;
            }
        }
// --------------------------------------------------------------------------------

        /**
         * @brief Private constructor - prevents direct instantiation
         *
         * @param capacity Initial element capacity to allocate
         * @param allocator Allocator to use for memory management
         *
         * @details Sets up an empty array with a pre-allocated buffer of the
         *          requested capacity. If allocation fails, data_ is left as
         *          nullptr so that init() can detect the failure and clean up.
         */
        Array(size_t capacity, Allocator& allocator)
            : data_(nullptr), len_(0), cap_(0), allocator_(&allocator) {

            auto buf_result = allocator.alloc(capacity * sizeof(T), false);
            if (!buf_result.hasValue()) {
                // Leave data_ as nullptr; init() will detect and clean up
                return;
            }

            data_ = static_cast<T*>(buf_result.value());
            cap_  = capacity;
        }
// --------------------------------------------------------------------------------

        /**
         * @brief Private destructor - cleanup is handled by ArrayDeleter
         *
         * @details Calls the destructor of every live element, then frees the
         *          buffer through the stored allocator.
         */
        ~Array() noexcept {
            if (data_ && allocator_) {
                // Destroy live elements in reverse order
                for (size_t i = len_; i > 0u; --i) {
                    data_[i - 1u].~T();
                }
                allocator_->return_element(static_cast<void*>(data_),
                                           cap_ * sizeof(T),
                                           allocator_->default_alignment());
                data_ = nullptr;
            }
        }
// --------------------------------------------------------------------------------

        // Prevent copying via constructor and all move operations
        Array(const Array&)            = delete;
        Array& operator=(const Array&) = delete;
        Array(Array&&)                 = delete;
        Array& operator=(Array&&)      = delete;

    public:

        // ========================================================================
        // Factory
        // ========================================================================

        /**
         * @brief Initialise an allocator-backed Array
         *
         * @param capacity Initial element capacity (must be > 0)
         * @param allocator Allocator to use for memory management
         * @return Expected<Array<T>*> containing a pointer to the Array, or an error
         *
         * @details Allocates the Array object itself and its internal element buffer
         *          through the provided allocator using placement new, mirroring the
         *          String::init() pattern.
         *
         * @par Error conditions:
         * - capacity == 0 (ArgumentError)
         * - Allocation of the Array struct fails (MemoryError)
         * - Allocation of the element buffer fails (MemoryError)
         *
         * @code{.cpp}
         * cslt::HeapAllocator alloc;
         * auto r = cslt::Array<float>::init(16, alloc);
         * if (r.hasValue()) {
         *     cslt::UniquePtr<cslt::Array<float>, cslt::ArrayDeleter<float>> arr(r.value());
         *     arr->push_back(3.14f);
         * }
         * @endcode
         */
        static Expected<Array<T>*> init(size_t capacity,
                                        Allocator& allocator) noexcept {
            Expected<Array<T>*> result;

            if (capacity == 0u) {
                result.setError(ArgumentError("Array::init: capacity must be > 0"));
                return result;
            }

            // Allocate memory for the Array object itself
            auto obj_result = allocator.alloc(sizeof(Array<T>), true);
            if (!obj_result.hasValue()) {
                result.setError(obj_result.error());
                return result;
            }

            // Construct the Array in-place via placement new
            Array<T>* a = new (obj_result.value()) Array<T>(capacity, allocator);

            // Check whether the constructor managed to allocate the element buffer
            if (!a->data_) {
                a->~Array<T>();
                allocator.return_element(obj_result.value(),
                                         sizeof(Array<T>),
                                         allocator.default_alignment());
                result.setError(MemoryError("Array::init: failed to allocate element buffer"));
                return result;
            }

            result.setValue(a);
            return result;
        }
// --------------------------------------------------------------------------------

        /**
         * @brief Create a deep copy of an existing Array using a caller-supplied
         *        allocator
         *
         * @param src      Array to copy from
         * @param allocator Allocator used to back the new Array and its buffer
         * @return Expected<Array<T>*> containing a pointer to the new Array,
         *         or an error
         *
         * @details Allocates a fresh Array with the same capacity as @p src,
         *          then copy-constructs each live element. On any failure the
         *          partially constructed Array is cleaned up and an error is
         *          returned.
         *
         * @code{.cpp}
         * cslt::HeapAllocator alloc;
         * auto r  = cslt::Array<int>::init(4, alloc);
         * // ... push elements ...
         * auto r2 = cslt::Array<int>::copy(*r.value(), alloc);
         * @endcode
         */
        static Expected<Array<T>*> copy(const Array<T>& src,
                                        Allocator& allocator) noexcept {
            Expected<Array<T>*> result;

            // Allocate the new Array struct
            auto obj_result = allocator.alloc(sizeof(Array<T>), true);
            if (!obj_result.hasValue()) {
                result.setError(obj_result.error());
                return result;
            }

            // Construct with same capacity as source
            Array<T>* a = new (obj_result.value()) Array<T>(src.cap_, allocator);

            if (!a->data_) {
                a->~Array<T>();
                allocator.return_element(obj_result.value(),
                                         sizeof(Array<T>),
                                         allocator.default_alignment());
                result.setError(MemoryError("Array::copy: failed to allocate element buffer"));
                return result;
            }

            // Copy-construct each live element from the source
            for (size_t i = 0u; i < src.len_; ++i) {
                new (static_cast<void*>(a->data_ + i)) T(src.data_[i]);
                a->len_ = i + 1u;  // update len_ incrementally for safe cleanup on throw
            }

            result.setValue(a);
            return result;
        }
// --------------------------------------------------------------------------------

        /**
         * @brief Create a deep copy of an existing Array using the source
         *        Array's own allocator
         *
         * @param src Array to copy from
         * @return Expected<Array<T>*> containing a pointer to the new Array,
         *         or an error
         *
         * @details Convenience overload that forwards to copy(src, allocator)
         *          using the allocator already associated with @p src.
         *
         * @code{.cpp}
         * auto r2 = cslt::Array<int>::copy(*arr);
         * @endcode
         */
        static Expected<Array<T>*> copy(const Array<T>& src) noexcept {
            return copy(src, *src.allocator_);
        }
// --------------------------------------------------------------------------------

        // ========================================================================
        // Insertion
        // ========================================================================

        /**
         * @brief Append an element to the end of the array
         *
         * @param value Element to copy into the array
         * @return true on success, false if reallocation failed
         *
         * @details Grows the buffer using the tiered strategy if at capacity,
         *          then copy-constructs the new element at the end.
         *
         * @code{.cpp}
         * arr->push_back(42);
         * @endcode
         */
        bool push_back(const T& value) noexcept {
            if (!data_ || !allocator_) return false;
            if (!_ensure_capacity()) return false;

            new (static_cast<void*>(data_ + len_)) T(value);
            ++len_;
            return true;
        }
// --------------------------------------------------------------------------------

        /**
         * @brief Insert an element at the beginning of the array
         *
         * @param value Element to copy into the array
         * @return true on success, false if reallocation failed
         *
         * @details Grows the buffer if at capacity, shifts all existing elements
         *          one position to the right via memmove (trivial T) or a
         *          placement-new loop (non-trivial T), then copy-constructs the
         *          new element at index 0.
         *
         * @code{.cpp}
         * arr->push_front(42);
         * @endcode
         */
        bool push_front(const T& value) noexcept {
            if (!data_ || !allocator_) return false;
            if (!_ensure_capacity()) return false;

            _shift_right(0u);
            new (static_cast<void*>(data_)) T(value);
            ++len_;
            return true;
        }
// --------------------------------------------------------------------------------

        /**
         * @brief Insert an element at a caller-specified index
         *
         * @param index Zero-based insertion position. Must be <= size().
         *              index == 0        is equivalent to push_front().
         *              index == size()   is equivalent to push_back().
         * @param value Element to copy into the array
         * @return true on success, false if index is out of range or
         *         reallocation failed
         *
         * @details Grows the buffer if at capacity, shifts all elements at
         *          [index, len_) one position to the right, then copy-constructs
         *          the new element at @p index.
         *
         * @code{.cpp}
         * // array contains {1, 2, 4}
         * arr->push_any(2, 3);  // array becomes {1, 2, 3, 4}
         * @endcode
         */
        bool push_any(size_t index, const T& value) noexcept {
            if (!data_ || !allocator_) return false;
            if (index > len_) return false;
            if (!_ensure_capacity()) return false;

            _shift_right(index);
            new (static_cast<void*>(data_ + index)) T(value);
            ++len_;
            return true;
        }
// --------------------------------------------------------------------------------

        // ========================================================================
        // Removal
        // ========================================================================

        /**
         * @brief Remove the last element from the array
         *
         * @return true if an element was removed, false if the array was empty
         *
         * @details Calls the destructor of the last element and decrements len_.
         *          The buffer capacity is not reduced.
         *
         * @code{.cpp}
         * arr->pop_back();
         * @endcode
         */
        bool pop_back() noexcept {
            if (!data_ || len_ == 0u) return false;

            --len_;
            data_[len_].~T();
            return true;
        }
// --------------------------------------------------------------------------------

        /**
         * @brief Remove the element at the beginning of the array
         *
         * @return true if an element was removed, false if the array was empty
         *
         * @details Destroys the element at index 0, shifts all remaining elements
         *          one position to the left, then decrements len_. The buffer
         *          capacity is not reduced.
         *
         * @code{.cpp}
         * arr->pop_front();
         * @endcode
         */
        bool pop_front() noexcept {
            if (!data_ || len_ == 0u) return false;

            data_[0].~T();
            _shift_left(0u);
            --len_;
            return true;
        }
// --------------------------------------------------------------------------------

        /**
         * @brief Remove the element at a caller-specified index
         *
         * @param index Zero-based position of the element to remove.
         *              Must be < size().
         *              index == 0          is equivalent to pop_front().
         *              index == size()-1   is equivalent to pop_back().
         * @return true on success, false if the array is empty or index is
         *         out of range
         *
         * @details Destroys the element at @p index, shifts all elements at
         *          (index, len_) one position to the left, then decrements len_.
         *          The buffer capacity is not reduced.
         *
         * @code{.cpp}
         * // array contains {1, 2, 3, 4}
         * arr->pop_any(2);  // array becomes {1, 2, 4}
         * @endcode
         */
        bool pop_any(size_t index) noexcept {
            if (!data_ || len_ == 0u) return false;
            if (index >= len_) return false;

            data_[index].~T();
            _shift_left(index);
            --len_;
            return true;
        }
// --------------------------------------------------------------------------------

        /**
         * @brief Destroy all live elements and reset the populated length to zero
         *
         * @details Calls the destructor of every live element in reverse order,
         *          then sets len_ to 0. The allocated buffer and its capacity are
         *          preserved so the Array can be reused without a further
         *          allocation.
         *
         * @code{.cpp}
         * arr->clear();
         * // arr->size() == 0, arr->capacity() unchanged
         * @endcode
         */
        void clear() noexcept {
            for (size_t i = len_; i > 0u; --i) {
                data_[i - 1u].~T();
            }
            len_ = 0u;
        }
// --------------------------------------------------------------------------------

        // ========================================================================
        // Indexed access
        // ========================================================================

        /**
         * @brief Write or append an element at a given index
         *
         * @param index Zero-based position to write to. Must be <= size().
         * @param value Element to copy into the array at @p index
         * @return Expected<bool> — true on success; error on failure
         *
         * @details Three cases are handled:
         *          - index <  len_  : destroys the existing element and
         *                             copy-constructs @p value in its place
         *          - index == len_  : calls _ensure_capacity(), copy-constructs
         *                             @p value, and increments len_
         *          - index >  len_  : returns an OutOfBoundsError
         *
         * @code{.cpp}
         * arr->set(0, 42);            // overwrite existing element
         * arr->set(arr->size(), 99);  // append via set
         * @endcode
         */
        Expected<bool> set(size_t index, const T& value) noexcept {
            Expected<bool> result;

            if (!data_ || !allocator_) {
                result.setError(ArgumentError("Array::set: array is not initialised"));
                return result;
            }

            if (index < len_) {
                data_[index].~T();
                new (static_cast<void*>(data_ + index)) T(value);
                result.setValue(true);
                return result;
            }

            if (index == len_) {
                if (!_ensure_capacity()) {
                    result.setError(MemoryError("Array::set: failed to grow buffer"));
                    return result;
                }
                new (static_cast<void*>(data_ + len_)) T(value);
                ++len_;
                result.setValue(true);
                return result;
            }

            result.setError(OutOfBoundsError("Array::set: index out of range"));
            return result;
        }
// --------------------------------------------------------------------------------

        /**
         * @brief Read-only access to an element at a given index
         *
         * @param index Zero-based position to read from. Must be < size().
         * @return Expected<T> containing a copy of the element on success,
         *         or an error if @p index is out of the populated range
         *
         * @code{.cpp}
         * auto r = (*arr)[1];
         * if (r.hasValue()) {
         *     int val = r.value();
         * }
         * @endcode
         */
        Expected<T> operator[](size_t index) const noexcept {
            Expected<T> result;

            if (!data_ || index >= len_) {
                result.setError(OutOfBoundsError("Array::operator[]: index out of range"));
                return result;
            }

            result.setValue(data_[index]);
            return result;
        }
// --------------------------------------------------------------------------------

        // ========================================================================
        // Accessors
        // ========================================================================

        /**
         * @brief Return a const pointer to the beginning of the element buffer
         *
         * @return Const pointer to the first element, or nullptr if the array
         *         has not been successfully initialised
         *
         * @details The pointer is valid until the next push operation that
         *          triggers a reallocation.
         *
         * @code{.cpp}
         * const int* ptr = arr->data();
         * for (size_t i = 0; i < arr->size(); ++i)
         *     std::cout << ptr[i] << '\n';
         * @endcode
         */
        const T* data() const noexcept {
            return data_;
        }
// --------------------------------------------------------------------------------

        /**
         * @brief Return the number of elements currently stored
         *
         * @return Element count
         */
        size_t size() const noexcept {
            return len_;
        }
// --------------------------------------------------------------------------------

        /**
         * @brief Return the current element capacity of the buffer
         *
         * @return Capacity in number of elements
         */
        size_t capacity() const noexcept {
            return cap_;
        }
// --------------------------------------------------------------------------------

        /**
         * @brief Return true if no elements are currently stored
         *
         * @return true if size() == 0
         *
         * @code{.cpp}
         * if (arr->is_empty()) { ... }
         * @endcode
         */
        bool is_empty() const noexcept {
            return len_ == 0u;
        }
// --------------------------------------------------------------------------------

        /**
         * @brief Return true if the populated length equals the allocated capacity
         *
         * @return true if size() == capacity()
         *
         * @details Note that a push operation on a full array will attempt to
         *          grow the buffer automatically. is_full() is useful for callers
         *          that need to know whether the next push will trigger a
         *          reallocation.
         *
         * @code{.cpp}
         * if (arr->is_full()) { ... }
         * @endcode
         */
        bool is_full() const noexcept {
            return len_ == cap_;
        }
// --------------------------------------------------------------------------------

        /**
         * @brief Return true if a pointer falls within the populated region
         *        and is aligned to an element boundary
         *
         * @param ptr Pointer to test
         * @return true if @p ptr points to a valid element in [data_, data_ + len_)
         *         and satisfies (ptr - data_) % sizeof(T) == 0
         *
         * @details A pointer obtained after a reallocation (e.g. following a
         *          push that triggered a grow) may no longer be valid; callers
         *          are responsible for not retaining pointers across reallocations.
         *
         * @code{.cpp}
         * const int* p = &(*arr->data());
         * if (arr->is_ptr(p)) { ... }
         * @endcode
         */
        bool is_ptr(const T* ptr) const noexcept {
            if (!data_ || !ptr || len_ == 0u) return false;

            if (ptr < data_ || ptr >= data_ + len_) return false;

            // Verify the pointer is index-aligned within the buffer
            size_t const byte_offset =
                static_cast<size_t>(
                    reinterpret_cast<const char*>(ptr) -
                    reinterpret_cast<const char*>(data_));

            return (byte_offset % sizeof(T)) == 0u;
        }
// --------------------------------------------------------------------------------

        // ========================================================================
        // Transformations
        // ========================================================================

        /**
         * @brief Produce a new Array containing the prefix-sum (running total)
         *        of this array using a caller-supplied accumulation function
         *
         * @tparam Func  Callable with signature `void(T&, const T&)`.
         *               The first argument is the current output element
         *               (modified in place); the second is the incoming source
         *               element. Any callable is accepted: plain function,
         *               lambda, or functor.
         *
         * @param add       Accumulation callable
         * @param allocator Allocator used to back the returned Array
         * @return Expected<Array<T>*> containing the cumulative Array on
         *         success, or an error
         *
         * @details Mirrors the C cumulative_array() prefix-sum algorithm:
         *          - result[0] = src[0]  (seed — no identity element required)
         *          - result[i] = result[i-1] + src[i]  for i >= 1
         *
         *          The output Array is allocated with exactly len_ slots (no
         *          extra capacity), making it a fixed-length snapshot.
         *
         * @par Error conditions:
         * - Array is empty (EmptyError)
         * - Allocation of the output struct or buffer fails (MemoryError)
         *
         * @code{.cpp}
         * auto r = cslt::Array<int>::cumulative(*src,
         *     [](int& accum, const int& elem) { accum += elem; },
         *     alloc);
         * @endcode
         */
        template <typename Func>
        static Expected<Array<T>*> cumulative(const Array<T>& src,
                                              Func            add,
                                              Allocator&      allocator) noexcept {
            Expected<Array<T>*> result;
 
            if (src.len_ == 0u) {
                result.setError(EmptyError("Array::cumulative: source array is empty"));
                return result;
            }
 
            // Construct the destination array via init() — same validated path
            // as all other callers, capacity fixed to src.len_ (snapshot).
            auto init_result = Array<T>::init(src.len_, allocator);
            if (!init_result.hasValue()) {
                result.setError(init_result.error());
                return result;
            }
 
            // Hold ownership so any early return is leak-free.
            // ArrayDeleter is defined after this class, so use a lambda deleter
            // to avoid the forward-reference problem.
            auto array_deleter = [](Array<T>* p) noexcept {
                if (!p) return;
                Allocator* a = p->allocator_;
                p->~Array<T>();
                if (a) a->return_element(static_cast<void*>(p),
                                         sizeof(Array<T>),
                                         a->default_alignment());
            };
            UniquePtr<Array<T>, decltype(array_deleter)> dst(init_result.value(), array_deleter);
 
            // Seed: output[0] = src[0]
            if (!dst->push_back(src.data_[0])) {
                result.setError(MemoryError("Array::cumulative: failed to seed output"));
                return result;
            }
 
            // Prefix-accumulation pass: output[i] = output[i-1] `add` src[i]
            for (size_t i = 1u; i < src.len_; ++i) {
                // Read the previous cumulative value through the const overload.
                auto prev = (*const_cast<const Array<T>*>(dst.get()))[i - 1u];
                if (!prev.hasValue()) {
                    result.setError(MemoryError("Array::cumulative: failed to read previous element"));
                    return result;
                }
                // Start the new slot as a copy of the previous cumulative value,
                // then apply the caller's accumulation operation in place.
                T next_val = prev.value();
                add(next_val, src.data_[i]);
                if (!dst->push_back(next_val)) {
                    result.setError(MemoryError("Array::cumulative: failed to push accumulated element"));
                    return result;
                }
            }
 
            // Release ownership to the caller.
            result.setValue(dst.release());
            return result;
        }
// --------------------------------------------------------------------------------

        /**
         * @brief Produce a new Array containing the prefix-sum of this array
         *        using the source array's own allocator
         *
         * @tparam Func  Callable with signature `void(T&, const T&)`
         * @param add    Accumulation callable
         * @return Expected<Array<T>*> on success, or an error
         *
         * @details Convenience overload that forwards to
         *          cumulative(src, add, allocator) using the allocator
         *          already associated with this array.
         *
         * @code{.cpp}
         * auto r = cslt::Array<int>::cumulative(*src,
         *     [](int& accum, const int& elem) { accum += elem; });
         * @endcode
         */
        template <typename Func>
        static Expected<Array<T>*> cumulative(const Array<T>& src,
                                              Func            add) noexcept {
            return cumulative(src, add, *src.allocator_);
        }
// --------------------------------------------------------------------------------

        /**
         * @brief Return a new Array containing the elements in [start, end)
         *
         * @param start     Zero-based index of the first element to include
         * @param end       One-past-the-last index (exclusive upper bound).
         *                  Must satisfy start < end <= size().
         * @param allocator Allocator used to back the returned Array
         * @return Expected<Array<T>*> containing the slice on success, or
         *         an error
         *
         * @details Mirrors the C slice_array() semantics exactly:
         *          - start >= end  is an ArgumentError
         *          - end > len_    is an OutOfBoundsError
         *          The returned Array has capacity == (end - start) and is a
         *          fixed-length snapshot of the source range. For trivially
         *          copyable T the copy uses std::memcpy; for non-trivial T
         *          each element is copy-constructed individually.
         *
         * @par Error conditions:
         * - start >= end (ArgumentError)
         * - end > len_   (OutOfBoundsError)
         * - Allocation failure (MemoryError)
         *
         * @code{.cpp}
         * // array contains {0, 1, 2, 3, 4}
         * auto r = cslt::Array<int>::slice(*arr, 1, 4, alloc);
         * // result contains {1, 2, 3}
         * @endcode
         */
        static Expected<Array<T>*> slice(const Array<T>& src,
                                         size_t          start,
                                         size_t          end,
                                         Allocator&      allocator) noexcept {
            Expected<Array<T>*> result;
 
            if (start >= end) {
                result.setError(ArgumentError("Array::slice: start must be < end"));
                return result;
            }
            if (end > src.len_) {
                result.setError(OutOfBoundsError("Array::slice: end exceeds array size"));
                return result;
            }
 
            size_t const slice_len = end - start;
 
            // Allocate the output Array struct directly, mirroring the copy()
            // pattern — no UniquePtr or ArrayDeleter needed here.
            auto obj_result = allocator.alloc(sizeof(Array<T>), true);
            if (!obj_result.hasValue()) {
                result.setError(obj_result.error());
                return result;
            }
 
            Array<T>* dst = new (obj_result.value()) Array<T>(slice_len, allocator);
 
            if (!dst->data_) {
                dst->~Array<T>();
                allocator.return_element(obj_result.value(),
                                         sizeof(Array<T>),
                                         allocator.default_alignment());
                result.setError(MemoryError("Array::slice: failed to allocate buffer"));
                return result;
            }
 
            // Copy-construct each element from the source range, incrementing
            // len_ after each one so the destructor cleans up correctly on throw.
            for (size_t i = 0u; i < slice_len; ++i) {
                new (static_cast<void*>(dst->data_ + i)) T(src.data_[start + i]);
                dst->len_ = i + 1u;
            }
 
            result.setValue(dst);
            return result;
        }
// --------------------------------------------------------------------------------

        /**
         * @brief Return a new Array containing the elements in [start, end)
         *        using the source array's own allocator
         *
         * @param start Zero-based index of the first element to include
         * @param end   One-past-the-last index (exclusive upper bound)
         * @return Expected<Array<T>*> on success, or an error
         *
         * @details Convenience overload that forwards to
         *          slice(src, start, end, allocator) using the allocator
         *          already associated with this array.
         *
         * @code{.cpp}
         * auto r = cslt::Array<int>::slice(*arr, 1, 4);
         * @endcode
         */
        static Expected<Array<T>*> slice(const Array<T>& src,
                                         size_t          start,
                                         size_t          end) noexcept {
            return slice(src, start, end, *src.allocator_);
        }
// --------------------------------------------------------------------------------

        /**
         * @brief Append all elements of another Array to the end of this array
         *
         * @param other Array whose elements are appended to this array
         * @return true on success, false if any reallocation fails
         *
         * @details Iterates over the populated elements of @p other and appends
         *          each one via push_back(), which uses the tiered growth
         *          strategy as needed. If any individual push_back() fails
         *          (reallocation error), the method returns false immediately;
         *          elements already appended before the failure remain in the
         *          array.
         *
         * @note Self-concatenation (passing this array as @p other) is safe
         *       because push_back() always reads from @p other before writing,
         *       and the snapshot of other.len_ is taken once before the loop.
         *
         * @code{.cpp}
         * // a contains {1, 2, 3}, b contains {4, 5, 6}
         * a->concat(*b);
         * // a now contains {1, 2, 3, 4, 5, 6}
         * @endcode
         */
        bool concat(const Array<T>& other) noexcept {
            if (!data_ || !allocator_) return false;

            size_t const other_len = other.len_;
            if (other_len == 0u) return true;

            // Snapshot the current length before any reallocation so that
            // self-concatenation (other == *this) is handled correctly
            size_t const old_len = len_;
            size_t const new_len = old_len + other_len;

            // Overflow guard
            if (new_len < old_len) return false;

            // Grow once upfront to a tiered capacity that fits new_len
            if (new_len > cap_) {
                size_t new_cap = cap_;
                while (new_cap < new_len) {
                    new_cap = _compute_new_cap(new_cap);
                }
                if (!_grow(new_cap)) return false;
            }

            // Copy elements from other into the newly available slots
            if constexpr (std::is_trivially_copyable_v<T>) {
                std::memcpy(static_cast<void*>(data_ + old_len),
                            static_cast<const void*>(other.data_),
                            other_len * sizeof(T));
                len_ = new_len;
            } else {
                for (size_t i = 0u; i < other_len; ++i) {
                    new (static_cast<void*>(data_ + old_len + i)) T(other.data_[i]);
                    ++len_;
                }
            }

            return true;
        }
// -------------------------------------------------------------------------------- 

       /**
         * @brief Reverse the order of all elements in the array in place
         *
         * @details For trivially copyable T the reversal is delegated to
         *          simd_reverse_uint8(), which is resolved at compile time to
         *          the best available SIMD back-end (AVX-512, AVX2, AVX, SSE4.1,
         *          SSSE3, SSE2, NEON, SVE, SVE2) or a portable scalar fallback.
         *          The function treats each element as an opaque bag of
         *          sizeof(T) bytes and swaps element positions — it does not
         *          reverse the bytes within an individual element.
         *
         *          For non-trivial T, a scalar swap loop is used that respects
         *          move construction and destruction semantics.
         *
         *          Arrays with fewer than two elements are left unchanged.
         *
         * @code{.cpp}
         * // arr contains {1, 2, 3, 4, 5}
         * arr->reverse();
         * // arr now contains {5, 4, 3, 2, 1}
         * @endcode
         */
        void reverse() noexcept {
            if (len_ < 2u) return;
 
            if constexpr (std::is_trivially_copyable_v<T> && sizeof(T) == 1u) {
                // Single-byte trivially copyable elements (e.g. uint8_t, char):
                // delegate to the SIMD back-end selected at compile time.
                // reinterpret_cast to uint8_t* is legal — the standard permits
                // copying object representations through unsigned char / uint8_t.
                simd_reverse_uint8(reinterpret_cast<uint8_t*>(data_),
                                   len_,
                                   sizeof(T));
            } else if constexpr (std::is_trivially_copyable_v<T>) {
                // Multi-byte trivially copyable elements (int, double, structs,
                // etc.): use a scalar memcpy-swap loop.  memcpy is correct here
                // because the standard explicitly permits copying trivially
                // copyable object representations.  The compiler will typically
                // auto-vectorise this loop for common element sizes.
                size_t lo = 0u;
                size_t hi = len_;
                while (lo < hi) {
                    --hi;
                    T tmp;
                    std::memcpy(&tmp,       data_ + lo, sizeof(T));
                    std::memcpy(data_ + lo, data_ + hi, sizeof(T));
                    std::memcpy(data_ + hi, &tmp,       sizeof(T));
                    ++lo;
                }
            } else {
                // Non-trivial T: honour construction and destruction semantics
                // via a move-swap loop.
                size_t lo = 0u;
                size_t hi = len_;
                while (lo < hi) {
                    --hi;
                    T tmp(std::move(data_[lo]));
                    data_[lo].~T();
                    new (static_cast<void*>(data_ + lo)) T(std::move(data_[hi]));
                    data_[hi].~T();
                    new (static_cast<void*>(data_ + hi)) T(std::move(tmp));
                    ++lo;
                }
            }
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Sort the array in place using an iterative quicksort
         *
         * @tparam Func  Callable with signature `int(const T&, const T&)`.
         *               Must return negative if a < b, zero if a == b, and
         *               positive if a > b — identical to the C qsort comparator
         *               convention.  Lambdas, functors, and plain function
         *               pointers are all accepted.
         *
         * @param cmp  Comparator callable
         * @param dir  Direction::FORWARD for ascending order;
         *             Direction::REVERSE for descending order
         * @return true on success; false if the array is uninitialised or
         *         contains fewer than two elements (no-op in those cases)
         *
         * @details Implements an iterative median-of-three quicksort with an
         *          insertion sort fallback for partitions smaller than 10
         *          elements and tail-call optimisation to keep stack depth at
         *          O(log n) in the worst case.  Element swaps use memcpy for
         *          trivially copyable T and move construction / destruction for
         *          non-trivial T.
         *
         * @code{.cpp}
         * arr->sort([](const int& a, const int& b) { return a - b; },
         *           cslt::Direction::FORWARD);   // ascending
         *
         * arr->sort([](const int& a, const int& b) { return a - b; },
         *           cslt::Direction::REVERSE);   // descending
         * @endcode
         */

        template <typename Func>
        bool sort(Func cmp, Direction dir) noexcept {
            if (!data_ || !allocator_) return false;
            if (len_ < 2u)             return false;
 
            _quicksort(0u, len_ - 1u, cmp, dir);
            return true;
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Return the minimum element in the array
         *
         * @tparam Func  Callable with signature `int(const T&, const T&)`.
         *               The comparator must satisfy:
         *               - returns a negative value when a <  b
         *               - returns zero              when a == b
         *               - returns a positive value  when a >  b
         *
         *               Any callable is accepted: lambda, functor, or plain
         *               function pointer.  For T == uint8_t the comparator is
         *               ignored — the SIMD back-end uses hardware min directly.
         *
         * @param cmp  Comparator callable
         * @return Expected<T> containing a copy of the minimum element on
         *         success, or an EmptyError if the array has no elements
         *
         * @par Comparator examples
         *
         * Scalar types — subtract or use the branchless idiom:
         * @code{.cpp}
         * // int array — branchless (avoids signed overflow risk of a - b)
         * auto r = arr->min([](const int& a, const int& b) {
         *     return (a > b) - (a < b);
         * });
         *
         * // float array — branchless
         * auto r = arr->min([](const float& a, const float& b) {
         *     return (a > b) - (a < b);
         * });
         * @endcode
         *
         * Struct / class types — compare on a specific field:
         * @code{.cpp}
         * // Particle struct with fields mass, charge, id
         * // Find the particle with the lowest mass
         * auto r = arr->min([](const Particle& a, const Particle& b) {
         *     return (a.mass > b.mass) - (a.mass < b.mass);
         * });
         *
         * // Find the particle with the most negative charge
         * auto r = arr->min([](const Particle& a, const Particle& b) {
         *     return (a.charge > b.charge) - (a.charge < b.charge);
         * });
         * @endcode
         *
         * File-scope comparator function (reusable across multiple calls):
         * @code{.cpp}
         * static int cmp_particle_mass(const Particle& a, const Particle& b) {
         *     return (a.mass > b.mass) - (a.mass < b.mass);
         * }
         * auto r = arr->min(cmp_particle_mass);
         * if (r.hasValue()) {
         *     Particle lightest = r.value();
         * }
         * @endcode
         *
         * @par Error conditions
         * - Array is empty (EmptyError)
         */    
        template <typename Func>
        Expected<T> min(Func cmp) const noexcept {
            Expected<T> result;
            if (!data_ || len_ == 0u) {
                result.setError(EmptyError("Array::min: array is empty"));
                return result;
            }
            result.setValue(data_[_min_index(cmp)]);
            return result;
        }
// --------------------------------------------------------------------------------
 
        /**
         * @brief Return the maximum element in the array
         *
         * @tparam Func  Callable with signature `int(const T&, const T&)`.
         *               The comparator must satisfy:
         *               - returns a negative value when a <  b
         *               - returns zero              when a == b
         *               - returns a positive value  when a >  b
         *
         *               Any callable is accepted: lambda, functor, or plain
         *               function pointer.  For T == uint8_t the comparator is
         *               ignored — the SIMD back-end uses hardware max directly.
         *
         * @param cmp  Comparator callable
         * @return Expected<T> containing a copy of the maximum element on
         *         success, or an EmptyError if the array has no elements
         *
         * @par Comparator examples
         *
         * Scalar types — use the branchless idiom:
         * @code{.cpp}
         * // int array
         * auto r = arr->max([](const int& a, const int& b) {
         *     return (a > b) - (a < b);
         * });
         *
         * // float array
         * auto r = arr->max([](const float& a, const float& b) {
         *     return (a > b) - (a < b);
         * });
         * @endcode
         *
         * Struct / class types — compare on a specific field:
         * @code{.cpp}
         * // Particle struct with fields mass, charge, id
         * // Find the particle with the highest mass
         * auto r = arr->max([](const Particle& a, const Particle& b) {
         *     return (a.mass > b.mass) - (a.mass < b.mass);
         * });
         *
         * // Find the particle with the highest charge
         * auto r = arr->max([](const Particle& a, const Particle& b) {
         *     return (a.charge > b.charge) - (a.charge < b.charge);
         * });
         * @endcode
         *
         * File-scope comparator function (reusable across multiple calls):
         * @code{.cpp}
         * static int cmp_particle_mass(const Particle& a, const Particle& b) {
         *     return (a.mass > b.mass) - (a.mass < b.mass);
         * }
         * auto r = arr->max(cmp_particle_mass);
         * if (r.hasValue()) {
         *     Particle heaviest = r.value();
         * }
         * @endcode
         *
         * @par Error conditions
         * - Array is empty (EmptyError)
         */ 
        template <typename Func>
        Expected<T> max(Func cmp) const noexcept {
            Expected<T> result;
            if (!data_ || len_ == 0u) {
                result.setError(EmptyError("Array::max: array is empty"));
                return result;
            }
            result.setValue(data_[_max_index(cmp)]);
            return result;
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Search for the first element whose byte representation matches
         *        @p value and return its index
         *
         * @param value  Element to search for.  Comparison is performed by
         *               comparing the raw byte representation of each array
         *               element against the raw bytes of @p value using
         *               simd_contains_uint8() (for the SIMD-accelerated path)
         *               or memcmp() (for the scalar remainder).
         *
         * @return Expected<size_t> containing the zero-based index of the first
         *         match on success, or an OutOfBoundsError if no match is found
         *
         * @par Compile-time restriction
         * This overload is only available when T satisfies both:
         * - std::is_trivially_copyable_v<T> == true
         * - std::is_floating_point_v<T>     == false
         *
         * Floating-point types are excluded because -0.0 and +0.0 are equal
         * by value but differ in bit pattern, so memcmp-based search would
         * produce incorrect results.  Use the predicate overload (coming soon)
         * for float, double, and long double.
         *
         * Structs with padding bytes are also subject to this caveat — padding
         * content is unspecified by the standard and may differ between two
         * otherwise identical objects.  Prefer the predicate overload for
         * structs unless you can guarantee there are no padding bytes.
         *
         * @par SIMD dispatch
         * The search delegates to simd_contains_uint8(), which is resolved at
         * compile time to the best available SIMD back-end (AVX-512, AVX2, AVX,
         * SSE4.1, SSSE3, SSE2, NEON, SVE, SVE2) or a portable scalar fallback.
         * For element sizes 1, 2, 4, and 8 bytes the SIMD path is taken; all
         * other sizes fall through to a scalar memcmp loop.
         *
         * @par Usage examples
         * @code{.cpp}
         * // Search for an integer value
         * auto r = arr->contains(42);
         * if (r.hasValue()) {
         *     size_t idx = r.value();  // index of first 42
         * }
         *
         * // Search for a plain struct (no padding, no floats)
         * struct Vec3i { int x, y, z; };
         * auto r = arr->contains(Vec3i{1, 2, 3});
         * @endcode
         *
         * @par Error conditions
         * - Value not found in the array (OutOfBoundsError)
         */
        Expected<size_t> contains(const T& value) const noexcept {
            static_assert(std::is_trivially_copyable_v<T>,
                "Array::contains: T must be trivially copyable. "
                "Use the predicate overload for non-trivial types.");
            static_assert(!std::is_floating_point_v<T>,
                "Array::contains: floating-point types are not safe for "
                "byte comparison (-0.0 != +0.0 bitwise). "
                "Use the predicate overload for float, double, and long double.");
 
            Expected<size_t> result;
 
            if (!data_ || len_ == 0u) {
                result.setError(OutOfBoundsError(
                    "Array::contains: value not found"));
                return result;
            }
 
            size_t idx = simd_contains_uint8(
                reinterpret_cast<const uint8_t*>(data_),
                0u,
                len_,
                sizeof(T),
                reinterpret_cast<const uint8_t*>(&value));
 
            if (idx == SIZE_MAX) {
                result.setError(OutOfBoundsError(
                    "Array::contains: value not found"));
                return result;
            }
 
            result.setValue(idx);
            return result;
        } 
// -------------------------------------------------------------------------------- 

        /**
         * @brief Search for the first element for which a caller-supplied
         *        equality predicate returns true and return its index
         *
         * @tparam Func  Callable with signature `bool(const T&, const T&)`.
         *               Must return true when the two arguments are considered
         *               equal.  Any callable is accepted: lambda, functor, or
         *               plain function pointer.
         *
         * @param value  Element to search for, passed as the first argument
         *               to @p eq at each position
         * @param eq     Equality predicate
         *
         * @return Expected<size_t> containing the zero-based index of the first
         *         match on success, or an OutOfBoundsError if no match is found
         *
         * @details Performs a linear scan from index 0 to size()-1, calling
         *          eq(value, data_[i]) at each position.  No SIMD acceleration
         *          is applied — equality semantics are entirely defined by the
         *          caller's predicate, so no byte-level optimisation is possible.
         *
         *          This overload is intended for:
         *          - Non-trivially-copyable types (classes with user-defined
         *            copy constructors or destructors)
         *          - Trivially copyable types where value equality differs from
         *            bitwise equality (e.g. structs with padding bytes)
         *          - Any type where a custom notion of equality is required
         *            (e.g. case-insensitive string matching, epsilon comparison)
         *
         * @par Usage examples
         * @code{.cpp}
         * // Non-trivial class with a user-defined copy constructor
         * auto r = arr->contains(target,
         *     [](const MyClass& a, const MyClass& b) {
         *         return a.id() == b.id();
         *     });
         * if (r.hasValue()) {
         *     size_t idx = r.value();
         * }
         *
         * // Struct with padding — use operator== rather than memcmp
         * struct Padded { char c; int x; };   // likely has 3 padding bytes
         * auto r = arr->contains(needle,
         *     [](const Padded& a, const Padded& b) {
         *         return a.c == b.c && a.x == b.x;
         *     });
         *
         * // File-scope predicate (reusable across multiple calls)
         * static bool eq_by_id(const MyClass& a, const MyClass& b) {
         *     return a.id() == b.id();
         * }
         * auto r = arr->contains(target, eq_by_id);
         * @endcode
         *
         * @par Error conditions
         * - Value not found in the array (OutOfBoundsError)
         */
        template <typename Func>
        Expected<size_t> contains(const T& value, Func eq) const noexcept {
            Expected<size_t> result;
 
            if (!data_ || len_ == 0u) {
                result.setError(OutOfBoundsError(
                    "Array::contains: value not found"));
                return result;
            }
 
            for (size_t i = 0u; i < len_; ++i) {
                if (eq(value, data_[i])) {
                    result.setValue(i);
                    return result;
                }
            }
 
            result.setError(OutOfBoundsError(
                "Array::contains: value not found"));
            return result;
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Search for @p value using binary search and return its index
         *
         * @tparam Func  Callable with signature `int(const T&, const T&)`.
         *               Same convention as sort() and min()/max():
         *               - negative when a <  b
         *               - zero     when a == b
         *               - positive when a >  b
         *
         * @param value     Element to search for
         * @param cmp       Comparator defining the sort order
         * @param dir       Direction::FORWARD if the array is (or should be)
         *                  sorted ascending; Direction::REVERSE if descending.
         *                  Must be consistent with @p cmp.
         * @param is_sorted Pass true if the array is already sorted in the
         *                  order defined by @p cmp and @p dir.  Pass false to
         *                  sort the array in place before searching.
         *
         * @return Expected<size_t> containing the zero-based index of a
         *         matching element on success, or an OutOfBoundsError if no
         *         match is found or the array is empty.
         *
         * @details Implements a standard iterative binary search.  When
         *          @p is_sorted is false the array is sorted in place via
         *          the same quicksort used by sort() before the search begins,
         *          so subsequent calls with is_sorted == true are valid.
         *
         *          If multiple elements compare equal to @p value the index
         *          of the first (lowest) occurrence is returned.  Once a match
         *          is found the search continues leftward until no earlier
         *          match exists.
         *
         * @par Complexity
         * - O(n log n) when is_sorted == false (sort step dominates)
         * - O(log n)   when is_sorted == true
         *
         * @par Usage examples
         * @code{.cpp}
         * auto cmp = [](const int& a, const int& b) {
         *     return (a > b) - (a < b);
         * };
         *
         * // Array not yet sorted — sort in place then search
         * arr->binary_search(30, cmp, cslt::Direction::FORWARD, false);
         *
         * // Array already sorted ascending — skip the sort step
         * auto r = arr->binary_search(30, cmp, cslt::Direction::FORWARD, true);
         * if (r.hasValue()) {
         *     size_t idx = r.value();
         * }
         *
         * // Struct array sorted descending by x field
         * auto cmp_pt = [](const Point& a, const Point& b) {
         *     return (a.x > b.x) - (a.x < b.x);
         * };
         * auto r = arr->binary_search(target, cmp_pt,
         *                             cslt::Direction::REVERSE, true);
         * @endcode
         *
         * @par Error conditions
         * - Value not found (OutOfBoundsError)
         * - Array is empty (OutOfBoundsError)
         */
        template <typename Func>
        Expected<size_t> binary_search(const T& value,
                                       Func     cmp,
                                       Direction dir,
                                       bool     is_sorted) noexcept {
            Expected<size_t> result;
 
            if (!data_ || len_ == 0u) {
                result.setError(OutOfBoundsError(
                    "Array::binary_search: array is empty"));
                return result;
            }
 
            // Sort in place if the caller has not guaranteed ordering
            if (!is_sorted) {
                _quicksort(0u, len_ - 1u, cmp, dir);
            }
 
            // Iterative binary search — finds the leftmost matching index
            // when duplicates are present.  The standard binary search is run
            // first to confirm the value exists, then hi is tightened to find
            // the first occurrence.
            size_t lo    = 0u;
            size_t hi    = len_ - 1u;
            size_t found = SIZE_MAX;
 
            while (lo <= hi) {
                size_t const mid = lo + (hi - lo) / 2u;
                int    const cmp_result = _apply_dir<Func>(
                                              cmp(data_[mid], value), dir);
 
                if (cmp_result == 0) {
                    // Record this match and keep searching left for an
                    // earlier occurrence
                    found = mid;
                    if (mid == 0u) break;
                    hi = mid - 1u;
                } else if (cmp_result < 0) {
                    lo = mid + 1u;
                } else {
                    if (mid == 0u) break;
                    hi = mid - 1u;
                }
            }
 
            if (found == SIZE_MAX) {
                result.setError(OutOfBoundsError(
                    "Array::binary_search: value not found"));
                return result;
            }
 
            result.setValue(found);
            return result;
        }
// -------------------------------------------------------------------------------- 

        /**
         * @brief Search for the bracketing pair of indices surrounding @p value
         *        in the sorted array
         *
         * @tparam Func  Callable with signature `int(const T&, const T&)`.
         *               Same convention as sort() and binary_search():
         *               - negative when a <  b
         *               - zero     when a == b
         *               - positive when a >  b
         *
         * @param value     Element to bracket
         * @param cmp       Comparator defining the sort order
         * @param dir       Direction::FORWARD for ascending order;
         *                  Direction::REVERSE for descending order.
         *                  Must be consistent with @p cmp.
         * @param is_sorted Pass true if the array is already sorted in the
         *                  order defined by @p cmp and @p dir.  Pass false to
         *                  sort the array in place before searching.
         *
         * @return Expected<std::pair<size_t, size_t>> where:
         *         - first  is the index of the largest element <= @p value
         *                  (lower bound)
         *         - second is the index of the smallest element >= @p value
         *                  (upper bound)
         *
         *         When @p value exists in the array both bounds are the index
         *         of the first (lowest) matching element.
         *
         *         Returns OutOfBoundsError when:
         *         - the array is empty
         *         - @p value is less than the smallest element (no lower bound)
         *         - @p value is greater than the largest element (no upper bound)
         *
         * @details The algorithm performs a single binary search pass to locate
         *          the insertion point of @p value, then derives the lower and
         *          upper bound indices from the elements immediately surrounding
         *          that point.
         *
         * @par Complexity
         * - O(n log n) when is_sorted == false (sort step dominates)
         * - O(log n)   when is_sorted == true
         *
         * @par Usage examples
         * @code{.cpp}
         * // Array {10, 20, 30, 40, 50} — bracket 25
         * auto r = arr->bracketed_binary_search(25, cmp,
         *                                       cslt::Direction::FORWARD, true);
         * if (r.hasValue()) {
         *     size_t lo = r.value().first;   // index of 20
         *     size_t hi = r.value().second;  // index of 30
         * }
         *
         * // Exact match — bracket 30
         * auto r = arr->bracketed_binary_search(30, cmp,
         *                                       cslt::Direction::FORWARD, true);
         * // r.value().first == r.value().second == 2 (index of 30)
         * @endcode
         *
         * @par Error conditions
         * - Array is empty (OutOfBoundsError)
         * - Value below the smallest element — no lower bound (OutOfBoundsError)
         * - Value above the largest element  — no upper bound (OutOfBoundsError)
         */
        template <typename Func>
        Expected<std::pair<size_t, size_t>>
        bracketed_binary_search(const T& value,
                                Func     cmp,
                                Direction dir,
                                bool     is_sorted) noexcept {
            Expected<std::pair<size_t, size_t>> result;
 
            if (!data_ || len_ == 0u) {
                result.setError(OutOfBoundsError(
                    "Array::bracketed_binary_search: array is empty"));
                return result;
            }
 
            if (!is_sorted) {
                _quicksort(0u, len_ - 1u, cmp, dir);
            }
 
            // Check whether value is within the range of the array.
            // For FORWARD: data_[0] <= value <= data_[len_-1]
            // For REVERSE: data_[0] >= value >= data_[len_-1]
            // _apply_dir normalises so that cmp(a,b) < 0 means a < b
            // in the sorted order.
            if (_apply_dir<Func>(cmp(value, data_[0u]), dir) < 0) {
                result.setError(OutOfBoundsError(
                    "Array::bracketed_binary_search: "
                    "value is below the smallest element"));
                return result;
            }
            if (_apply_dir<Func>(cmp(value, data_[len_ - 1u]), dir) > 0) {
                result.setError(OutOfBoundsError(
                    "Array::bracketed_binary_search: "
                    "value is above the largest element"));
                return result;
            }
 
            // Binary search for the insertion point: find the smallest index
            // whose element is >= value (in sorted order).
            size_t lo = 0u;
            size_t hi = len_ - 1u;
 
            while (lo < hi) {
                size_t const mid = lo + (hi - lo) / 2u;
                int    const c   = _apply_dir<Func>(cmp(data_[mid], value), dir);
 
                if (c < 0) {
                    // data_[mid] < value — insertion point is to the right
                    lo = mid + 1u;
                } else {
                    // data_[mid] >= value — insertion point is here or left
                    hi = mid;
                }
            }
            // lo == hi is now the index of the first element >= value
 
            int const exact = _apply_dir<Func>(cmp(data_[lo], value), dir);
 
            if (exact == 0) {
                // Exact match — walk left to find the first occurrence
                size_t first = lo;
                while (first > 0u &&
                       _apply_dir<Func>(cmp(data_[first - 1u], value), dir) == 0) {
                    --first;
                }
                result.setValue(std::make_pair(first, first));
            } else {
                // lo is the upper bound; lo-1 is the lower bound
                // (range check above guarantees lo > 0)
                result.setValue(std::make_pair(lo - 1u, lo));
            }
 
            return result;
        }
// --------------------------------------------------------------------------------

        // ArrayDeleter needs access to private members for cleanup
        template <typename U>
        friend class ArrayDeleter;

        template <typename TT, typename CC>
        friend class Heap;
    };
// ================================================================================
// ================================================================================

    /**
     * @class ArrayDeleter
     * @brief Custom deleter for Array instances
     *
     * @details Implements proper cleanup for Array objects by:
     *          1. Calling the Array destructor (which destroys elements and frees the buffer)
     *          2. Freeing the Array object itself
     *          Both operations use the allocator stored inside the Array.
     *
     * This deleter is intended for use with UniquePtr to provide RAII semantics.
     *
     * @tparam T Element type of the Array being deleted
     *
     * @code{.cpp}
     * cslt::HeapAllocator allocator;
     * auto r = cslt::Array<int>::init(8, allocator);
     *
     * if (r.hasValue()) {
     *     cslt::UniquePtr<cslt::Array<int>, cslt::ArrayDeleter<int>> arr(r.value());
     *     arr->push_back(1);
     * } // ArrayDeleter called here - buffer and struct are both freed
     * @endcode
     */
    template <typename T>
    class ArrayDeleter {
    public:
        /**
         * @brief Delete an Array instance
         *
         * @param a Array to delete (may be nullptr)
         *
         * @details Frees the element buffer and the Array structure using the
         *          allocator stored within the Array. Safe to call with nullptr.
         */
        void operator()(Array<T>* a) const noexcept {
            if (!a) return;

            // Save the allocator pointer before the destructor clears it
            Allocator* allocator = a->allocator_;

            // Destroy elements and free the element buffer
            a->~Array<T>();

            // Free the Array object itself
            if (allocator) {
                allocator->return_element(static_cast<void*>(a),
                                          sizeof(Array<T>),
                                          allocator->default_alignment());
            }
        }
    };
// ================================================================================ 
// ================================================================================ 

    // Forward declaration so HeapDeleter is visible inside Heap
    template <typename T, typename Compare> class HeapDeleter;
     
    // ================================================================================
    // ================================================================================
     
    /**
     * @class Heap
     * @brief Allocator-backed generic binary heap backed by cslt::Array<T>.
     *
     * @details Implements a binary heap whose element storage, capacity management,
     *          and tiered growth strategy are provided entirely by an owned
     *          cslt::Array<T> instance.  The Heap class itself is responsible only
     *          for maintaining the heap property via sift-up and sift-down
     *          operations, and for managing the comparator.
     *
     * The template parameter @p Compare determines heap ordering.  It must be a
     * callable with signature @c bool(const T&, const T&) that returns @c true
     * when its first argument has higher priority than its second:
     *
     * @code{.cpp}
     * // Min-heap — smallest element at root
     * cslt::Heap<int, std::greater<int>> min_heap;
     *
     * // Max-heap — largest element at root
     * cslt::Heap<int, std::less<int>> max_heap;
     *
     * // Lambda comparator — custom priority
     * auto cmp = [](const Task& a, const Task& b){ return a.priority > b.priority; };
     * cslt::Heap<Task, decltype(cmp)> task_heap;
     * @endcode
     *
     * Key features:
     * - All storage, growth, and element lifetime delegated to Array<T>
     * - No duplicated growth logic — changes to Array<T> growth strategy
     *   automatically apply to Heap<T, Compare>
     * - Template parameters T and Compare fix element type and ordering at
     *   compile time — no runtime type tags or function pointer casts
     * - Factory pattern via init() prevents uninitialised instances
     * - RAII cleanup through HeapDeleter and UniquePtr
     * - Returns explicit success/error state using the Expected<T> pattern
     * - foreach() iteration accepts any callable with signature void(const T&)
     *
     * @tparam T        Element type.  Must be default-constructible,
     *                  copy-constructible, and destructible.
     *                  Default-constructibility is required by Expected<T>.
     * @tparam Compare  Comparator type.  Must be default-constructible and
     *                  callable as bool(const T&, const T&).  Returns true when
     *                  the first argument has higher priority than the second.
     *
     * @code{.cpp}
     * cslt::HeapAllocator alloc;
     * auto r = cslt::Heap<int, std::greater<int>>::init(16, true, alloc);
     * if (r.hasValue()) {
     *     cslt::UniquePtr<cslt::Heap<int, std::greater<int>>,
     *                     cslt::HeapDeleter<int, std::greater<int>>> h(r.value());
     *     h->push(5);
     *     h->push(1);
     *     h->push(3);
     *     auto pr = h->pop();   // Expected<int> containing 1  (min-heap)
     * }
     * @endcode
     */
    template <typename T, typename Compare>
    class Heap {
     
    private:
     
        // ========================================================================
        // Private data members
        // ========================================================================
     
        Array<T>*  array_;      ///< Owned backing array — provides storage and growth
        bool       growth_;     ///< If false, push() returns false when array is full 
        Compare    cmp_;        ///< Comparator instance defining heap order
        Allocator* allocator_;  ///< Allocator used for the Heap struct itself
     
        // ========================================================================
        // Private constructor / destructor
        // ========================================================================
       Heap(Array<T>*  array,
                 bool       growth,
                 Compare    cmp,
                 Allocator& allocator) noexcept
                : array_(array)
                , growth_(growth)   // <-- this line is missing
                , cmp_(cmp)
                , allocator_(&allocator)
            {} 
     
        ~Heap() noexcept {
            if (array_ && allocator_) {
                // Destroy and free the backing array through its own deleter
                ArrayDeleter<T> deleter;
                deleter(array_);
                array_ = nullptr;
            }
        }
     
        Heap(const Heap&)            = delete;
        Heap& operator=(const Heap&) = delete;
        Heap(Heap&&)                 = delete;
        Heap& operator=(Heap&&)      = delete;
     
        // ========================================================================
        // Private helpers — direct element access via Array friendship
        // ========================================================================
     
        /**
         * @brief Return a reference to the element at index i.
         *
         * Bypasses Expected<T> overhead for tight sift loops by accessing
         * Array::data_ directly.  Safe because Heap is a friend of Array.
         */
        T& _at(size_t i) noexcept {
            return array_->data_[i];
        }
     
        const T& _at(size_t i) const noexcept {
            return array_->data_[i];
        }
     
        // ========================================================================
        // Private helpers — swap
        // ========================================================================
     
        /**
         * @brief Swap the elements at indices i and j.
         *
         * Uses memcpy for trivially copyable T and move semantics for
         * non-trivial T, consistent with Array's own swap strategy.
         */
        void _swap(size_t i, size_t j) noexcept {
            if (i == j) return;
            if constexpr (std::is_trivially_copyable_v<T>) {
                alignas(T) uint8_t tmp[sizeof(T)];
                std::memcpy(tmp,                   &array_->data_[i], sizeof(T));
                std::memcpy(&array_->data_[i],     &array_->data_[j], sizeof(T));
                std::memcpy(&array_->data_[j],     tmp,               sizeof(T));
            } else {
                T tmp(std::move(array_->data_[i]));
                new (&array_->data_[i]) T(std::move(array_->data_[j]));
                array_->data_[j].~T();
                new (&array_->data_[j]) T(std::move(tmp));
            }
        }
     
        // ========================================================================
        // Private helpers — sift operations
        // ========================================================================
     
        /**
         * @brief Sift the element at index i upward until the heap property is
         *        restored.  Called after push_back() appends to the array.
         *
         * The heap property holds when cmp_(parent, child) is true for every
         * parent-child pair — the root always has higher priority than every
         * descendant under the stored comparator.
         */
        void _sift_up(size_t i) noexcept {
            while (i > 0u) {
                size_t parent = (i - 1u) / 2u;
                // If parent already has priority over child, stop
                if (cmp_(_at(parent), _at(i))) break;
                _swap(parent, i);
                i = parent;
            }
        }
     
        /**
         * @brief Sift the element at index i downward until the heap property is
         *        restored.  Called after the root is replaced during pop().
         */
        void _sift_down(size_t i) noexcept {
            size_t const n = array_->len_;
     
            for (;;) {
                size_t left  = 2u * i + 1u;
                size_t right = 2u * i + 2u;
                size_t best  = i;
     
                if (left  < n && cmp_(_at(left),  _at(best))) best = left;
                if (right < n && cmp_(_at(right), _at(best))) best = right;
     
                if (best == i) break;
     
                _swap(i, best);
                i = best;
            }
        }
     
        // HeapDeleter needs access to private members
        template <typename TT, typename CC>
        friend class HeapDeleter;
     
    public:
     
        // ========================================================================
        // Factory
        // ========================================================================
     
        /**
         * @brief Allocate and initialise a new Heap
         *
         * @param capacity   Initial element capacity passed to Array<T>::init().
         *                   Must be > 0.
         * @param growth     Forwarded to Array<T>::init().  If true, the backing
         *                   array grows automatically when full using Array's
         *                   tiered strategy.  If false, push() returns false when
         *                   the array is at capacity.
         * @param allocator  Allocator for the Heap struct and, via Array, the
         *                   backing element buffer.
         * @param cmp        Comparator instance (default-constructed if omitted).
         *                   Returns true when its first argument has higher
         *                   priority than its second.
         * @return Expected<Heap<T,Compare>*> on success, or an ArgumentError /
         *         MemoryError on failure
         *
         * @par Error conditions
         * - capacity == 0 (ArgumentError, propagated from Array::init)
         * - Allocation of the backing Array fails (MemoryError)
         * - Allocation of the Heap struct fails (MemoryError)
         *
         * @code{.cpp}
         * // Min-heap of integers (smallest value at root)
         * cslt::HeapAllocator alloc;
         * auto r = cslt::Heap<int, std::greater<int>>::init(16, true, alloc);
         * if (!r.hasValue()) return;
         * cslt::UniquePtr<cslt::Heap<int, std::greater<int>>,
         *                 cslt::HeapDeleter<int, std::greater<int>>> h(r.value());
         *
         * h->push(5);
         * h->push(1);
         * h->push(3);
         * // h->peek().value() == 1  (smallest element is at the root)
         * @endcode
         */
        static Expected<Heap<T, Compare>*> init(size_t     capacity,
                                                 bool       growth,
                                                 Allocator& allocator,
                                                 Compare    cmp = Compare{}) noexcept {
            Expected<Heap<T, Compare>*> result;

            // Delegate array creation — this validates capacity > 0 and handles
            // all buffer allocation and error reporting
            auto arr_r = Array<T>::init(capacity, allocator);
            if (!arr_r.hasValue()) {
                result.setError(arr_r.error());
                return result;
            }

            // Allocate the Heap struct itself
            auto obj_r = allocator.alloc(sizeof(Heap<T, Compare>), true);
            if (!obj_r.hasValue()) {
                ArrayDeleter<T>{}(arr_r.value());
                result.setError(MemoryError(
                    "Heap::init: failed to allocate Heap struct"));
                return result;
            }

            Heap<T, Compare>* h = new (obj_r.value()) Heap<T, Compare>(
                arr_r.value(), growth, cmp, allocator);

            result.setValue(h);
            return result;
        } 
    // --------------------------------------------------------------------------------
     
        /**
         * @brief Create a deep copy of @p src using a caller-supplied allocator
         *
         * @param src       Heap to copy from
         * @param allocator Allocator for the new Heap and its backing array
         * @return Expected<Heap<T,Compare>*> on success, or an error
         *
         * @code{.cpp}
         * cslt::HeapAllocator alloc;
         * auto r = cslt::Heap<int, std::greater<int>>::init(8, true, alloc);
         * if (!r.hasValue()) return;
         * cslt::UniquePtr<cslt::Heap<int, std::greater<int>>,
         *                 cslt::HeapDeleter<int, std::greater<int>>> src(r.value());
         * src->push(3);
         * src->push(1);
         * src->push(2);
         *
         * // Deep copy — src and dst are fully independent
         * auto cr = cslt::Heap<int, std::greater<int>>::copy(*src, alloc);
         * if (!cr.hasValue()) return;
         * cslt::UniquePtr<cslt::Heap<int, std::greater<int>>,
         *                 cslt::HeapDeleter<int, std::greater<int>>> dst(cr.value());
         * @endcode
         */
        static Expected<Heap<T, Compare>*> copy(const Heap<T, Compare>& src,
                                                  Allocator&              allocator) noexcept {
            Expected<Heap<T, Compare>*> result;
     
            // Deep-copy the backing array — Array::copy handles buffer and elements
            auto arr_r = Array<T>::copy(*src.array_, allocator);
            if (!arr_r.hasValue()) {
                result.setError(arr_r.error());
                return result;
            }
     
            // Allocate the Heap struct
            auto obj_r = allocator.alloc(sizeof(Heap<T, Compare>), true);
            if (!obj_r.hasValue()) {
                ArrayDeleter<T>{}(arr_r.value());
                result.setError(MemoryError(
                    "Heap::copy: failed to allocate Heap struct"));
                return result;
            }
     
            Heap<T, Compare>* h = new (obj_r.value()) Heap<T, Compare>(
                arr_r.value(), src.growth_, src.cmp_, allocator); 
            result.setValue(h);
            return result;
        }
    // --------------------------------------------------------------------------------
     
        /**
         * @brief Create a deep copy using the source heap's own allocator
         *
         * @param src  Heap to copy from
         * @return Expected<Heap<T,Compare>*> on success, or an error
         *
         * @code{.cpp}
         * auto cr = cslt::Heap<int, std::greater<int>>::copy(*src);
         * @endcode
         */
        static Expected<Heap<T, Compare>*> copy(const Heap<T, Compare>& src) noexcept {
            return copy(src, *src.allocator_);
        }
     
        // ========================================================================
        // Mutation operations
        // ========================================================================
     
        /**
         * @brief Insert an element into the heap and restore the heap property
         *
         * Delegates the actual insertion and any required growth to
         * Array::push_back(), then sifts the new element upward.
         *
         * @param value  Element to copy into the heap
         * @return true on success; false if the backing array is full and growth
         *         is disabled, or if allocation fails
         *
         * @code{.cpp}
         * cslt::HeapAllocator alloc;
         * auto r = cslt::Heap<int, std::greater<int>>::init(8, true, alloc);
         * if (!r.hasValue()) return;
         * cslt::UniquePtr<cslt::Heap<int, std::greater<int>>,
         *                 cslt::HeapDeleter<int, std::greater<int>>> h(r.value());
         *
         * h->push(5);
         * h->push(1);   // root becomes 1 (min-heap)
         * h->push(3);
         * // h->peek().value() == 1
         * @endcode
         */
        bool push(const T& value) noexcept {
            if (!array_) return false;

            // Enforce fixed-capacity mode before delegating to Array
            if (!growth_ && array_->len_ == array_->cap_) return false;

            if (!array_->push_back(value)) return false;

            _sift_up(array_->len_ - 1u);
            return true;
        }
    // --------------------------------------------------------------------------------
     
        /**
         * @brief Remove the root (highest-priority) element and return its value
         *
         * Copies the root value, moves the last element into the root slot via
         * Array's internal buffer, decrements the array length, then sifts
         * downward to restore the heap property.
         *
         * @return Expected<T> containing the root value on success, or an
         *         EmptyError if the heap contains no elements
         *
         * @code{.cpp}
         * cslt::HeapAllocator alloc;
         * auto r = cslt::Heap<int, std::greater<int>>::init(8, true, alloc);
         * if (!r.hasValue()) return;
         * cslt::UniquePtr<cslt::Heap<int, std::greater<int>>,
         *                 cslt::HeapDeleter<int, std::greater<int>>> h(r.value());
         *
         * h->push(5);
         * h->push(1);
         * h->push(3);
         *
         * auto pr = h->pop();
         * if (pr.hasValue()) {
         *     int v = pr.value();   // v == 1 (min-heap root)
         * }
         * @endcode
         */
        Expected<T> pop() noexcept {
            Expected<T> result;
     
            if (!array_ || array_->len_ == 0u) {
                result.setError(EmptyError("Heap::pop: heap is empty"));
                return result;
            }
     
            // Copy the root value out before we modify the buffer
            T val(array_->data_[0]);
     
            size_t const last = array_->len_ - 1u;
     
            if (last == 0u) {
                // Only one element — just pop it from the array
                array_->pop_back();
                result.setValue(std::move(val));
                return result;
            }
     
            // Move the last element into the root slot in-place
            array_->data_[0].~T();
            new (&array_->data_[0]) T(std::move(array_->data_[last]));
     
            // Decrement the array length without calling the destructor again
            // (we already moved out of data_[last])
            array_->data_[last].~T();
            --array_->len_;
     
            // Restore heap property from the root downward
            _sift_down(0u);
     
            result.setValue(std::move(val));
            return result;
        }
     
        // ========================================================================
        // Inspection operations
        // ========================================================================
     
        /**
         * @brief Return a copy of the root (highest-priority) element
         *
         * @return Expected<T> containing a copy of the root value, or an
         *         EmptyError if the heap is empty
         *
         * @code{.cpp}
         * cslt::HeapAllocator alloc;
         * auto r = cslt::Heap<int, std::greater<int>>::init(8, true, alloc);
         * if (!r.hasValue()) return;
         * cslt::UniquePtr<cslt::Heap<int, std::greater<int>>,
         *                 cslt::HeapDeleter<int, std::greater<int>>> h(r.value());
         *
         * h->push(5);
         * h->push(1);
         * h->push(3);
         *
         * auto pr = h->peek();
         * if (pr.hasValue()) {
         *     int root = pr.value();   // root == 1 (min-heap)
         *     // h->size() still == 3 — peek does not remove the element
         * }
         * @endcode
         */
        Expected<T> peek() const noexcept {
            Expected<T> result;
            if (!array_ || array_->len_ == 0u) {
                result.setError(EmptyError("Heap::peek: heap is empty"));
                return result;
            }
            result.setValue(array_->data_[0]);
            return result;
        }
     
        // ========================================================================
        // Iteration
        // ========================================================================
     
        /**
         * @brief Call @p fn once for every element in heap-internal order
         *
         * @details Traversal is in backing-array order, which satisfies the heap
         *          property but is NOT sorted order.  Do not rely on elements
         *          being visited highest-priority first.  For ordered traversal,
         *          copy the heap and pop from the copy in a loop.
         *
         * @tparam Func  Callable with signature `void(const T&)`.
         *               Lambdas, functors, and function pointers are accepted.
         *               The callback must not call push() or pop() during
         *               traversal.
         *
         * @param fn  Callback invoked for each element
         * @return true on success; false if the heap is uninitialised or empty
         *
         * @code{.cpp}
         * cslt::HeapAllocator alloc;
         * auto r = cslt::Heap<int, std::greater<int>>::init(8, true, alloc);
         * if (!r.hasValue()) return;
         * cslt::UniquePtr<cslt::Heap<int, std::greater<int>>,
         *                 cslt::HeapDeleter<int, std::greater<int>>> h(r.value());
         *
         * h->push(5);
         * h->push(1);
         * h->push(3);
         *
         * int total = 0;
         * h->foreach([&total](const int& v) { total += v; });
         * // total == 9  (visit order is unspecified beyond heap property)
         * @endcode
         */
        template <typename Func>
        bool foreach(Func fn) const noexcept {
            if (!array_ || array_->len_ == 0u) return false;
            for (size_t i = 0u; i < array_->len_; ++i)
                fn(array_->data_[i]);
            return true;
        }
     
        // ========================================================================
        // Introspection
        // ========================================================================
     
        /**
         * @brief Number of elements currently stored in the heap
         *
         * Delegates directly to the backing array.
         */
        size_t size() const noexcept {
            return array_ ? array_->len_ : 0u;
        }
     
        /**
         * @brief Allocated capacity in elements
         *
         * Delegates directly to the backing array.
         */
        size_t capacity() const noexcept {
            return array_ ? array_->cap_ : 0u;
        }
     
        /**
         * @brief true if no elements are stored
         */
        bool is_empty() const noexcept {
            return !array_ || array_->len_ == 0u;
        }
     
        /**
         * @brief true if the heap is at full capacity and growth is disabled
         */
        bool is_full() const noexcept {
            return array_ && !growth_ && array_->len_ == array_->cap_;
        }
    };
    // ================================================================================
    // ================================================================================
     
    /**
     * @class HeapDeleter
     * @brief Custom deleter for Heap instances, for use with UniquePtr
     *
     * @details Calls the Heap destructor (which in turn calls ArrayDeleter on the
     *          backing array, destroying all elements and freeing the buffer)
     *          then returns the Heap struct memory to the allocator.
     *
     * @tparam T        Element type of the Heap being deleted
     * @tparam Compare  Comparator type of the Heap being deleted
     *
     * @code{.cpp}
     * cslt::UniquePtr<cslt::Heap<int, std::greater<int>>,
     *                 cslt::HeapDeleter<int, std::greater<int>>> h(r.value());
     * @endcode
     */
    template <typename T, typename Compare>
    class HeapDeleter {
    public:
        void operator()(Heap<T, Compare>* h) const noexcept {
            if (!h) return;
            Allocator* allocator = h->allocator_;
            h->~Heap<T, Compare>();
            if (allocator) {
                allocator->return_element(
                    static_cast<void*>(h),
                    sizeof(Heap<T, Compare>),
                    allocator->default_alignment());
            }
        }
    };
// ================================================================================
// ================================================================================

} // namespace cslt
// ================================================================================ 
// ================================================================================ 
#endif /* file_name_HPP */
// ================================================================================
// ================================================================================
// eof
