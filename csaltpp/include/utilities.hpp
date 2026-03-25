// ================================================================================
// ================================================================================
// - File:    utilities.hpp
// - Purpose: This file contains template utility classes and functions
//
// Source Metadata
// - Author:  Jonathan A. Webb
// - Date:    January 04, 2026
// - Version: 1.0
// - Copyright: Copyright 2022, Jon Webb Inc.
// ================================================================================
// ================================================================================

#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <cstddef>
#include <atomic>

// ================================================================================
// ================================================================================

namespace cslt {

    using nullptr_t = decltype(nullptr);
// ================================================================================
// ================================================================================ 

    template <bool B>
    struct BoolConstant { static constexpr bool value = B; };

    using TrueType  = BoolConstant<true>;
    using FalseType = BoolConstant<false>;

// ================================================================================
// ================================================================================

    template <class T>
    struct RemoveRef { using type = T; };
// --------------------------------------------------------------------------------

    template <class T>
    struct RemoveRef<T&> { using type = T; };
// --------------------------------------------------------------------------------

    template <class T>
    struct RemoveRef<T&&> { using type = T; };
// --------------------------------------------------------------------------------

    template <class T>
    using RemoveRefT = typename RemoveRef<T>::type;

// ================================================================================
// ================================================================================

    template <class T>
    struct IsLValueRef { static constexpr bool value = false; };
// --------------------------------------------------------------------------------

    template <class T>
    struct IsLValueRef<T&> { static constexpr bool value = true; };

// ================================================================================
// ================================================================================

    template <bool Cond, class T = void>
    struct EnableIf { };
// --------------------------------------------------------------------------------

    template <class T>
    struct EnableIf<true, T> { using type = T; };
// --------------------------------------------------------------------------------

    template <bool Cond, class T = void>
    using EnableIfT = typename EnableIf<Cond, T>::type;

// ================================================================================
// ================================================================================

    template <class A, class B>
    struct IsSame { static constexpr bool value = false; };
// --------------------------------------------------------------------------------

    template <class A>
    struct IsSame<A, A> { static constexpr bool value = true; };

// ================================================================================
// ================================================================================

    template <class T>
    struct IsArray { static constexpr bool value = false; };
// --------------------------------------------------------------------------------

    template <class T>
    struct IsArray<T[]> { static constexpr bool value = true; };
// --------------------------------------------------------------------------------

    template <class T, ::size_t N>
    struct IsArray<T[N]> { static constexpr bool value = true; };

// ================================================================================
// ================================================================================

    template <class T>
    constexpr RemoveRefT<T>&& move(T&& x) noexcept {
        return static_cast<RemoveRefT<T>&&>(x);
    }
// --------------------------------------------------------------------------------

    template <class T>
    constexpr T&& forward(RemoveRefT<T>& x) noexcept {
        return static_cast<T&&>(x);
    }
// --------------------------------------------------------------------------------

    template <class T>
    constexpr T&& forward(RemoveRefT<T>&& x) noexcept {
        static_assert(!IsLValueRef<T>::value,
                      "cslt::forward<T>(T&&): T is an lvalue reference");
        return static_cast<T&&>(x);
    }

// ================================================================================
// ================================================================================

    template <class T>
    constexpr void swap(T& a, T& b) noexcept {
        T tmp = cslt::move(a);
        a = cslt::move(b);
        b = cslt::move(tmp);
    }

// ================================================================================
// ================================================================================

    template <class T>
    T&& declval() noexcept; // declaration only (no definition needed)

// ================================================================================

    namespace detail {

        template <class To>
        static void _accept(To); // declaration only
// --------------------------------------------------------------------------------

        template <class From, class To>
        struct IsConvertibleImpl {
        private:
            template <class F, class = decltype(_accept<To>(declval<F>()))>
            static char test(int);

            template <class>
            static long test(...);

        public:
            static constexpr bool value = (sizeof(test<From>(0)) == sizeof(char));
        };

// ================================================================================
// IsConstructibleImpl
// ================================================================================

        template <class T, class... Args>
        struct IsConstructibleImpl {
        private:
            template <class U, class = decltype(U(declval<Args>()...))>
            static char test(int);

            template <class, class...>
            static long test(...);

        public:
            static constexpr bool value = (sizeof(test<T>(0)) == sizeof(char));
        };

// ================================================================================
// IsAssignableImpl
// ================================================================================

        template <class T, class U>
        struct IsAssignableImpl {
        private:
            template <class X, class Y, class = decltype(declval<X>() = declval<Y>())>
            static char test(int);

            template <class, class>
            static long test(...);

        public:
            static constexpr bool value = (sizeof(test<T, U>(0)) == sizeof(char));
        };

        struct ControlBlock {
            std::atomic<size_t> strong;
            std::atomic<size_t> weak;

            ControlBlock() : strong(1u), weak(1u) {}
            virtual ~ControlBlock() {}

            virtual void destroy_object() noexcept = 0;
            virtual void* get_ptr() const noexcept = 0;
        };
        // struct ControlBlock {
        //     std::atomic<size_t> strong;
        //
        //     ControlBlock() : strong(1u) {}
        //     virtual ~ControlBlock() {}
        //
        //     virtual void destroy_object() noexcept = 0;
        //     virtual void* get_ptr() const noexcept = 0;
        // };

        // Control block for pointer + deleter stored by value
        template <class T, class Deleter>
        struct ControlBlockPtr final : ControlBlock {
            T* ptr;
            Deleter del;

            ControlBlockPtr(T* p, const Deleter& d) : ControlBlock(), ptr(p), del(d) {}
            ControlBlockPtr(T* p, Deleter&& d)      : ControlBlock(), ptr(p), del(cslt::move(d)) {}

            void destroy_object() noexcept override {
                if (ptr) {
                    del(ptr);
                    ptr = nullptr;
                }
            }

            void* get_ptr() const noexcept override { return (void*)ptr; }
        };

        // Strong reference operations
        inline void incref_strong(ControlBlock* cb) noexcept {
            if (cb) {
                cb->strong.fetch_add(1u, std::memory_order_relaxed);
            }
        }

        inline bool decref_strong(ControlBlock* cb) noexcept {
            if (!cb) return false;
            if (cb->strong.fetch_sub(1u, std::memory_order_acq_rel) == 1u) {
                std::atomic_thread_fence(std::memory_order_acquire);
                return true;
            }
            return false;
        }

        // Weak reference operations
        inline void incref_weak(ControlBlock* cb) noexcept {
            if (cb) {
                cb->weak.fetch_add(1u, std::memory_order_relaxed);
            }
        }

        inline bool decref_weak(ControlBlock* cb) noexcept {
            if (!cb) return false;
            if (cb->weak.fetch_sub(1u, std::memory_order_acq_rel) == 1u) {
                std::atomic_thread_fence(std::memory_order_acquire);
                return true;
            }
            return false;
        }

        inline size_t get_strong_count(const ControlBlock* cb) noexcept {
            if (!cb) return 0u;
            return cb->strong.load(std::memory_order_relaxed);
        }

        inline size_t get_weak_count(const ControlBlock* cb) noexcept {
            if (!cb) return 0u;
            return cb->weak.load(std::memory_order_relaxed);
        }

        // Try to increment strong count (for weak->shared conversion)
        // Returns true if successful (strong count was > 0)
        inline bool try_incref_strong(ControlBlock* cb) noexcept {
            if (!cb) return false;
            
            size_t count = cb->strong.load(std::memory_order_relaxed);
            while (count != 0u) {
                // Try to increment if still non-zero
                if (cb->strong.compare_exchange_weak(count, count + 1u,
                                                     std::memory_order_relaxed,
                                                     std::memory_order_relaxed)) {
                    return true;
                }
                // count was updated by compare_exchange_weak on failure, retry
            }
            return false;
        }
        // Thread-safe refcount operations using atomics
        // inline void incref(ControlBlock* cb) noexcept {
        //     if (cb) {
        //         cb->strong.fetch_add(1u, std::memory_order_relaxed);
        //     }
        // }
        //
        // inline bool decref(ControlBlock* cb) noexcept {
        //     if (!cb) return false;
        //     // fetch_sub returns the value BEFORE subtraction
        //     // Use acquire-release ordering for proper synchronization
        //     if (cb->strong.fetch_sub(1u, std::memory_order_acq_rel) == 1u) {
        //         // We were the last reference
        //         // Add an acquire fence to ensure all writes from other threads are visible
        //         std::atomic_thread_fence(std::memory_order_acquire);
        //         return true;
        //     }
        //     return false;
        // }
        //
        // inline size_t get_refcount(const ControlBlock* cb) noexcept {
        //     if (!cb) return 0u;
        //     return cb->strong.load(std::memory_order_relaxed);
        // }
        // struct ControlBlock {
        //     size_t strong;
        //
        //     ControlBlock() : strong(1u) {}
        //     virtual ~ControlBlock() {}
        //
        //     virtual void destroy_object() noexcept = 0;
        //     virtual void* get_ptr() const noexcept = 0;
        // };
        //
        //  // Control block for pointer + deleter stored by value
        // template <class T, class Deleter>
        // struct ControlBlockPtr final : ControlBlock {
        //     T* ptr;
        //     Deleter del;
        //
        //     ControlBlockPtr(T* p, const Deleter& d) : ControlBlock(), ptr(p), del(d) {}
        //     ControlBlockPtr(T* p, Deleter&& d)      : ControlBlock(), ptr(p), del(cslt::move(d)) {}
        //
        //     void destroy_object() noexcept override {
        //         if (ptr) {
        //             del(ptr);
        //             ptr = nullptr;
        //         }
        //     }
        //
        //     void* get_ptr() const noexcept override { return (void*)ptr; }
        // };
        //
        // // Simple refcount ops (non-atomic for now)
        // inline void incref(ControlBlock* cb) noexcept {
        //     if (cb) { ++cb->strong; }
        // }
        //
        // inline bool decref(ControlBlock* cb) noexcept {
        //     if (!cb) return false;
        //     // returns true if reached zero
        //     return (--cb->strong == 0u);
        // }
    } // namespace detail

// ================================================================================
// ================================================================================

    template <class From, class To>
    struct IsConvertible {
        static constexpr bool value = detail::IsConvertibleImpl<From, To>::value;
    };

// ================================================================================
// IsConstructible (public wrapper)
// ================================================================================

    template <class T, class... Args>
    struct IsConstructible {
        static constexpr bool value = detail::IsConstructibleImpl<T, Args...>::value;
    };
// --------------------------------------------------------------------------------

    template <class T>
    struct IsDefaultConstructible {
        static constexpr bool value = IsConstructible<T>::value;
    };
// --------------------------------------------------------------------------------

    template <class T>
    struct IsMoveConstructible {
        static constexpr bool value = IsConstructible<T, T&&>::value;
    };
// --------------------------------------------------------------------------------

    template <class T>
    struct IsCopyConstructible {
        static constexpr bool value = IsConstructible<T, const T&>::value;
    };

// ================================================================================
// IsAssignable (public wrapper)
// ================================================================================

    template <class T, class U>
    struct IsAssignable {
        static constexpr bool value = detail::IsAssignableImpl<T, U>::value;
    };
// --------------------------------------------------------------------------------

    template <class T>
    struct IsMoveAssignable {
        static constexpr bool value = IsAssignable<T&, T&&>::value;
    };
// --------------------------------------------------------------------------------

    template <class T>
    struct IsCopyAssignable {
        static constexpr bool value = IsAssignable<T&, const T&>::value;
    };

// ================================================================================
// ================================================================================
} // namespace cslt
#endif // UTILITIES_HPP
// ================================================================================
// ================================================================================
// eof

