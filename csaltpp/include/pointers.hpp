// ================================================================================
// ================================================================================
// - File:    pointers.hpp
// - Purpose: Standalone smart pointers (cslt::UniquePtr) with custom deleters
//
// Source Metadata
// - Author:  Jonathan A. Webb
// - Date:    January 04, 2026
// - Version: 1.0
// - Copyright: Copyright 2022, Jon Webb Inc.
// ================================================================================
// ================================================================================

#ifndef POINTERS_HPP
#define POINTERS_HPP

#include "utilities.hpp"  // cslt::move/forward/swap + traits
#include <cstddef>        // ::size_t

// ================================================================================
// ================================================================================

namespace cslt {

// ================================================================================
// Small meta helpers (standalone detection idiom)
// ================================================================================

    template <class...>
    struct VoidTImpl { using type = void; };

    template <class... Ts>
    using VoidT = typename VoidTImpl<Ts...>::type;

// --------------------------------------------------------------------------------
// HasPointerType<D>::value == true if RemoveRefT<D> has nested ::pointer
// --------------------------------------------------------------------------------

    template <class D, class = void>
    struct HasPointerType : FalseType { };

    template <class D>
    struct HasPointerType<D, VoidT<typename RemoveRefT<D>::pointer>> : TrueType { };

// --------------------------------------------------------------------------------
// PointerType<T, D>::type:
// - RemoveRefT<D>::pointer if present
// - otherwise T*
// --------------------------------------------------------------------------------

    template <class T, class D, bool = HasPointerType<D>::value>
    struct PointerType { using type = T*; };

    template <class T, class D>
    struct PointerType<T, D, true> { using type = typename RemoveRefT<D>::pointer; };

    template <class T, class D>
    using PointerTypeT = typename PointerType<T, D>::type;

// ================================================================================
// DefaultDelete
// ================================================================================

// ================================================================================
// DefaultDelete
// ================================================================================

    template <class T>
    struct DefaultDelete {
        constexpr DefaultDelete() noexcept = default;

        // Converting ctor: DefaultDelete<U> -> DefaultDelete<T>
        // Enabled when U* is convertible to T* (e.g., Derived* -> Base*)
        template <class U,
                  class = EnableIfT<IsConvertible<U*, T*>::value>>
        constexpr DefaultDelete(const DefaultDelete<U>&) noexcept {}

        void operator()(T* ptr) const noexcept {
            static_assert(sizeof(T) > 0,
                          "cslt::DefaultDelete<T>: T must be a complete type");
            delete ptr;
        }
    };

    template <class T>
    struct DefaultDelete<T[]> {
        constexpr DefaultDelete() noexcept = default;

        // Converting ctor: DefaultDelete<U[]> -> DefaultDelete<T[]>
        // This mirrors std::default_delete for arrays, loosely:
        // allow when U(*)[] convertible to T(*)[] (covers derived->base in arrays too).
        template <class U,
                  class = EnableIfT<IsConvertible<U(*)[], T(*)[]>::value>>
        constexpr DefaultDelete(const DefaultDelete<U[]>&) noexcept {}

        template <class U>
        void operator()(U* ptr) const noexcept {
            static_assert(sizeof(U) > 0,
                          "cslt::DefaultDelete<T[]>: element type must be a complete type");
            delete[] ptr;
        }
    };

// ================================================================================
// Deleter storage (value deleter vs reference deleter)
// - If Deleter is D, store D by value.
// - If Deleter is D&, store a pointer to D.
// Mirrors std::unique_ptr support for reference deleters.
// ================================================================================

    template <class Deleter, bool IsRef = IsLValueRef<Deleter>::value>
    class DeleterStorage;

// --------------------------------------------------------------------------------
// Value deleter
// --------------------------------------------------------------------------------

    template <class Deleter>
    class DeleterStorage<Deleter, false> {
    public:
        using stored_type = Deleter;

        constexpr DeleterStorage() : del_() {}
        constexpr DeleterStorage(const stored_type& d) : del_(d) {}
        constexpr DeleterStorage(stored_type&& d) : del_(cslt::move(d)) {}

        stored_type& get() noexcept { return del_; }
        const stored_type& get() const noexcept { return del_; }

    private:
        stored_type del_;
    };

// --------------------------------------------------------------------------------
// Reference deleter (D&): store pointer to D
// --------------------------------------------------------------------------------

    template <class Deleter>
    class DeleterStorage<Deleter, true> {
    public:
        using raw_type    = RemoveRefT<Deleter>;
        using stored_type = raw_type*;

        DeleterStorage() = delete;  // cannot default-construct reference deleter
        explicit constexpr DeleterStorage(raw_type& d) : del_(&d) {}

        raw_type& get() noexcept { return *del_; }
        const raw_type& get() const noexcept { return *del_; }

    private:
        stored_type del_;
    };

// ================================================================================
// UniquePtr forward declaration (for comparisons)
// ================================================================================

    template <class T, class Deleter = DefaultDelete<T>>
    class UniquePtr;

// ================================================================================
// UniquePtr (array specialization forward decl)
// ================================================================================

    template <class T, class Deleter>
    class UniquePtr<T[], Deleter>;

// ================================================================================
// UniquePtr (single object)
// ================================================================================

    template <class T, class Deleter>
    class UniquePtr {
    public:
        using element_type = T;
        using deleter_type = Deleter;
        using pointer      = PointerTypeT<T, Deleter>;

    private:
        using storage_type = DeleterStorage<deleter_type>;

    public:
        // ----------------------------------------------------------------------------
        // Constructors (STL-like enablement)
        // ----------------------------------------------------------------------------

        template <class D = deleter_type,
                  class = EnableIfT<!IsLValueRef<D>::value && IsDefaultConstructible<D>::value>>
        constexpr UniquePtr() noexcept
            : ptr_(pointer()), del_() {}

        template <class D = deleter_type,
                  class = EnableIfT<!IsLValueRef<D>::value && IsDefaultConstructible<D>::value>>
        constexpr UniquePtr(nullptr_t) noexcept
            : ptr_(pointer()), del_() {}

        template <class D = deleter_type,
                  class = EnableIfT<!IsLValueRef<D>::value && IsDefaultConstructible<D>::value>>
        explicit UniquePtr(pointer p) noexcept
            : ptr_(p), del_() {}

        template <class D = deleter_type,
                  class = EnableIfT<
                      (!IsLValueRef<D>::value && IsConstructible<D, const D&>::value) ||
                      ( IsLValueRef<D>::value )>>
        UniquePtr(pointer p, const RemoveRefT<deleter_type>& d) noexcept
            : ptr_(p), del_(make_deleter_from_lvalue_(d)) {}

        template <class D = deleter_type,
                  class = EnableIfT<!IsLValueRef<D>::value && IsConstructible<D, D&&>::value>>
        UniquePtr(pointer p, RemoveRefT<deleter_type>&& d) noexcept
            : ptr_(p), del_(cslt::move(d)) {}

        UniquePtr(const UniquePtr&)            = delete;
        UniquePtr& operator=(const UniquePtr&) = delete;

        template <class D = deleter_type,
                  class = EnableIfT<
                      (!IsLValueRef<D>::value && IsMoveConstructible<D>::value) ||
                      ( IsLValueRef<D>::value )>>
        UniquePtr(UniquePtr&& other) noexcept
            : ptr_(other.release()), del_(move_deleter_from_(other)) {}

        template <class D = deleter_type,
                  class = EnableIfT<
                      (!IsLValueRef<D>::value && IsMoveAssignable<D>::value) ||
                      ( IsLValueRef<D>::value )>>
        UniquePtr& operator=(UniquePtr&& other) noexcept {
            if (this != &other) {
                reset(other.release());
                assign_deleter_from_(other);
            }
            return *this;
        }

        UniquePtr& operator=(nullptr_t) noexcept {
            reset();
            return *this;
        }

        // ----------------------------------------------------------------------------
        // Converting move ctor / assignment (single overload; dispatch internally)
        // ----------------------------------------------------------------------------

        template <class U, class D2,
                  class = EnableIfT<
                      IsConvertible<typename UniquePtr<U, D2>::pointer, pointer>::value &&
                      !IsArray<U>::value && !IsArray<T>::value &&
                      (
                          (!IsLValueRef<deleter_type>::value &&
                           IsConstructible<deleter_type, D2&&>::value)
                          ||
                          (IsLValueRef<deleter_type>::value &&
                           IsSame<RemoveRefT<deleter_type>, RemoveRefT<D2>>::value)
                      )>>
        UniquePtr(UniquePtr<U, D2>&& other) noexcept
            : ptr_(other.release()),
              del_(make_converting_deleter_(other)) {}

        template <class U, class D2,
                  class = EnableIfT<
                      IsConvertible<typename UniquePtr<U, D2>::pointer, pointer>::value &&
                      !IsArray<U>::value && !IsArray<T>::value &&
                      (
                          (!IsLValueRef<deleter_type>::value &&
                           IsAssignable<deleter_type&, D2&&>::value)
                          ||
                          (IsLValueRef<deleter_type>::value &&
                           IsSame<RemoveRefT<deleter_type>, RemoveRefT<D2>>::value)
                      )>>
        UniquePtr& operator=(UniquePtr<U, D2>&& other) noexcept {
            reset(other.release());
            assign_converting_deleter_(other);
            return *this;
        }

        // ----------------------------------------------------------------------------
        // Destructor
        // ----------------------------------------------------------------------------
        ~UniquePtr() noexcept { reset(); }

        // ----------------------------------------------------------------------------
        // Observers
        // ----------------------------------------------------------------------------
        pointer get() const noexcept { return ptr_; }

        RemoveRefT<deleter_type>& get_deleter() noexcept { return del_.get(); }
        const RemoveRefT<deleter_type>& get_deleter() const noexcept { return del_.get(); }

        explicit operator bool() const noexcept { return !(ptr_ == pointer()); }

        element_type& operator*() const noexcept { return *to_raw_(ptr_); }
        element_type* operator->() const noexcept { return to_raw_(ptr_); }

        // ----------------------------------------------------------------------------
        // Modifiers
        // ----------------------------------------------------------------------------
        pointer release() noexcept {
            pointer tmp = ptr_;
            ptr_ = pointer();
            return tmp;
        }

        void reset(pointer p = pointer()) noexcept {
            if (!(ptr_ == p)) {
                pointer old = ptr_;
                ptr_ = p;
                if (!(old == pointer())) {
                    invoke_deleter_(old);
                }
            }
        }

        void reset(nullptr_t) noexcept { reset(pointer()); }

        void swap(UniquePtr& other) noexcept {
            cslt::swap(ptr_, other.ptr_);
            cslt::swap(del_, other.del_);
        }

    private:
        pointer      ptr_;
        storage_type del_;

        // bind deleter from lvalue (value stores copy; reference binds)
        static storage_type make_deleter_from_lvalue_(const RemoveRefT<deleter_type>& d) noexcept {
            return make_deleter_from_lvalue_dispatch_(d, BoolConstant<IsLValueRef<deleter_type>::value>{});
        }

        static storage_type make_deleter_from_lvalue_dispatch_(const RemoveRefT<deleter_type>& d, FalseType) noexcept {
            return storage_type(d);
        }

        static storage_type make_deleter_from_lvalue_dispatch_(const RemoveRefT<deleter_type>& d, TrueType) noexcept {
            return storage_type(const_cast<RemoveRefT<deleter_type>&>(d));
        }

        // move deleter from same-type UniquePtr
        static storage_type move_deleter_from_(UniquePtr& other) noexcept {
            return move_deleter_from_dispatch_(other, BoolConstant<IsLValueRef<deleter_type>::value>{});
        }

        static storage_type move_deleter_from_dispatch_(UniquePtr& other, FalseType) noexcept {
            return storage_type(cslt::move(other.get_deleter()));
        }

        static storage_type move_deleter_from_dispatch_(UniquePtr& other, TrueType) noexcept {
            return storage_type(other.get_deleter());
        }

        // assign deleter from same-type UniquePtr
        void assign_deleter_from_(UniquePtr& other) noexcept {
            assign_deleter_from_dispatch_(other, BoolConstant<IsLValueRef<deleter_type>::value>{});
        }

        void assign_deleter_from_dispatch_(UniquePtr& other, FalseType) noexcept {
            del_.get() = cslt::move(other.get_deleter());
        }

        void assign_deleter_from_dispatch_(UniquePtr&, TrueType) noexcept {
            // reference deleter stays bound
        }

        // converting deleter for converting move-ctor
        template <class U, class D2>
        static storage_type make_converting_deleter_(UniquePtr<U, D2>& other) noexcept {
            return make_converting_deleter_dispatch_(other, BoolConstant<IsLValueRef<deleter_type>::value>{});
        }

        template <class U, class D2>
        static storage_type make_converting_deleter_dispatch_(UniquePtr<U, D2>& other, FalseType) noexcept {
            return storage_type(cslt::move(other.get_deleter()));
        }

        template <class U, class D2>
        static storage_type make_converting_deleter_dispatch_(UniquePtr<U, D2>& other, TrueType) noexcept {
            return storage_type(other.get_deleter());
        }

        // converting deleter assignment for converting move-assignment
        template <class U, class D2>
        void assign_converting_deleter_(UniquePtr<U, D2>& other) noexcept {
            assign_converting_deleter_dispatch_(other, BoolConstant<IsLValueRef<deleter_type>::value>{});
        }

        template <class U, class D2>
        void assign_converting_deleter_dispatch_(UniquePtr<U, D2>& other, FalseType) noexcept {
            del_.get() = cslt::move(other.get_deleter());
        }

        template <class U, class D2>
        void assign_converting_deleter_dispatch_(UniquePtr<U, D2>&, TrueType) noexcept {
            // reference deleter remains bound
        }

        // Raw pointer access (fancy pointers could override later)
        static element_type* to_raw_(pointer p) noexcept { return p; }

        void invoke_deleter_(pointer p) noexcept {
            del_.get()(to_raw_(p));
        }
    };

// ================================================================================
// UniquePtr (array specialization)
// ================================================================================

    template <class T, class Deleter>
    class UniquePtr<T[], Deleter> {
    public:
        using element_type = T;
        using deleter_type = Deleter;
        using pointer      = PointerTypeT<T, Deleter>;

    private:
        using storage_type = DeleterStorage<deleter_type>;

    public:
        template <class D = deleter_type,
                  class = EnableIfT<!IsLValueRef<D>::value && IsDefaultConstructible<D>::value>>
        constexpr UniquePtr() noexcept
            : ptr_(pointer()), del_() {}

        template <class D = deleter_type,
                  class = EnableIfT<!IsLValueRef<D>::value && IsDefaultConstructible<D>::value>>
        constexpr UniquePtr(nullptr_t) noexcept
            : ptr_(pointer()), del_() {}

        template <class D = deleter_type,
                  class = EnableIfT<!IsLValueRef<D>::value && IsDefaultConstructible<D>::value>>
        explicit UniquePtr(pointer p) noexcept
            : ptr_(p), del_() {}

        template <class D = deleter_type,
                  class = EnableIfT<
                      (!IsLValueRef<D>::value && IsConstructible<D, const D&>::value) ||
                      ( IsLValueRef<D>::value )>>
        UniquePtr(pointer p, const RemoveRefT<deleter_type>& d) noexcept
            : ptr_(p), del_(make_deleter_from_lvalue_(d)) {}

        template <class D = deleter_type,
                  class = EnableIfT<!IsLValueRef<D>::value && IsConstructible<D, D&&>::value>>
        UniquePtr(pointer p, RemoveRefT<deleter_type>&& d) noexcept
            : ptr_(p), del_(cslt::move(d)) {}

        UniquePtr(const UniquePtr&)            = delete;
        UniquePtr& operator=(const UniquePtr&) = delete;

        template <class D = deleter_type,
                  class = EnableIfT<
                      (!IsLValueRef<D>::value && IsMoveConstructible<D>::value) ||
                      ( IsLValueRef<D>::value )>>
        UniquePtr(UniquePtr&& other) noexcept
            : ptr_(other.release()), del_(move_deleter_from_(other)) {}

        template <class D = deleter_type,
                  class = EnableIfT<
                      (!IsLValueRef<D>::value && IsMoveAssignable<D>::value) ||
                      ( IsLValueRef<D>::value )>>
        UniquePtr& operator=(UniquePtr&& other) noexcept {
            if (this != &other) {
                reset(other.release());
                assign_deleter_from_(other);
            }
            return *this;
        }

        UniquePtr& operator=(nullptr_t) noexcept {
            reset();
            return *this;
        }

        ~UniquePtr() noexcept { reset(); }

        pointer get() const noexcept { return ptr_; }

        RemoveRefT<deleter_type>& get_deleter() noexcept { return del_.get(); }
        const RemoveRefT<deleter_type>& get_deleter() const noexcept { return del_.get(); }

        explicit operator bool() const noexcept { return !(ptr_ == pointer()); }

        element_type& operator[](::size_t i) const noexcept { return to_raw_(ptr_)[i]; }

        pointer release() noexcept {
            pointer tmp = ptr_;
            ptr_ = pointer();
            return tmp;
        }

        void reset(pointer p = pointer()) noexcept {
            if (!(ptr_ == p)) {
                pointer old = ptr_;
                ptr_ = p;
                if (!(old == pointer())) {
                    invoke_deleter_(old);
                }
            }
        }

        void reset(nullptr_t) noexcept { reset(pointer()); }

        void swap(UniquePtr& other) noexcept {
            cslt::swap(ptr_, other.ptr_);
            cslt::swap(del_, other.del_);
        }

    private:
        pointer      ptr_;
        storage_type del_;

        static storage_type make_deleter_from_lvalue_(const RemoveRefT<deleter_type>& d) noexcept {
            return make_deleter_from_lvalue_dispatch_(d, BoolConstant<IsLValueRef<deleter_type>::value>{});
        }

        static storage_type make_deleter_from_lvalue_dispatch_(const RemoveRefT<deleter_type>& d, FalseType) noexcept {
            return storage_type(d);
        }

        static storage_type make_deleter_from_lvalue_dispatch_(const RemoveRefT<deleter_type>& d, TrueType) noexcept {
            return storage_type(const_cast<RemoveRefT<deleter_type>&>(d));
        }

        static storage_type move_deleter_from_(UniquePtr& other) noexcept {
            return move_deleter_from_dispatch_(other, BoolConstant<IsLValueRef<deleter_type>::value>{});
        }

        static storage_type move_deleter_from_dispatch_(UniquePtr& other, FalseType) noexcept {
            return storage_type(cslt::move(other.get_deleter()));
        }

        static storage_type move_deleter_from_dispatch_(UniquePtr& other, TrueType) noexcept {
            return storage_type(other.get_deleter());
        }

        void assign_deleter_from_(UniquePtr& other) noexcept {
            assign_deleter_from_dispatch_(other, BoolConstant<IsLValueRef<deleter_type>::value>{});
        }

        void assign_deleter_from_dispatch_(UniquePtr& other, FalseType) noexcept {
            del_.get() = cslt::move(other.get_deleter());
        }

        void assign_deleter_from_dispatch_(UniquePtr&, TrueType) noexcept {
            // reference deleter stays bound
        }

        static element_type* to_raw_(pointer p) noexcept { return p; }

        void invoke_deleter_(pointer p) noexcept {
            del_.get()(to_raw_(p));
        }
    };

// ================================================================================
// Non-member swap
// ================================================================================

    template <class T, class D>
    inline void swap(UniquePtr<T, D>& a, UniquePtr<T, D>& b) noexcept { a.swap(b); }

    template <class T, class D>
    inline void swap(UniquePtr<T[], D>& a, UniquePtr<T[], D>& b) noexcept { a.swap(b); }

// ================================================================================
// Comparisons (UniquePtr vs UniquePtr) and nullptr comparisons
// ================================================================================

    template <class T1, class D1, class T2, class D2>
    inline bool operator==(const UniquePtr<T1, D1>& a, const UniquePtr<T2, D2>& b) noexcept {
        return a.get() == b.get();
    }

    template <class T1, class D1, class T2, class D2>
    inline bool operator!=(const UniquePtr<T1, D1>& a, const UniquePtr<T2, D2>& b) noexcept {
        return !(a == b);
    }

    template <class T1, class D1, class T2, class D2>
    inline bool operator<(const UniquePtr<T1, D1>& a, const UniquePtr<T2, D2>& b) noexcept {
        return a.get() < b.get();
    }

    template <class T1, class D1, class T2, class D2>
    inline bool operator<=(const UniquePtr<T1, D1>& a, const UniquePtr<T2, D2>& b) noexcept {
        return !(b < a);
    }

    template <class T1, class D1, class T2, class D2>
    inline bool operator>(const UniquePtr<T1, D1>& a, const UniquePtr<T2, D2>& b) noexcept {
        return b < a;
    }

    template <class T1, class D1, class T2, class D2>
    inline bool operator>=(const UniquePtr<T1, D1>& a, const UniquePtr<T2, D2>& b) noexcept {
        return !(a < b);
    }

    template <class T, class D>
    inline bool operator==(const UniquePtr<T, D>& p, nullptr_t) noexcept { return !p; }

    template <class T, class D>
    inline bool operator==(nullptr_t, const UniquePtr<T, D>& p) noexcept { return !p; }

    template <class T, class D>
    inline bool operator!=(const UniquePtr<T, D>& p, nullptr_t) noexcept { return static_cast<bool>(p); }

    template <class T, class D>
    inline bool operator!=(nullptr_t, const UniquePtr<T, D>& p) noexcept { return static_cast<bool>(p); }

// ================================================================================
// Optional helpers: make_unique (standalone)
// ================================================================================

    template <class T, class... Args>
    inline UniquePtr<T> make_unique(Args&&... args) {
        return UniquePtr<T>(new T(cslt::forward<Args>(args)...));
    }

    template <class T>
    inline UniquePtr<T[]> make_unique_array(::size_t n) {
        return UniquePtr<T[]>(new T[n]());
    }

// ================================================================================
// ================================================================================

    template <class T>
    class SharedPtr {
    public:
        using element_type = T;

        // ----------------------------------------------------------------------------
        // Constructors
        // ----------------------------------------------------------------------------
        constexpr SharedPtr() noexcept : ptr_(nullptr), cb_(nullptr) {}
        constexpr SharedPtr(nullptr_t) noexcept : ptr_(nullptr), cb_(nullptr) {}

        // Construct from raw pointer with default delete
        explicit SharedPtr(T* p) : ptr_(p), cb_(nullptr) {
            if (p) {
                cb_ = new detail::ControlBlockPtr<T, DefaultDelete<T>>(p, DefaultDelete<T>{});
            }
        }

        // Construct from raw pointer + deleter (value deleter)
        template <class Deleter,
                  class = EnableIfT<!IsLValueRef<Deleter>::value>>
        SharedPtr(T* p, Deleter d) : ptr_(p), cb_(nullptr) {
            if (p) {
                cb_ = new detail::ControlBlockPtr<T, Deleter>(p, cslt::move(d));
            }
        }

        // Copy
        SharedPtr(const SharedPtr& other) noexcept : ptr_(other.ptr_), cb_(other.cb_) {
            detail::incref_strong(cb_);
        }
     
        // Move
        SharedPtr(SharedPtr&& other) noexcept : ptr_(other.ptr_), cb_(other.cb_) {
            other.ptr_ = nullptr;
            other.cb_  = nullptr;
        }

        // ADD THIS: Converting move constructor (Derived -> Base)
        template <class U,
                  class = EnableIfT<IsConvertible<U*, T*>::value>>
        SharedPtr(SharedPtr<U>&& other) noexcept : ptr_(other.ptr_), cb_(other.cb_) {
            other.ptr_ = nullptr;
            other.cb_  = nullptr;
        }

        template <class U,
                  class = EnableIfT<IsConvertible<U*, T*>::value>>
        SharedPtr(const SharedPtr<U>& other) noexcept : ptr_(other.ptr_), cb_(other.cb_) {
            detail::incref_strong(cb_);
        }
      
        // Copy assignment
        SharedPtr& operator=(const SharedPtr& other) noexcept {
            if (this != &other) {
                release_();
                ptr_ = other.ptr_;
                cb_  = other.cb_;
                detail::incref_strong(cb_);
            }
            return *this;
        }

        // Converting copy assignment
        template <class U,
                  class = EnableIfT<IsConvertible<U*, T*>::value>>
        SharedPtr& operator=(const SharedPtr<U>& other) noexcept {
            if ((void*)this != (void*)&other) {
                release_();
                ptr_ = other.ptr_;
                cb_  = other.cb_;
                detail::incref_strong(cb_);
            }
            return *this;
        }
       
        // ----------------------------------------------------------------------------
        // Destructor
        // ----------------------------------------------------------------------------
        ~SharedPtr() noexcept {
            release_();
        }

        SharedPtr& operator=(SharedPtr&& other) noexcept {
            if (this != &other) {
                release_();
                ptr_ = other.ptr_;
                cb_  = other.cb_;
                other.ptr_ = nullptr;
                other.cb_  = nullptr;
            }
            return *this;
        }

        SharedPtr& operator=(nullptr_t) noexcept {
            reset();
            return *this;
        }

        template <class U,
                  class = EnableIfT<IsConvertible<U*, T*>::value>>
        SharedPtr& operator=(SharedPtr<U>&& other) noexcept {
            release_();
            ptr_ = other.ptr_;
            cb_  = other.cb_;
            other.ptr_ = nullptr;
            other.cb_  = nullptr;
            return *this;
        }

        // ----------------------------------------------------------------------------
        // Observers
        // ----------------------------------------------------------------------------
        T* get() const noexcept { return ptr_; }
        T& operator*() const noexcept { return *ptr_; }
        T* operator->() const noexcept { return ptr_; }

        explicit operator bool() const noexcept { return ptr_ != nullptr; }

        size_t use_count() const noexcept { 
            return detail::get_strong_count(cb_);
        }
       
        bool unique() const noexcept { return use_count() == 1u; }

        // ----------------------------------------------------------------------------
        // Modifiers
        // ----------------------------------------------------------------------------
        void reset() noexcept { release_(); }

        void reset(T* p) {
            release_();
            ptr_ = p;
            cb_  = nullptr;
            if (p) {
                cb_ = new detail::ControlBlockPtr<T, DefaultDelete<T>>(p, DefaultDelete<T>{});
            }
        }

        template <class Deleter,
                  class = EnableIfT<!IsLValueRef<Deleter>::value>>
        void reset(T* p, Deleter d) {
            release_();
            ptr_ = p;
            cb_  = nullptr;
            if (p) {
                cb_ = new detail::ControlBlockPtr<T, Deleter>(p, cslt::move(d));
            }
        }

        void swap(SharedPtr& other) noexcept {
            cslt::swap(ptr_, other.ptr_);
            cslt::swap(cb_, other.cb_);
        }

    private:
        template <class U>
        friend class SharedPtr;

        template <class U>
        friend class WeakPtr;

        void release_() noexcept {
            if (!cb_) {
                ptr_ = nullptr;
                return;
            }
            if (detail::decref_strong(cb_)) {
                cb_->destroy_object();
                // Decrement weak count (shared_ptr holds one weak ref to control block)
                if (detail::decref_weak(cb_)) {
                    delete cb_;
                }
            }
            ptr_ = nullptr;
            cb_  = nullptr;
        }
        
        T* ptr_;
        detail::ControlBlock* cb_;
    };

    // ================================================================================
    // Non-member swap + comparisons
    // ================================================================================

    template <class T>
    inline void swap(SharedPtr<T>& a, SharedPtr<T>& b) noexcept { a.swap(b); }

    template <class T, class U>
    inline bool operator==(const SharedPtr<T>& a, const SharedPtr<U>& b) noexcept {
        return a.get() == b.get();
    }

    template <class T, class U>
    inline bool operator!=(const SharedPtr<T>& a, const SharedPtr<U>& b) noexcept {
        return !(a == b);
    }

    template <class T>
    inline bool operator==(const SharedPtr<T>& p, nullptr_t) noexcept { return !p; }

    template <class T>
    inline bool operator==(nullptr_t, const SharedPtr<T>& p) noexcept { return !p; }

    template <class T>
    inline bool operator!=(const SharedPtr<T>& p, nullptr_t) noexcept { return static_cast<bool>(p); }

    template <class T>
    inline bool operator!=(nullptr_t, const SharedPtr<T>& p) noexcept { return static_cast<bool>(p); }

    // ================================================================================
    // make_shared (simple version)
    // - Note: true std::make_shared uses single allocation (control block + object).
    // - This version uses two allocations (new T + new control block) for simplicity.
    // ================================================================================

    template <class T, class... Args>
    inline SharedPtr<T> make_shared(Args&&... args) {
        return SharedPtr<T>(new T(cslt::forward<Args>(args)...));
    }
// ================================================================================ 
// ================================================================================ 

    // Equality comparisons between SharedPtr<T> and SharedPtr<U>
    template <class T, class U>
    inline bool operator<(const SharedPtr<T>& a, const SharedPtr<U>& b) noexcept {
        return a.get() < b.get();
    }

    template <class T, class U>
    inline bool operator<=(const SharedPtr<T>& a, const SharedPtr<U>& b) noexcept {
        return !(b < a);
    }

    template <class T, class U>
    inline bool operator>(const SharedPtr<T>& a, const SharedPtr<U>& b) noexcept {
        return b < a;
    }

    template <class T, class U>
    inline bool operator>=(const SharedPtr<T>& a, const SharedPtr<U>& b) noexcept {
        return !(a < b);
    }

    // Comparisons with nullptr
    template <class T>
    inline bool operator<(const SharedPtr<T>& p, nullptr_t) noexcept {
        return p.get() < static_cast<T*>(nullptr);
    }

    template <class T>
    inline bool operator<(nullptr_t, const SharedPtr<T>& p) noexcept {
        return static_cast<T*>(nullptr) < p.get();
    }

    template <class T>
    inline bool operator<=(const SharedPtr<T>& p, nullptr_t) noexcept {
        return !(nullptr < p);
    }

    template <class T>
    inline bool operator<=(nullptr_t, const SharedPtr<T>& p) noexcept {
        return !(p < nullptr);
    }

    template <class T>
    inline bool operator>(const SharedPtr<T>& p, nullptr_t) noexcept {
        return nullptr < p;
    }

    template <class T>
    inline bool operator>(nullptr_t, const SharedPtr<T>& p) noexcept {
        return p < nullptr;
    }

    template <class T>
    inline bool operator>=(const SharedPtr<T>& p, nullptr_t) noexcept {
        return !(p < nullptr);
    }

    template <class T>
    inline bool operator>=(nullptr_t, const SharedPtr<T>& p) noexcept {
        return !(nullptr < p);
    }
// ================================================================================ 
// ================================================================================ 

    template <class T>
    class WeakPtr {
    public:
        using element_type = T;

        // ----------------------------------------------------------------------------
        // Constructors
        // ----------------------------------------------------------------------------
        constexpr WeakPtr() noexcept : ptr_(nullptr), cb_(nullptr) {}

        // Construct from SharedPtr
        WeakPtr(const SharedPtr<T>& sp) noexcept : ptr_(sp.ptr_), cb_(sp.cb_) {
            detail::incref_weak(cb_);
        }

        // Converting constructor from SharedPtr<U>
        template <class U,
                  class = EnableIfT<IsConvertible<U*, T*>::value>>
        WeakPtr(const SharedPtr<U>& sp) noexcept : ptr_(sp.ptr_), cb_(sp.cb_) {
            detail::incref_weak(cb_);
        }

        // Copy constructor
        WeakPtr(const WeakPtr& other) noexcept : ptr_(other.ptr_), cb_(other.cb_) {
            detail::incref_weak(cb_);
        }

        // Converting copy constructor
        template <class U,
                  class = EnableIfT<IsConvertible<U*, T*>::value>>
        WeakPtr(const WeakPtr<U>& other) noexcept : ptr_(other.ptr_), cb_(other.cb_) {
            detail::incref_weak(cb_);
        }

        // Move constructor
        WeakPtr(WeakPtr&& other) noexcept : ptr_(other.ptr_), cb_(other.cb_) {
            other.ptr_ = nullptr;
            other.cb_  = nullptr;
        }

        // Converting move constructor
        template <class U,
                  class = EnableIfT<IsConvertible<U*, T*>::value>>
        WeakPtr(WeakPtr<U>&& other) noexcept : ptr_(other.ptr_), cb_(other.cb_) {
            other.ptr_ = nullptr;
            other.cb_  = nullptr;
        }

        // ----------------------------------------------------------------------------
        // Destructor
        // ----------------------------------------------------------------------------
        ~WeakPtr() noexcept {
            release_();
        }

        // ----------------------------------------------------------------------------
        // Assignment
        // ----------------------------------------------------------------------------
        WeakPtr& operator=(const WeakPtr& other) noexcept {
            if (this != &other) {
                release_();
                ptr_ = other.ptr_;
                cb_  = other.cb_;
                detail::incref_weak(cb_);
            }
            return *this;
        }

        WeakPtr& operator=(WeakPtr&& other) noexcept {
            if (this != &other) {
                release_();
                ptr_ = other.ptr_;
                cb_  = other.cb_;
                other.ptr_ = nullptr;
                other.cb_  = nullptr;
            }
            return *this;
        }

        template <class U,
                  class = EnableIfT<IsConvertible<U*, T*>::value>>
        WeakPtr& operator=(const WeakPtr<U>& other) noexcept {
            release_();
            ptr_ = other.ptr_;
            cb_  = other.cb_;
            detail::incref_weak(cb_);
            return *this;
        }

        template <class U,
                  class = EnableIfT<IsConvertible<U*, T*>::value>>
        WeakPtr& operator=(WeakPtr<U>&& other) noexcept {
            release_();
            ptr_ = other.ptr_;
            cb_  = other.cb_;
            other.ptr_ = nullptr;
            other.cb_  = nullptr;
            return *this;
        }

        WeakPtr& operator=(const SharedPtr<T>& sp) noexcept {
            release_();
            ptr_ = sp.ptr_;
            cb_  = sp.cb_;
            detail::incref_weak(cb_);
            return *this;
        }

        template <class U,
                  class = EnableIfT<IsConvertible<U*, T*>::value>>
        WeakPtr& operator=(const SharedPtr<U>& sp) noexcept {
            release_();
            ptr_ = sp.ptr_;
            cb_  = sp.cb_;
            detail::incref_weak(cb_);
            return *this;
        }

        // ----------------------------------------------------------------------------
        // Observers
        // ----------------------------------------------------------------------------
        std::size_t use_count() const noexcept {
            return detail::get_strong_count(cb_);
        }

        bool expired() const noexcept {
            return use_count() == 0u;
        }

        // Convert to SharedPtr (returns empty SharedPtr if expired)
        SharedPtr<T> lock() const noexcept {
            if (!cb_) {
                return SharedPtr<T>();
            }
            
            // Try to atomically increment strong count if it's not zero
            if (detail::try_incref_strong(cb_)) {
                SharedPtr<T> sp;
                sp.ptr_ = ptr_;
                sp.cb_  = cb_;
                return sp;
            }
            
            return SharedPtr<T>();
        }

        // ----------------------------------------------------------------------------
        // Modifiers
        // ----------------------------------------------------------------------------
        void reset() noexcept {
            release_();
        }

        void swap(WeakPtr& other) noexcept {
            cslt::swap(ptr_, other.ptr_);
            cslt::swap(cb_, other.cb_);
        }

    private:
        template <class U>
        friend class WeakPtr;
        
        template <class U>
        friend class SharedPtr;

        void release_() noexcept {
            if (cb_ && detail::decref_weak(cb_)) {
                delete cb_;
            }
            ptr_ = nullptr;
            cb_  = nullptr;
        }

        T* ptr_;
        detail::ControlBlock* cb_;
    };

    // ================================================================================
    // Non-member swap for WeakPtr
    // ================================================================================

    template <class T>
    inline void swap(WeakPtr<T>& a, WeakPtr<T>& b) noexcept { 
        a.swap(b); 
    }
// ================================================================================ 
// ================================================================================ 
} // namespace cslt

#endif /* POINTERS_HPP */

// ================================================================================
// ================================================================================
// eof

