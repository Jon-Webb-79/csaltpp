// ================================================================================
// ================================================================================
// - File:    list.hpp
// - Purpose: Describe the file purpose here
//
// Source Metadata
// - Author:  Jonathan A. Webb
// - Date:    April 01, 2026
// - Version: 1.0
// - Copyright: Copyright 2026, Jon Webb Inc.
// ================================================================================
// ================================================================================
// Include modules here

#ifndef list_HPP
#define list_HPP

#include "error.hpp"
#include "allocator.hpp"
 
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <type_traits>
#include <utility>
// ================================================================================ 
// ================================================================================ 

namespace cslt {
#ifndef CSLT_HAS_EQUALITY_OP_V
#define CSLT_HAS_EQUALITY_OP_V
    namespace detail {
        template <typename T, typename = void>
        struct has_eq_op : std::false_type {};

        template <typename T>
        struct has_eq_op<T,
            std::void_t<decltype(std::declval<const T&>() ==
                                 std::declval<const T&>())>>
            : std::true_type {};
    } // namespace detail

    template <typename T>
    static constexpr bool has_eq_op_v = detail::has_eq_op<T>::value;
#endif /* CSLT_HAS_EQUALITY_OP_V */
    // ================================================================================
    // ================================================================================

    // Forward declaration so SListDeleter is visible inside SList
    template <typename T> class SListDeleter;

    // ================================================================================
    // ================================================================================

    /**
     * @class SList
     * @brief Allocator-backed generic singly-linked list container.
     *
     * @details
     * Implements a singly-linked list backed by a pre-allocated contiguous node
     * slab. Nodes obtained from the slab are physically adjacent in memory,
     * improving cache behavior during traversal.
     *
     * Popped slab nodes are recycled through an internal free list and reused
     * before consuming fresh slab slots. Once all reusable slab slots are
     * exhausted, behavior is controlled by @p allow_overflow:
     *
     * - @c true  — overflow nodes are allocated individually through the allocator
     * - @c false — push operations fail once no slab nodes remain available
     *
     * @tparam T Element type.
     */
    template <typename T>
    class SList {

    private:

        // =========================================================================
        // Node layout
        // =========================================================================

        struct Node {
            Node*              next;
            alignas(T) uint8_t value_[sizeof(T)];
        };

        // =========================================================================
        // Private data members
        // =========================================================================

        Node*      head_;             ///< First live node
        Node*      tail_;             ///< Last live node
        size_t     len_;              ///< Number of live nodes currently in the list

        uint8_t*   slab_;             ///< Backing contiguous slab memory
        size_t     slab_cap_;         ///< Maximum number of slab nodes
        size_t     slab_used_;        ///< Number of slab-backed nodes currently live
        Node*      slab_free_;        ///< Recycled slab nodes available for reuse
        size_t     slab_free_count_;  ///< Count of nodes on slab_free_

        bool       allow_overflow_;   ///< True = allocator fallback, false = hard cap
        size_t     overflow_live_;    ///< Number of currently live overflow nodes
        Allocator* allocator_;        ///< Allocator for all internal memory

        // =========================================================================
        // Private constructor / destructor
        // =========================================================================

        SList(uint8_t*   slab,
              size_t     slab_cap,
              bool       allow_overflow,
              Allocator& allocator) noexcept
            : head_(nullptr)
            , tail_(nullptr)
            , len_(0u)
            , slab_(slab)
            , slab_cap_(slab_cap)
            , slab_used_(0u)
            , slab_free_(nullptr)
            , slab_free_count_(0u)
            , allow_overflow_(allow_overflow)
            , overflow_live_(0u)
            , allocator_(&allocator)
        {}

        ~SList() noexcept {
            Node* cur = head_;
            while (cur != nullptr) {
                Node* nxt = cur->next;
                _node_value(cur)->~T();
                _return_node(cur);
                cur = nxt;
            }

            head_ = nullptr;
            tail_ = nullptr;
            len_ = 0u;
            slab_used_ = 0u;
            slab_free_ = nullptr;
            slab_free_count_ = 0u;
            overflow_live_ = 0u;

            if (slab_ && allocator_) {
                allocator_->return_element(
                    static_cast<void*>(slab_),
                    slab_cap_ * _node_stride(),
                    allocator_->default_alignment());
                slab_ = nullptr;
            }
        }

        SList(const SList&)            = delete;
        SList& operator=(const SList&) = delete;
        SList(SList&&)                 = delete;
        SList& operator=(SList&&)      = delete;

        // =========================================================================
        // Private helpers — node stride and value access
        // =========================================================================

        static constexpr size_t _node_stride() noexcept {
            constexpr size_t raw   = sizeof(Node);
            constexpr size_t align = alignof(Node);
            return (raw + align - 1u) & ~(align - 1u);
        }

        static T* _node_value(Node* n) noexcept {
            return reinterpret_cast<T*>(n->value_);
        }

        static const T* _node_value_c(const Node* n) noexcept {
            return reinterpret_cast<const T*>(n->value_);
        }

        // =========================================================================
        // Private helpers — slab bounds / slab free list
        // =========================================================================

        bool _in_slab(const Node* node) const noexcept {
            if (!node || !slab_) return false;

            const uint8_t* p  = reinterpret_cast<const uint8_t*>(node);
            const uint8_t* lo = slab_;
            const uint8_t* hi = slab_ + slab_cap_ * _node_stride();

            return (p >= lo) && (p < hi);
        }

        void _push_slab_free(Node* node) noexcept {
            if (!node) return;
            node->next = slab_free_;
            slab_free_ = node;
            ++slab_free_count_;
        }

        Node* _pop_slab_free() noexcept {
            if (!slab_free_) return nullptr;

            Node* n = slab_free_;
            slab_free_ = slab_free_->next;
            n->next = nullptr;

            if (slab_free_count_ > 0u) {
                --slab_free_count_;
            }
            return n;
        }

        // =========================================================================
        // Private helpers — node allocation / deallocation
        // =========================================================================

        Node* _alloc_node_raw() noexcept {
            // 1. Reuse recycled slab node first
            if (slab_free_) {
                Node* n = _pop_slab_free();
                if (n) {
                    ++slab_used_;
                }
                return n;
            }

            // 2. Consume fresh slab slot
            size_t touched = slab_used_ + slab_free_count_;
            if (touched < slab_cap_) {
                Node* n = reinterpret_cast<Node*>(slab_ + touched * _node_stride());
                n->next = nullptr;
                ++slab_used_;
                return n;
            }

            // 3. Fall back to overflow allocation
            if (!allow_overflow_) {
                return nullptr;
            }

            auto r = allocator_->alloc(sizeof(Node), true);
            if (!r.hasValue()) {
                return nullptr;
            }

            Node* n = static_cast<Node*>(r.value());
            n->next = nullptr;
            ++overflow_live_;
            return n;
        }

        void _return_node(Node* node) noexcept {
            if (!node || !allocator_) return;

            if (_in_slab(node)) {
                if (slab_used_ > 0u) {
                    --slab_used_;
                }
                _push_slab_free(node);
                return;
            }

            allocator_->return_element(
                static_cast<void*>(node),
                sizeof(Node),
                allocator_->default_alignment());

            if (overflow_live_ > 0u) {
                --overflow_live_;
            }
        }

        // =========================================================================
        // Private helpers — key equality
        // =========================================================================

        static bool _values_equal(const T& a, const T& b) noexcept {
            if constexpr (has_eq_op_v<T>) {
                return a == b;
            } else {
                return std::memcmp(&a, &b, sizeof(T)) == 0;
            }
        }

        friend class SListDeleter<T>;

    public:

        // =========================================================================
        // Factory
        // =========================================================================

        static Expected<SList<T>*> init(size_t     num_nodes,
                                        bool       allow_overflow,
                                        Allocator& allocator) noexcept {
            Expected<SList<T>*> result;

            if (num_nodes == 0u) {
                result.setError(ArgumentError(
                    "SList::init: num_nodes must be > 0"));
                return result;
            }

            auto obj_r = allocator.alloc(sizeof(SList<T>), true);
            if (!obj_r.hasValue()) {
                result.setError(MemoryError(
                    "SList::init: failed to allocate SList struct"));
                return result;
            }

            auto slab_r = allocator.alloc(num_nodes * _node_stride(), true);
            if (!slab_r.hasValue()) {
                allocator.return_element(obj_r.value(),
                                         sizeof(SList<T>),
                                         allocator.default_alignment());
                result.setError(MemoryError(
                    "SList::init: failed to allocate node slab"));
                return result;
            }

            SList<T>* l = new (obj_r.value()) SList<T>(
                static_cast<uint8_t*>(slab_r.value()),
                num_nodes,
                allow_overflow,
                allocator);

            result.setValue(l);
            return result;
        }
    // -----------------------------------------------------------------------------

        static Expected<SList<T>*> copy(const SList<T>& src,
                                        Allocator&      allocator) noexcept {
            Expected<SList<T>*> result;

            auto ir = SList<T>::init(src.slab_cap_, src.allow_overflow_, allocator);
            if (!ir.hasValue()) {
                result.setError(ir.error());
                return result;
            }

            SList<T>* dst = ir.value();

            for (const Node* cur = src.head_; cur != nullptr; cur = cur->next) {
                if (!dst->push_back(*_node_value_c(cur))) {
                    Allocator* a = dst->allocator_;
                    dst->~SList<T>();
                    a->return_element(static_cast<void*>(dst),
                                      sizeof(SList<T>),
                                      a->default_alignment());
                    result.setError(MemoryError(
                        "SList::copy: failed to push during copy"));
                    return result;
                }
            }

            result.setValue(dst);
            return result;
        }
    // -----------------------------------------------------------------------------

        static Expected<SList<T>*> copy(const SList<T>& src) noexcept {
            return copy(src, *src.allocator_);
        }

        // =========================================================================
        // Push operations
        // =========================================================================

        bool push_back(const T& value) noexcept {
            if (!slab_ || !allocator_) return false;

            Node* node = _alloc_node_raw();
            if (!node) return false;

            new (static_cast<void*>(node->value_)) T(value);

            if (!tail_) {
                head_ = node;
                tail_ = node;
            } else {
                tail_->next = node;
                tail_ = node;
            }

            ++len_;
            return true;
        }
    // -----------------------------------------------------------------------------

        bool push_front(const T& value) noexcept {
            if (!slab_ || !allocator_) return false;

            Node* node = _alloc_node_raw();
            if (!node) return false;

            new (static_cast<void*>(node->value_)) T(value);
            node->next = head_;
            head_ = node;

            if (!tail_) {
                tail_ = node;
            }

            ++len_;
            return true;
        }
    // -----------------------------------------------------------------------------

        bool push_at(size_t index, const T& value) noexcept {
            if (!slab_ || !allocator_) return false;
            if (index > len_) return false;

            if (index == 0u)   return push_front(value);
            if (index == len_) return push_back(value);

            Node* node = _alloc_node_raw();
            if (!node) return false;

            new (static_cast<void*>(node->value_)) T(value);

            Node* prev = head_;
            for (size_t i = 0u; i < index - 1u; ++i) {
                prev = prev->next;
            }

            node->next = prev->next;
            prev->next = node;
            ++len_;
            return true;
        }

        // =========================================================================
        // Pop operations
        // =========================================================================

        Expected<T> pop_front() noexcept {
            Expected<T> result;

            if (!head_) {
                result.setError(OutOfBoundsError("SList::pop_front: list is empty"));
                return result;
            }

            Node* old_head = head_;
            T val(*_node_value(old_head));

            head_ = old_head->next;
            if (!head_) {
                tail_ = nullptr;
            }

            _node_value(old_head)->~T();
            _return_node(old_head);
            --len_;

            result.setValue(std::move(val));
            return result;
        }
    // -----------------------------------------------------------------------------

        Expected<T> pop_back() noexcept {
            Expected<T> result;

            if (!head_) {
                result.setError(OutOfBoundsError("SList::pop_back: list is empty"));
                return result;
            }

            if (head_ == tail_) {
                return pop_front();
            }

            Node* prev = head_;
            while (prev->next != tail_) {
                prev = prev->next;
            }

            Node* old_tail = tail_;
            T val(*_node_value(old_tail));

            prev->next = nullptr;
            tail_ = prev;

            _node_value(old_tail)->~T();
            _return_node(old_tail);
            --len_;

            result.setValue(std::move(val));
            return result;
        }
    // -----------------------------------------------------------------------------

        Expected<T> pop_at(size_t index) noexcept {
            Expected<T> result;

            if (!head_) {
                result.setError(OutOfBoundsError("SList::pop_at: list is empty"));
                return result;
            }

            if (index >= len_) {
                result.setError(OutOfBoundsError("SList::pop_at: index out of range"));
                return result;
            }

            if (index == 0u) {
                return pop_front();
            }
            if (index == len_ - 1u) {
                return pop_back();
            }

            Node* prev = head_;
            for (size_t i = 0u; i < index - 1u; ++i) {
                prev = prev->next;
            }

            Node* target = prev->next;
            T val(*_node_value(target));

            prev->next = target->next;

            _node_value(target)->~T();
            _return_node(target);
            --len_;

            result.setValue(std::move(val));
            return result;
        }

        // =========================================================================
        // Get and peek
        // =========================================================================

        Expected<T> get(size_t index) const noexcept {
            Expected<T> result;

            if (index >= len_) {
                result.setError(OutOfBoundsError("SList::get: index out of range"));
                return result;
            }

            const Node* node = head_;
            for (size_t i = 0u; i < index; ++i) {
                node = node->next;
            }

            result.setValue(*_node_value_c(node));
            return result;
        }
    // -----------------------------------------------------------------------------

        Expected<T> peek_front() const noexcept {
            Expected<T> result;

            if (!head_) {
                result.setError(OutOfBoundsError(
                    "SList::peek_front: list is empty"));
                return result;
            }

            result.setValue(*_node_value_c(head_));
            return result;
        }
    // -----------------------------------------------------------------------------

        Expected<T> peek_back() const noexcept {
            Expected<T> result;

            if (!tail_) {
                result.setError(OutOfBoundsError(
                    "SList::peek_back: list is empty"));
                return result;
            }

            result.setValue(*_node_value_c(tail_));
            return result;
        }

        // =========================================================================
        // Utility operations
        // =========================================================================

        bool clear() noexcept {
            Node* cur = head_;
            while (cur != nullptr) {
                Node* nxt = cur->next;
                _node_value(cur)->~T();
                _return_node(cur);
                cur = nxt;
            }

            head_ = nullptr;
            tail_ = nullptr;
            len_ = 0u;
            slab_used_ = 0u;
            slab_free_ = nullptr;
            slab_free_count_ = 0u;
            overflow_live_ = 0u;

            return true;
        }
    // -----------------------------------------------------------------------------

        bool reverse() noexcept {
            if (!slab_) return false;
            if (len_ <= 1u) return true;

            Node* prev = nullptr;
            Node* cur = head_;
            Node* new_tail = head_;

            while (cur != nullptr) {
                Node* nxt = cur->next;
                cur->next = prev;
                prev = cur;
                cur = nxt;
            }

            head_ = prev;
            tail_ = new_tail;
            return true;
        }
    // -----------------------------------------------------------------------------

        bool concat(const SList<T>& src) noexcept {
            for (const Node* cur = src.head_; cur != nullptr; cur = cur->next) {
                if (!push_back(*_node_value_c(cur))) {
                    return false;
                }
            }
            return true;
        }

        // =========================================================================
        // Search
        // =========================================================================

        Expected<size_t> contains(const T& needle) const noexcept {
            Expected<size_t> result;

            const Node* cur = head_;
            for (size_t i = 0u; cur != nullptr; ++i, cur = cur->next) {
                if (_values_equal(*_node_value_c(cur), needle)) {
                    result.setValue(i);
                    return result;
                }
            }

            result.setError(OutOfBoundsError("SList::contains: value not found"));
            return result;
        }

        // =========================================================================
        // Iteration
        // =========================================================================

        template <typename Func>
        bool foreach(Func fn) const noexcept {
            if (!slab_) return false;

            const Node* cur = head_;
            for (size_t i = 0u; cur != nullptr; ++i, cur = cur->next) {
                fn(*_node_value_c(cur), i);
            }

            return true;
        }

        // =========================================================================
        // Introspection
        // =========================================================================

        size_t size() const noexcept { return len_; }

        size_t slab_capacity() const noexcept { return slab_cap_; }

        size_t slab_used() const noexcept { return slab_used_; }

        size_t slab_free_count() const noexcept { return slab_free_count_; }

        size_t slab_live_count() const noexcept {
            return slab_used_;
        }

        size_t slab_remaining() const noexcept {
            return (slab_cap_ > slab_used_) ? (slab_cap_ - slab_used_) : 0u;
        }

        size_t overflow_count() const noexcept { return overflow_live_; }

        bool is_empty() const noexcept { return len_ == 0u; }

        bool is_slab_full() const noexcept {
            return slab_used_ >= slab_cap_;
        }

        bool in_overflow() const noexcept {
            return overflow_live_ > 0u;
        }
    };
    // ================================================================================
    // ================================================================================

    /**
     * @class SListDeleter
     * @brief Custom deleter enabling RAII cleanup for ::SList.
     *
     * @tparam T Element type stored by the list.
     *
     * @details
     * Intended for use with ``UniquePtr<SList<T>, SListDeleter<T>>``.
     * The deleter runs the list destructor and then returns the list object
     * itself to the allocator that created it.
     */
    template <typename T>
    class SListDeleter {
    public:
        /**
         * @brief Destroy a list and return its storage to the allocator.
         *
         * @param l
         *     Pointer to the list to destroy. May be @c nullptr.
         *
         * @note
         * This method first calls ``l->~SList<T>()`` to release slab and node
         * contents, then returns the list object memory itself to the allocator.
         */
        void operator()(SList<T>* l) const noexcept {
            if (!l) return;

            Allocator* allocator = l->allocator_;
            l->~SList<T>();

            if (allocator) {
                allocator->return_element(static_cast<void*>(l),
                                          sizeof(SList<T>),
                                          allocator->default_alignment());
            }
        }
    }; 
} // end cslt namespace
// ================================================================================ 
// ================================================================================ 
#endif /* file_name_HPP */
// ================================================================================
// ================================================================================
// eof
