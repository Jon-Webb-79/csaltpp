// ================================================================================
// ================================================================================
// - File:    dict.hpp
// - Purpose: Describe the file purpose here
//
// Source Metadata
// - Author:  Jonathan A. Webb
// - Date:    March 25, 2026
// - Version: 1.0
// - Copyright: Copyright 2022, Jon Webb Inc.
// ================================================================================
// ================================================================================
// Include modules here

#ifndef dict_HPP
#define dict_HPP
// ================================================================================ 
// ================================================================================ 

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
 
// ============================================================================
// C++17 detection trait — has_equality_op<T> is true when T provides operator==
// ============================================================================
 
    namespace detail {
        template <typename T, typename = void>
        struct has_equality_op : std::false_type {};
     
        template <typename T>
        struct has_equality_op<T,
            std::void_t<decltype(std::declval<const T&>() ==
                                 std::declval<const T&>())>>
            : std::true_type {};
    } // namespace detail
     
    template <typename T>
    static constexpr bool has_equality_op_v = detail::has_equality_op<T>::value;
    // ================================================================================
    // ================================================================================
     
    // Forward declaration so DictDeleter is visible inside Dict
    template <typename K, typename V> class DictDeleter;
     
    // ================================================================================
    // ================================================================================
     
    /**
     * @class Dict
     * @brief Allocator-backed generic hash dictionary mapping keys of type K to
     *        values of type V.
     *
     * @details Implements a chained hash table using the MurmurHash3-inspired hash
     *          function from the companion C implementation.  Both the Dict object
     *          and every internal node are allocated through the provided Allocator,
     *          enabling deterministic memory behaviour across heap, arena, buddy,
     *          and slab allocators.
     *
     * Key features:
     * - Template parameters K and V decouple key and value types completely
     * - K must be trivially copyable; keys are hashed over their raw bytes
     * - Key equality uses operator== when available, memcmp as fallback
     * - V may be any copy-constructible, destructible type
     * - Optional automatic growth when load factor exceeds 0.75
     * - Factory pattern via init() prevents uninitialised instances
     * - RAII cleanup through DictDeleter and UniquePtr
     * - foreach() iteration accepts any callable (lambda, functor, function pointer)
     *
     * @tparam K Key type — must satisfy std::is_trivially_copyable_v<K>
     * @tparam V Value type — must be default-constructible, copy-constructible,
     *           and destructible.  Default-constructibility is required because
     *           get() and pop() declare a local Expected<V> which default-constructs
     *           V before a value is available.
     *
     * @code{.cpp}
     * cslt::HeapAllocator alloc;
     * auto r = cslt::Dict<int, float>::init(8, true, alloc);
     * if (r.hasValue()) {
     *     cslt::UniquePtr<cslt::Dict<int,float>,
     *                     cslt::DictDeleter<int,float>> d(r.value());
     *     d->insert(42, 3.14f);
     *     auto val = d->get(42);          // Expected<float>
     *     auto popped = d->pop(42);       // Expected<float>
     * }
     * @endcode
     */
    template <typename K, typename V>
    class Dict {
     
        static_assert(std::is_trivially_copyable_v<K>,
            "Dict<K,V>: K must be trivially copyable. "
            "Raw-byte hashing and memcmp-based equality are only correct for "
            "trivially copyable types.");
     
    private:
     
        // ============================================================================
        // Internal constants
        // ============================================================================
     
        static constexpr double LOAD_FACTOR   = 0.75;
        static constexpr size_t GROWTH_FACTOR = 2u;
        static constexpr size_t LARGE_THRESH  = 4096u;
        static constexpr size_t LARGE_STEP    = 1024u;
        static constexpr size_t MIN_ALLOC     = 8u;
        static constexpr uint32_t HASH_SEED   = 0xdeadbeef;
     
        // ============================================================================
        // Node layout
        // ============================================================================
     
        /**
         * @brief A single chained node in the hash table.
         *
         * The node header is followed immediately in the same allocation by a
         * copy-constructed V value.  Access via _node_value() / _node_value_c().
         * The key is stored as a separately allocated copy of sizeof(K) bytes.
         */
        struct Node {
            Node*    next;     ///< Next node in the same bucket, or nullptr
            uint8_t* key_data; ///< Allocator-managed copy of the key bytes
        };
     
        /**
         * @brief Bucket sentinel — head of the linked list for one bucket slot.
         */
        struct Bucket {
            Node* next = nullptr;
        };
     
        // ============================================================================
        // Private data members
        // ============================================================================
     
        Bucket*    buckets_;    ///< Array of bucket sentinels, length alloc_
        size_t     len_;        ///< Number of occupied buckets (non-empty chains)
        size_t     hash_size_;  ///< Total number of key-value pairs stored
        size_t     alloc_;      ///< Number of buckets allocated (always power of two)
        bool       growth_;     ///< If true, resize automatically when load > 0.75
        Allocator* allocator_;  ///< Allocator used for all internal memory
     
        // ============================================================================
        // Private constructor / destructor
        // ============================================================================
     
        Dict(Bucket* buckets, size_t alloc, bool growth, Allocator& allocator)
            : buckets_(buckets)
            , len_(0u)
            , hash_size_(0u)
            , alloc_(alloc)
            , growth_(growth)
            , allocator_(&allocator)
        {}
     
        ~Dict() noexcept {
            if (!buckets_ || !allocator_) return;
            for (size_t i = 0u; i < alloc_; ++i) {
                Node* cur = buckets_[i].next;
                while (cur != nullptr) {
                    Node* nxt = cur->next;
                    _free_node(cur);
                    cur = nxt;
                }
            }
            allocator_->return_element(static_cast<void*>(buckets_),
                                       alloc_ * sizeof(Bucket),
                                       allocator_->default_alignment());
            buckets_ = nullptr;
        }
     
        Dict(const Dict&)            = delete;
        Dict& operator=(const Dict&) = delete;
        Dict(Dict&&)                 = delete;
        Dict& operator=(Dict&&)      = delete;
     
        // ============================================================================
        // Private helpers — node value access
        // ============================================================================
     
        /** @brief Return a pointer to the V value stored inline after the node header */
        static V* _node_value(Node* n) noexcept {
            return reinterpret_cast<V*>(
                reinterpret_cast<uint8_t*>(n) + sizeof(Node));
        }
     
        static const V* _node_value_c(const Node* n) noexcept {
            return reinterpret_cast<const V*>(
                reinterpret_cast<const uint8_t*>(n) + sizeof(Node));
        }
     
        // ============================================================================
        // Private helpers — key equality
        // ============================================================================
     
        /**
         * @brief Compare two keys for equality.
         *
         * Uses operator== when K provides it; falls back to memcmp otherwise.
         * Both paths are correct for trivially copyable K.
         */
        static bool _keys_equal(const K& a, const K& b) noexcept {
            if constexpr (has_equality_op_v<K>) {
                return a == b;
            } else {
                return std::memcmp(&a, &b, sizeof(K)) == 0;
            }
        }
     
        // ============================================================================
        // Private helpers — hashing
        // ============================================================================
     
        /**
         * @brief MurmurHash3-inspired hash over the raw bytes of key.
         *
         * Operates on sizeof(K) bytes, which is safe for trivially copyable K.
         */
        static size_t _hash_key(const K& key) noexcept {
            return _hash_bytes(&key, sizeof(K), HASH_SEED);
        }
     
        static size_t _hash_bytes(const void* data,
                                   size_t      len,
                                   uint32_t    seed) noexcept {
            if (!data || len == 0u) return 0u;
     
            const uint32_t c1 = 0xcc9e2d51u;
            const uint32_t c2 = 0x1b873593u;
            uint32_t h1 = seed;
     
            const unsigned char* p       = static_cast<const unsigned char*>(data);
            const size_t         nblocks = len / 4u;
     
            for (size_t i = 0u; i < nblocks; ++i) {
                uint32_t k1;
                std::memcpy(&k1, p + i * 4u, 4u);
                k1 *= c1;
                k1  = (k1 << 15) | (k1 >> 17);
                k1 *= c2;
                h1 ^= k1;
                h1  = (h1 << 13) | (h1 >> 19);
                h1  = h1 * 5u + 0xe6546b64u;
            }
     
            const unsigned char* tail = p + nblocks * 4u;
            uint32_t k1 = 0u;
            switch (len & 3u) {
                case 3: k1 ^= static_cast<uint32_t>(tail[2]) << 16; [[fallthrough]];
                case 2: k1 ^= static_cast<uint32_t>(tail[1]) <<  8; [[fallthrough]];
                case 1: k1 ^= static_cast<uint32_t>(tail[0]);
                        k1 *= c1;
                        k1  = (k1 << 15) | (k1 >> 17);
                        k1 *= c2;
                        h1 ^= k1;
                        break;
                default: break;
            }
     
            h1 ^= static_cast<uint32_t>(len);
            h1 ^= h1 >> 16;
            h1 *= 0x85ebca6bu;
            h1 ^= h1 >> 13;
            h1 *= 0xc2b2ae35u;
            h1 ^= h1 >> 16;
     
            return static_cast<size_t>(h1);
        }
     
        size_t _bucket_index(const K& key) const noexcept {
            return _hash_key(key) % alloc_;
        }
     
        // ============================================================================
        // Private helpers — next power of two
        // ============================================================================
     
        static size_t _next_pow2(size_t n) noexcept {
            if (n < MIN_ALLOC) return MIN_ALLOC;
            --n;
            n |= n >>  1;
            n |= n >>  2;
            n |= n >>  4;
            n |= n >>  8;
            n |= n >> 16;
#if SIZE_MAX > 0xFFFFFFFFu
            n |= n >> 32;
#endif
            return n + 1u;
        }
     
        // ============================================================================
        // Private helpers — node allocation / deallocation
        // ============================================================================
     
        /**
         * @brief Allocate a node with an inline copy-constructed V value.
         *
         * Layout: [Node header][V value (placement-new)]
         * Key is stored in a separate allocation.
         *
         * @return nullptr on any allocation failure (node cleaned up internally)
         */
        Node* _alloc_node(const K& key, const V& value) noexcept {
            // Single allocation: node header + sizeof(V) for the inline value
            size_t const node_bytes = sizeof(Node) + sizeof(V);
            auto nr = allocator_->alloc(node_bytes, true);
            if (!nr.hasValue()) return nullptr;
     
            Node* node = static_cast<Node*>(nr.value());
     
            // Key copy — separate allocation so key lifetime is explicit
            auto kr = allocator_->alloc(sizeof(K), true);
            if (!kr.hasValue()) {
                allocator_->return_element(static_cast<void*>(node),
                                           node_bytes,
                                           allocator_->default_alignment());
                return nullptr;
            }
     
            node->key_data = static_cast<uint8_t*>(kr.value());
            std::memcpy(node->key_data, &key, sizeof(K));
            node->next = nullptr;
     
            // Placement-new the value into the inline buffer
            new (static_cast<void*>(_node_value(node))) V(value);
     
            return node;
        }
     
        /** @brief Destroy the inline V and free both the key and node allocations */
        void _free_node(Node* node) noexcept {
            if (!node) return;
            _node_value(node)->~V();
            allocator_->return_element(static_cast<void*>(node->key_data),
                                       sizeof(K),
                                       allocator_->default_alignment());
            allocator_->return_element(static_cast<void*>(node),
                                       sizeof(Node) + sizeof(V),
                                       allocator_->default_alignment());
        }
     
        // ============================================================================
        // Private helpers — bucket lookup
        // ============================================================================
     
        Node* _find_node(size_t bucket_idx, const K& key) noexcept {
            for (Node* n = buckets_[bucket_idx].next; n != nullptr; n = n->next) {
                if (_keys_equal(*reinterpret_cast<const K*>(n->key_data), key))
                    return n;
            }
            return nullptr;
        }
     
        const Node* _find_node_c(size_t bucket_idx, const K& key) const noexcept {
            for (const Node* n = buckets_[bucket_idx].next; n != nullptr; n = n->next) {
                if (_keys_equal(*reinterpret_cast<const K*>(n->key_data), key))
                    return n;
            }
            return nullptr;
        }
     
        // ============================================================================
        // Private helpers — resize
        // ============================================================================
     
        bool _resize(size_t new_alloc) noexcept {
            new_alloc = _next_pow2(new_alloc);
     
            auto br = allocator_->alloc(new_alloc * sizeof(Bucket), true);
            if (!br.hasValue()) return false;
     
            Bucket* new_buckets = static_cast<Bucket*>(br.value());
     
            // Initialise all next pointers to nullptr
            for (size_t i = 0u; i < new_alloc; ++i)
                new_buckets[i].next = nullptr;
     
            // Rehash all existing nodes into the new bucket array
            for (size_t i = 0u; i < alloc_; ++i) {
                Node* cur = buckets_[i].next;
                while (cur != nullptr) {
                    Node* nxt = cur->next;
                    const K& k = *reinterpret_cast<const K*>(cur->key_data);
                    size_t idx = _hash_key(k) % new_alloc;
                    cur->next = new_buckets[idx].next;
                    new_buckets[idx].next = cur;
                    cur = nxt;
                }
            }
     
            // Recompute occupied-bucket count
            size_t new_len = 0u;
            for (size_t i = 0u; i < new_alloc; ++i)
                if (new_buckets[i].next != nullptr) ++new_len;
     
            allocator_->return_element(static_cast<void*>(buckets_),
                                       alloc_ * sizeof(Bucket),
                                       allocator_->default_alignment());
            buckets_ = new_buckets;
            alloc_   = new_alloc;
            len_     = new_len;
     
            return true;
        }
     
        // DictDeleter needs access to private members and destructor
        template <typename KK, typename VV>
        friend class DictDeleter;
     
    public:
     
        // ============================================================================
        // Factory
        // ============================================================================
     
        /**
         * @brief Allocate and initialise a new Dict
         *
         * @param capacity  Initial bucket count.  Must be > 0.  Rounded up to the
         *                  next power of two (minimum 8).
         * @param growth    If true the table resizes automatically when the load
         *                  factor exceeds 0.75.  If false, insert() returns false
         *                  when all buckets are full.
         * @param allocator Allocator for all internal memory.
         * @return Expected<Dict<K,V>*> on success, or a MemoryError / ArgumentError
         *
         * @par Error conditions
         * - capacity == 0 (ArgumentError)
         * - Allocation of the Dict struct fails (MemoryError)
         * - Allocation of the bucket array fails (MemoryError)
         *
         * @code{.cpp}
         * // Create an int->float dictionary with an initial capacity of 16 buckets
         * // and automatic growth enabled.  The UniquePtr owns the dict and frees
         * // it automatically when it goes out of scope.
         * cslt::HeapAllocator alloc;
         * auto r = cslt::Dict<int, float>::init(16, true, alloc);
         * if (!r.hasValue()) {
         *     // handle error — r.error() describes the failure
         *     return;
         * }
         * cslt::UniquePtr<cslt::Dict<int, float>,
         *                 cslt::DictDeleter<int, float>> d(r.value());
         *
         * d->insert(1, 1.1f);
         * d->insert(2, 2.2f);
         * // d is freed here when it leaves scope
         * @endcode
         */
        static Expected<Dict<K,V>*> init(size_t     capacity,
                                         bool       growth,
                                         Allocator& allocator) noexcept {
            Expected<Dict<K,V>*> result;
     
            if (capacity == 0u) {
                result.setError(ArgumentError(
                    "Dict::init: capacity must be > 0"));
                return result;
            }
     
            size_t const alloc = _next_pow2(capacity);
     
            // Allocate the Dict struct itself
            auto obj_r = allocator.alloc(sizeof(Dict<K,V>), true);
            if (!obj_r.hasValue()) {
                result.setError(MemoryError(
                    "Dict::init: failed to allocate Dict struct"));
                return result;
            }
     
            // Allocate the zeroed bucket array
            auto bkt_r = allocator.alloc(alloc * sizeof(Bucket), true);
            if (!bkt_r.hasValue()) {
                allocator.return_element(obj_r.value(),
                                         sizeof(Dict<K,V>),
                                         allocator.default_alignment());
                result.setError(MemoryError(
                    "Dict::init: failed to allocate bucket array"));
                return result;
            }
     
            Bucket* buckets = static_cast<Bucket*>(bkt_r.value());
            for (size_t i = 0u; i < alloc; ++i)
                buckets[i].next = nullptr;
     
            Dict<K,V>* d = new (obj_r.value()) Dict<K,V>(buckets, alloc,
                                                           growth, allocator);
            result.setValue(d);
            return result;
        }
    // --------------------------------------------------------------------------------
     
        /**
         * @brief Create a deep copy of @p src using a caller-supplied allocator
         *
         * All key-value pairs are rehashed into a new bucket array of the same
         * capacity as @p src.  Nodes are copy-constructed individually so the
         * copy is fully independent of the source.
         *
         * @param src       Dict to copy from
         * @param allocator Allocator for the new Dict and all its nodes.
         *                  May differ from the allocator used to create @p src.
         * @return Expected<Dict<K,V>*> on success, or a MemoryError on allocation
         *         failure
         *
         * @code{.cpp}
         * cslt::HeapAllocator alloc;
         *
         * // Build the source dictionary
         * auto r = cslt::Dict<int, float>::init(8, true, alloc);
         * if (!r.hasValue()) return;
         * cslt::UniquePtr<cslt::Dict<int, float>,
         *                 cslt::DictDeleter<int, float>> src(r.value());
         * src->insert(1, 1.1f);
         * src->insert(2, 2.2f);
         *
         * // Deep-copy into a second dictionary backed by the same allocator
         * auto cr = cslt::Dict<int, float>::copy(*src, alloc);
         * if (!cr.hasValue()) return;
         * cslt::UniquePtr<cslt::Dict<int, float>,
         *                 cslt::DictDeleter<int, float>> dst(cr.value());
         *
         * // src and dst are independent — mutating one does not affect the other
         * dst->insert(3, 3.3f);
         * @endcode
         */
        static Expected<Dict<K,V>*> copy(const Dict<K,V>& src,
                                          Allocator&       allocator) noexcept {
            Expected<Dict<K,V>*> result;
     
            auto ir = Dict<K,V>::init(src.alloc_, src.growth_, allocator);
            if (!ir.hasValue()) {
                result.setError(ir.error());
                return result;
            }
     
            Dict<K,V>* dst = ir.value();
     
            for (size_t i = 0u; i < src.alloc_; ++i) {
                for (const Node* cur = src.buckets_[i].next;
                     cur != nullptr; cur = cur->next) {
                    const K& k = *reinterpret_cast<const K*>(cur->key_data);
                    if (!dst->insert(k, *_node_value_c(cur))) {
                        // Clean up partially built copy
                        Allocator* a = dst->allocator_;
                        dst->~Dict<K,V>();
                        a->return_element(static_cast<void*>(dst),
                                          sizeof(Dict<K,V>),
                                          a->default_alignment());
                        result.setError(MemoryError(
                            "Dict::copy: failed to insert during copy"));
                        return result;
                    }
                }
            }
     
            result.setValue(dst);
            return result;
        }
    // --------------------------------------------------------------------------------
     
        /**
         * @brief Create a deep copy using the source dict's own allocator
         *
         * Convenience overload that forwards to copy(src, allocator) using the
         * allocator already associated with @p src.
         *
         * @param src  Dict to copy from
         * @return Expected<Dict<K,V>*> on success, or a MemoryError
         *
         * @code{.cpp}
         * cslt::HeapAllocator alloc;
         *
         * auto r = cslt::Dict<int, float>::init(8, true, alloc);
         * if (!r.hasValue()) return;
         * cslt::UniquePtr<cslt::Dict<int, float>,
         *                 cslt::DictDeleter<int, float>> src(r.value());
         * src->insert(10, 9.9f);
         *
         * // Copy using src's own allocator — no need to pass alloc explicitly
         * auto cr = cslt::Dict<int, float>::copy(*src);
         * if (!cr.hasValue()) return;
         * cslt::UniquePtr<cslt::Dict<int, float>,
         *                 cslt::DictDeleter<int, float>> dst(cr.value());
         * @endcode
         */
        static Expected<Dict<K,V>*> copy(const Dict<K,V>& src) noexcept {
            return copy(src, *src.allocator_);
        }
    // --------------------------------------------------------------------------------
     
        /**
         * @brief Merge two dicts into a new dict
         *
         * All entries from @p a are inserted first.  Entries from @p b are then
         * processed:
         * - If the key does not exist in the result it is inserted.
         * - If the key exists and @p overwrite is true the value is replaced.
         * - If the key exists and @p overwrite is false the original value is kept.
         *
         * @p a and @p b are not modified.
         *
         * @param a         First source dict
         * @param b         Second source dict
         * @param overwrite If true, @p b's values win on key conflicts
         * @param allocator Allocator for the new dict
         * @return Expected<Dict<K,V>*> on success, or a MemoryError on allocation
         *         failure
         *
         * @code{.cpp}
         * cslt::HeapAllocator alloc;
         *
         * // Build the first source dictionary
         * auto ra = cslt::Dict<int, float>::init(8, true, alloc);
         * if (!ra.hasValue()) return;
         * cslt::UniquePtr<cslt::Dict<int, float>,
         *                 cslt::DictDeleter<int, float>> a(ra.value());
         * a->insert(1, 1.0f);
         * a->insert(2, 2.0f);
         *
         * // Build the second source dictionary — key 2 conflicts with a
         * auto rb = cslt::Dict<int, float>::init(8, true, alloc);
         * if (!rb.hasValue()) return;
         * cslt::UniquePtr<cslt::Dict<int, float>,
         *                 cslt::DictDeleter<int, float>> b(rb.value());
         * b->insert(2, 99.0f);   // conflict — same key as in a
         * b->insert(3, 3.0f);
         *
         * // Merge: b wins on conflicts (overwrite == true)
         * // Result contains: {1->1.0, 2->99.0, 3->3.0}
         * auto mr = cslt::Dict<int, float>::merge(*a, *b, true, alloc);
         * if (!mr.hasValue()) return;
         * cslt::UniquePtr<cslt::Dict<int, float>,
         *                 cslt::DictDeleter<int, float>> merged(mr.value());
         * @endcode
         */
        static Expected<Dict<K,V>*> merge(const Dict<K,V>& a,
                                           const Dict<K,V>& b,
                                           bool             overwrite,
                                           Allocator&       allocator) noexcept {
            Expected<Dict<K,V>*> result;
     
            // Start with a full copy of a
            auto cr = copy(a, allocator);
            if (!cr.hasValue()) {
                result.setError(cr.error());
                return result;
            }
            Dict<K,V>* dst = cr.value();
     
            for (size_t i = 0u; i < b.alloc_; ++i) {
                for (const Node* cur = b.buckets_[i].next;
                     cur != nullptr; cur = cur->next) {
                    const K& k = *reinterpret_cast<const K*>(cur->key_data);
                    const V& v = *_node_value_c(cur);
     
                    if (dst->has_key(k)) {
                        if (overwrite) dst->update(k, v);
                    } else {
                        if (!dst->insert(k, v)) {
                            Allocator* al = dst->allocator_;
                            dst->~Dict<K,V>();
                            al->return_element(static_cast<void*>(dst),
                                               sizeof(Dict<K,V>),
                                               al->default_alignment());
                            result.setError(MemoryError(
                                "Dict::merge: failed to insert during merge"));
                            return result;
                        }
                    }
                }
            }
     
            result.setValue(dst);
            return result;
        }
     
        // ============================================================================
        // Insert / remove / update
        // ============================================================================
     
        /**
         * @brief Insert a new key-value pair
         *
         * @param key    Key to insert.  Must not already exist in the dict.
         * @param value  Value to associate with the key.  Copy-constructed into
         *               the node's inline buffer.
         * @return true on success; false if the key already exists, allocation
         *         fails, or the dict is full and growth is disabled.
         *
         * @code{.cpp}
         * cslt::HeapAllocator alloc;
         * auto r = cslt::Dict<int, float>::init(8, true, alloc);
         * if (!r.hasValue()) return;
         * cslt::UniquePtr<cslt::Dict<int, float>,
         *                 cslt::DictDeleter<int, float>> d(r.value());
         *
         * bool ok = d->insert(42, 3.14f);   // ok == true
         * bool dup = d->insert(42, 9.99f);  // dup == false — duplicate key rejected
         * @endcode
         */
        bool insert(const K& key, const V& value) noexcept {
            if (!buckets_ || !allocator_) return false;
     
            // Auto-grow if load factor exceeded
            if (growth_ &&
                hash_size_ >= static_cast<size_t>(alloc_ * LOAD_FACTOR)) {
                size_t new_alloc = (alloc_ < LARGE_THRESH)
                                   ? alloc_ * GROWTH_FACTOR
                                   : alloc_ + LARGE_STEP;
                if (!_resize(new_alloc)) return false;
            }
     
            if (!growth_ && hash_size_ >= alloc_) return false;
     
            size_t const idx = _bucket_index(key);
     
            // Reject duplicate keys
            if (_find_node(idx, key) != nullptr) return false;
     
            Node* node = _alloc_node(key, value);
            if (!node) return false;
     
            bool const was_empty = (buckets_[idx].next == nullptr);
            node->next = buckets_[idx].next;
            buckets_[idx].next = node;
     
            ++hash_size_;
            if (was_empty) ++len_;
     
            return true;
        }
    // --------------------------------------------------------------------------------
     
        /**
         * @brief Overwrite the value of an existing key
         *
         * The key is not re-hashed and no allocation is performed — only the
         * stored value is replaced via destroy-then-copy-construct in place.
         *
         * @param key   Key to update.  Must already exist in the dict.
         * @param value New value to store.
         * @return true on success; false if the key is not found.
         *
         * @code{.cpp}
         * cslt::HeapAllocator alloc;
         * auto r = cslt::Dict<int, float>::init(8, true, alloc);
         * if (!r.hasValue()) return;
         * cslt::UniquePtr<cslt::Dict<int, float>,
         *                 cslt::DictDeleter<int, float>> d(r.value());
         *
         * d->insert(42, 3.14f);
         *
         * bool ok = d->update(42, 6.28f);   // ok == true,  value is now 6.28f
         * bool nf = d->update(99, 1.00f);   // nf == false, key 99 does not exist
         * @endcode
         */
        bool update(const K& key, const V& value) noexcept {
            if (!buckets_ || !allocator_) return false;
     
            size_t const idx  = _bucket_index(key);
            Node*        node = _find_node(idx, key);
            if (!node) return false;
     
            // Destroy old value and copy-construct the new one in place
            V* slot = _node_value(node);
            slot->~V();
            new (static_cast<void*>(slot)) V(value);
     
            return true;
        }
    // --------------------------------------------------------------------------------
     
        /**
         * @brief Remove a key-value pair and return the removed value
         *
         * The node is unlinked from its bucket chain and freed.  The value is
         * copy-constructed into the Expected before the node is destroyed, so
         * the caller always receives the value regardless of what type V is.
         *
         * @param key  Key to remove.
         * @return Expected<V> containing the value on success, or an
         *         OutOfBoundsError if the key is not found.
         *
         * @code{.cpp}
         * cslt::HeapAllocator alloc;
         * auto r = cslt::Dict<int, float>::init(8, true, alloc);
         * if (!r.hasValue()) return;
         * cslt::UniquePtr<cslt::Dict<int, float>,
         *                 cslt::DictDeleter<int, float>> d(r.value());
         *
         * d->insert(42, 3.14f);
         * d->insert(99, 9.99f);
         *
         * auto pr = d->pop(42);
         * if (pr.hasValue()) {
         *     float v = pr.value();   // v == 3.14f; key 42 no longer in dict
         * }
         *
         * auto missing = d->pop(7);
         * // missing.hasValue() == false — key 7 was never inserted
         * @endcode
         */
        Expected<V> pop(const K& key) noexcept {
            Expected<V> result;
     
            if (!buckets_ || !allocator_) {
                result.setError(OutOfBoundsError("Dict::pop: key not found"));
                return result;
            }
     
            size_t  const idx     = _bucket_index(key);
            Node**        prevnxt = &buckets_[idx].next;
            Node*         cur     = buckets_[idx].next;
     
            while (cur != nullptr) {
                if (_keys_equal(*reinterpret_cast<const K*>(cur->key_data), key)) {
                    // Copy value out before destruction
                    V val(*_node_value(cur));
     
                    *prevnxt = cur->next;
                    _free_node(cur);
     
                    --hash_size_;
                    if (buckets_[idx].next == nullptr) --len_;
     
                    result.setValue(std::move(val));
                    return result;
                }
                prevnxt = &cur->next;
                cur     = cur->next;
            }
     
            result.setError(OutOfBoundsError("Dict::pop: key not found"));
            return result;
        }
     
        // ============================================================================
        // Lookup
        // ============================================================================
     
        /**
         * @brief Return a copy of the value associated with a key
         *
         * The dict is not modified.  The returned Expected owns a copy of the
         * value; mutations to the copy do not affect the stored value.
         *
         * @param key  Key to look up.
         * @return Expected<V> containing a copy of the value on success, or an
         *         OutOfBoundsError if the key is not found.
         *
         * @code{.cpp}
         * cslt::HeapAllocator alloc;
         * auto r = cslt::Dict<int, float>::init(8, true, alloc);
         * if (!r.hasValue()) return;
         * cslt::UniquePtr<cslt::Dict<int, float>,
         *                 cslt::DictDeleter<int, float>> d(r.value());
         *
         * d->insert(42, 3.14f);
         *
         * auto gr = d->get(42);
         * if (gr.hasValue()) {
         *     float v = gr.value();   // v == 3.14f
         * }
         *
         * auto missing = d->get(99);
         * // missing.hasValue() == false
         * @endcode
         */
        Expected<V> get(const K& key) const noexcept {
            Expected<V> result;
     
            if (!buckets_) {
                result.setError(OutOfBoundsError("Dict::get: key not found"));
                return result;
            }
     
            size_t      const idx  = _bucket_index(key);
            const Node*       node = _find_node_c(idx, key);
     
            if (!node) {
                result.setError(OutOfBoundsError("Dict::get: key not found"));
                return result;
            }
     
            result.setValue(*_node_value_c(node));
            return result;
        }
    // --------------------------------------------------------------------------------
     
        /**
         * @brief Return a read-only pointer directly into the node's value slot
         *
         * The pointer remains valid until the next mutation of the dict
         * (insert, update, pop, clear, or any operation that triggers a resize).
         * The caller must not free or write through it.
         *
         * Prefer get() when a copy is acceptable.  Use get_ptr() only when
         * avoiding the copy is important and the pointer lifetime can be
         * guaranteed.
         *
         * @param key  Key to look up.
         * @return Const pointer to the value on success, nullptr if not found.
         *
         * @code{.cpp}
         * cslt::HeapAllocator alloc;
         * auto r = cslt::Dict<int, float>::init(8, true, alloc);
         * if (!r.hasValue()) return;
         * cslt::UniquePtr<cslt::Dict<int, float>,
         *                 cslt::DictDeleter<int, float>> d(r.value());
         *
         * d->insert(42, 3.14f);
         *
         * const float* ptr = d->get_ptr(42);   // ptr != nullptr
         * if (ptr) {
         *     float v = *ptr;   // v == 3.14f — zero-copy read
         * }
         *
         * const float* missing = d->get_ptr(99);   // missing == nullptr
         * @endcode
         */
        const V* get_ptr(const K& key) const noexcept {
            if (!buckets_) return nullptr;
            size_t      const idx  = _bucket_index(key);
            const Node*       node = _find_node_c(idx, key);
            return (node != nullptr) ? _node_value_c(node) : nullptr;
        }
    // --------------------------------------------------------------------------------
     
        /**
         * @brief Test whether a key exists without retrieving its value
         *
         * @param key  Key to test.
         * @return true if the key exists, false otherwise.
         *
         * @code{.cpp}
         * cslt::HeapAllocator alloc;
         * auto r = cslt::Dict<int, float>::init(8, true, alloc);
         * if (!r.hasValue()) return;
         * cslt::UniquePtr<cslt::Dict<int, float>,
         *                 cslt::DictDeleter<int, float>> d(r.value());
         *
         * d->insert(42, 3.14f);
         *
         * bool found   = d->has_key(42);   // true
         * bool missing = d->has_key(99);   // false
         * @endcode
         */
        bool has_key(const K& key) const noexcept {
            if (!buckets_) return false;
            size_t const idx = _bucket_index(key);
            return _find_node_c(idx, key) != nullptr;
        }
     
        // ============================================================================
        // Utility
        // ============================================================================
     
        /**
         * @brief Remove all entries without freeing the Dict or its bucket array
         *
         * Every node (key copy + inline value) is destroyed and freed via the
         * allocator.  The bucket array is retained and zeroed so the Dict can
         * be reused immediately without a further allocation.  size() and
         * hash_size() are reset to 0; bucket_count() is unchanged.
         *
         * @code{.cpp}
         * cslt::HeapAllocator alloc;
         * auto r = cslt::Dict<int, float>::init(8, true, alloc);
         * if (!r.hasValue()) return;
         * cslt::UniquePtr<cslt::Dict<int, float>,
         *                 cslt::DictDeleter<int, float>> d(r.value());
         *
         * d->insert(1, 1.1f);
         * d->insert(2, 2.2f);
         * // d->hash_size() == 2
         *
         * d->clear();
         * // d->hash_size() == 0, d->is_empty() == true
         * // d->bucket_count() is unchanged — the bucket array is reused
         *
         * d->insert(3, 3.3f);   // can insert immediately after clear
         * @endcode
         */
        void clear() noexcept {
            if (!buckets_) return;
            for (size_t i = 0u; i < alloc_; ++i) {
                Node* cur = buckets_[i].next;
                while (cur != nullptr) {
                    Node* nxt = cur->next;
                    _free_node(cur);
                    cur = nxt;
                }
                buckets_[i].next = nullptr;
            }
            hash_size_ = 0u;
            len_       = 0u;
        }
     
        // ============================================================================
        // Iteration
        // ============================================================================
     
        /**
         * @brief Call @p fn once for every key-value pair in the dict
         *
         * Traversal order is bucket order (not insertion order).  The callback
         * receives const references to both the key and the value.
         *
         * @tparam Func  Callable with signature `void(const K&, const V&)`.
         *               Lambdas, functors, and plain function pointers are all
         *               accepted.  The callback must not insert, update, remove,
         *               or clear entries during traversal.
         *
         * @param fn  Callback invoked for each entry.
         * @return true on success; false if the dict is uninitialised.
         *
         * @code{.cpp}
         * cslt::HeapAllocator alloc;
         * auto r = cslt::Dict<int, float>::init(8, true, alloc);
         * if (!r.hasValue()) return;
         * cslt::UniquePtr<cslt::Dict<int, float>,
         *                 cslt::DictDeleter<int, float>> d(r.value());
         *
         * d->insert(1, 1.1f);
         * d->insert(2, 2.2f);
         * d->insert(3, 3.3f);
         *
         * // Lambda — accumulate the sum of all values
         * float total = 0.0f;
         * d->foreach([&total](const int& key, const float& value) {
         *     total += value;
         * });
         * // total == 6.6f (approximately)
         *
         * // File-scope function pointer
         * static void print_entry(const int& key, const float& value) {
         *     // log key and value
         * }
         * d->foreach(print_entry);
         * @endcode
         */
        template <typename Func>
        bool foreach(Func fn) const noexcept {
            if (!buckets_) return false;
            for (size_t i = 0u; i < alloc_; ++i) {
                for (const Node* cur = buckets_[i].next;
                     cur != nullptr; cur = cur->next) {
                    fn(*reinterpret_cast<const K*>(cur->key_data),
                       *_node_value_c(cur));
                }
            }
            return true;
        }
     
        // ============================================================================
        // Introspection
        // ============================================================================
     
        /**
         * @brief Number of occupied buckets (chains with at least one entry)
         *
         * This is the number of distinct hash slots in use, not the total number
         * of key-value pairs.  Multiple keys may hash to the same bucket and form
         * a chain; each such chain counts as one occupied bucket.  Use hash_size()
         * for the total number of pairs.
         *
         * @code{.cpp}
         * cslt::HeapAllocator alloc;
         * auto r = cslt::Dict<int, float>::init(8, true, alloc);
         * if (!r.hasValue()) return;
         * cslt::UniquePtr<cslt::Dict<int, float>,
         *                 cslt::DictDeleter<int, float>> d(r.value());
         *
         * d->insert(1, 1.0f);
         * d->insert(2, 2.0f);
         * size_t occ = d->size();   // <= 2 (may be 1 if both keys hash to same bucket)
         * @endcode
         */
        size_t size() const noexcept { return len_; }
     
        /**
         * @brief Total number of key-value pairs stored
         *
         * Incremented by insert() and decremented by pop().  Reset to 0 by
         * clear().  This is the value to use when you need to know how many
         * distinct keys are in the dict.
         *
         * @code{.cpp}
         * cslt::HeapAllocator alloc;
         * auto r = cslt::Dict<int, float>::init(8, true, alloc);
         * if (!r.hasValue()) return;
         * cslt::UniquePtr<cslt::Dict<int, float>,
         *                 cslt::DictDeleter<int, float>> d(r.value());
         *
         * d->insert(1, 1.0f);
         * d->insert(2, 2.0f);
         * size_t n = d->hash_size();   // n == 2
         * @endcode
         */
        size_t hash_size() const noexcept { return hash_size_; }
     
        /**
         * @brief Number of buckets allocated (always a power of two)
         *
         * This is the current size of the internal bucket array.  It grows
         * automatically when growth is enabled and the load factor exceeds 0.75.
         * The initial value is the smallest power of two >= the capacity passed
         * to init(), with a minimum of 8.
         *
         * @code{.cpp}
         * cslt::HeapAllocator alloc;
         * auto r = cslt::Dict<int, float>::init(10, true, alloc);
         * if (!r.hasValue()) return;
         * cslt::UniquePtr<cslt::Dict<int, float>,
         *                 cslt::DictDeleter<int, float>> d(r.value());
         *
         * size_t bc = d->bucket_count();   // bc == 16 (next power of two >= 10)
         * @endcode
         */
        size_t bucket_count() const noexcept { return alloc_; }
     
        /**
         * @brief true if no key-value pairs are stored
         *
         * Equivalent to hash_size() == 0.  Returns true on a freshly initialised
         * dict and after clear().
         *
         * @code{.cpp}
         * cslt::HeapAllocator alloc;
         * auto r = cslt::Dict<int, float>::init(8, true, alloc);
         * if (!r.hasValue()) return;
         * cslt::UniquePtr<cslt::Dict<int, float>,
         *                 cslt::DictDeleter<int, float>> d(r.value());
         *
         * bool empty_before = d->is_empty();   // true
         * d->insert(1, 1.0f);
         * bool empty_after  = d->is_empty();   // false
         * d->clear();
         * bool empty_again  = d->is_empty();   // true
         * @endcode
         */
        bool is_empty() const noexcept { return hash_size_ == 0u; }
    };
    // ================================================================================
    // ================================================================================
     
    /**
     * @class DictDeleter
     * @brief Custom deleter for Dict instances, for use with UniquePtr
     *
     * @details Calls the Dict destructor (which frees all nodes and the bucket
     *          array) then returns the Dict struct memory to the allocator.
     *
     * @tparam K Key type
     * @tparam V Value type
     *
     * @code{.cpp}
     * cslt::UniquePtr<cslt::Dict<int,float>,
     *                 cslt::DictDeleter<int,float>> d(r.value());
     * @endcode
     */
    template <typename K, typename V>
    class DictDeleter {
    public:
        void operator()(Dict<K,V>* d) const noexcept {
            if (!d) return;
            Allocator* allocator = d->allocator_;
            d->~Dict<K,V>();
            if (allocator) {
                allocator->return_element(static_cast<void*>(d),
                                          sizeof(Dict<K,V>),
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
