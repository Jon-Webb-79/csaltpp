// ================================================================================
// ================================================================================
// - File:    tree.hpp
// - Purpose: Describe the file purpose here
//
// Source Metadata
// - Author:  Jonathan A. Webb
// - Date:    April 02, 2026
// - Version: 1.0
// - Copyright: Copyright 2026, Jon Webb Inc.
// ================================================================================
// ================================================================================
// Include modules here

#ifndef tree_HPP
#define tree_HPP

#include "error.hpp"
#include "allocator.hpp"

#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>
#include <utility>
// ================================================================================ 
// ================================================================================ 

namespace cslt {

// ================================================================================
// ================================================================================

    /**
     * @brief Default three-way comparator for scalar-like types.
     *
     * Returns:
     * - negative when lhs < rhs
     * - zero     when lhs == rhs
     * - positive when lhs > rhs
     *
     * This matches the comparator style used in the C AVL implementation.
     */
    template <typename T>
    struct DefaultCompare {
        int operator()(const T& lhs, const T& rhs) const noexcept {
            return (lhs > rhs) - (lhs < rhs);
        }
    };

    // ================================================================================
    // Forward declaration for deleter
    // ================================================================================
    template <typename T, typename Compare>
    class AVLTree;

    template <typename T, typename Compare = DefaultCompare<T>>
    class AVLTreeDeleter;

    // ================================================================================
    // ================================================================================

    /**
     * @class AVLTree
     * @brief Allocator-backed generic AVL tree container
     *
     * @details
     * Provides a self-balancing binary search tree that stores values of type `T`
     * inline within each node. The tree object itself, its primary slab of nodes,
     * and any overflow nodes are all allocated through the supplied allocator.
     *
     * Key features:
     * - Custom allocator support
     * - AVL balancing with O(log n) insert / remove / find
     * - Slab-backed primary node storage
     * - Internal free-list reuse for removed nodes
     * - Optional overflow allocation when the slab is exhausted
     * - Comparator-driven ordering
     * - Optional duplicate support
     * - RAII cleanup through AVLTreeDeleter
     *
     * Usage pattern:
     * - Must be initialised via the static factory function init()
     * - Cannot be constructed directly (private constructor)
     * - Cleaned up through AVLTreeDeleter
     *
     * @tparam T        Stored element type
     * @tparam Compare  Comparator type returning negative / zero / positive
     */
    template <typename T, typename Compare>
    class AVLTree {
    private:
        // ============================================================================
        // Node
        // ============================================================================
        struct Node {
            Node* left;     ///< Left child, or next free-list entry when recycled
            Node* right;    ///< Right child
            int   height;   ///< Cached subtree height
            T     value;    ///< Inline stored value
        };

        // ============================================================================
        // Members
        // ============================================================================
        Node*      root_;               ///< Root node of the AVL tree
        Node*      slab_;               ///< Primary slab allocation
        Node*      free_list_;          ///< Recycled-node free list
        size_t     slab_cap_;           ///< Number of nodes in the slab
        size_t     slab_used_;          ///< Number of slab slots ever carved
        size_t     len_;                ///< Number of active elements in the tree
        bool       overflow_;           ///< Allow overflow node allocation
        bool       allow_duplicates_;   ///< Allow equal keys to be inserted
        Allocator* allocator_;          ///< Allocator used for all memory operations
        Compare    compare_;            ///< Comparator used for ordering

        // ============================================================================
        // Private helpers
        // ============================================================================
        static bool _is_no_error(const Error& err) noexcept {
            return std::strcmp(err.what(), "No Error") == 0;
        }

        /**
         * @brief Return the cached height of a node, or 0 for nullptr.
         */
        static int _height(const Node* n) noexcept {
            return n ? n->height : 0;
        }
    // --------------------------------------------------------------------------------

        /**
         * @brief Recompute the cached height of a node from its children.
         */
        static void _update_height(Node* n) noexcept {
            int lh = _height(n->left);
            int rh = _height(n->right);
            n->height = 1 + ((lh > rh) ? lh : rh);
        }
    // --------------------------------------------------------------------------------

        /**
         * @brief Return the AVL balance factor of a node.
         *
         * Defined as height(right) - height(left).
         */
        static int _balance(const Node* n) noexcept {
            return _height(n->right) - _height(n->left);
        }
    // --------------------------------------------------------------------------------

        /**
         * @brief Destroy the value stored in a node.
         *
         * Used when removing active nodes or clearing the tree.
         */
        static void _destroy_value(Node* n) noexcept {
            n->value.~T();
        }
    // --------------------------------------------------------------------------------

        /**
         * @brief Return true if a node lies outside the slab allocation.
         *
         * Such nodes are overflow nodes and must be individually returned
         * to the allocator.
         */
        bool _is_overflow_node(const Node* node) const noexcept {
            if (!slab_) return true;

            const uint8_t* slab_begin = reinterpret_cast<const uint8_t*>(slab_);
            const uint8_t* slab_end   = slab_begin + slab_cap_ * sizeof(Node);
            const uint8_t* ptr        = reinterpret_cast<const uint8_t*>(node);

            return (ptr < slab_begin || ptr >= slab_end);
        }
    // --------------------------------------------------------------------------------

        /**
         * @brief Allocate one node using free-list, slab, or overflow allocation.
         *
         * Constructs `value` in-place.
         *
         * @return Pointer to a live node on success, nullptr on failure.
         */
        Node* _alloc_node(const T& value) noexcept {
            Node* node = nullptr;

            // Reuse a recycled node first
            if (free_list_ != nullptr) {
                node       = free_list_;
                free_list_ = free_list_->left;

                node->left   = nullptr;
                node->right  = nullptr;
                node->height = 1;
                new (static_cast<void*>(&node->value)) T(value);
                return node;
            }

            // Then carve from the slab
            if (slab_used_ < slab_cap_) {
                node = slab_ + slab_used_;
                ++slab_used_;

                node->left   = nullptr;
                node->right  = nullptr;
                node->height = 1;
                new (static_cast<void*>(&node->value)) T(value);
                return node;
            }

            // Finally allocate overflow node if enabled
            if (overflow_) {
                auto obj_result = allocator_->alloc(sizeof(Node), true);
                if (!obj_result.hasValue()) {
                    return nullptr;
                }

                node = static_cast<Node*>(obj_result.value());
                node->left   = nullptr;
                node->right  = nullptr;
                node->height = 1;
                new (static_cast<void*>(&node->value)) T(value);
                return node;
            }

            return nullptr;
        }
    // --------------------------------------------------------------------------------

        /**
         * @brief Return a removed node to the free list.
         *
         * The value must already have been destroyed by the caller.
         */
        void _free_node(Node* node) noexcept {
            node->left  = free_list_;
            node->right = nullptr;
            node->height = 0;
            free_list_  = node;
        }
    // --------------------------------------------------------------------------------

        /**
         * @brief Right rotation.
         */
        static Node* _rotate_right(Node* y) noexcept {
            Node* x = y->left;
            Node* B = x->right;

            x->right = y;
            y->left  = B;

            _update_height(y);
            _update_height(x);
            return x;
        }
    // --------------------------------------------------------------------------------

        /**
         * @brief Left rotation.
         */
        static Node* _rotate_left(Node* x) noexcept {
            Node* y = x->right;
            Node* B = y->left;

            y->left  = x;
            x->right = B;

            _update_height(x);
            _update_height(y);
            return y;
        }
    // --------------------------------------------------------------------------------

        /**
         * @brief Rebalance a subtree if required.
         */
        static Node* _rebalance(Node* n) noexcept {
            _update_height(n);
            int bf = _balance(n);

            // Left-heavy
            if (bf < -1) {
                if (_balance(n->left) > 0) {
                    n->left = _rotate_left(n->left);
                }
                return _rotate_right(n);
            }

            // Right-heavy
            if (bf > 1) {
                if (_balance(n->right) < 0) {
                    n->right = _rotate_right(n->right);
                }
                return _rotate_left(n);
            }

            return n;
        }
    // --------------------------------------------------------------------------------

        /**
         * @brief Recursive insert helper.
         *
         * @param node Existing subtree root
         * @param value Value to insert
         * @param err Error output
         * @return New subtree root
         */
        Node* _insert(Node* node, const T& value, Error& err) noexcept {
            if (node == nullptr) {
                Node* fresh = _alloc_node(value);
                if (fresh == nullptr) {
                    if (overflow_) {
                        err = OutOfMemoryError(
                            "AVLTree::insert: overflow allocation failed"
                        );
                    } else {
                        err = CapacityOverflowError(
                            "AVLTree::insert: slab capacity exceeded"
                        );
                    }
                    return nullptr;
                }

                ++len_;
                return fresh;
            }

            int cmp = compare_(value, node->value);

            if (cmp < 0) {
                node->left = _insert(node->left, value, err);
            } else if (cmp > 0 || allow_duplicates_) {
                node->right = _insert(node->right, value, err);
            } else {
                err = InvalidArgError("AVLTree::insert: duplicate value not allowed");
                return node;
            }

            if (!_is_no_error(err)) {
                return node;
            }

            return _rebalance(node);
        } 
    // --------------------------------------------------------------------------------

        /**
         * @brief Detach the leftmost node in a subtree.
         *
         * Used to find the in-order successor during two-child removal.
         */
        static Node* _detach_min(Node* node, Node*& min_out) noexcept {
            if (node->left == nullptr) {
                min_out = node;
                return node->right;
            }

            node->left = _detach_min(node->left, min_out);
            return _rebalance(node);
        }
    // --------------------------------------------------------------------------------

        /**
         * @brief Recursive remove helper.
         *
         * @param node Existing subtree root
         * @param key Value to remove
         * @param removed Receives detached node on success
         * @param err Error output
         * @return New subtree root
         */
        Node* _remove(Node* node, const T& key, Node*& removed, Error& err) noexcept {
            if (node == nullptr) {
                err = NotFoundError("AVLTree::remove: value not found");
                return nullptr;
            }

            int cmp = compare_(key, node->value);

            if (cmp < 0) {
                node->left = _remove(node->left, key, removed, err);
            } else if (cmp > 0) {
                node->right = _remove(node->right, key, removed, err);
            } else {
                removed = node;

                if (node->left == nullptr) {
                    return node->right;
                }

                if (node->right == nullptr) {
                    return node->left;
                }

                Node* successor = nullptr;
                Node* new_right = _detach_min(node->right, successor);

                successor->left  = node->left;
                successor->right = new_right;
                return _rebalance(successor);
            }

            if (!_is_no_error(err)) {
                return node;
            }

            return _rebalance(node);
        } 
    // --------------------------------------------------------------------------------

        /**
         * @brief Find a node matching key.
         */
        const Node* _find_node(const T& key) const noexcept {
            const Node* node = root_;
            while (node != nullptr) {
                int cmp = compare_(key, node->value);
                if (cmp < 0) {
                    node = node->left;
                } else if (cmp > 0) {
                    node = node->right;
                } else {
                    return node;
                }
            }
            return nullptr;
        }
    // --------------------------------------------------------------------------------

        /**
         * @brief Mutable pre-order traversal over active tree nodes.
         */
        template <typename Fn>
        static void _preorder(Node* node, Fn&& fn) noexcept {
            if (node == nullptr) return;

            Node* left  = node->left;
            Node* right = node->right;

            fn(node);
            _preorder(left,  static_cast<Fn&&>(fn));
            _preorder(right, static_cast<Fn&&>(fn));
        }
    // --------------------------------------------------------------------------------

        /**
         * @brief Const in-order traversal over active tree nodes.
         */
        template <typename Fn>
        static void _inorder(const Node* node, Fn&& fn) noexcept {
            if (node == nullptr) return;

            _inorder(node->left, static_cast<Fn&&>(fn));
            fn(node->value);
            _inorder(node->right, static_cast<Fn&&>(fn));
        }
    // --------------------------------------------------------------------------------

        /**
         * @brief Selective in-order traversal within inclusive range [low, high].
         */
        template <typename Fn>
        static void _inorder_range(const Node* node,
                                   const T& low,
                                   const T& high,
                                   const Compare& compare,
                                   Fn&& fn) noexcept {
            if (node == nullptr) return;

            int cmp_low  = compare(low,  node->value);
            int cmp_high = compare(high, node->value);

            if (cmp_low < 0) {
                _inorder_range(node->left,
                               low,
                               high,
                               compare,
                               static_cast<Fn&&>(fn));
            }

            if (cmp_low <= 0 && cmp_high >= 0) {
                fn(node->value);
            }

            if (cmp_high > 0) {
                _inorder_range(node->right,
                               low,
                               high,
                               compare,
                               static_cast<Fn&&>(fn));
            }
        }
    // --------------------------------------------------------------------------------

        /**
         * @brief Private constructor.
         *
         * Allocates the slab through the allocator. The tree object itself is
         * allocated separately by init() using placement new.
         */
        AVLTree(size_t capacity,
                bool overflow,
                bool allow_duplicates,
                Allocator& allocator,
                const Compare& compare) noexcept
            : root_(nullptr),
              slab_(nullptr),
              free_list_(nullptr),
              slab_cap_(capacity),
              slab_used_(0u),
              len_(0u),
              overflow_(overflow),
              allow_duplicates_(allow_duplicates),
              allocator_(&allocator),
              compare_(compare) {
            auto slab_result = allocator.alloc(sizeof(Node) * capacity, true);
            if (slab_result.hasValue()) {
                slab_ = static_cast<Node*>(slab_result.value());
            }
        }
    // --------------------------------------------------------------------------------

        /**
         * @brief Private destructor.
         *
         * Cleans active nodes and overflow allocations. The tree object itself
         * is freed by AVLTreeDeleter.
         */
        ~AVLTree() noexcept {
            clear();

            if (slab_ && allocator_) {
                allocator_->return_element(static_cast<void*>(slab_),
                                           slab_cap_ * sizeof(Node),
                                           allocator_->default_alignment());
                slab_ = nullptr;
            }
        }
    // --------------------------------------------------------------------------------

        AVLTree(const AVLTree&)            = delete;
        AVLTree& operator=(const AVLTree&) = delete;
        AVLTree(AVLTree&&)                 = delete;
        AVLTree& operator=(AVLTree&&)      = delete;

    public:
        // ============================================================================
        // Factory
        // ============================================================================

        /**
         * @brief Initialise an allocator-backed AVL tree.
         *
         * @param capacity         Number of primary slab nodes (must be > 0)
         * @param allocator        Allocator used for all tree memory
         * @param overflow         If true, allocate overflow nodes beyond the slab
         * @param allow_duplicates If true, equal values are inserted on the right
         * @param compare          Comparator for tree ordering
         *
         * @return Expected<AVLTree*> containing a pointer to the tree or an error
         */
        static Expected<AVLTree*> init(size_t capacity,
                                       Allocator& allocator,
                                       bool overflow = true,
                                       bool allow_duplicates = false,
                                       const Compare& compare = Compare{}) noexcept {
            Expected<AVLTree*> result;

            if (capacity == 0u) {
                result.setError(ArgumentError("AVLTree::init: capacity must be > 0"));
                return result;
            }

            auto obj_result = allocator.alloc(sizeof(AVLTree), true);
            if (!obj_result.hasValue()) {
                result.setError(obj_result.error());
                return result;
            }

            AVLTree* tree = new (obj_result.value())
                AVLTree(capacity, overflow, allow_duplicates, allocator, compare);

            if (!tree->slab_) {
                tree->~AVLTree();
                allocator.return_element(obj_result.value(),
                                         sizeof(AVLTree),
                                         allocator.default_alignment());
                result.setError(MemoryError("AVLTree::init: failed to allocate node slab"));
                return result;
            }

            result.setValue(tree);
            return result;
        }
    // --------------------------------------------------------------------------------

        /**
         * @brief Create a deep copy of an existing AVL tree using a caller-supplied
         * allocator.
         */
        static Expected<AVLTree*> copy(const AVLTree& src,
                                       Allocator& allocator) noexcept {
            Expected<AVLTree*> result;

            size_t cap = (src.len_ > 0u) ? src.len_ : 1u;
            auto init_result = AVLTree::init(cap,
                                             allocator,
                                             src.overflow_,
                                             src.allow_duplicates_,
                                             src.compare_);
            if (!init_result.hasValue()) {
                result.setError(init_result.error());
                return result;
            }

            AVLTree* dst = init_result.value();

            Error err = NoError();
            _preorder(src.root_, [dst, &err](Node* n) noexcept {
                if (!dst->_is_no_error(err)) return;
                err = dst->insert(n->value);
            });

            if (!dst->_is_no_error(err)) {
                AVLTreeDeleter<T, Compare>{}(dst);
                result.setError(err);
                return result;
            }

            result.setValue(dst);
            return result;
        } 
    // --------------------------------------------------------------------------------

        /**
         * @brief Remove all elements from the tree.
         *
         * Active node values are destroyed. Overflow nodes are individually returned
         * to the allocator. Recycled free-list nodes are also released if they are
         * overflow nodes. Slab memory remains allocated for reuse by the tree.
         */
        void clear() noexcept {
            // Destroy active tree nodes
            _preorder(root_, [this](Node* n) noexcept {
                _destroy_value(n);
                if (_is_overflow_node(n)) {
                    allocator_->return_element(static_cast<void*>(n),
                                               sizeof(Node),
                                               allocator_->default_alignment());
                }
            });

            // Free recycled overflow nodes from the free list.
            // Their values were already destroyed at removal time.
            Node* fl = free_list_;
            while (fl != nullptr) {
                Node* next = fl->left;
                if (_is_overflow_node(fl)) {
                    allocator_->return_element(static_cast<void*>(fl),
                                               sizeof(Node),
                                               allocator_->default_alignment());
                }
                fl = next;
            }

            root_      = nullptr;
            free_list_ = nullptr;
            slab_used_ = 0u;
            len_       = 0u;
        }
    // --------------------------------------------------------------------------------

        /**
         * @brief Insert a value into the tree.
         *
         * @return NoError on success, or an appropriate error object.
         */
        Error insert(const T& value) noexcept {
            Error err = NoError();
            root_ = _insert(root_, value, err);
            return err;
        }
    // --------------------------------------------------------------------------------

        /**
         * @brief Remove a value from the tree.
         *
         * @return Expected<T> containing the removed value or an error.
         */
        Expected<T> remove(const T& key) noexcept {
            Expected<T> result;

            if (len_ == 0u) {
                result.setError(EmptyError("AVLTree::remove: tree is empty"));
                return result;
            }

            Error err = NoError();
            Node* removed = nullptr;

            root_ = _remove(root_, key, removed, err);

            if (!_is_no_error(err)) {
                result.setError(err);
                return result;
            }

            T out = removed->value;
            _destroy_value(removed);
            _free_node(removed);
            --len_;

            result.setValue(cslt::move(out));
            return result;
        } 
    // --------------------------------------------------------------------------------

        /**
         * @brief Return true if the tree contains key.
         */
        bool contains(const T& key) const noexcept {
            return _find_node(key) != nullptr;
        }
    // --------------------------------------------------------------------------------

        /**
         * @brief Find a value equal to key.
         */
        Expected<T> find(const T& key) const noexcept {
            Expected<T> result;

            if (len_ == 0u) {
                result.setError(EmptyError("AVLTree::find: tree is empty"));
                return result;
            }

            const Node* node = _find_node(key);
            if (!node) {
                result.setError(NotFoundError("AVLTree::find: value not found"));
                return result;
            }

            result.setValue(node->value);
            return result;
        }
    // --------------------------------------------------------------------------------

        /**
         * @brief Get the minimum value in the tree.
         */
        Expected<T> min() const noexcept {
            Expected<T> result;

            if (len_ == 0u) {
                result.setError(EmptyError("AVLTree::min: tree is empty"));
                return result;
            }

            const Node* node = root_;
            while (node->left != nullptr) node = node->left;

            result.setValue(node->value);
            return result;
        }
    // --------------------------------------------------------------------------------

        /**
         * @brief Get the maximum value in the tree.
         */
        Expected<T> max() const noexcept {
            Expected<T> result;

            if (len_ == 0u) {
                result.setError(EmptyError("AVLTree::max: tree is empty"));
                return result;
            }

            const Node* node = root_;
            while (node->right != nullptr) node = node->right;

            result.setValue(node->value);
            return result;
        }
    // --------------------------------------------------------------------------------

        /**
         * @brief Visit every element in sorted order.
         *
         * @tparam Fn Callable taking `const T&`
         * @return NoError on success, or EmptyError if the tree is empty.
         */
        template <typename Fn>
        Error foreach(Fn&& fn) const noexcept {
            if (len_ == 0u) {
                return EmptyError("AVLTree::foreach: tree is empty");
            }

            _inorder(root_, static_cast<Fn&&>(fn));
            return NoError();
        }
    // --------------------------------------------------------------------------------

        /**
         * @brief Visit every element in inclusive range [low, high] in sorted order.
         *
         * @tparam Fn Callable taking `const T&`
         * @return NoError on success, or an error if the range is invalid or tree empty.
         */
        template <typename Fn>
        Error foreach_range(const T& low,
                            const T& high,
                            Fn&& fn) const noexcept {
            if (len_ == 0u) {
                return EmptyError("AVLTree::foreach_range: tree is empty");
            }

            if (compare_(low, high) > 0) {
                return InvalidArgError("AVLTree::foreach_range: low must be <= high");
            }

            _inorder_range(root_, low, high, compare_, static_cast<Fn&&>(fn));
            return NoError();
        }
    // --------------------------------------------------------------------------------

        /**
         * @brief Number of active elements in the tree.
         */
        size_t size() const noexcept {
            return len_;
        }
    // --------------------------------------------------------------------------------

        /**
         * @brief Return true if the tree is empty.
         */
        bool empty() const noexcept {
            return len_ == 0u;
        }
    // --------------------------------------------------------------------------------

        /**
         * @brief Return the height of the tree.
         */
        int height() const noexcept {
            return _height(root_);
        }
    // --------------------------------------------------------------------------------

        /**
         * @brief Number of nodes in the primary slab allocation.
         */
        size_t slab_capacity() const noexcept {
            return slab_cap_;
        }
    // --------------------------------------------------------------------------------

        /**
         * @brief Number of slab slots ever carved since last clear().
         */
        size_t slab_used() const noexcept {
            return slab_used_;
        }
    // --------------------------------------------------------------------------------

        /**
         * @brief Return whether overflow allocation is enabled.
         */
        bool overflow_enabled() const noexcept {
            return overflow_;
        }
    // --------------------------------------------------------------------------------

        /**
         * @brief Return whether duplicates are allowed.
         */
        bool duplicates_allowed() const noexcept {
            return allow_duplicates_;
        }

        template <typename U, typename C>
        friend class AVLTreeDeleter;
    };

    // ================================================================================
    // ================================================================================

    /**
     * @class AVLTreeDeleter
     * @brief Custom deleter for AVLTree instances
     *
     * @details
     * Properly destroys the tree, frees overflow nodes and slab memory via the
     * tree destructor, then returns the AVLTree object itself using its allocator.
     */
    template <typename T, typename Compare>
    class AVLTreeDeleter {
    public:
        void operator()(AVLTree<T, Compare>* tree) const noexcept {
            if (!tree) return;

            Allocator* allocator = tree->allocator_;
            tree->~AVLTree<T, Compare>();

            if (allocator) {
                allocator->return_element(static_cast<void*>(tree),
                                          sizeof(AVLTree<T, Compare>),
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
