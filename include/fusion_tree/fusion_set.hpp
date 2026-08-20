#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <type_traits>
#include <utility>

#if defined(__BMI2__) && (defined(__x86_64__) || defined(__amd64__) || defined(_M_X64))
#include <immintrin.h>
#endif

namespace fusion_tree {

namespace detail {

// A fusion node stores the compressed sketches of all keys in guarded lanes.
// Subtracting a replicated query sketch compares every lane simultaneously;
// the guard bits then encode the lower-bound result.
template <std::unsigned_integral Key, std::size_t Capacity> class fusion_index {
  static_assert(Capacity >= 1 && Capacity <= 8);

public:
  void rebuild(std::span<const Key> keys) noexcept {
    assert(keys.size() <= Capacity);

    size_ = static_cast<std::uint8_t>(keys.size());
    relevant_mask_ = 0;
    packed_ = 0;
    repeat_mask_ = 0;

    for (std::size_t i = 1; i < keys.size(); ++i) {
      assert(keys[i - 1] < keys[i]);
      const auto difference = static_cast<Key>(keys[i - 1] ^ keys[i]);
      assert(difference != 0);
      const auto position = static_cast<unsigned>(std::bit_width(difference)) - 1U;
      relevant_mask_ |= std::uint64_t{1} << position;
    }

    relevant_count_ = static_cast<std::uint8_t>(std::popcount(relevant_mask_));
    lane_width_ = static_cast<std::uint8_t>(relevant_count_ + 1U);

    const auto guard = std::uint64_t{1} << relevant_count_;
    for (std::size_t i = 0; i < keys.size(); ++i) {
      const auto shift = static_cast<unsigned>(i * lane_width_);
      packed_ |= (guard | sketch(keys[i])) << shift;
      repeat_mask_ |= std::uint64_t{1} << shift;
    }
  }

  [[nodiscard]] std::size_t lower_bound(std::span<const Key> keys, Key query) const noexcept {
    assert(keys.size() == size_);
    if (keys.empty()) {
      return 0;
    }

    const auto approximate = sketch_lower_bound(sketch(query));
    std::size_t closest = 0;
    if (approximate == keys.size()) {
      closest = keys.size() - 1U;
    } else if (approximate == 0) {
      closest = 0;
    } else {
      const auto left_difference = static_cast<Key>(query ^ keys[approximate - 1U]);
      const auto right_difference = static_cast<Key>(query ^ keys[approximate]);
      closest = std::bit_width(left_difference) <= std::bit_width(right_difference)
                    ? approximate - 1U
                    : approximate;
    }

    if (keys[closest] == query) {
      return closest;
    }

    const auto difference = static_cast<Key>(query ^ keys[closest]);
    const auto branching_bit = static_cast<unsigned>(std::bit_width(difference)) - 1U;
    const auto lower_bits = static_cast<Key>((std::uint64_t{1} << branching_bit) - 1U);

    if (query < keys[closest]) {
      const auto minimum_on_query_side = static_cast<Key>(query & ~lower_bits);
      return sketch_lower_bound(sketch(minimum_on_query_side));
    }

    const auto maximum_on_query_side = static_cast<Key>(query | lower_bits);
    const auto maximum_sketch = sketch(maximum_on_query_side);
    auto position = sketch_lower_bound(maximum_sketch);
    if (position != keys.size() && sketch(keys[position]) == maximum_sketch) {
      ++position;
    }
    return position;
  }

  [[nodiscard]] std::size_t upper_bound(std::span<const Key> keys, Key query) const noexcept {
    assert(keys.size() == size_);
    auto position = lower_bound(keys, query);
    if (position != keys.size() && keys[position] == query) {
      ++position;
    }
    return position;
  }

  friend bool operator==(const fusion_index &, const fusion_index &) noexcept = default;

private:
  [[nodiscard]] std::uint64_t sketch(Key key) const noexcept {
#if defined(__BMI2__) && (defined(__x86_64__) || defined(__amd64__) || defined(_M_X64))
    return _pext_u64(static_cast<std::uint64_t>(key), relevant_mask_);
#else
    auto mask = relevant_mask_;
    std::uint64_t result = 0;
    std::uint64_t output_bit = 1;
    while (mask != 0) {
      const auto input_bit = mask & (~mask + 1U);
      result |= (static_cast<std::uint64_t>(key) & input_bit) != 0 ? output_bit : 0;
      mask &= mask - 1U;
      output_bit <<= 1U;
    }
    return result;
#endif
  }

  [[nodiscard]] std::size_t sketch_lower_bound(std::uint64_t query_sketch) const noexcept {
    if (size_ == 0) {
      return 0;
    }

    const auto differences = packed_ - query_sketch * repeat_mask_;
    const auto greater_or_equal = differences & (repeat_mask_ << relevant_count_);
    if (greater_or_equal == 0) {
      return size_;
    }
    return static_cast<std::size_t>(std::countr_zero(greater_or_equal)) /
           static_cast<std::size_t>(lane_width_);
  }

  std::uint64_t relevant_mask_{};
  std::uint64_t packed_{};
  std::uint64_t repeat_mask_{};
  std::uint8_t size_{};
  std::uint8_t relevant_count_{};
  std::uint8_t lane_width_{1};
};

} // namespace detail

// A dynamic fusion tree for machine-word, unsigned integer keys. The leaves and
// internal nodes form an allocator-aware B+ tree; every in-node search is
// accelerated by a portable broadword fusion index.
template <std::unsigned_integral Key, std::size_t Branching = 8,
          class Allocator = std::allocator<Key>>
class fusion_set {
  static_assert(!std::same_as<Key, bool>, "bool is not a useful fusion-tree key");
  static_assert(std::numeric_limits<Key>::digits <= 64,
                "the portable fusion node supports keys up to 64 bits");
  static_assert(Branching >= 3 && Branching <= 8,
                "the packed portable fusion node supports branching factors 3..8");
  static_assert(std::same_as<typename std::allocator_traits<Allocator>::value_type, Key>,
                "Allocator::value_type must be the key type");

  struct internal_node;

  struct node {
    explicit constexpr node(bool leaf) noexcept : is_leaf(leaf) {}

    detail::fusion_index<Key, Branching> index{};
    internal_node *parent{};
    Key minimum{};
    std::uint8_t count{};
    bool is_leaf{};
  };

  struct leaf_node final : node {
    constexpr leaf_node() noexcept : node(true) {}

    std::array<Key, Branching + 1> keys{};
    leaf_node *previous{};
    leaf_node *next{};
  };

  struct internal_node final : node {
    constexpr internal_node() noexcept : node(false) {}

    std::array<Key, Branching> keys{};
    std::array<node *, Branching + 1> children{};
  };

  using allocator_traits = std::allocator_traits<Allocator>;
  using leaf_allocator = typename allocator_traits::template rebind_alloc<leaf_node>;
  using leaf_allocator_traits = std::allocator_traits<leaf_allocator>;
  using internal_allocator = typename allocator_traits::template rebind_alloc<internal_node>;
  using internal_allocator_traits = std::allocator_traits<internal_allocator>;

public:
  using key_type = Key;
  using value_type = Key;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using allocator_type = Allocator;
  using reference = const value_type &;
  using const_reference = const value_type &;

  class const_iterator {
    friend class fusion_set;

  public:
    using iterator_category = std::bidirectional_iterator_tag;
    using iterator_concept = std::bidirectional_iterator_tag;
    using value_type = Key;
    using difference_type = std::ptrdiff_t;
    using pointer = const Key *;
    using reference = const Key &;

    constexpr const_iterator() noexcept = default;

    [[nodiscard]] constexpr reference operator*() const noexcept {
      assert(leaf_ != nullptr && position_ < leaf_->count);
      return leaf_->keys[position_];
    }

    [[nodiscard]] constexpr pointer operator->() const noexcept {
      return std::addressof(operator*());
    }

    constexpr const_iterator &operator++() noexcept {
      assert(leaf_ != nullptr);
      ++position_;
      if (position_ == leaf_->count) {
        leaf_ = leaf_->next;
        position_ = 0;
      }
      return *this;
    }

    constexpr const_iterator operator++(int) noexcept {
      auto copy = *this;
      ++*this;
      return copy;
    }

    constexpr const_iterator &operator--() noexcept {
      assert(owner_ != nullptr);
      if (leaf_ == nullptr) {
        leaf_ = owner_->last_leaf_;
        assert(leaf_ != nullptr);
        position_ = leaf_->count - 1U;
      } else if (position_ != 0) {
        --position_;
      } else {
        leaf_ = leaf_->previous;
        assert(leaf_ != nullptr);
        position_ = leaf_->count - 1U;
      }
      return *this;
    }

    constexpr const_iterator operator--(int) noexcept {
      auto copy = *this;
      --*this;
      return copy;
    }

    friend constexpr bool operator==(const const_iterator &,
                                     const const_iterator &) noexcept = default;

  private:
    constexpr const_iterator(const fusion_set *owner, const leaf_node *leaf,
                             size_type position) noexcept
        : owner_(owner), leaf_(leaf), position_(position) {}

    const fusion_set *owner_{};
    const leaf_node *leaf_{};
    size_type position_{};
  };

  using iterator = const_iterator;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;
  using reverse_iterator = const_reverse_iterator;

  constexpr fusion_set() noexcept(std::is_nothrow_default_constructible_v<Allocator>) = default;

  explicit constexpr fusion_set(const Allocator &allocator) noexcept(
      std::is_nothrow_copy_constructible_v<Allocator>)
      : allocator_(allocator) {}

  template <std::input_iterator Iterator, std::sentinel_for<Iterator> Sentinel>
    requires std::convertible_to<std::iter_reference_t<Iterator>, key_type>
  fusion_set(Iterator first, Sentinel last, const Allocator &allocator = Allocator{})
      : allocator_(allocator) {
    try {
      insert(first, last);
    } catch (...) {
      clear();
      throw;
    }
  }

  fusion_set(std::initializer_list<key_type> values, const Allocator &allocator = Allocator{})
      : fusion_set(values.begin(), values.end(), allocator) {}

  fusion_set(const fusion_set &other)
      : fusion_set(other,
                   allocator_traits::select_on_container_copy_construction(other.allocator_)) {}

  fusion_set(const fusion_set &other, const Allocator &allocator) : allocator_(allocator) {
    try {
      insert(other.begin(), other.end());
    } catch (...) {
      clear();
      throw;
    }
  }

  fusion_set(fusion_set &&other) noexcept(std::is_nothrow_move_constructible_v<Allocator>)
      : allocator_(std::move(other.allocator_)) {
    steal_from(other);
  }

  fusion_set(fusion_set &&other, const Allocator &allocator) : allocator_(allocator) {
    if constexpr (allocator_traits::is_always_equal::value) {
      steal_from(other);
    } else if (allocator_ == other.allocator_) {
      steal_from(other);
    } else {
      try {
        insert(other.begin(), other.end());
      } catch (...) {
        clear();
        throw;
      }
      other.clear();
    }
  }

  ~fusion_set() { clear(); }

  fusion_set &operator=(const fusion_set &other) {
    if (this == std::addressof(other)) {
      return *this;
    }

    if constexpr (allocator_traits::propagate_on_container_copy_assignment::value) {
      fusion_set replacement(other, other.allocator_);
      clear();
      allocator_ = other.allocator_;
      steal_from(replacement);
    } else {
      fusion_set replacement(other, allocator_);
      clear();
      steal_from(replacement);
    }
    return *this;
  }

  fusion_set &operator=(fusion_set &&other) noexcept(
      allocator_traits::propagate_on_container_move_assignment::value
          ? std::is_nothrow_move_assignable_v<Allocator>
          : allocator_traits::is_always_equal::value) {
    if (this == std::addressof(other)) {
      return *this;
    }

    if constexpr (allocator_traits::propagate_on_container_move_assignment::value) {
      clear();
      allocator_ = std::move(other.allocator_);
      steal_from(other);
    } else if constexpr (allocator_traits::is_always_equal::value) {
      clear();
      steal_from(other);
    } else if (allocator_ == other.allocator_) {
      clear();
      steal_from(other);
    } else {
      fusion_set replacement(other.begin(), other.end(), allocator_);
      clear();
      steal_from(replacement);
      other.clear();
    }
    return *this;
  }

  [[nodiscard]] constexpr allocator_type get_allocator() const
      noexcept(std::is_nothrow_copy_constructible_v<Allocator>) {
    return allocator_;
  }

  [[nodiscard]] constexpr iterator begin() const noexcept { return iterator{this, first_leaf_, 0}; }
  [[nodiscard]] constexpr iterator end() const noexcept { return iterator{this, nullptr, 0}; }
  [[nodiscard]] constexpr const_iterator cbegin() const noexcept { return begin(); }
  [[nodiscard]] constexpr const_iterator cend() const noexcept { return end(); }
  [[nodiscard]] constexpr reverse_iterator rbegin() const noexcept {
    return reverse_iterator{end()};
  }
  [[nodiscard]] constexpr reverse_iterator rend() const noexcept {
    return reverse_iterator{begin()};
  }
  [[nodiscard]] constexpr const_reverse_iterator crbegin() const noexcept { return rbegin(); }
  [[nodiscard]] constexpr const_reverse_iterator crend() const noexcept { return rend(); }

  [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }
  [[nodiscard]] constexpr size_type size() const noexcept { return size_; }
  [[nodiscard]] constexpr size_type max_size() const noexcept {
    return std::numeric_limits<size_type>::max() / sizeof(key_type);
  }

  std::pair<iterator, bool> insert(key_type key) {
    if (root_ == nullptr) {
      leaf_node *new_root = create_leaf();
      new_root->keys[0] = key;
      new_root->count = 1;
      refresh(new_root);
      root_ = first_leaf_ = last_leaf_ = new_root;
      size_ = 1;
      return {iterator{this, new_root, 0}, true};
    }

    leaf_node *leaf = find_leaf(key);
    const auto keys = key_span(leaf);
    const auto position = leaf->index.lower_bound(keys, key);
    if (position != leaf->count && leaf->keys[position] == key) {
      return {iterator{this, leaf, position}, false};
    }

    if (leaf->count < Branching) {
      const bool minimum_changed = position == 0;
      insert_key(leaf, position, key);
      ++size_;
      refresh(leaf);
      if (minimum_changed) {
        propagate_minimum(leaf);
      }
      return {iterator{this, leaf, position}, true};
    }

    return {insert_into_full_leaf(leaf, position, key), true};
  }

  template <std::input_iterator Iterator, std::sentinel_for<Iterator> Sentinel>
    requires std::convertible_to<std::iter_reference_t<Iterator>, key_type>
  void insert(Iterator first, Sentinel last) {
    for (; first != last; ++first) {
      insert(static_cast<key_type>(*first));
    }
  }

  template <std::ranges::input_range Range>
    requires std::convertible_to<std::ranges::range_reference_t<Range>, key_type>
  void insert_range(Range &&range) {
    insert(std::ranges::begin(range), std::ranges::end(range));
  }

  size_type erase(key_type key) noexcept {
    if (root_ == nullptr) {
      return 0;
    }

    leaf_node *leaf = find_leaf(key);
    const auto position = leaf->index.lower_bound(key_span(leaf), key);
    if (position == leaf->count || leaf->keys[position] != key) {
      return 0;
    }

    const bool minimum_changed = position == 0;
    for (size_type i = position + 1U; i < leaf->count; ++i) {
      leaf->keys[i - 1U] = leaf->keys[i];
    }
    --leaf->count;
    --size_;

    if (leaf == root_) {
      if (leaf->count == 0) {
        destroy_node(root_);
        root_ = first_leaf_ = last_leaf_ = nullptr;
      } else {
        refresh(leaf);
      }
      return 1;
    }

    if (leaf->count >= minimum_occupancy) {
      refresh(leaf);
      if (minimum_changed) {
        propagate_minimum(leaf);
      }
    } else {
      rebalance_after_erase(leaf, minimum_changed);
    }
    return 1;
  }

  iterator erase(const_iterator position) noexcept {
    assert(position.owner_ == this && position != end());
    const auto next = std::next(position);
    const auto next_key =
        next == end() ? std::optional<key_type>{} : std::optional<key_type>{*next};
    erase(*position);
    return next_key ? lower_bound(*next_key) : end();
  }

  void clear() noexcept {
    destroy_subtree(root_);
    root_ = first_leaf_ = last_leaf_ = nullptr;
    size_ = 0;
  }

  [[nodiscard]] iterator find(key_type key) const noexcept {
    if (root_ == nullptr) {
      return end();
    }
    const leaf_node *leaf = find_leaf(key);
    const auto position = leaf->index.lower_bound(key_span(leaf), key);
    return position != leaf->count && leaf->keys[position] == key ? iterator{this, leaf, position}
                                                                  : end();
  }

  [[nodiscard]] bool contains(key_type key) const noexcept { return find(key) != end(); }
  [[nodiscard]] size_type count(key_type key) const noexcept { return contains(key) ? 1U : 0U; }

  [[nodiscard]] iterator lower_bound(key_type key) const noexcept {
    if (root_ == nullptr) {
      return end();
    }
    const leaf_node *leaf = find_leaf(key);
    const auto position = leaf->index.lower_bound(key_span(leaf), key);
    return iterator_at(leaf, position);
  }

  [[nodiscard]] iterator upper_bound(key_type key) const noexcept {
    if (root_ == nullptr) {
      return end();
    }
    const leaf_node *leaf = find_leaf(key);
    const auto position = leaf->index.upper_bound(key_span(leaf), key);
    return iterator_at(leaf, position);
  }

  [[nodiscard]] std::pair<iterator, iterator> equal_range(key_type key) const noexcept {
    if (root_ == nullptr) {
      return {end(), end()};
    }
    const leaf_node *leaf = find_leaf(key);
    const auto lower = leaf->index.lower_bound(key_span(leaf), key);
    const auto upper =
        lower + static_cast<size_type>(lower != leaf->count && leaf->keys[lower] == key);
    return {iterator_at(leaf, lower), iterator_at(leaf, upper)};
  }

  // Strict predecessor/successor queries. floor() and ceiling() are the
  // inclusive counterparts.
  [[nodiscard]] iterator predecessor(key_type key) const noexcept {
    auto result = lower_bound(key);
    if (result == begin()) {
      return end();
    }
    return --result;
  }

  [[nodiscard]] iterator successor(key_type key) const noexcept { return upper_bound(key); }

  [[nodiscard]] iterator floor(key_type key) const noexcept {
    auto result = upper_bound(key);
    if (result == begin()) {
      return end();
    }
    return --result;
  }

  [[nodiscard]] iterator ceiling(key_type key) const noexcept { return lower_bound(key); }

  void swap(fusion_set &other) noexcept(allocator_traits::propagate_on_container_swap::value
                                            ? std::is_nothrow_swappable_v<Allocator>
                                            : allocator_traits::is_always_equal::value) {
    using std::swap;
    if constexpr (allocator_traits::propagate_on_container_swap::value) {
      swap(allocator_, other.allocator_);
    } else if constexpr (!allocator_traits::is_always_equal::value) {
      assert(allocator_ == other.allocator_);
    }
    swap(root_, other.root_);
    swap(first_leaf_, other.first_leaf_);
    swap(last_leaf_, other.last_leaf_);
    swap(size_, other.size_);
  }

  // Performs a complete, allocation-free structural check. This is useful for
  // fuzzing, defensive diagnostics, and validating custom allocators.
  [[nodiscard]] bool validate() const noexcept {
    if (root_ == nullptr) {
      return size_ == 0 && first_leaf_ == nullptr && last_leaf_ == nullptr;
    }
    if (size_ == 0 || root_->parent != nullptr || first_leaf_ == nullptr || last_leaf_ == nullptr ||
        first_leaf_->previous != nullptr || last_leaf_->next != nullptr) {
      return false;
    }

    size_type observed_size = 0;
    size_type leaf_depth = std::numeric_limits<size_type>::max();
    const leaf_node *previous_leaf = nullptr;
    std::optional<key_type> previous_key;

    const auto inspect = [&](const auto &self, const node *current, size_type depth,
                             bool is_root) noexcept -> bool {
      if (current == nullptr || current->count == 0 || current->count > Branching) {
        return false;
      }

      if (current->is_leaf) {
        const auto *leaf = static_cast<const leaf_node *>(current);
        if (!is_root && current->count < minimum_occupancy) {
          return false;
        }
        if (leaf_depth == std::numeric_limits<size_type>::max()) {
          leaf_depth = depth;
        } else if (leaf_depth != depth) {
          return false;
        }
        if ((previous_leaf == nullptr && leaf != first_leaf_) || leaf->previous != previous_leaf ||
            (previous_leaf != nullptr && previous_leaf->next != current)) {
          return false;
        }

        detail::fusion_index<Key, Branching> expected;
        expected.rebuild(key_span(leaf));
        if (expected != current->index || current->minimum != leaf->keys[0]) {
          return false;
        }
        for (size_type i = 0; i < current->count; ++i) {
          if (previous_key && *previous_key >= leaf->keys[i]) {
            return false;
          }
          previous_key = leaf->keys[i];
          ++observed_size;
        }
        previous_leaf = leaf;
        return true;
      }

      if ((is_root && current->count < 2U) || (!is_root && current->count < minimum_occupancy)) {
        return false;
      }

      const auto *internal = static_cast<const internal_node *>(current);
      detail::fusion_index<Key, Branching> expected;
      expected.rebuild(internal_key_span(internal));
      if (expected != current->index || current->minimum != internal->children[0]->minimum) {
        return false;
      }

      for (size_type i = 0; i < current->count; ++i) {
        if (internal->children[i] == nullptr || internal->children[i]->parent != current) {
          return false;
        }
        if (i != 0 && internal->keys[i - 1U] != subtree_min(internal->children[i])) {
          return false;
        }
        if (!self(self, internal->children[i], depth + 1U, false)) {
          return false;
        }
      }
      return true;
    };

    return inspect(inspect, root_, 0, true) && observed_size == size_ &&
           previous_leaf == last_leaf_;
  }

  friend bool operator==(const fusion_set &left, const fusion_set &right) noexcept {
    return left.size() == right.size() && std::ranges::equal(left, right);
  }

  friend void swap(fusion_set &left, fusion_set &right) noexcept(noexcept(left.swap(right))) {
    left.swap(right);
  }

private:
  static constexpr size_type minimum_occupancy = (Branching + 1U) / 2U;

  [[nodiscard]] static constexpr std::span<const key_type>
  key_span(const leaf_node *current) noexcept {
    return {current->keys.data(), current->count};
  }

  [[nodiscard]] static constexpr std::span<const key_type>
  internal_key_span(const internal_node *current) noexcept {
    assert(current->count >= 1);
    return {current->keys.data(), static_cast<size_type>(current->count - 1U)};
  }

  [[nodiscard]] constexpr iterator iterator_at(const leaf_node *leaf,
                                               size_type position) const noexcept {
    assert(position <= leaf->count);
    return position != leaf->count ? iterator{this, leaf, position} : iterator{this, leaf->next, 0};
  }

  [[nodiscard]] leaf_node *create_leaf() {
    leaf_allocator allocator{allocator_};
    auto storage = leaf_allocator_traits::allocate(allocator, 1);
    leaf_node *result = std::to_address(storage);
    try {
      leaf_allocator_traits::construct(allocator, result);
    } catch (...) {
      leaf_allocator_traits::deallocate(allocator, storage, 1);
      throw;
    }
    return result;
  }

  [[nodiscard]] internal_node *create_internal() {
    internal_allocator allocator{allocator_};
    auto storage = internal_allocator_traits::allocate(allocator, 1);
    internal_node *result = std::to_address(storage);
    try {
      internal_allocator_traits::construct(allocator, result);
    } catch (...) {
      internal_allocator_traits::deallocate(allocator, storage, 1);
      throw;
    }
    return result;
  }

  void destroy_leaf(leaf_node *current) noexcept {
    leaf_allocator allocator{allocator_};
    const auto storage =
        std::pointer_traits<typename leaf_allocator_traits::pointer>::pointer_to(*current);
    leaf_allocator_traits::destroy(allocator, current);
    leaf_allocator_traits::deallocate(allocator, storage, 1);
  }

  void destroy_internal(internal_node *current) noexcept {
    internal_allocator allocator{allocator_};
    const auto storage =
        std::pointer_traits<typename internal_allocator_traits::pointer>::pointer_to(*current);
    internal_allocator_traits::destroy(allocator, current);
    internal_allocator_traits::deallocate(allocator, storage, 1);
  }

  void destroy_node(node *current) noexcept {
    if (current->is_leaf) {
      destroy_leaf(static_cast<leaf_node *>(current));
    } else {
      destroy_internal(static_cast<internal_node *>(current));
    }
  }

  void destroy_subtree(node *current) noexcept {
    if (current == nullptr) {
      return;
    }
    if (!current->is_leaf) {
      auto *internal = static_cast<internal_node *>(current);
      for (size_type i = 0; i < current->count; ++i) {
        destroy_subtree(internal->children[i]);
      }
    }
    destroy_node(current);
  }

  [[nodiscard]] static constexpr key_type subtree_min(const node *current) noexcept {
    assert(current != nullptr && current->count != 0);
    return current->minimum;
  }

  static void refresh(node *current) noexcept {
    if (current->is_leaf) {
      auto *leaf = static_cast<leaf_node *>(current);
      current->minimum = leaf->keys[0];
      current->index.rebuild(key_span(leaf));
      return;
    }
    auto *internal = static_cast<internal_node *>(current);
    assert(current->count >= 1 && current->count <= Branching);
    current->minimum = internal->children[0]->minimum;
    for (size_type i = 1; i < current->count; ++i) {
      internal->keys[i - 1U] = internal->children[i]->minimum;
    }
    current->index.rebuild(internal_key_span(internal));
  }

  [[nodiscard]] static constexpr size_type index_of_child(const internal_node *parent,
                                                          const node *child) noexcept {
    for (size_type i = 0; i < parent->count; ++i) {
      if (parent->children[i] == child) {
        return i;
      }
    }
    assert(false && "child is not attached to its parent");
    return 0;
  }

  [[nodiscard]] leaf_node *find_leaf(key_type key) noexcept {
    return const_cast<leaf_node *>(std::as_const(*this).find_leaf(key));
  }

  [[nodiscard]] const leaf_node *find_leaf(key_type key) const noexcept {
    const node *current = root_;
    while (!current->is_leaf) {
      const auto *internal = static_cast<const internal_node *>(current);
      const auto child = current->index.upper_bound(internal_key_span(internal), key);
      current = internal->children[child];
    }
    return static_cast<const leaf_node *>(current);
  }

  static constexpr void insert_key(leaf_node *leaf, size_type position, key_type key) noexcept {
    assert(position <= leaf->count && leaf->count <= Branching);
    for (size_type i = leaf->count; i > position; --i) {
      leaf->keys[i] = leaf->keys[i - 1U];
    }
    leaf->keys[position] = key;
    ++leaf->count;
  }

  [[nodiscard]] iterator insert_into_full_leaf(leaf_node *leaf, size_type position, key_type key) {
    constexpr auto maximum_allocations = std::numeric_limits<size_type>::digits + 2U;
    std::array<node *, maximum_allocations> prepared{};
    size_type prepared_count = 0;

    try {
      prepared[prepared_count++] = create_leaf();
      internal_node *first_non_full = leaf->parent;
      while (first_non_full != nullptr && first_non_full->count == Branching) {
        prepared[prepared_count++] = create_internal();
        first_non_full = first_non_full->parent;
      }
      if (first_non_full == nullptr) {
        prepared[prepared_count++] = create_internal();
      }
    } catch (...) {
      while (prepared_count != 0) {
        destroy_node(prepared[--prepared_count]);
      }
      throw;
    }

    size_type used = 0;
    const auto take_prepared = [&]() noexcept -> node * {
      assert(used < prepared_count);
      return prepared[used++];
    };

    const bool minimum_changed = position == 0;
    insert_key(leaf, position, key);
    leaf_node *result_leaf = nullptr;
    size_type result_position = 0;
    node *current = leaf;

    while (current->count > Branching) {
      node *right = take_prepared();
      const bool splitting_leaf = current->is_leaf;
      split_node(current, right);
      if (splitting_leaf) {
        constexpr auto left_count = (Branching + 1U) / 2U;
        if (position < left_count) {
          result_leaf = static_cast<leaf_node *>(current);
          result_position = position;
        } else {
          result_leaf = static_cast<leaf_node *>(right);
          result_position = position - left_count;
        }
      }

      internal_node *parent = current->parent;
      if (parent == nullptr) {
        auto *new_root = static_cast<internal_node *>(take_prepared());
        new_root->count = 2;
        new_root->children[0] = current;
        new_root->children[1] = right;
        current->parent = new_root;
        right->parent = new_root;
        refresh(new_root);
        root_ = new_root;
        break;
      }

      const auto child_position = index_of_child(parent, current);
      for (size_type i = parent->count; i > child_position + 1U; --i) {
        parent->children[i] = parent->children[i - 1U];
      }
      parent->children[child_position + 1U] = right;
      right->parent = parent;
      ++parent->count;

      if (parent->count <= Branching) {
        refresh(parent);
        break;
      }
      current = parent;
    }

    assert(used == prepared_count && result_leaf != nullptr);
    if (minimum_changed) {
      propagate_minimum(leaf);
    }
    ++size_;
    return iterator{this, result_leaf, result_position};
  }

  static void propagate_minimum(node *current) noexcept {
    const auto minimum = current->minimum;
    while (current->parent != nullptr) {
      internal_node *parent = current->parent;
      const auto position = index_of_child(parent, current);
      if (position != 0) {
        parent->keys[position - 1U] = minimum;
        parent->index.rebuild(internal_key_span(parent));
        return;
      }
      parent->minimum = minimum;
      current = parent;
    }
  }

  void split_node(node *left, node *right) noexcept {
    assert(left->count == Branching + 1U);
    assert(right->is_leaf == left->is_leaf);
    right->parent = left->parent;

    const auto left_count = static_cast<size_type>((Branching + 1U) / 2U);
    const auto right_count = static_cast<size_type>(left->count) - left_count;

    if (left->is_leaf) {
      auto *left_leaf = static_cast<leaf_node *>(left);
      auto *right_leaf = static_cast<leaf_node *>(right);
      for (size_type i = 0; i < right_count; ++i) {
        right_leaf->keys[i] = left_leaf->keys[left_count + i];
      }
      right_leaf->previous = left_leaf;
      right_leaf->next = left_leaf->next;
      if (right_leaf->next != nullptr) {
        right_leaf->next->previous = right_leaf;
      } else {
        last_leaf_ = right_leaf;
      }
      left_leaf->next = right_leaf;
    } else {
      auto *left_internal = static_cast<internal_node *>(left);
      auto *right_internal = static_cast<internal_node *>(right);
      for (size_type i = 0; i < right_count; ++i) {
        right_internal->children[i] = left_internal->children[left_count + i];
        right_internal->children[i]->parent = right_internal;
      }
    }

    left->count = static_cast<std::uint8_t>(left_count);
    right->count = static_cast<std::uint8_t>(right_count);
    refresh(left);
    refresh(right);
  }

  void rebalance_after_erase(node *current, bool minimum_changed) noexcept {
    while (current != root_ && current->count < minimum_occupancy) {
      internal_node *parent = current->parent;
      const auto position = index_of_child(parent, current);
      const bool parent_minimum_changed = minimum_changed && position == 0;
      node *left = position == 0 ? nullptr : parent->children[position - 1U];
      node *right = position + 1U == parent->count ? nullptr : parent->children[position + 1U];

      if (left != nullptr && left->count > minimum_occupancy) {
        borrow_from_left(left, current);
        refresh(parent);
        if (parent_minimum_changed) {
          propagate_minimum(parent);
        }
        return;
      }
      if (right != nullptr && right->count > minimum_occupancy) {
        borrow_from_right(current, right);
        refresh(parent);
        if (parent_minimum_changed) {
          propagate_minimum(parent);
        }
        return;
      }

      if (left != nullptr) {
        merge_nodes(left, current);
        remove_child(parent, position);
        destroy_node(current);
      } else {
        assert(right != nullptr);
        merge_nodes(current, right);
        remove_child(parent, position + 1U);
        destroy_node(right);
      }

      refresh(parent);
      if (parent == root_) {
        if (parent->count == 1U) {
          node *new_root = parent->children[0];
          new_root->parent = nullptr;
          root_ = new_root;
          destroy_node(parent);
        }
        return;
      }
      if (parent->count >= minimum_occupancy) {
        if (parent_minimum_changed) {
          propagate_minimum(parent);
        }
        return;
      }
      current = parent;
      minimum_changed = parent_minimum_changed;
    }
  }

  static void borrow_from_left(node *left, node *current) noexcept {
    assert(left->is_leaf == current->is_leaf && left->count > minimum_occupancy);
    if (current->is_leaf) {
      auto *left_leaf = static_cast<leaf_node *>(left);
      auto *current_leaf = static_cast<leaf_node *>(current);
      for (size_type i = current->count; i > 0; --i) {
        current_leaf->keys[i] = current_leaf->keys[i - 1U];
      }
      current_leaf->keys[0] = left_leaf->keys[left->count - 1U];
    } else {
      auto *left_internal = static_cast<internal_node *>(left);
      auto *current_internal = static_cast<internal_node *>(current);
      for (size_type i = current->count; i > 0; --i) {
        current_internal->children[i] = current_internal->children[i - 1U];
      }
      current_internal->children[0] = left_internal->children[left->count - 1U];
      current_internal->children[0]->parent = current_internal;
    }
    --left->count;
    ++current->count;
    refresh(left);
    refresh(current);
  }

  static void borrow_from_right(node *current, node *right) noexcept {
    assert(right->is_leaf == current->is_leaf && right->count > minimum_occupancy);
    if (current->is_leaf) {
      auto *current_leaf = static_cast<leaf_node *>(current);
      auto *right_leaf = static_cast<leaf_node *>(right);
      current_leaf->keys[current->count] = right_leaf->keys[0];
      for (size_type i = 1; i < right->count; ++i) {
        right_leaf->keys[i - 1U] = right_leaf->keys[i];
      }
    } else {
      auto *current_internal = static_cast<internal_node *>(current);
      auto *right_internal = static_cast<internal_node *>(right);
      current_internal->children[current->count] = right_internal->children[0];
      current_internal->children[current->count]->parent = current_internal;
      for (size_type i = 1; i < right->count; ++i) {
        right_internal->children[i - 1U] = right_internal->children[i];
      }
    }
    ++current->count;
    --right->count;
    refresh(current);
    refresh(right);
  }

  void merge_nodes(node *left, node *right) noexcept {
    assert(left->is_leaf == right->is_leaf && left->count + right->count <= Branching);
    const auto offset = static_cast<size_type>(left->count);
    if (left->is_leaf) {
      auto *left_leaf = static_cast<leaf_node *>(left);
      auto *right_leaf = static_cast<leaf_node *>(right);
      for (size_type i = 0; i < right->count; ++i) {
        left_leaf->keys[offset + i] = right_leaf->keys[i];
      }
      left_leaf->next = right_leaf->next;
      if (right_leaf->next != nullptr) {
        right_leaf->next->previous = left_leaf;
      } else {
        last_leaf_ = left_leaf;
      }
    } else {
      auto *left_internal = static_cast<internal_node *>(left);
      auto *right_internal = static_cast<internal_node *>(right);
      for (size_type i = 0; i < right->count; ++i) {
        left_internal->children[offset + i] = right_internal->children[i];
        left_internal->children[offset + i]->parent = left_internal;
      }
    }
    left->count = static_cast<std::uint8_t>(left->count + right->count);
    refresh(left);
  }

  static constexpr void remove_child(internal_node *parent, size_type position) noexcept {
    assert(position < parent->count);
    for (size_type i = position + 1U; i < parent->count; ++i) {
      parent->children[i - 1U] = parent->children[i];
    }
    --parent->count;
  }

  constexpr void steal_from(fusion_set &other) noexcept {
    root_ = std::exchange(other.root_, nullptr);
    first_leaf_ = std::exchange(other.first_leaf_, nullptr);
    last_leaf_ = std::exchange(other.last_leaf_, nullptr);
    size_ = std::exchange(other.size_, 0);
  }

  [[no_unique_address]] Allocator allocator_{};
  node *root_{};
  leaf_node *first_leaf_{};
  leaf_node *last_leaf_{};
  size_type size_{};
};

} // namespace fusion_tree
