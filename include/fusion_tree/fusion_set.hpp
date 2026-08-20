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

namespace fusion_tree {

namespace detail {

// A fusion node stores the compressed sketches of all keys in guarded lanes.
// Subtracting a replicated query sketch compares every lane simultaneously;
// the guard bits then encode the lower-bound result.
template <std::unsigned_integral Key, std::size_t Capacity> class fusion_index {
  static_assert(Capacity >= 1 && Capacity <= 8);

public:
  constexpr void rebuild(std::span<const Key> keys) noexcept {
    assert(keys.size() <= Capacity);

    size_ = static_cast<std::uint8_t>(keys.size());
    relevant_count_ = 0;
    positions_.fill(0);
    packed_ = 0;
    guard_mask_ = 0;
    repeat_mask_ = 0;

    for (std::size_t i = 1; i < keys.size(); ++i) {
      assert(keys[i - 1] < keys[i]);
      const auto difference = static_cast<Key>(keys[i - 1] ^ keys[i]);
      assert(difference != 0);
      const auto position =
          static_cast<std::uint8_t>(static_cast<unsigned>(std::bit_width(difference)) - 1U);

      bool already_present = false;
      for (std::size_t j = 0; j < relevant_count_; ++j) {
        already_present |= positions_[j] == position;
      }
      if (!already_present) {
        positions_[relevant_count_++] = position;
      }
    }

    std::sort(positions_.begin(), positions_.begin() + relevant_count_, std::greater<>{});
    lane_width_ = static_cast<std::uint8_t>(relevant_count_ + 1U);

    const auto guard = std::uint64_t{1} << relevant_count_;
    for (std::size_t i = 0; i < keys.size(); ++i) {
      const auto shift = static_cast<unsigned>(i * lane_width_);
      packed_ |= (guard | sketch(keys[i])) << shift;
      guard_mask_ |= guard << shift;
      repeat_mask_ |= std::uint64_t{1} << shift;
    }
  }

  [[nodiscard]] constexpr std::size_t lower_bound(std::span<const Key> keys,
                                                  Key query) const noexcept {
    assert(keys.size() == size_);
    if (keys.empty()) {
      return 0;
    }

    const auto approximate = sketch_lower_bound(query);
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
      return sketch_lower_bound(minimum_on_query_side);
    }

    const auto maximum_on_query_side = static_cast<Key>(query | lower_bits);
    auto position = sketch_lower_bound(maximum_on_query_side);
    if (position != keys.size() && sketch(keys[position]) == sketch(maximum_on_query_side)) {
      ++position;
    }
    return position;
  }

  [[nodiscard]] constexpr std::size_t upper_bound(std::span<const Key> keys,
                                                  Key query) const noexcept {
    assert(keys.size() == size_);
    auto position = lower_bound(keys, query);
    if (position != keys.size() && keys[position] == query) {
      ++position;
    }
    return position;
  }

  friend constexpr bool operator==(const fusion_index &, const fusion_index &) noexcept = default;

private:
  [[nodiscard]] constexpr std::uint64_t sketch(Key key) const noexcept {
    std::uint64_t result = 0;
    for (std::size_t i = 0; i < relevant_count_; ++i) {
      result = (result << 1U) | ((static_cast<std::uint64_t>(key) >> positions_[i]) & 1U);
    }
    return result;
  }

  [[nodiscard]] constexpr std::size_t sketch_lower_bound(Key query) const noexcept {
    if (size_ == 0) {
      return 0;
    }

    const auto differences = packed_ - sketch(query) * repeat_mask_;
    const auto greater_or_equal = differences & guard_mask_;
    if (greater_or_equal == 0) {
      return size_;
    }
    return static_cast<std::size_t>(std::countr_zero(greater_or_equal)) /
           static_cast<std::size_t>(lane_width_);
  }

  std::array<std::uint8_t, Capacity - 1> positions_{};
  std::uint64_t packed_{};
  std::uint64_t guard_mask_{};
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

  struct node {
    explicit constexpr node(bool leaf) noexcept : is_leaf(leaf) {}

    std::array<Key, Branching + 1> keys{};
    std::array<node *, Branching + 1> children{};
    detail::fusion_index<Key, Branching> index{};
    node *parent{};
    node *previous{};
    node *next{};
    std::uint8_t count{};
    bool is_leaf{};
  };

  using allocator_traits = std::allocator_traits<Allocator>;
  using node_allocator = typename allocator_traits::template rebind_alloc<node>;
  using node_allocator_traits = std::allocator_traits<node_allocator>;

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
    constexpr const_iterator(const fusion_set *owner, const node *leaf, size_type position) noexcept
        : owner_(owner), leaf_(leaf), position_(position) {}

    const fusion_set *owner_{};
    const node *leaf_{};
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
      node *new_root = create_node(true);
      new_root->keys[0] = key;
      new_root->count = 1;
      refresh(new_root);
      root_ = first_leaf_ = last_leaf_ = new_root;
      size_ = 1;
      return {iterator{this, new_root, 0}, true};
    }

    node *leaf = find_leaf(key);
    const auto keys = key_span(leaf);
    const auto position = leaf->index.lower_bound(keys, key);
    if (position != leaf->count && leaf->keys[position] == key) {
      return {iterator{this, leaf, position}, false};
    }

    if (leaf->count < Branching) {
      insert_key(leaf, position, key);
      ++size_;
      refresh_upward(leaf);
      return {iterator{this, leaf, position}, true};
    }

    constexpr auto maximum_allocations = std::numeric_limits<size_type>::digits + 2U;
    std::array<node *, maximum_allocations> prepared{};
    size_type prepared_count = 0;
    size_type needed = 0;
    for (node *current = leaf; current != nullptr && current->count == Branching;
         current = current->parent) {
      ++needed;
    }
    node *first_non_full = leaf;
    while (first_non_full != nullptr && first_non_full->count == Branching) {
      first_non_full = first_non_full->parent;
    }
    if (first_non_full == nullptr) {
      ++needed;
    }
    assert(needed <= prepared.size());

    try {
      while (prepared_count != needed) {
        prepared[prepared_count++] = create_node(false);
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

    insert_key(leaf, position, key);
    node *current = leaf;
    while (current->count > Branching) {
      node *right = take_prepared();
      split_node(current, right);
      node *parent = current->parent;

      if (parent == nullptr) {
        node *new_root = take_prepared();
        new_root->is_leaf = false;
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
        refresh_upward(parent);
        break;
      }
      current = parent;
    }

    assert(used == prepared_count);
    ++size_;
    return {find(key), true};
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

    node *leaf = find_leaf(key);
    const auto position = leaf->index.lower_bound(key_span(leaf), key);
    if (position == leaf->count || leaf->keys[position] != key) {
      return 0;
    }

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
      refresh_upward(leaf);
    } else {
      rebalance_after_erase(leaf);
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
    const node *leaf = find_leaf(key);
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
    const node *leaf = find_leaf(key);
    const auto position = leaf->index.lower_bound(key_span(leaf), key);
    if (position != leaf->count) {
      return iterator{this, leaf, position};
    }
    return iterator{this, leaf->next, 0};
  }

  [[nodiscard]] iterator upper_bound(key_type key) const noexcept {
    if (root_ == nullptr) {
      return end();
    }
    const node *leaf = find_leaf(key);
    const auto position = leaf->index.upper_bound(key_span(leaf), key);
    if (position != leaf->count) {
      return iterator{this, leaf, position};
    }
    return iterator{this, leaf->next, 0};
  }

  [[nodiscard]] std::pair<iterator, iterator> equal_range(key_type key) const noexcept {
    return {lower_bound(key), upper_bound(key)};
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
    const node *previous_leaf = nullptr;
    std::optional<key_type> previous_key;

    const auto inspect = [&](const auto &self, const node *current, size_type depth,
                             bool is_root) noexcept -> bool {
      if (current == nullptr || current->count == 0 || current->count > Branching) {
        return false;
      }

      if (current->is_leaf) {
        if (!is_root && current->count < minimum_occupancy) {
          return false;
        }
        if (leaf_depth == std::numeric_limits<size_type>::max()) {
          leaf_depth = depth;
        } else if (leaf_depth != depth) {
          return false;
        }
        if ((previous_leaf == nullptr && current != first_leaf_) ||
            current->previous != previous_leaf ||
            (previous_leaf != nullptr && previous_leaf->next != current)) {
          return false;
        }

        detail::fusion_index<Key, Branching> expected;
        expected.rebuild(key_span(current));
        if (expected != current->index) {
          return false;
        }
        for (size_type i = 0; i < current->count; ++i) {
          if (previous_key && *previous_key >= current->keys[i]) {
            return false;
          }
          previous_key = current->keys[i];
          ++observed_size;
        }
        previous_leaf = current;
        return true;
      }

      if ((is_root && current->count < 2U) || (!is_root && current->count < minimum_occupancy)) {
        return false;
      }

      detail::fusion_index<Key, Branching> expected;
      expected.rebuild(internal_key_span(current));
      if (expected != current->index) {
        return false;
      }

      for (size_type i = 0; i < current->count; ++i) {
        if (current->children[i] == nullptr || current->children[i]->parent != current) {
          return false;
        }
        if (i != 0 && current->keys[i - 1U] != subtree_min(current->children[i])) {
          return false;
        }
        if (!self(self, current->children[i], depth + 1U, false)) {
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

  [[nodiscard]] static constexpr std::span<const key_type> key_span(const node *current) noexcept {
    return {current->keys.data(), current->count};
  }

  [[nodiscard]] static constexpr std::span<const key_type>
  internal_key_span(const node *current) noexcept {
    assert(!current->is_leaf && current->count >= 1);
    return {current->keys.data(), static_cast<size_type>(current->count - 1U)};
  }

  [[nodiscard]] node *create_node(bool leaf) {
    node_allocator allocator{allocator_};
    node *result = node_allocator_traits::allocate(allocator, 1);
    try {
      node_allocator_traits::construct(allocator, result, leaf);
    } catch (...) {
      node_allocator_traits::deallocate(allocator, result, 1);
      throw;
    }
    return result;
  }

  void destroy_node(node *current) noexcept {
    node_allocator allocator{allocator_};
    node_allocator_traits::destroy(allocator, current);
    node_allocator_traits::deallocate(allocator, current, 1);
  }

  void destroy_subtree(node *current) noexcept {
    if (current == nullptr) {
      return;
    }
    if (!current->is_leaf) {
      for (size_type i = 0; i < current->count; ++i) {
        destroy_subtree(current->children[i]);
      }
    }
    destroy_node(current);
  }

  [[nodiscard]] static constexpr key_type subtree_min(const node *current) noexcept {
    while (!current->is_leaf) {
      current = current->children[0];
    }
    assert(current->count != 0);
    return current->keys[0];
  }

  static constexpr void refresh(node *current) noexcept {
    if (current->is_leaf) {
      current->index.rebuild(key_span(current));
      return;
    }
    assert(current->count >= 1 && current->count <= Branching);
    for (size_type i = 1; i < current->count; ++i) {
      current->keys[i - 1U] = subtree_min(current->children[i]);
    }
    current->index.rebuild(internal_key_span(current));
  }

  static constexpr void refresh_upward(node *current) noexcept {
    while (current != nullptr) {
      refresh(current);
      current = current->parent;
    }
  }

  [[nodiscard]] static constexpr size_type index_of_child(const node *parent,
                                                          const node *child) noexcept {
    for (size_type i = 0; i < parent->count; ++i) {
      if (parent->children[i] == child) {
        return i;
      }
    }
    assert(false && "child is not attached to its parent");
    return 0;
  }

  [[nodiscard]] node *find_leaf(key_type key) noexcept {
    return const_cast<node *>(std::as_const(*this).find_leaf(key));
  }

  [[nodiscard]] const node *find_leaf(key_type key) const noexcept {
    const node *current = root_;
    while (!current->is_leaf) {
      const auto child = current->index.upper_bound(internal_key_span(current), key);
      current = current->children[child];
    }
    return current;
  }

  static constexpr void insert_key(node *leaf, size_type position, key_type key) noexcept {
    assert(leaf->is_leaf && position <= leaf->count && leaf->count <= Branching);
    for (size_type i = leaf->count; i > position; --i) {
      leaf->keys[i] = leaf->keys[i - 1U];
    }
    leaf->keys[position] = key;
    ++leaf->count;
  }

  void split_node(node *left, node *right) noexcept {
    assert(left->count == Branching + 1U);
    right->is_leaf = left->is_leaf;
    right->parent = left->parent;

    const auto left_count = static_cast<size_type>((Branching + 1U) / 2U);
    const auto right_count = static_cast<size_type>(left->count) - left_count;

    if (left->is_leaf) {
      for (size_type i = 0; i < right_count; ++i) {
        right->keys[i] = left->keys[left_count + i];
      }
      right->previous = left;
      right->next = left->next;
      if (right->next != nullptr) {
        right->next->previous = right;
      } else {
        last_leaf_ = right;
      }
      left->next = right;
    } else {
      for (size_type i = 0; i < right_count; ++i) {
        right->children[i] = left->children[left_count + i];
        right->children[i]->parent = right;
      }
    }

    left->count = static_cast<std::uint8_t>(left_count);
    right->count = static_cast<std::uint8_t>(right_count);
    refresh(left);
    refresh(right);
  }

  void rebalance_after_erase(node *current) noexcept {
    while (current != root_ && current->count < minimum_occupancy) {
      node *parent = current->parent;
      const auto position = index_of_child(parent, current);
      node *left = position == 0 ? nullptr : parent->children[position - 1U];
      node *right = position + 1U == parent->count ? nullptr : parent->children[position + 1U];

      if (left != nullptr && left->count > minimum_occupancy) {
        borrow_from_left(left, current);
        refresh_upward(parent);
        return;
      }
      if (right != nullptr && right->count > minimum_occupancy) {
        borrow_from_right(current, right);
        refresh_upward(parent);
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
        } else {
          refresh_upward(parent);
        }
        return;
      }
      if (parent->count >= minimum_occupancy) {
        refresh_upward(parent);
        return;
      }
      current = parent;
    }

    refresh_upward(current);
  }

  static constexpr void borrow_from_left(node *left, node *current) noexcept {
    assert(left->is_leaf == current->is_leaf && left->count > minimum_occupancy);
    if (current->is_leaf) {
      for (size_type i = current->count; i > 0; --i) {
        current->keys[i] = current->keys[i - 1U];
      }
      current->keys[0] = left->keys[left->count - 1U];
    } else {
      for (size_type i = current->count; i > 0; --i) {
        current->children[i] = current->children[i - 1U];
      }
      current->children[0] = left->children[left->count - 1U];
      current->children[0]->parent = current;
    }
    --left->count;
    ++current->count;
    refresh(left);
    refresh(current);
  }

  static constexpr void borrow_from_right(node *current, node *right) noexcept {
    assert(right->is_leaf == current->is_leaf && right->count > minimum_occupancy);
    if (current->is_leaf) {
      current->keys[current->count] = right->keys[0];
      for (size_type i = 1; i < right->count; ++i) {
        right->keys[i - 1U] = right->keys[i];
      }
    } else {
      current->children[current->count] = right->children[0];
      current->children[current->count]->parent = current;
      for (size_type i = 1; i < right->count; ++i) {
        right->children[i - 1U] = right->children[i];
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
      for (size_type i = 0; i < right->count; ++i) {
        left->keys[offset + i] = right->keys[i];
      }
      left->next = right->next;
      if (right->next != nullptr) {
        right->next->previous = left;
      } else {
        last_leaf_ = left;
      }
    } else {
      for (size_type i = 0; i < right->count; ++i) {
        left->children[offset + i] = right->children[i];
        left->children[offset + i]->parent = left;
      }
    }
    left->count = static_cast<std::uint8_t>(left->count + right->count);
    refresh(left);
  }

  static constexpr void remove_child(node *parent, size_type position) noexcept {
    assert(!parent->is_leaf && position < parent->count);
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
  node *first_leaf_{};
  node *last_leaf_{};
  size_type size_{};
};

} // namespace fusion_tree
