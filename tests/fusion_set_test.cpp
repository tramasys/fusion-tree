#include <fusion_tree/fusion_set.hpp>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <random>
#include <set>
#include <source_location>
#include <stdexcept>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(std::string_view expression,
                       std::source_location location = std::source_location::current()) {
  std::cerr << location.file_name() << ':' << location.line() << ": check failed: " << expression
            << '\n';
  std::abort();
}

#define CHECK(expression) ((expression) ? static_cast<void>(0) : fail(#expression))

template <class ActualIterator, class ExpectedIterator, class ActualSentinel,
          class ExpectedSentinel>
void check_optional_iterator(ActualIterator actual, ExpectedIterator expected,
                             ActualSentinel actual_end, ExpectedSentinel expected_end) {
  CHECK((actual == actual_end) == (expected == expected_end));
  if (actual != actual_end) {
    CHECK(*actual == *expected);
  }
}

template <class Tree, class Reference>
void check_equivalent(const Tree &tree, const Reference &reference) {
  CHECK(tree.validate());
  CHECK(tree.size() == reference.size());
  CHECK(tree.empty() == reference.empty());
  CHECK(std::ranges::equal(tree, reference));
  CHECK(std::ranges::equal(tree | std::views::reverse, reference | std::views::reverse));

  using key_type = typename Tree::key_type;
  std::array<key_type, 8> probes{
      0,
      1,
      2,
      static_cast<key_type>(7),
      static_cast<key_type>(31),
      static_cast<key_type>(127),
      static_cast<key_type>(std::numeric_limits<key_type>::max() / 2),
      std::numeric_limits<key_type>::max(),
  };

  for (const auto probe : probes) {
    CHECK(tree.contains(probe) == reference.contains(probe));
    CHECK(tree.count(probe) == reference.count(probe));
    check_optional_iterator(tree.find(probe), reference.find(probe), tree.end(), reference.end());
    check_optional_iterator(tree.lower_bound(probe), reference.lower_bound(probe), tree.end(),
                            reference.end());
    check_optional_iterator(tree.upper_bound(probe), reference.upper_bound(probe), tree.end(),
                            reference.end());
    const auto [actual_equal_begin, actual_equal_end] = tree.equal_range(probe);
    const auto [expected_equal_begin, expected_equal_end] = reference.equal_range(probe);
    check_optional_iterator(actual_equal_begin, expected_equal_begin, tree.end(), reference.end());
    check_optional_iterator(actual_equal_end, expected_equal_end, tree.end(), reference.end());
    check_optional_iterator(tree.successor(probe), reference.upper_bound(probe), tree.end(),
                            reference.end());
    check_optional_iterator(tree.ceiling(probe), reference.lower_bound(probe), tree.end(),
                            reference.end());

    auto expected_predecessor = reference.lower_bound(probe);
    if (expected_predecessor == reference.begin()) {
      expected_predecessor = reference.end();
    } else {
      --expected_predecessor;
    }
    check_optional_iterator(tree.predecessor(probe), expected_predecessor, tree.end(),
                            reference.end());

    auto expected_floor = reference.upper_bound(probe);
    if (expected_floor == reference.begin()) {
      expected_floor = reference.end();
    } else {
      --expected_floor;
    }
    check_optional_iterator(tree.floor(probe), expected_floor, tree.end(), reference.end());
  }
}

void test_sketch_collision_queries() {
  fusion_tree::fusion_set<std::uint64_t> tree{4, 7};
  CHECK(*tree.lower_bound(5) == 7);
  CHECK(*tree.lower_bound(6) == 7);
  CHECK(*tree.predecessor(6) == 4);
  CHECK(*tree.floor(6) == 4);

  const std::array<std::uint64_t, 18> values{
      0,
      1,
      2,
      3,
      0x0f,
      0x10,
      0x11,
      0xff,
      0x100,
      0x101,
      0xffff,
      0x10000,
      0x10001,
      std::uint64_t{1} << 32U,
      (std::uint64_t{1} << 32U) + 1U,
      std::uint64_t{1} << 63U,
      (std::uint64_t{1} << 63U) + 1U,
      std::numeric_limits<std::uint64_t>::max(),
  };
  fusion_tree::fusion_set<std::uint64_t> larger;
  std::set<std::uint64_t> reference(values.begin(), values.end());
  for (const auto value : values) {
    larger.insert(value);
    CHECK(larger.validate());
  }
  check_equivalent(larger, reference);
  for (std::uint64_t query = 0; query < 100'000; query += 37) {
    check_optional_iterator(larger.lower_bound(query), reference.lower_bound(query), larger.end(),
                            reference.end());
    check_optional_iterator(larger.upper_bound(query), reference.upper_bound(query), larger.end(),
                            reference.end());
  }

  std::mt19937 random(0x5eedU);
  for (std::size_t round = 0; round < 1'000; ++round) {
    fusion_tree::fusion_set<std::uint8_t> byte_tree;
    std::set<std::uint8_t> byte_reference;
    while (byte_reference.size() < 8) {
      const auto value = static_cast<std::uint8_t>(random());
      byte_tree.insert(value);
      byte_reference.insert(value);
    }
    for (unsigned query = 0; query <= std::numeric_limits<std::uint8_t>::max(); ++query) {
      const auto key = static_cast<std::uint8_t>(query);
      check_optional_iterator(byte_tree.lower_bound(key), byte_reference.lower_bound(key),
                              byte_tree.end(), byte_reference.end());
      check_optional_iterator(byte_tree.upper_bound(key), byte_reference.upper_bound(key),
                              byte_tree.end(), byte_reference.end());
    }
  }
}

template <std::size_t Branching> void test_ordered_insert_and_erase() {
  fusion_tree::fusion_set<std::uint32_t, Branching> tree;
  std::set<std::uint32_t> reference;

  for (std::uint32_t value = 0; value < 4'000; ++value) {
    const auto [where, inserted] = tree.insert(value);
    CHECK(inserted);
    CHECK(*where == value);
    reference.insert(value);
  }
  check_equivalent(tree, reference);

  for (std::uint32_t value = 0; value < 4'000; value += 2) {
    CHECK(tree.erase(value) == 1);
    reference.erase(value);
    if (value % 128 == 0) {
      CHECK(tree.validate());
    }
  }
  check_equivalent(tree, reference);

  for (std::uint32_t value = 3'999;; value -= 2) {
    CHECK(tree.erase(value) == 1);
    reference.erase(value);
    if (value % 127 == 0) {
      CHECK(tree.validate());
    }
    if (value == 1) {
      break;
    }
  }
  check_equivalent(tree, reference);
  CHECK(tree.empty());
}

template <class Key, std::size_t Branching>
void test_randomized(std::uint64_t seed, std::size_t operations) {
  fusion_tree::fusion_set<Key, Branching> tree;
  std::set<Key> reference;
  std::mt19937_64 random(seed);

  for (std::size_t operation = 0; operation < operations; ++operation) {
    const auto key = static_cast<Key>(random());
    switch (random() % 5U) {
    case 0:
    case 1: {
      const auto actual = tree.insert(key);
      const auto expected = reference.insert(key);
      CHECK(actual.second == expected.second);
      CHECK(*actual.first == *expected.first);
      break;
    }
    case 2:
      CHECK(tree.erase(key) == reference.erase(key));
      break;
    case 3:
      check_optional_iterator(tree.lower_bound(key), reference.lower_bound(key), tree.end(),
                              reference.end());
      break;
    case 4:
      check_optional_iterator(tree.upper_bound(key), reference.upper_bound(key), tree.end(),
                              reference.end());
      break;
    default:
      std::unreachable();
    }

    if (operation % 257U == 0) {
      CHECK(tree.validate());
      CHECK(tree.size() == reference.size());
      CHECK(std::ranges::equal(tree, reference));
    }
  }
  check_equivalent(tree, reference);
}

template <std::size_t Branching> void test_existing_key_erase(std::uint64_t seed) {
  fusion_tree::fusion_set<std::uint64_t, Branching> tree;
  std::set<std::uint64_t> reference;
  std::vector<std::uint64_t> keys;
  std::mt19937_64 random(seed);
  keys.reserve(30'000);

  while (keys.size() != keys.capacity()) {
    const auto key = random();
    if (reference.insert(key).second) {
      tree.insert(key);
      keys.push_back(key);
    }
  }
  std::ranges::shuffle(keys, random);

  for (std::size_t i = 0; i < keys.size(); ++i) {
    CHECK(tree.erase(keys[i]) == reference.erase(keys[i]));
    if (i % 113U == 0) {
      CHECK(tree.validate());
      CHECK(tree.size() == reference.size());
      CHECK(std::ranges::equal(tree, reference));
    }
  }
  CHECK(tree.empty());
  CHECK(tree.validate());
}

void test_value_semantics_and_ranges() {
  std::vector<std::uint64_t> values{9, 1, 7, 3, 5, 3, 1};
  fusion_tree::fusion_set<std::uint64_t> original;
  original.insert_range(values);
  CHECK(original == fusion_tree::fusion_set<std::uint64_t>({1, 3, 5, 7, 9}));

  auto copy = original;
  CHECK(copy == original);
  copy.insert(11);
  CHECK(copy != original);

  fusion_tree::fusion_set<std::uint64_t> assigned;
  assigned = copy;
  CHECK(assigned == copy);

  auto moved = std::move(copy);
  CHECK(moved.contains(11));
  CHECK(copy.empty());
  CHECK(copy.validate());

  fusion_tree::fusion_set<std::uint64_t> move_assigned;
  move_assigned = std::move(moved);
  CHECK(move_assigned.contains(11));
  CHECK(moved.empty());

  auto iterator = move_assigned.find(5);
  iterator = move_assigned.erase(iterator);
  CHECK(*iterator == 7);
  CHECK(!move_assigned.contains(5));

  original.swap(move_assigned);
  CHECK(original.contains(11));
  CHECK(move_assigned == fusion_tree::fusion_set<std::uint64_t>({1, 3, 5, 7, 9}));
}

struct allocation_state {
  std::size_t allocated{};
  std::size_t deallocated{};
  std::size_t remaining = std::numeric_limits<std::size_t>::max();
};

template <class T> class counting_allocator {
public:
  using value_type = T;

  counting_allocator() : state(std::make_shared<allocation_state>()) {}
  explicit counting_allocator(std::shared_ptr<allocation_state> value) : state(std::move(value)) {}

  template <class U>
  counting_allocator(const counting_allocator<U> &other) noexcept : state(other.state) {}

  [[nodiscard]] T *allocate(std::size_t count) {
    if (state->remaining == 0) {
      throw std::bad_alloc{};
    }
    --state->remaining;
    state->allocated += count;
    return std::allocator<T>{}.allocate(count);
  }

  void deallocate(T *pointer, std::size_t count) noexcept {
    state->deallocated += count;
    std::allocator<T>{}.deallocate(pointer, count);
  }

  template <class U> friend class counting_allocator;

  template <class U>
  friend bool operator==(const counting_allocator &left,
                         const counting_allocator<U> &right) noexcept {
    return left.state == right.state;
  }

  std::shared_ptr<allocation_state> state;
};

template <class T> class fancy_pointer {
public:
  using element_type = T;
  using difference_type = std::ptrdiff_t;
  template <class U> using rebind = fancy_pointer<U>;

  constexpr fancy_pointer() noexcept = default;
  explicit constexpr fancy_pointer(T *pointer) noexcept : pointer_(pointer) {}

  template <class U>
    requires std::convertible_to<U *, T *>
  constexpr fancy_pointer(const fancy_pointer<U> &other) noexcept : pointer_(other.get()) {}

  [[nodiscard]] static constexpr fancy_pointer pointer_to(T &value) noexcept {
    return fancy_pointer{std::addressof(value)};
  }
  [[nodiscard]] constexpr T *get() const noexcept { return pointer_; }
  [[nodiscard]] constexpr T &operator*() const noexcept { return *pointer_; }
  [[nodiscard]] constexpr T *operator->() const noexcept { return pointer_; }
  explicit constexpr operator bool() const noexcept { return pointer_ != nullptr; }

  friend constexpr bool operator==(const fancy_pointer &, const fancy_pointer &) noexcept = default;
  friend constexpr bool operator==(const fancy_pointer &pointer, std::nullptr_t) noexcept {
    return pointer.pointer_ == nullptr;
  }

private:
  template <class U> friend class fancy_pointer;
  T *pointer_{};
};

template <class T> class fancy_allocator {
public:
  using value_type = T;
  using pointer = fancy_pointer<T>;

  fancy_allocator() : state(std::make_shared<allocation_state>()) {}
  explicit fancy_allocator(std::shared_ptr<allocation_state> value) : state(std::move(value)) {}

  template <class U>
  fancy_allocator(const fancy_allocator<U> &other) noexcept : state(other.state) {}

  [[nodiscard]] pointer allocate(std::size_t count) {
    state->allocated += count;
    return pointer{std::allocator<T>{}.allocate(count)};
  }

  void deallocate(pointer storage, std::size_t count) noexcept {
    state->deallocated += count;
    std::allocator<T>{}.deallocate(storage.get(), count);
  }

  template <class U> friend class fancy_allocator;

  template <class U>
  friend bool operator==(const fancy_allocator &left, const fancy_allocator<U> &right) noexcept {
    return left.state == right.state;
  }

  std::shared_ptr<allocation_state> state;
};

void test_allocator_and_allocation_failure() {
  auto state = std::make_shared<allocation_state>();
  using allocator = counting_allocator<std::uint64_t>;
  using tree_type = fusion_tree::fusion_set<std::uint64_t, 8, allocator>;

  {
    tree_type tree{allocator{state}};
    for (std::uint64_t key = 0; key < 8; ++key) {
      tree.insert(key);
    }
    CHECK(tree.validate());

    state->remaining = 1;
    try {
      tree.insert(8);
      fail("insertion should have thrown std::bad_alloc");
    } catch (const std::bad_alloc &) {
    }
    CHECK(tree.size() == 8);
    CHECK(!tree.contains(8));
    CHECK(tree.validate());

    state->remaining = 10;
    CHECK(tree.insert(8).second);
    CHECK(tree.validate());
  }
  CHECK(state->allocated == state->deallocated);

  auto constructor_state = std::make_shared<allocation_state>();
  constructor_state->remaining = 2;
  const std::array<std::uint64_t, 9> values{0, 1, 2, 3, 4, 5, 6, 7, 8};
  try {
    const tree_type tree{values.begin(), values.end(), allocator{constructor_state}};
    static_cast<void>(tree);
    fail("range construction should have thrown std::bad_alloc");
  } catch (const std::bad_alloc &) {
  }
  CHECK(constructor_state->allocated == constructor_state->deallocated);

  using cascading_tree = fusion_tree::fusion_set<std::uint64_t, 3, allocator>;
  for (std::uint64_t element_count = 1; element_count < 200; ++element_count) {
    for (std::size_t allocation_budget = 0; allocation_budget < 5; ++allocation_budget) {
      auto cascade_state = std::make_shared<allocation_state>();
      {
        cascading_tree tree{allocator{cascade_state}};
        for (std::uint64_t key = 0; key < element_count; ++key) {
          tree.insert(key);
        }
        cascade_state->remaining = allocation_budget;
        try {
          tree.insert(element_count);
          CHECK(tree.size() == element_count + 1U);
        } catch (const std::bad_alloc &) {
          CHECK(tree.size() == element_count);
          CHECK(!tree.contains(element_count));
        }
        CHECK(tree.validate());
      }
      CHECK(cascade_state->allocated == cascade_state->deallocated);
    }
  }

  auto fancy_state = std::make_shared<allocation_state>();
  using fancy_tree = fusion_tree::fusion_set<std::uint64_t, 8, fancy_allocator<std::uint64_t>>;
  {
    fancy_tree tree{fancy_allocator<std::uint64_t>{fancy_state}};
    for (std::uint64_t key = 0; key < 10'000; ++key) {
      tree.insert(key * 17U);
    }
    for (std::uint64_t key = 0; key < 10'000; key += 3U) {
      tree.erase(key * 17U);
    }
    CHECK(tree.validate());
  }
  CHECK(fancy_state->allocated == fancy_state->deallocated);
}

} // namespace

int main() {
  static_assert(
      std::bidirectional_iterator<fusion_tree::fusion_set<std::uint64_t>::const_iterator>);
  static_assert(std::ranges::bidirectional_range<fusion_tree::fusion_set<std::uint64_t>>);
  static_assert(std::same_as<fusion_tree::fusion_set<std::uint64_t>::iterator,
                             fusion_tree::fusion_set<std::uint64_t>::const_iterator>);

  test_sketch_collision_queries();
  test_ordered_insert_and_erase<3>();
  test_ordered_insert_and_erase<5>();
  test_ordered_insert_and_erase<8>();
  test_randomized<std::uint16_t, 3>(0x01a2b3c4U, 40'000);
  test_randomized<std::uint32_t, 5>(0xdeadbeefU, 70'000);
  test_randomized<std::uint64_t, 8>(0x123456789abcdef0ULL, 100'000);
  test_existing_key_erase<3>(0xa54ff53a5f1d36f1ULL);
  test_existing_key_erase<8>(0x510e527fade682d1ULL);
  test_value_semantics_and_ranges();
  test_allocator_and_allocation_failure();

  std::cout << "all fusion-tree tests passed\n";
}
