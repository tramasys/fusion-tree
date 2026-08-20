#include <fusion_tree/fusion_set.hpp>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <set>
#include <string_view>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

template <class Function> double time_seconds(Function &&function) {
  const auto start = clock_type::now();
  std::forward<Function>(function)();
  return std::chrono::duration<double>(clock_type::now() - start).count();
}

template <class Set>
std::uint64_t lookup_all(const Set &values, const std::vector<std::uint64_t> &queries) {
  std::uint64_t checksum = 0;
  for (const auto query : queries) {
    const auto where = values.lower_bound(query);
    checksum ^= where == values.end() ? query : *where;
  }
  return checksum;
}

} // namespace

int main(int argc, char **argv) {
  std::size_t count = 500'000;
  if (argc == 2) {
    const std::string_view argument{argv[1]};
    const auto [end, error] =
        std::from_chars(argument.data(), argument.data() + argument.size(), count);
    if (error != std::errc{} || end != argument.data() + argument.size()) {
      std::cerr << "usage: fusion_tree_benchmark [element-count]\n";
      return 2;
    }
  }

  std::mt19937_64 random(0x6a09e667f3bcc909ULL);
  std::vector<std::uint64_t> keys(count);
  std::ranges::generate(keys, [&] { return random(); });
  auto queries = keys;
  std::ranges::shuffle(queries, random);

  fusion_tree::fusion_set<std::uint64_t> fusion;
  std::set<std::uint64_t> standard;
  const auto fusion_build = time_seconds([&] { fusion.insert_range(keys); });
  const auto standard_build = time_seconds([&] { standard.insert(keys.begin(), keys.end()); });

  std::uint64_t fusion_checksum = 0;
  std::uint64_t standard_checksum = 0;
  const auto fusion_lookup = time_seconds([&] { fusion_checksum = lookup_all(fusion, queries); });
  const auto standard_lookup =
      time_seconds([&] { standard_checksum = lookup_all(standard, queries); });

  if (fusion_checksum != standard_checksum || fusion.size() != standard.size() ||
      !fusion.validate()) {
    std::cerr << "benchmark validation failed\n";
    return 1;
  }

  std::cout << "elements: " << fusion.size() << '\n'
            << "build fusion/std::set (s): " << fusion_build << " / " << standard_build << '\n'
            << "lookup fusion/std::set (s): " << fusion_lookup << " / " << standard_lookup << '\n'
            << "checksum: " << fusion_checksum << '\n';
}
