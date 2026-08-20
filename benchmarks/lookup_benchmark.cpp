#include <fusion_tree/fusion_set.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>
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

struct lookup_timings {
  double fusion_seconds{};
  double standard_seconds{};
  std::uint64_t checksum{};
};

template <class FusionSet, class StandardSet>
lookup_timings measure_lookups(const FusionSet &fusion, const StandardSet &standard,
                               const std::vector<std::uint64_t> &queries) {
  constexpr std::size_t repetitions = 5;
  std::array<double, repetitions> fusion_times{};
  std::array<double, repetitions> standard_times{};
  std::uint64_t expected_checksum = lookup_all(fusion, queries);
  if (expected_checksum != lookup_all(standard, queries)) {
    throw std::runtime_error{"lookup checksums differ"};
  }

  for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
    std::uint64_t fusion_checksum = 0;
    std::uint64_t standard_checksum = 0;
    const auto run_fusion = [&] {
      fusion_times[repetition] =
          time_seconds([&] { fusion_checksum = lookup_all(fusion, queries); });
    };
    const auto run_standard = [&] {
      standard_times[repetition] =
          time_seconds([&] { standard_checksum = lookup_all(standard, queries); });
    };

    if (repetition % 2U == 0) {
      run_fusion();
      run_standard();
    } else {
      run_standard();
      run_fusion();
    }
    if (fusion_checksum != expected_checksum || standard_checksum != expected_checksum) {
      throw std::runtime_error{"unstable lookup checksum"};
    }
  }

  std::ranges::sort(fusion_times);
  std::ranges::sort(standard_times);
  return {fusion_times[repetitions / 2U], standard_times[repetitions / 2U], expected_checksum};
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

  std::vector<std::uint64_t> misses;
  misses.reserve(queries.size());
  for (const auto key : queries) {
    auto candidate = ~key;
    while (standard.contains(candidate)) {
      ++candidate;
    }
    misses.push_back(candidate);
  }
  std::vector<std::uint64_t> mixed;
  mixed.reserve(queries.size());
  for (std::size_t i = 0; i < queries.size(); ++i) {
    mixed.push_back(i % 2U == 0 ? queries[i] : misses[i]);
  }

  lookup_timings hit_timings;
  lookup_timings miss_timings;
  lookup_timings mixed_timings;
  try {
    hit_timings = measure_lookups(fusion, standard, queries);
    miss_timings = measure_lookups(fusion, standard, misses);
    mixed_timings = measure_lookups(fusion, standard, mixed);
  } catch (const std::runtime_error &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }

  if (fusion.size() != standard.size() || !fusion.validate()) {
    std::cerr << "benchmark validation failed\n";
    return 1;
  }

  std::size_t fusion_erased = 0;
  std::size_t standard_erased = 0;
  const auto fusion_erase = time_seconds([&] {
    for (const auto key : queries) {
      fusion_erased += fusion.erase(key);
    }
  });
  const auto standard_erase = time_seconds([&] {
    for (const auto key : queries) {
      standard_erased += standard.erase(key);
    }
  });
  if (fusion_erased != standard_erased || !fusion.empty() || !standard.empty() ||
      !fusion.validate()) {
    std::cerr << "erase validation failed\n";
    return 1;
  }

  std::cout << "elements: " << fusion_erased << '\n'
            << "build fusion/std::set (s): " << fusion_build << " / " << standard_build << '\n'
            << "hit lower_bound median (s): " << hit_timings.fusion_seconds << " / "
            << hit_timings.standard_seconds << '\n'
            << "miss lower_bound median (s): " << miss_timings.fusion_seconds << " / "
            << miss_timings.standard_seconds << '\n'
            << "mixed lower_bound median (s): " << mixed_timings.fusion_seconds << " / "
            << mixed_timings.standard_seconds << '\n'
            << "erase fusion/std::set (s): " << fusion_erase << " / " << standard_erase << '\n'
            << "checksums: " << hit_timings.checksum << ' ' << miss_timings.checksum << ' '
            << mixed_timings.checksum << '\n';
}
