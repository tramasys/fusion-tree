#include <fusion_tree/fusion_set.hpp>

#include <cstdint>

int main() {
  const fusion_tree::fusion_set<std::uint64_t> values{2, 3, 5, 7};
  return values.contains(5) && values.validate() ? 0 : 1;
}
