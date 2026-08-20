# fusion_tree

A header-only C++23 fusion-tree set for unsigned integer keys. It combines an
8-way B+ tree with portable broadword fusion nodes: distinguishing-bit sketches
are compared in parallel using guarded 64-bit lanes. No compiler intrinsics or
non-standard integer types are required.

```cpp
#include <fusion_tree/fusion_set.hpp>

fusion_tree::fusion_set<std::uint64_t> values{4, 7, 12};
values.insert(9);
auto next = values.lower_bound(8);       // 9
auto previous = values.predecessor(8);   // 7
```

The container provides ordered iteration, `find`, bounds, strict predecessor /
successor, inclusive `floor` / `ceiling`, insertion, erasure, allocator support,
and structural validation. Keys must be unsigned integral types; branching can
be configured from 3 through 8. Mutating operations may invalidate iterators.

```sh
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=clang++
cmake --build build
ctest --test-dir build --output-on-failure
```

Enable `FUSION_TREE_ENABLE_SANITIZERS` for ASan/UBSan or
`FUSION_TREE_BUILD_BENCHMARKS` for the standalone lookup benchmark.
