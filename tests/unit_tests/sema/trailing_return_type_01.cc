// RUN: %cxx -verify -fcheck -dump-symbols %s | %filecheck %s

struct list {
  using value_type = int;

  struct iterator {};

  auto begin() -> iterator;
  auto at(value_type) -> value_type;
};

auto list::begin() -> iterator { return {}; }
auto list::at(value_type) -> value_type { return 0; }

// clang-format off
//      CHECK:namespace
// CHECK-NEXT:  class list
// CHECK-NEXT:    typealias int value_type
// CHECK-NEXT:    class iterator
// CHECK-NEXT:    function ::list::iterator begin()
// CHECK-NEXT:      block
// CHECK-NEXT:    function int at(int)
// CHECK-NEXT:      parameters
// CHECK-NEXT:        parameter int
// CHECK-NEXT:        block