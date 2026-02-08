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
// CHECK-NEXT:    constructor defaulted void list()
// CHECK-NEXT:    constructor defaulted void list(const ::list&)
// CHECK-NEXT:    constructor defaulted void list(::list&&)
// CHECK-NEXT:    typealias int value_type
// CHECK-NEXT:    class iterator
// CHECK-NEXT:      constructor defaulted void iterator()
// CHECK-NEXT:      constructor defaulted void iterator(const ::list::iterator&)
// CHECK-NEXT:      constructor defaulted void iterator(::list::iterator&&)
// CHECK-NEXT:      function defaulted void ~iterator()
// CHECK-NEXT:    function ::list::iterator begin()
// CHECK-NEXT:      block
// CHECK-NEXT:        variable static constexpr const char __func__[6]
// CHECK-NEXT:    function int at(int)
// CHECK-NEXT:      parameters
// CHECK-NEXT:        parameter int
// CHECK-NEXT:        block
// CHECK-NEXT:          variable static constexpr const char __func__[3]
// CHECK-NEXT:    function defaulted void ~list()