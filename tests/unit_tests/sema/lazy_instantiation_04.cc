// RUN: %cxx -fcheck -dump-symbols %s | %filecheck %s

template <typename T>
struct Base {
  T value;
  auto get() -> T { return value; }
};

struct Derived : Base<int> {
  auto sum() -> int { return get() + 1; }
};

// clang-format off
//      CHECK:namespace
// CHECK-NEXT:  template class Base<type-param<0, 0>>
// CHECK-NEXT:    parameter typename<0, 0> T
// CHECK-NEXT:    injected class name Base
// CHECK-NEXT:    field type-param<0, 0> value
// CHECK-NEXT:    function inline type-param<0, 0> get()
//      CHECK:      class Base<int>
// CHECK-NEXT:        constructor defaulted void Base()
// CHECK-NEXT:        constructor defaulted void Base(const ::Base<int>&)
// CHECK-NEXT:        constructor defaulted void Base(::Base<int>&&)
// CHECK-NEXT:        injected class name Base
// CHECK-NEXT:        field int value
// CHECK-NEXT:        function int get()
