// RUN: %cxx -verify -dump-symbols %s | %filecheck %s

template <typename T>
struct List {
  List(T) {}
};

List(int) -> List<int>;

template <typename T>
List(T) -> List<T>;

// expected-no-diagnostics

// clang-format off
//      CHECK:namespace
// CHECK-NEXT:  template class List<type-param<0, 0>>
// CHECK-NEXT:    parameter typename<0, 0> T
// CHECK-NEXT:    constructor inline void List(type-param<0, 0>)
// CHECK-NEXT:      parameters
// CHECK-NEXT:        parameter type-param<0, 0>
// CHECK-NEXT:        block
// CHECK-NEXT:          variable static constexpr const char __func__[5]
// CHECK-NEXT:    injected class name List
// CHECK-NEXT:    deduction-guide List(int) -> ::List<int>
// CHECK-NEXT:    deduction-guide List(type-param<0, 0>) -> ::List<type-param<0, 0>>
// CHECK-NEXT:      parameter typename<0, 0> T
