// RUN: %cxx -verify -dump-symbols %s | %filecheck %s

struct Outer {
  template <typename T>
  struct Inner {
    Inner(T) {}
  };
  Inner(int) -> Inner<int>;
};

// expected-no-diagnostics

// clang-format off
//      CHECK:namespace
// CHECK-NEXT:  class Outer
// CHECK-NEXT:    constructor defaulted void Outer()
// CHECK-NEXT:    constructor defaulted void Outer(const ::Outer&)
// CHECK-NEXT:    constructor defaulted void Outer(::Outer&&)
// CHECK-NEXT:    injected class name Outer
// CHECK-NEXT:    template class Inner<type-param<0, 0>>
// CHECK-NEXT:      parameter typename<0, 0> T
// CHECK-NEXT:      constructor inline void Inner(type-param<0, 0>)
// CHECK-NEXT:        parameters
// CHECK-NEXT:          parameter type-param<0, 0>
// CHECK-NEXT:          block
// CHECK-NEXT:            variable static constexpr const char __func__[6]
// CHECK-NEXT:      injected class name Inner
// CHECK-NEXT:      deduction-guide Inner(int) -> ::Outer::Inner<int>
// CHECK-NEXT:      [specializations]
// CHECK-NEXT:        class Inner<int>
// CHECK-NEXT:    function defaulted ::Outer& operator =(const ::Outer&)
// CHECK-NEXT:    function defaulted ::Outer& operator =(::Outer&&)
// CHECK-NEXT:    function defaulted void ~Outer()
