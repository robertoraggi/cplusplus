// RUN: %cxx -verify -fcheck -dump-symbols %s | %filecheck %s
// expected-no-diagnostics

// Template class with explicit type parameter base - instantiation test.

struct Base {
  int value;
  Base() : value(0) {}
  explicit Base(int v) : value(v) {}
};

template <typename T, typename B = Base>
struct Wrapper : public B {
  Wrapper() = default;
  explicit Wrapper(T v) : B(static_cast<int>(v)) {}
  T extra;
};

Wrapper<double> w1;

// clang-format off
//      CHECK:namespace
// CHECK-NEXT:  class Base
// CHECK-NEXT:    constructor inline void Base()
// CHECK-NEXT:      block
// CHECK-NEXT:        variable static constexpr const char __func__[5]
// CHECK-NEXT:    constructor inline explicit void Base(int)
// CHECK-NEXT:      parameters
// CHECK-NEXT:        parameter int v
// CHECK-NEXT:        block
// CHECK-NEXT:          variable static constexpr const char __func__[5]
// CHECK-NEXT:    constructor defaulted void Base(const ::Base&)
// CHECK-NEXT:    constructor defaulted void Base(::Base&&)
// CHECK-NEXT:    injected class name Base
// CHECK-NEXT:    field int value
// CHECK-NEXT:    function defaulted ::Base& operator =(const ::Base&)
// CHECK-NEXT:    function defaulted ::Base& operator =(::Base&&)
// CHECK-NEXT:    function defaulted void ~Base()
// CHECK-NEXT:  template class Wrapper<type-param<0, 0>, type-param<1, 0>>
// CHECK-NEXT:    parameter typename<0, 0> T
// CHECK-NEXT:    parameter typename<1, 0> B
// CHECK-NEXT:    base class B
// CHECK-NEXT:    constructor inline defaulted void Wrapper()
// CHECK-NEXT:    constructor inline explicit void Wrapper(type-param<0, 0>)
// CHECK-NEXT:      parameters
// CHECK-NEXT:        parameter type-param<0, 0> v
// CHECK-NEXT:        block
// CHECK-NEXT:          variable static constexpr const char __func__[8]
// CHECK-NEXT:    injected class name Wrapper
// CHECK-NEXT:    field type-param<0, 0> extra
// CHECK-NEXT:    [specializations]
// CHECK-NEXT:      class Wrapper<double, ::Base>
// CHECK-NEXT:        base class Base
// CHECK-NEXT:        constructor void Wrapper()
// CHECK-NEXT:        constructor explicit void Wrapper(double)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter double v
// CHECK-NEXT:            block
// CHECK-NEXT:        constructor defaulted void Wrapper(const ::Wrapper<double, ::Base>&)
// CHECK-NEXT:        constructor defaulted void Wrapper(::Wrapper<double, ::Base>&&)
// CHECK-NEXT:        injected class name Wrapper
// CHECK-NEXT:        field double extra
// CHECK-NEXT:        function defaulted ::Wrapper<double, ::Base>& operator =(const ::Wrapper<double, ::Base>&)
// CHECK-NEXT:        function defaulted ::Wrapper<double, ::Base>& operator =(::Wrapper<double, ::Base>&&)
// CHECK-NEXT:        function defaulted void ~Wrapper()
// CHECK-NEXT:  variable ::Wrapper<double, ::Base> w1
