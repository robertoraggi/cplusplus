// RUN: %cxx -fcheck -dump-symbols %s | %filecheck %s

template <typename T>
struct X {
  X();
  X(int);

  auto self() const -> const X*;
  auto self() -> X*;
};

template <typename T>
X<T>::X() {}

template <typename T>
X<T>::X(int) {}

template <typename T>
auto X<T>::self() const -> const X* {
  return this;
}

template <typename T>
auto X<T>::self() -> X* {
  return this;
}

// clang-format off
//      CHECK:namespace
// CHECK-NEXT:  template class X<type-param<0, 0>>
// CHECK-NEXT:    parameter typename<0, 0> T
// CHECK-NEXT:    constructor void X()
// CHECK-NEXT:      [redeclarations]
// CHECK-NEXT:        constructor void X()
// CHECK-NEXT:          block
// CHECK-NEXT:            variable static constexpr const char __func__[2]
// CHECK-NEXT:    constructor void X(int)
// CHECK-NEXT:      [redeclarations]
// CHECK-NEXT:        constructor void X(int)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter int
// CHECK-NEXT:            block
// CHECK-NEXT:              variable static constexpr const char __func__[2]
// CHECK-NEXT:    function const ::X* self() const
// CHECK-NEXT:      [redeclarations]
// CHECK-NEXT:        function const ::X* self() const
// CHECK-NEXT:          block
// CHECK-NEXT:            variable static constexpr const char __func__[5]
// CHECK-NEXT:    function ::X* self()
// CHECK-NEXT:      [redeclarations]
// CHECK-NEXT:        function ::X* self()
// CHECK-NEXT:          block
// CHECK-NEXT:            variable static constexpr const char __func__[5]