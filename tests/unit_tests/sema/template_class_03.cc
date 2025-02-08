// RUN: %cxx -ftemplates -fcheck -dump-symbols %s | %filecheck %s

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
// CHECK-NEXT:  template class X
// CHECK-NEXT:    parameter typename<0, 0> T
// CHECK-NEXT:    constructor  X()
// CHECK-NEXT:      block
// CHECK-NEXT:    constructor  X(int)
// CHECK-NEXT:      parameters
// CHECK-NEXT:        parameter int
// CHECK-NEXT:        block
// CHECK-NEXT:    function const ::X* self() const
// CHECK-NEXT:      block
// CHECK-NEXT:    function ::X* self()
// CHECK-NEXT:      block