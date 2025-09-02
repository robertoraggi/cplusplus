// RUN: %cxx -fcheck -dump-symbols %s | %filecheck %s

template <typename X>
struct A {
  using T = X;
  int v;
};

template <typename T>
using B = const A<T>;

struct D : B<A<void>> {
  void f() {
    T t;

    this->v = 321;
    static_assert(__is_same(T, A<void>));
    static_assert(__is_same(decltype(t), A<void>));
  }
};

template <typename T>
struct D2 : B<double> {
  auto f() -> T {
    this->v = 42;
    T t;
    return t;
  }

  T t;
};

auto main() -> int {
  D d1;
  static_assert(__is_same(decltype(d1.f()), void));

  D2<int> d2;

  static_assert(__is_same(decltype(d2.f()), double));
  static_assert(__is_same(decltype(d2.t), double));
  static_assert(__is_same(decltype((d2.t)), double&));

  d2.t = 123;

  return 0;
}

// clang-format off
//      CHECK:namespace
// CHECK-NEXT:  template class A<type-param<0, 0>>
// CHECK-NEXT:    parameter typename<0, 0> X
// CHECK-NEXT:    typealias type-param<0, 0> T
// CHECK-NEXT:    field int v
// CHECK-NEXT:    [specializations]
// CHECK-NEXT:      class A<void>
// CHECK-NEXT:        typealias void T
// CHECK-NEXT:        field int v
// CHECK-NEXT:      class A<::A<void>>
// CHECK-NEXT:        typealias ::A<void> T
// CHECK-NEXT:        field int v
// CHECK-NEXT:      class A<double>
// CHECK-NEXT:        typealias double T
// CHECK-NEXT:        field int v
// CHECK-NEXT:  template typealias const ::A B
// CHECK-NEXT:    parameter typename<0, 0> T
// CHECK-NEXT:  class D
// CHECK-NEXT:    base class A
// CHECK-NEXT:    function void f()
// CHECK-NEXT:      block
// CHECK-NEXT:        variable ::A<void> t
// CHECK-NEXT:  template class D2<type-param<0, 0>>
// CHECK-NEXT:    parameter typename<0, 0> T
// CHECK-NEXT:    base class A
// CHECK-NEXT:    function double f()
// CHECK-NEXT:      block
// CHECK-NEXT:        variable double t
// CHECK-NEXT:    field double t
// CHECK-NEXT:    [specializations]
// CHECK-NEXT:      class D2<int>
// CHECK-NEXT:        base class A
// CHECK-NEXT:        function double f()
// CHECK-NEXT:          block
// CHECK-NEXT:            variable double t
// CHECK-NEXT:        field double t
// CHECK-NEXT:  function int main()
// CHECK-NEXT:    block
// CHECK-NEXT:      variable ::D d1
// CHECK-NEXT:      variable ::D2<int> d2
