// RUN: %cxx -dump-symbols %s | %filecheck %s

struct Foo {
  Foo* self() { return this; }
};

template <typename T>
struct Box {
  Box copy() { return *this; }
  Box<T> explicit_copy() { return *this; }
};

struct IntBox : Box<int> {
  Box get() { return *this; }  // Box
};

struct IntBoxDerived : IntBox {
  Box get2() { return *this; }
};

auto test() -> int {
  Foo f;
  Box<int> b;
  auto b2 = b.copy();
  auto b3 = b.explicit_copy();

  IntBox ib;
  auto ib2 = ib.get();

  IntBoxDerived ibd;
  auto ibd2 = ibd.get2();

  return 0;
}

// clang-format off

// Non-template class has injected class name
//      CHECK:class Foo
// CHECK-NEXT:    constructor defaulted void Foo()
// CHECK-NEXT:    constructor defaulted void Foo(const ::Foo&)
// CHECK-NEXT:    constructor defaulted void Foo(::Foo&&)
// CHECK-NEXT:    injected class name Foo

// Primary template has injected class name
//      CHECK:template class Box<type-param<0, 0>>
// CHECK-NEXT:    parameter typename<0, 0> T
// CHECK-NEXT:    injected class name Box

// Specializations have injected class name
//      CHECK:class Box<int>
// CHECK:        injected class name Box

// Non-template derived from template has injected class name
//      CHECK:class IntBox
// CHECK-NEXT:    base class Box
// CHECK:        injected class name IntBox

// Multiple inheritance levels preserve injected class name
//      CHECK:class IntBoxDerived
// CHECK-NEXT:    base class IntBox
// CHECK:        injected class name IntBoxDerived
