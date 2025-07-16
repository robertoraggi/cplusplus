// RUN: %cxx -fcheck -dump-symbols -xc %s | %filecheck %s --match-full-lines

enum E {
  v,
};

struct A {
  enum E e;
  int x;
};

int main() {
  enum K {
    w,
  };

  struct B {
    enum K k;
    enum E e;
    int y;
  };

  struct M;
}

// clang-format off
//      CHECK:namespace
// CHECK-NEXT:  enum E : int
// CHECK-NEXT:  enumerator E v
// CHECK-NEXT:  class A
// CHECK-NEXT:    field E e
// CHECK-NEXT:    field int x
// CHECK-NEXT:  function int main()
// CHECK-NEXT:    block
// CHECK-NEXT:      enum K : int
// CHECK-NEXT:      enumerator K w
// CHECK-NEXT:      class B
// CHECK-NEXT:        field K k
// CHECK-NEXT:        field E e
// CHECK-NEXT:        field int y
// CHECK-NEXT:      class M
