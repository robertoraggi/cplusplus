// RUN: %cxx -verify -fcheck -dump-symbols -xc %s | %filecheck %s

enum X {
  A,
  B,
  C,
};

enum X x;

//      CHECK:namespace
// CHECK-NEXT:enum X : int
// CHECK-NEXT:enumerator X A
// CHECK-NEXT:enumerator X B
// CHECK-NEXT:enumerator X C
// CHECK-NEXT:variable X x