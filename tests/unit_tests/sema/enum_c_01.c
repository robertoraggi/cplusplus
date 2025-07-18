// RUN: %cxx -verify -fcheck -dump-symbols -xc %s | %filecheck %s

enum X {
  A,
  B = 1,
  C = B,
  D,
};

static_assert(A == 0);
static_assert(B == 1);
static_assert(C == 1);
static_assert(D == 2);

enum X x;

//      CHECK:namespace
// CHECK-NEXT:enum X : int
// CHECK-NEXT:enumerator X A
// CHECK-NEXT:enumerator X B
// CHECK-NEXT:enumerator X C
// CHECK-NEXT:enumerator X D
// CHECK-NEXT:variable X x