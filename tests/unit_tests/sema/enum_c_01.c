// RUN: %cxx -verify -dump-symbols -xc %s | %filecheck %s

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
// CHECK-NEXT:  enum X : int
// CHECK-NEXT:  enumerator X A = 0
// CHECK-NEXT:  enumerator X B = 1
// CHECK-NEXT:  enumerator X C = 1
// CHECK-NEXT:  enumerator X D = 2
// CHECK-NEXT:  variable X x