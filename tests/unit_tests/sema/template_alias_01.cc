// RUN: %cxx -fcheck -dump-symbols %s | %filecheck %s

template <typename T>
using Pointer = T*;

using IntPointer = Pointer<int>;

// clang-format off
//      CHECK:namespace
// CHECK-NEXT:  template typealias type-param<0, 0>* Pointer
// CHECK-NEXT:    parameter typename<0, 0> T
// CHECK-NEXT:  typealias int* IntPointer
