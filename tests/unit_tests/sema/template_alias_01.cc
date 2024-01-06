// RUN: %cxx -fcheck -dump-symbols %s

template <typename T>
using Pointer = T*;

using IntPointer = Pointer<int>;

// clang-format off
// CHECK: namespace
// CHECK:   template typealias T* Pointer
// CHECK:     parameter typename<0, 0> T
// CHECK:   typealias T* IntPointer