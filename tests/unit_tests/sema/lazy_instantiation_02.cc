// RUN: %cxx -fcheck -dump-symbols %s | %filecheck %s

template <typename T>
struct S {
  T value;
};

using Alias = S<int>;
using AliasPtr = S<double>*;

// clang-format off
//      CHECK:namespace
// CHECK-NEXT:  template class S<type-param<0, 0>>
// CHECK-NEXT:    parameter typename<0, 0> T
// CHECK-NEXT:    injected class name S
// CHECK-NEXT:    field type-param<0, 0> value
// CHECK-NEXT:    [specializations]
// CHECK-NEXT:      class S<int>
// CHECK-NEXT:      class S<double>
// CHECK-NEXT:  typealias ::S<int> Alias
// CHECK-NEXT:  typealias ::S<double>* AliasPtr
