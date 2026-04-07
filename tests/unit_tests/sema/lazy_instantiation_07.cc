// RUN: %cxx -dump-symbols %s | %filecheck %s

template <typename T>
struct S {
  T value;
};

template <typename T>
using Ptr = S<T>*;

template <typename T>
using Ref = S<T>&;

Ptr<int> p = nullptr;

Ptr<double> p2 = nullptr;

// clang-format off
//      CHECK:namespace
// CHECK-NEXT:  template class S<type-param<0, 0>>
// CHECK-NEXT:    parameter typename<0, 0> T
// CHECK-NEXT:    injected class name S
// CHECK-NEXT:    field type-param<0, 0> value
// CHECK-NEXT:    [specializations]
// CHECK-NEXT:      class S<int>
// CHECK-NEXT:      class S<double>
// CHECK-NEXT:  template typealias ::S* Ptr
// CHECK-NEXT:    parameter typename<0, 0> T
// CHECK-NEXT:  template typealias ::S& Ref
// CHECK-NEXT:    parameter typename<0, 0> T
// CHECK-NEXT:  variable ::S<int>* p
// CHECK-NEXT:  variable ::S<double>* p2
