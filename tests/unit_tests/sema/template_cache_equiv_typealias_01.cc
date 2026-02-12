// RUN: %cxx -fcheck -dump-symbols %s | %filecheck %s

template <class T>
struct Box {};

template <class T>
using Alias = T;

using IntAlias = Alias<int>;

Box<int> a;
Box<IntAlias> b;

// clang-format off
//      CHECK:namespace
// CHECK-NEXT:  template class Box<type-param<0, 0>>
// CHECK-NEXT:    parameter typename<0, 0> T
// CHECK-NEXT:    [specializations]
// CHECK-NEXT:      class Box<int>
// CHECK-NEXT:        constructor defaulted void Box()
// CHECK-NEXT:        constructor defaulted void Box(const ::Box<int>&)
// CHECK-NEXT:        constructor defaulted void Box(::Box<int>&&)
// CHECK-NEXT:        function defaulted void ~Box()
// CHECK-NEXT:  template typealias type-param<0, 0> Alias
// CHECK-NEXT:    parameter typename<0, 0> T
// CHECK-NEXT:  typealias int IntAlias
// CHECK-NEXT:  variable ::Box<int> a
// CHECK-NEXT:  variable ::Box<int> b
