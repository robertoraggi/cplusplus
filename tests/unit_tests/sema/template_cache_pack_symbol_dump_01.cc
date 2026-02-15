// RUN: %cxx -fcheck -dump-symbols %s | %filecheck %s

template <typename... Ts>
struct Pack {};

template <typename T>
using Alias = T;

Pack<int, Alias<int>> p;
Pack<int, int> q;

// clang-format off
//      CHECK:namespace
// CHECK-NEXT:  template class Pack<type-param<0, 0>...>
// CHECK-NEXT:    parameter typename<0, 0>... Ts
// CHECK-NEXT:    [specializations]
// CHECK-NEXT:      class Pack<>
// CHECK-NEXT:        constructor defaulted void Pack()
// CHECK-NEXT:        constructor defaulted void Pack(const ::Pack<>&)
// CHECK-NEXT:        constructor defaulted void Pack(::Pack<>&&)
// CHECK-NEXT:        function defaulted ::Pack<>& operator =(const ::Pack<>&)
// CHECK-NEXT:        function defaulted ::Pack<>& operator =(::Pack<>&&)
// CHECK-NEXT:        function defaulted void ~Pack()
// CHECK-NEXT:  template typealias type-param<0, 0> Alias
// CHECK-NEXT:    parameter typename<0, 0> T
// CHECK-NEXT:  variable ::Pack<> p
// CHECK-NEXT:  variable ::Pack<> q
