// RUN: %cxx -fsyntax-only -dump-symbols %s | %filecheck %s

template <typename... Ts>
struct Pack {};

template <typename T>
using Alias = T;

template <int... Ns>
struct Indices {};

Pack<int, Alias<int>> p;
Pack<int, int> q;

// clang-format off
//      CHECK:namespace
// CHECK-NEXT:  template class Pack<type-param<0, 0>...>
// CHECK-NEXT:    parameter typename<0, 0>... Ts
// CHECK-NEXT:    injected class name Pack
// CHECK-NEXT:    [specializations]
// CHECK-NEXT:      class Pack<int, int>
// CHECK-NEXT:        constructor constexpr inline defaulted void Pack()
// CHECK-NEXT:        constructor constexpr inline defaulted void Pack(const ::Pack<int, int>&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter const ::Pack<int, int>&
// CHECK-NEXT:        constructor constexpr inline defaulted void Pack(::Pack<int, int>&&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter ::Pack<int, int>&&
// CHECK-NEXT:        injected class name Pack
// CHECK-NEXT:        function constexpr inline defaulted ::Pack<int, int>& operator =(const ::Pack<int, int>&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter const ::Pack<int, int>&
// CHECK-NEXT:        function constexpr inline defaulted ::Pack<int, int>& operator =(::Pack<int, int>&&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter ::Pack<int, int>&&
// CHECK-NEXT:        function constexpr inline defaulted void ~Pack()
// CHECK-NEXT:  template typealias type-param<0, 0> Alias
// CHECK-NEXT:    parameter typename<0, 0> T
// CHECK-NEXT:  template class Indices<int...>
// CHECK-NEXT:    parameter constant<0, 0, int>... Ns
// CHECK-NEXT:    injected class name Indices
// CHECK-NEXT:  variable ::Pack<int, int> p
// CHECK-NEXT:  variable ::Pack<int, int> q
