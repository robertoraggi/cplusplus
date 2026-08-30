// RUN: %cxx -fsyntax-only -dump-symbols %s | %filecheck %s

template <typename T>
struct Box {};

template <typename T>
using Alias = T;

Box<int> a;
Box<Alias<int>> b;

// clang-format off
//      CHECK:namespace
// CHECK-NEXT:  template class Box<type-param<0, 0>>
// CHECK-NEXT:    parameter typename<0, 0> T
// CHECK-NEXT:    injected class name Box
// CHECK-NEXT:    [specializations]
// CHECK-NEXT:      class Box<int>
// CHECK-NEXT:        constructor constexpr inline defaulted void Box()
// CHECK-NEXT:        constructor constexpr inline defaulted void Box(const ::Box<int>&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter const ::Box<int>&
// CHECK-NEXT:        constructor constexpr inline defaulted void Box(::Box<int>&&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter ::Box<int>&&
// CHECK-NEXT:        injected class name Box
// CHECK-NEXT:        function constexpr inline defaulted ::Box<int>& operator =(const ::Box<int>&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter const ::Box<int>&
// CHECK-NEXT:        function constexpr inline defaulted ::Box<int>& operator =(::Box<int>&&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter ::Box<int>&&
// CHECK-NEXT:        function constexpr inline defaulted void ~Box()
// CHECK-NEXT:  template typealias type-param<0, 0> Alias
// CHECK-NEXT:    parameter typename<0, 0> T
// CHECK-NEXT:  variable ::Box<int> a
// CHECK-NEXT:  variable ::Box<int> b
