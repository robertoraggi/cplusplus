// RUN: %cxx -fsyntax-only -dump-symbols %s | %filecheck %s

template <int N>
struct Box {};

constexpr int three() { return 3; }
constexpr int k = 3;

Box<three()> a;
Box<1 + 2> b;
Box<k> c;

// clang-format off
//      CHECK:namespace
// CHECK-NEXT:  template class Box<int>
// CHECK-NEXT:    parameter constant<0, 0, int> N
// CHECK-NEXT:    injected class name Box
// CHECK-NEXT:    [specializations]
// CHECK-NEXT:      class Box<3>
// CHECK-NEXT:        constructor constexpr inline defaulted void Box()
// CHECK-NEXT:        constructor constexpr inline defaulted void Box(const ::Box<3>&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter const ::Box<3>&
// CHECK-NEXT:        constructor constexpr inline defaulted void Box(::Box<3>&&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter ::Box<3>&&
// CHECK-NEXT:        injected class name Box
// CHECK-NEXT:        function constexpr inline defaulted ::Box<3>& operator =(const ::Box<3>&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter const ::Box<3>&
// CHECK-NEXT:        function constexpr inline defaulted ::Box<3>& operator =(::Box<3>&&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter ::Box<3>&&
// CHECK-NEXT:        function constexpr inline defaulted void ~Box()
// CHECK-NEXT:  function constexpr inline int three()
// CHECK-NEXT:    block
// CHECK-NEXT:      variable static constexpr const char __func__[6]
// CHECK-NEXT:  variable constexpr const int k
// CHECK-NEXT:  variable ::Box<3> a
// CHECK-NEXT:  variable ::Box<3> b
// CHECK-NEXT:  variable ::Box<3> c
