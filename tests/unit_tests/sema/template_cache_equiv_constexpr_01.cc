// RUN: %cxx -fcheck -dump-symbols %s | %filecheck %s

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
// CHECK-NEXT:    [specializations]
// CHECK-NEXT:      class Box<3>
// CHECK-NEXT:        constructor defaulted void Box()
// CHECK-NEXT:        constructor defaulted void Box(const ::Box<int>&)
// CHECK-NEXT:        constructor defaulted void Box(::Box<int>&&)
// CHECK-NEXT:        function defaulted void ~Box()
// CHECK-NEXT:  function constexpr int three()
// CHECK-NEXT:    block
// CHECK-NEXT:      variable static constexpr const char __func__[6]
// CHECK-NEXT:  variable constexpr const int k
// CHECK-NEXT:  variable ::Box<int> a
// CHECK-NEXT:  variable ::Box<int> b
// CHECK-NEXT:  variable ::Box<int> c
