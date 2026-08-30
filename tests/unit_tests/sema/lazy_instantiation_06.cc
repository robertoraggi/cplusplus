// RUN: %cxx -fsyntax-only -dump-symbols %s | %filecheck %s

template <typename T>
struct S {
  using type = T;
  static constexpr int size = sizeof(T);
};

using X = S<int>::type;

// clang-format off
//      CHECK:namespace
// CHECK-NEXT:  template class S<type-param<0, 0>>
// CHECK-NEXT:    parameter typename<0, 0> T
// CHECK-NEXT:    injected class name S
// CHECK-NEXT:    typealias type-param<0, 0> type
// CHECK-NEXT:    field static constexpr const int size
// CHECK-NEXT:    [specializations]
// CHECK-NEXT:      class S<int>
// CHECK-NEXT:        constructor constexpr inline defaulted void S()
// CHECK-NEXT:        constructor constexpr inline defaulted void S(const ::S<int>&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter const ::S<int>&
// CHECK-NEXT:        constructor constexpr inline defaulted void S(::S<int>&&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter ::S<int>&&
// CHECK-NEXT:        injected class name S
// CHECK-NEXT:        typealias int type
// CHECK-NEXT:        field static constexpr const int size
// CHECK-NEXT:        function constexpr inline defaulted ::S<int>& operator =(const ::S<int>&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter const ::S<int>&
// CHECK-NEXT:        function constexpr inline defaulted ::S<int>& operator =(::S<int>&&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter ::S<int>&&
// CHECK-NEXT:        function constexpr inline defaulted void ~S()
// CHECK-NEXT:  typealias int X
