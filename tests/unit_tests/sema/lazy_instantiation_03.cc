// RUN: %cxx -fcheck -dump-symbols %s | %filecheck %s

template <typename T>
struct S {
  T value;
};

S<int> var;

// clang-format off
//      CHECK:namespace
// CHECK-NEXT:  template class S<type-param<0, 0>>
// CHECK-NEXT:    parameter typename<0, 0> T
// CHECK-NEXT:    injected class name S
// CHECK-NEXT:    field type-param<0, 0> value
// CHECK-NEXT:    [specializations]
// CHECK-NEXT:      class S<int>
// CHECK-NEXT:        constructor defaulted void S()
// CHECK-NEXT:        constructor defaulted void S(const ::S<int>&)
// CHECK-NEXT:        constructor defaulted void S(::S<int>&&)
// CHECK-NEXT:        injected class name S
// CHECK-NEXT:        field int value
// CHECK-NEXT:        function defaulted ::S<int>& operator =(const ::S<int>&)
// CHECK-NEXT:        function defaulted ::S<int>& operator =(::S<int>&&)
// CHECK-NEXT:        function defaulted void ~S()
// CHECK-NEXT:  variable ::S<int> var
