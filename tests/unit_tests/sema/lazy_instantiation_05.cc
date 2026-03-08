// RUN: %cxx -fcheck -dump-symbols %s | %filecheck %s

template <typename T>
struct S {
  T value;
};

S<float>* p1 = nullptr;
S<float>& r1 = *p1;
using A1 = S<double>;

S<int> v1;
S<char> v2;

// clang-format off
//      CHECK:namespace
// CHECK-NEXT:  template class S<type-param<0, 0>>
// CHECK-NEXT:    parameter typename<0, 0> T
// CHECK-NEXT:    injected class name S
// CHECK-NEXT:    field type-param<0, 0> value
// CHECK-NEXT:    [specializations]
// CHECK-NEXT:      class S<float>
// CHECK-NEXT:      class S<double>
// CHECK-NEXT:      class S<int>
// CHECK-NEXT:        constructor defaulted void S()
// CHECK-NEXT:        constructor defaulted void S(const ::S<int>&)
// CHECK-NEXT:        constructor defaulted void S(::S<int>&&)
// CHECK-NEXT:        injected class name S
// CHECK-NEXT:        field int value
// CHECK-NEXT:        function defaulted ::S<int>& operator =(const ::S<int>&)
// CHECK-NEXT:        function defaulted ::S<int>& operator =(::S<int>&&)
// CHECK-NEXT:        function defaulted void ~S()
// CHECK-NEXT:      class S<char>
// CHECK-NEXT:        constructor defaulted void S()
// CHECK-NEXT:        constructor defaulted void S(const ::S<char>&)
// CHECK-NEXT:        constructor defaulted void S(::S<char>&&)
// CHECK-NEXT:        injected class name S
// CHECK-NEXT:        field char value
// CHECK-NEXT:        function defaulted ::S<char>& operator =(const ::S<char>&)
// CHECK-NEXT:        function defaulted ::S<char>& operator =(::S<char>&&)
// CHECK-NEXT:        function defaulted void ~S()
// CHECK-NEXT:  variable ::S<float>* p1
// CHECK-NEXT:  variable ::S<float>& r1
// CHECK-NEXT:  typealias ::S<double> A1
// CHECK-NEXT:  variable ::S<int> v1
// CHECK-NEXT:  variable ::S<char> v2
