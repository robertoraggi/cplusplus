// RUN: %cxx -fcheck -dump-symbols %s | %filecheck %s

template <typename T>
struct Outer {
  template <typename U>
  struct Inner {
    T t;
    U u;
  };
};

template struct Outer<float>;

template struct Outer<int>::Inner<char>;

// clang-format off
//      CHECK:namespace
// CHECK-NEXT:  template class Outer<type-param<0, 0>>
// CHECK-NEXT:    parameter typename<0, 0> T
// CHECK-NEXT:    template class Inner<type-param<0, 1>>
// CHECK-NEXT:      parameter typename<0, 1> U
// CHECK-NEXT:      field type-param<0, 0> t
// CHECK-NEXT:      field type-param<0, 1> u
// CHECK-NEXT:    [specializations]
// CHECK-NEXT:      class Outer<float>
// CHECK-NEXT:        template class Inner<type-param<0, 1>>
// CHECK-NEXT:          parameter typename<0, 1> U
// CHECK-NEXT:          field float t
// CHECK-NEXT:          field type-param<0, 1> u
// CHECK-NEXT:      class Outer<int>
// CHECK-NEXT:        template class Inner<type-param<0, 1>>
// CHECK-NEXT:          parameter typename<0, 1> U
// CHECK-NEXT:          field int t
// CHECK-NEXT:          field type-param<0, 1> u
// CHECK-NEXT:          [specializations]
// CHECK-NEXT:            class Inner<char>
// CHECK-NEXT:              field int t
// CHECK-NEXT:              field char u
