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
// CHECK-NEXT:        constructor defaulted void Outer()
// CHECK-NEXT:        constructor defaulted void Outer(const ::Outer<float>&)
// CHECK-NEXT:        constructor defaulted void Outer(::Outer<float>&&)
// CHECK-NEXT:        template class Inner<type-param<0, 1>>
// CHECK-NEXT:          parameter typename<0, 1> U
// CHECK-NEXT:          field float t
// CHECK-NEXT:          field type-param<0, 1> u
// CHECK-NEXT:        function defaulted ::Outer<float>& operator =(const ::Outer<float>&)
// CHECK-NEXT:        function defaulted ::Outer<float>& operator =(::Outer<float>&&)
// CHECK-NEXT:        function defaulted void ~Outer()
// CHECK-NEXT:      class Outer<int>
// CHECK-NEXT:        constructor defaulted void Outer()
// CHECK-NEXT:        constructor defaulted void Outer(const ::Outer<int>&)
// CHECK-NEXT:        constructor defaulted void Outer(::Outer<int>&&)
// CHECK-NEXT:        template class Inner<type-param<0, 1>>
// CHECK-NEXT:          parameter typename<0, 1> U
// CHECK-NEXT:          field int t
// CHECK-NEXT:          field type-param<0, 1> u
// CHECK-NEXT:          [specializations]
// CHECK-NEXT:            class Inner<char>
// CHECK-NEXT:              constructor defaulted void Inner()
// CHECK-NEXT:              constructor defaulted void Inner(const ::Outer<int>::Inner<char>&)
// CHECK-NEXT:              constructor defaulted void Inner(::Outer<int>::Inner<char>&&)
// CHECK-NEXT:              field int t
// CHECK-NEXT:              field char u
// CHECK-NEXT:              function defaulted ::Outer<int>::Inner<char>& operator =(const ::Outer<int>::Inner<char>&)
// CHECK-NEXT:              function defaulted ::Outer<int>::Inner<char>& operator =(::Outer<int>::Inner<char>&&)
// CHECK-NEXT:              function defaulted void ~Inner()
// CHECK-NEXT:        function defaulted ::Outer<int>& operator =(const ::Outer<int>&)
// CHECK-NEXT:        function defaulted ::Outer<int>& operator =(::Outer<int>&&)
// CHECK-NEXT:        function defaulted void ~Outer()
