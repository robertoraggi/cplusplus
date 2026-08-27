// RUN: %cxx -fsyntax-only -dump-symbols %s | %filecheck %s

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
// CHECK-NEXT:    injected class name Outer
// CHECK-NEXT:    template class Inner<type-param<0, 1>>
// CHECK-NEXT:      parameter typename<0, 1> U
// CHECK-NEXT:      injected class name Inner
// CHECK-NEXT:      field type-param<0, 0> t
// CHECK-NEXT:      field type-param<0, 1> u
// CHECK-NEXT:    [specializations]
// CHECK-NEXT:      class Outer<float>
// CHECK-NEXT:        constructor inline defaulted void Outer()
// CHECK-NEXT:        constructor inline defaulted void Outer(const ::Outer<float>&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter const ::Outer<float>&
// CHECK-NEXT:        constructor inline defaulted void Outer(::Outer<float>&&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter ::Outer<float>&&
// CHECK-NEXT:        injected class name Outer
// CHECK-NEXT:        template class Inner<type-param<0, 1>>
// CHECK-NEXT:          parameter typename<0, 1> U
// CHECK-NEXT:          injected class name Inner
// CHECK-NEXT:          field float t
// CHECK-NEXT:          field type-param<0, 1> u
// CHECK-NEXT:        function inline defaulted ::Outer<float>& operator =(const ::Outer<float>&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter const ::Outer<float>&
// CHECK-NEXT:        function inline defaulted ::Outer<float>& operator =(::Outer<float>&&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter ::Outer<float>&&
// CHECK-NEXT:        function inline defaulted void ~Outer()
// CHECK-NEXT:      class Outer<int>
// CHECK-NEXT:        constructor inline defaulted void Outer()
// CHECK-NEXT:        constructor inline defaulted void Outer(const ::Outer<int>&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter const ::Outer<int>&
// CHECK-NEXT:        constructor inline defaulted void Outer(::Outer<int>&&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter ::Outer<int>&&
// CHECK-NEXT:        injected class name Outer
// CHECK-NEXT:        template class Inner<type-param<0, 1>>
// CHECK-NEXT:          parameter typename<0, 1> U
// CHECK-NEXT:          injected class name Inner
// CHECK-NEXT:          field int t
// CHECK-NEXT:          field type-param<0, 1> u
// CHECK-NEXT:          [specializations]
// CHECK-NEXT:            class Inner<char>
// CHECK-NEXT:              constructor inline defaulted void Inner()
// CHECK-NEXT:              constructor inline defaulted void Inner(const ::Outer<int>::Inner<char>&)
// CHECK-NEXT:                parameters
// CHECK-NEXT:                  parameter const ::Outer<int>::Inner<char>&
// CHECK-NEXT:              constructor inline defaulted void Inner(::Outer<int>::Inner<char>&&)
// CHECK-NEXT:                parameters
// CHECK-NEXT:                  parameter ::Outer<int>::Inner<char>&&
// CHECK-NEXT:              injected class name Inner
// CHECK-NEXT:              field int t
// CHECK-NEXT:              field char u
// CHECK-NEXT:              function inline defaulted ::Outer<int>::Inner<char>& operator =(const ::Outer<int>::Inner<char>&)
// CHECK-NEXT:                parameters
// CHECK-NEXT:                  parameter const ::Outer<int>::Inner<char>&
// CHECK-NEXT:              function inline defaulted ::Outer<int>::Inner<char>& operator =(::Outer<int>::Inner<char>&&)
// CHECK-NEXT:                parameters
// CHECK-NEXT:                  parameter ::Outer<int>::Inner<char>&&
// CHECK-NEXT:              function inline defaulted void ~Inner()
// CHECK-NEXT:        function inline defaulted ::Outer<int>& operator =(const ::Outer<int>&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter const ::Outer<int>&
// CHECK-NEXT:        function inline defaulted ::Outer<int>& operator =(::Outer<int>&&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter ::Outer<int>&&
// CHECK-NEXT:        function inline defaulted void ~Outer()
