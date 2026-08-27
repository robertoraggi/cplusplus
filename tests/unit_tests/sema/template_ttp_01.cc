// RUN: %cxx -verify -fsyntax-only -dump-symbols %s | %filecheck %s

template <typename T>
struct Holder {
  T value;
};

template <template <typename> class C, typename T>
struct Wrap {
  C<T> member;
};

Wrap<Holder, int> w1;

template <template <typename> class C>
struct ApplyInt {
  using type = C<int>;
};

ApplyInt<Holder> a1;

template <template <typename> class C>
struct Multi {
  C<int> a;
  C<double> b;
};

Multi<Holder> m1;

// clang-format off
//      CHECK:namespace
// CHECK-NEXT:  template class Holder<type-param<0, 0>>
// CHECK-NEXT:    parameter typename<0, 0> T
// CHECK-NEXT:    injected class name Holder
// CHECK-NEXT:    field type-param<0, 0> value
// CHECK-NEXT:    [specializations]
// CHECK-NEXT:      class Holder<int>
// CHECK-NEXT:        constructor inline defaulted void Holder()
// CHECK-NEXT:        constructor inline defaulted void Holder(const ::Holder<int>&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter const ::Holder<int>&
// CHECK-NEXT:        constructor inline defaulted void Holder(::Holder<int>&&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter ::Holder<int>&&
// CHECK-NEXT:        injected class name Holder
// CHECK-NEXT:        field int value
// CHECK-NEXT:        function inline defaulted ::Holder<int>& operator =(const ::Holder<int>&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter const ::Holder<int>&
// CHECK-NEXT:        function inline defaulted ::Holder<int>& operator =(::Holder<int>&&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter ::Holder<int>&&
// CHECK-NEXT:        function inline defaulted void ~Holder()
// CHECK-NEXT:      class Holder<double>
// CHECK-NEXT:        constructor inline defaulted void Holder()
// CHECK-NEXT:        constructor inline defaulted void Holder(const ::Holder<double>&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter const ::Holder<double>&
// CHECK-NEXT:        constructor inline defaulted void Holder(::Holder<double>&&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter ::Holder<double>&&
// CHECK-NEXT:        injected class name Holder
// CHECK-NEXT:        field double value
// CHECK-NEXT:        function inline defaulted ::Holder<double>& operator =(const ::Holder<double>&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter const ::Holder<double>&
// CHECK-NEXT:        function inline defaulted ::Holder<double>& operator =(::Holder<double>&&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter ::Holder<double>&&
// CHECK-NEXT:        function inline defaulted void ~Holder()
// CHECK-NEXT:  template class Wrap<template-type-param<0, 0>, type-param<1, 0>>
// CHECK-NEXT:    parameter template<0, 0> C
// CHECK-NEXT:    parameter typename<1, 0> T
// CHECK-NEXT:    injected class name Wrap
// CHECK-NEXT:    field template-type-param<0, 0> member
// CHECK-NEXT:    [specializations]
// CHECK-NEXT:      class Wrap<::Holder, int>
// CHECK-NEXT:        constructor inline defaulted void Wrap()
// CHECK-NEXT:        constructor inline defaulted void Wrap(const ::Wrap<::Holder, int>&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter const ::Wrap<::Holder, int>&
// CHECK-NEXT:        constructor inline defaulted void Wrap(::Wrap<::Holder, int>&&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter ::Wrap<::Holder, int>&&
// CHECK-NEXT:        injected class name Wrap
// CHECK-NEXT:        field ::Holder<int> member
// CHECK-NEXT:        function inline defaulted ::Wrap<::Holder, int>& operator =(const ::Wrap<::Holder, int>&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter const ::Wrap<::Holder, int>&
// CHECK-NEXT:        function inline defaulted ::Wrap<::Holder, int>& operator =(::Wrap<::Holder, int>&&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter ::Wrap<::Holder, int>&&
// CHECK-NEXT:        function inline defaulted void ~Wrap()
// CHECK-NEXT:  variable ::Wrap<::Holder, int> w1
// CHECK-NEXT:  template class ApplyInt<template-type-param<0, 0>>
// CHECK-NEXT:    parameter template<0, 0> C
// CHECK-NEXT:    injected class name ApplyInt
// CHECK-NEXT:    typealias template-type-param<0, 0> type
// CHECK-NEXT:    [specializations]
// CHECK-NEXT:      class ApplyInt<::Holder>
// CHECK-NEXT:        constructor inline defaulted void ApplyInt()
// CHECK-NEXT:        constructor inline defaulted void ApplyInt(const ::ApplyInt<::Holder>&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter const ::ApplyInt<::Holder>&
// CHECK-NEXT:        constructor inline defaulted void ApplyInt(::ApplyInt<::Holder>&&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter ::ApplyInt<::Holder>&&
// CHECK-NEXT:        injected class name ApplyInt
// CHECK-NEXT:        typealias ::Holder<int> type
// CHECK-NEXT:        function inline defaulted ::ApplyInt<::Holder>& operator =(const ::ApplyInt<::Holder>&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter const ::ApplyInt<::Holder>&
// CHECK-NEXT:        function inline defaulted ::ApplyInt<::Holder>& operator =(::ApplyInt<::Holder>&&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter ::ApplyInt<::Holder>&&
// CHECK-NEXT:        function inline defaulted void ~ApplyInt()
// CHECK-NEXT:  variable ::ApplyInt<::Holder> a1
// CHECK-NEXT:  template class Multi<template-type-param<0, 0>>
// CHECK-NEXT:    parameter template<0, 0> C
// CHECK-NEXT:    injected class name Multi
// CHECK-NEXT:    field template-type-param<0, 0> a
// CHECK-NEXT:    field template-type-param<0, 0> b
// CHECK-NEXT:    [specializations]
// CHECK-NEXT:      class Multi<::Holder>
// CHECK-NEXT:        constructor inline defaulted void Multi()
// CHECK-NEXT:        constructor inline defaulted void Multi(const ::Multi<::Holder>&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter const ::Multi<::Holder>&
// CHECK-NEXT:        constructor inline defaulted void Multi(::Multi<::Holder>&&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter ::Multi<::Holder>&&
// CHECK-NEXT:        injected class name Multi
// CHECK-NEXT:        field ::Holder<int> a
// CHECK-NEXT:        field ::Holder<double> b
// CHECK-NEXT:        function inline defaulted ::Multi<::Holder>& operator =(const ::Multi<::Holder>&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter const ::Multi<::Holder>&
// CHECK-NEXT:        function inline defaulted ::Multi<::Holder>& operator =(::Multi<::Holder>&&)
// CHECK-NEXT:          parameters
// CHECK-NEXT:            parameter ::Multi<::Holder>&&
// CHECK-NEXT:        function inline defaulted void ~Multi()
// CHECK-NEXT:  variable ::Multi<::Holder> m1
