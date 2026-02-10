// RUN: %cxx -verify -fcheck -dump-symbols %s | %filecheck %s

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
// CHECK-NEXT:    field type-param<0, 0> value
// CHECK-NEXT:    [specializations]
// CHECK-NEXT:      class Holder<int>
// CHECK-NEXT:        constructor defaulted void Holder()
// CHECK-NEXT:        constructor defaulted void Holder(const ::Holder<int>&)
// CHECK-NEXT:        constructor defaulted void Holder(::Holder<int>&&)
// CHECK-NEXT:        field int value
// CHECK-NEXT:        function defaulted void ~Holder()
// CHECK-NEXT:      class Holder<double>
// CHECK-NEXT:        constructor defaulted void Holder()
// CHECK-NEXT:        constructor defaulted void Holder(const ::Holder<double>&)
// CHECK-NEXT:        constructor defaulted void Holder(::Holder<double>&&)
// CHECK-NEXT:        field double value
// CHECK-NEXT:        function defaulted void ~Holder()
// CHECK-NEXT:  template class Wrap<template-type-param<0, 0>, type-param<1, 0>>
// CHECK-NEXT:    parameter template<0, 0> C
// CHECK-NEXT:    parameter typename<1, 0> T
// CHECK-NEXT:    field template-type-param<0, 0> member
// CHECK-NEXT:    [specializations]
// CHECK-NEXT:      class Wrap<::Holder, int>
// CHECK-NEXT:        constructor defaulted void Wrap()
// CHECK-NEXT:        constructor defaulted void Wrap(const ::Wrap<::Holder, int>&)
// CHECK-NEXT:        constructor defaulted void Wrap(::Wrap<::Holder, int>&&)
// CHECK-NEXT:        field ::Holder<int> member
// CHECK-NEXT:        function defaulted void ~Wrap()
// CHECK-NEXT:  variable ::Wrap<::Holder, int> w1
// CHECK-NEXT:  template class ApplyInt<template-type-param<0, 0>>
// CHECK-NEXT:    parameter template<0, 0> C
// CHECK-NEXT:    typealias template-type-param<0, 0> type
// CHECK-NEXT:    [specializations]
// CHECK-NEXT:      class ApplyInt<::Holder>
// CHECK-NEXT:        constructor defaulted void ApplyInt()
// CHECK-NEXT:        constructor defaulted void ApplyInt(const ::ApplyInt<::Holder>&)
// CHECK-NEXT:        constructor defaulted void ApplyInt(::ApplyInt<::Holder>&&)
// CHECK-NEXT:        typealias ::Holder<int> type
// CHECK-NEXT:        function defaulted void ~ApplyInt()
// CHECK-NEXT:  variable ::ApplyInt<::Holder> a1
// CHECK-NEXT:  template class Multi<template-type-param<0, 0>>
// CHECK-NEXT:    parameter template<0, 0> C
// CHECK-NEXT:    field template-type-param<0, 0> a
// CHECK-NEXT:    field template-type-param<0, 0> b
// CHECK-NEXT:    [specializations]
// CHECK-NEXT:      class Multi<::Holder>
// CHECK-NEXT:        constructor defaulted void Multi()
// CHECK-NEXT:        constructor defaulted void Multi(const ::Multi<::Holder>&)
// CHECK-NEXT:        constructor defaulted void Multi(::Multi<::Holder>&&)
// CHECK-NEXT:        field ::Holder<int> a
// CHECK-NEXT:        field ::Holder<double> b
// CHECK-NEXT:        function defaulted void ~Multi()
// CHECK-NEXT:  variable ::Multi<::Holder> m1
