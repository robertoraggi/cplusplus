// clang-format off
// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

struct A {};
struct B {};

template <class T>
struct Foo { using type = T; };

// Non-template: Foo<A>::type as base
struct test1 : Foo<A>::type {};

// Template: Foo<A>::type as base
template <class T>
struct test2 : Foo<A>::type {};
test2<int> t2;

// Partial specialization + NNS base
template <bool _Bp, class _If, class _Then>
struct conditional { using type = _If; };

template <class _If, class _Then>
struct conditional<false, _If, _Then> { using type = _Then; };

// conditional NNS with literal true — template context
template <class T>
struct test3 : conditional<true, A, B>::type {};
test3<int> t3;

// conditional NNS with computed bool (integral_constant::value)
template <class _Tp, _Tp __v>
struct integral_constant {
  static constexpr _Tp value = __v;
};

template <bool __v>
using BoolConstant = integral_constant<bool, __v>;

template <class _Tp, class _Up>
using IsSame = BoolConstant<__is_same(_Tp, _Up)>;

// Nested bool value through alias template as NNS
template <bool _Bp, class _If, class _Then>
using conditional_t = typename conditional<_Bp, _If, _Then>::type;

template <class T, class U>
struct test4 : conditional_t<IsSame<T, U>::value, A, B> {};

test4<int, int> t4_same;
test4<int, double> t4_diff;
