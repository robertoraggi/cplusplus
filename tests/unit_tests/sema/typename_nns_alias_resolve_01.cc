// clang-format off
// RUN: %cxx -verify -fcheck %s
// expected-no-diagnostics

struct A {};
struct B {};

template <bool _Bp, class _If, class _Then>
struct conditional { using type = _If; };

template <class _If, class _Then>
struct conditional<false, _If, _Then> { using type = _Then; };

// typename through class template NNS — non-template context
using R1 = typename conditional<true, A, B>::type;
static_assert(__is_same(R1, A), "R1 should be A");

using R2 = typename conditional<false, A, B>::type;
static_assert(__is_same(R2, B), "R2 should be B");

// Alias template using typename NNS
template <bool _Bp, class _If, class _Then>
using conditional_t = typename conditional<_Bp, _If, _Then>::type;

// Alias template as base class — non-template context
struct test1 : conditional_t<true, A, B> {};

// Alias template as base class — template context
template <class T>
struct test2 : conditional_t<true, A, B> {};
test2<int> t2;

// Alias template with dependent typename NNS
template <class T>
struct wrap { using type = T; };

template <class T>
using unwrap_t = typename wrap<T>::type;

struct test3 : unwrap_t<A> {};
static_assert(__is_same(unwrap_t<A>, A), "unwrap_t<A> should be A");
