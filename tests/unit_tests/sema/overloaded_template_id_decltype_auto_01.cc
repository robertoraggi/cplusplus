// RUN: %cxx -std=c++20 -verify -fsyntax-only %s
// expected-no-diagnostics

template <bool, class T = void>
struct Enable {};

template <class T>
struct Enable<true, T> {
  using type = T;
};

template <class T, typename Enable<(T(-1) < T(0)), int>::type = 0>
constexpr int f(T) {
  return 1;
}

template <class T, typename Enable<(T(-1) > T(0)), int>::type = 0>
constexpr int f(T) {
  return 2;
}

decltype(auto) reference = (f<int>);
static_assert(__is_same(decltype(reference), int (&)(int)));
static_assert(reference(0) == 1);
