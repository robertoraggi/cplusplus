// RUN: %cxx -verify %s
// expected-no-diagnostics

template <class T>
T identity(T x) {
  return x;
}

static_assert(__is_same(decltype(identity(42)), int));
static_assert(__is_same(decltype(identity(3.14)), double));
static_assert(__is_same(decltype(identity('c')), char));

template <class T, class U>
T first(T a, U b) {
  return a;
}

static_assert(__is_same(decltype(first(1, 2.0)), int));
static_assert(__is_same(decltype(first(2.0, 1)), double));
