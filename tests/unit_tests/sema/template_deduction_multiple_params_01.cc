// RUN: %cxx -verify %s
// expected-no-diagnostics

template <class T>
T same(T a, T b) {
  return a;
}

static_assert(__is_same(decltype(same(1, 2)), int));
static_assert(__is_same(decltype(same(1.0, 2.0)), double));
