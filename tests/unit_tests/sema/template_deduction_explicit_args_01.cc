// RUN: %cxx -verify %s
// expected-no-diagnostics

template <class T, class U>
T pair_first(T a, U b) {
  return a;
}

int r1 = pair_first(1, 2.0);
double r2 = pair_first(2.0, 1);

static_assert(__is_same(decltype(pair_first(1, 2.0)), int));
static_assert(__is_same(decltype(pair_first(2.0, 1)), double));
