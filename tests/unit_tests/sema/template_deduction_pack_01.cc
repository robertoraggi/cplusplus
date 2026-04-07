// RUN: %cxx -verify %s
// expected-no-diagnostics

template <class... Ts>
void sink(Ts...) {}

void test_pack() {
  sink(1, 2, 3);
  sink(1);
  sink();
}

template <class T, class... Us>
T first_of(T a, Us...) {
  return a;
}

static_assert(__is_same(decltype(first_of(42, 'a', 3.14)), int));
static_assert(__is_same(decltype(first_of('x', 1, 2)), char));
