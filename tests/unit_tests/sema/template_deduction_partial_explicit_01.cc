// RUN: %cxx -verify %s
// expected-no-diagnostics

template <typename R, typename T>
R convert(T t) {
  return static_cast<R>(t);
}

template <typename T, typename U>
T first_of_two(T a, U b) {
  return a;
}

void test_partial_explicit_args() {
  int i = 42;

  double d = convert<double>(i);
  static_assert(__is_same(decltype(d), double));

  float f = convert<float>(3.14);
  static_assert(__is_same(decltype(f), float));

  long l = convert<long>(i);
  static_assert(__is_same(decltype(l), long));
}

void test_explicit_first_deduce_second() {
  auto r = first_of_two<int>(1, 2.0);
  static_assert(__is_same(decltype(r), int));

  auto r2 = first_of_two<double>(1.0, 42);
  static_assert(__is_same(decltype(r2), double));
}
