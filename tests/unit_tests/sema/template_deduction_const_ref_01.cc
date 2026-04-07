// RUN: %cxx -verify %s
// expected-no-diagnostics

template <typename T>
const T& cref(const T& t) {
  return t;
}

template <typename T>
T val(T t) {
  return t;
}

void test_sequential_const_ref_deduction() {
  int i = 42;
  double d = 3.14;
  long l = 100;

  auto a = cref(i);
  static_assert(__is_same(decltype(a), int));

  auto b = cref(d);
  static_assert(__is_same(decltype(b), double));

  auto c = cref(l);
  static_assert(__is_same(decltype(c), long));

  auto d2 = cref(i);
  static_assert(__is_same(decltype(d2), int));
}

void test_const_ref_with_const_source() {
  const int ci = 10;
  auto a = cref(ci);
  static_assert(__is_same(decltype(a), int));
}

void test_sequential_value_deduction() {
  int i = 42;
  double d = 3.14;

  auto a = val(i);
  static_assert(__is_same(decltype(a), int));

  auto b = val(d);
  static_assert(__is_same(decltype(b), double));
}
