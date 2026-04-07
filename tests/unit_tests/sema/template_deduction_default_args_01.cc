// RUN: %cxx -verify %s
// expected-no-diagnostics

template <typename T, typename U = int>
T with_default(T t, U u) {
  return t;
}

template <typename T, typename U = double, typename V = int>
T multi_default(T t, U u) {
  return t;
}

void test_default_template_args() {
  auto a = with_default(42, 0);
  static_assert(__is_same(decltype(a), int));

  auto c = with_default(42, 3.14);
  static_assert(__is_same(decltype(c), int));
}

void test_multiple_defaults() {
  auto a = multi_default(42, 1.0);
  static_assert(__is_same(decltype(a), int));

  auto b = multi_default(3.14, 0);
  static_assert(__is_same(decltype(b), double));
}
