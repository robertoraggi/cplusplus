// RUN: %cxx -verify %s
// expected-no-diagnostics

int f_exact(int x) { return x; }
int f_variadic(int x, ...) { return x + 1; }

void test_prefer_nonvariadic() {
  auto a = f_exact(1);
  static_assert(__is_same(decltype(a), int));
}

void test_variadic_accepts_extra() {
  auto b = f_variadic(1, 2, 3);
  static_assert(__is_same(decltype(b), int));
}
