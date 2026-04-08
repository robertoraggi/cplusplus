// RUN: %cxx -verify %s
// expected-no-diagnostics

struct L {};
struct CL {};
struct V {};
struct CV {};

L overloaded(int&);
CL overloaded(const int&);

void test_ref_cv_overload_resolution() {
  int x = 0;
  const int cx = 0;

  static_assert(__is_same(decltype(overloaded(x)), L));
  static_assert(__is_same(decltype(overloaded(cx)), CL));
}

template <typename T>
const T& as_const(const T& t) {
  return t;
}

void test_template_const_ref_overload() {
  int i = 42;
  double d = 3.14;

  auto a = as_const(i);
  auto b = as_const(d);

  static_assert(__is_same(decltype(a), int));
  static_assert(__is_same(decltype(b), double));
}
