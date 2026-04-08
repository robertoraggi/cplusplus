// RUN: %cxx -verify %s
// expected-no-diagnostics

struct A {
  int x;
  A() : x(0) {}
  A(int a) : x(a) {}
  A(int a, int b) : x(a + b) {}
};

void test_ctor_overload() {
  A a1;
  A a2(10);
  A a3(10, 20);
  static_assert(__is_same(decltype(a1), A));
  static_assert(__is_same(decltype(a2), A));
  static_assert(__is_same(decltype(a3), A));
}

struct B {
  int x;
  B(int a, int b = 0) : x(a + b) {}
};

void test_ctor_default_arg() {
  B b1(5);
  B b2(5, 10);
  static_assert(__is_same(decltype(b1), B));
  static_assert(__is_same(decltype(b2), B));
}
