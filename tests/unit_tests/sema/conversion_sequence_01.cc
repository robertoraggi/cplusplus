// RUN: %cxx -verify -fcheck %s

struct Wrapper {
  Wrapper(int x) {}
};

void take_wrapper(Wrapper w) {}

void test_converting_ctor() {
  Wrapper w1 = 42;
  Wrapper w2(42);
  take_wrapper(42);
}

struct FromDouble {
  FromDouble(double d) {}
};

void test_double_ctor() {
  FromDouble f = 3.14;
  FromDouble f2(2.71);
}

struct IntLike {
  operator int() { return 0; }
};

void take_int(int x) {}

void test_conversion_function() {
  IntLike il;
  int x = il;
  take_int(il);
}

struct Explicit {
  explicit Explicit(int x) {}
};

void take_explicit(Explicit e) {}

void test_explicit() {
  Explicit e(42);
  // clang-format off
  // expected-error@+1 {{invalid argument of type 'int' for parameter of type '::Explicit'}}
  take_explicit(42);
  // clang-format on
}
